import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.set_printoptions(threshold=100)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Xavier initialization of weights
        self.W = nn.Parameter(torch.zeros((in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros((2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, states, kc_input, adj):

        # Linear project: concepts_embeddings --> concepts_space
        h = torch.matmul(states, self.W)
        batch_size = h.shape[0]
        N = h.size()[1]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(batch_size, N, -1, 2 * self.out_features)

        # project dim=2 to scalar get attention matrix: (N, N, 2*out_features) --> (N, N)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) 

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=2)
        # print(attention[1, 2, :])
        # Randomly zero tensor
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # (bs, n_kc, h_dim)
        h_prime_ = torch.matmul(attention, h)
        extracted_rows = adj[kc_input] # (bs, n_kc)
        mask = extracted_rows.unsqueeze(2).repeat(1, 1, self.in_features)
        h_prime = torch.where(mask > 0, h_prime_, states)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MKT(nn.Module):

    def __init__(self, n_q, n_kc, n_kb, q_dim, lambda_p, graph, hidden_num=90, response_dim=50, kb_dim=10, dropout=0.2):
        super().__init__()
        self.n_q = n_q
        self.n_kc = n_kc
        self.n_kb = n_kb
        self.q_dim = q_dim
        self.lambda_p = lambda_p
        self.graph = graph
        self.h_dim = hidden_num
        self.r_dim = response_dim
        self.kb_dim = kb_dim
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

        # Embeddings of questions
        self.q_embed = nn.Embedding(self.n_q + 1, self.q_dim)
        self.qd_embed = nn.Embedding(self.lambda_p + 1, self.kb_dim)
        self.kb_embed = nn.Embedding(self.n_kb + 1, self.kb_dim)

        # Temproal effect by gru
        self.Temporal_effect = nn.GRUCell(self.q_dim + self.r_dim + 2*self.kb_dim, self.h_dim)

        # Two gatlayer
        self.GAT_layer1 = GATLayer(self.h_dim, self.h_dim)
        self.GAT_layer2 = GATLayer(self.h_dim, self.h_dim)

        # Prediction layer
        self.output_layer = nn.Linear(self.q_dim + self.h_dim + 2*self.kb_dim, self.h_dim)

    def forward(self, kc, r, q, qd, kb, states=None):
        device = q.device
        batch_size, seq_len = q.shape

        q_embed_data = self.q_embed(q)
        qd_embed_data =self.qd_embed(qd)
        kb_embed_data =self.kb_embed(kb)
        r_data = r.view(-1, 1).repeat(1, self.r_dim).view(batch_size, -1, self.r_dim)

        if states is None:
            # Initial knowledge state (bs, n_kc, h_dim)
            states = nn.init.xavier_uniform_(torch.zeros(batch_size, self.n_kc + 1, self.h_dim)).to(device)
        # print("original", states)

        pred = torch.zeros(batch_size, seq_len).to(device)
        for i in range(seq_len - 1):
            
            kc_i = kc[:, i] # Knowledge concept: (bs,)
            r_i = r_data[:, i] # Response: (bs, r_dim)
            q_embed_i = q_embed_data[:, i] # Question: (bs, q_dim)
            qd_embed_i = qd_embed_data[:, i] # Question difficulty: (bs, q_dim)
            kb_embed_i = kb_embed_data[:, i] # Knowledge burden: (bs, h_dim)

            cl_embed_i = torch.cat((qd_embed_i, kb_embed_i), dim=1)
            # Self-influence
            self_state = states.gather(1, kc_i.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.h_dim))
            # (bs, h_dim)
            self_state = self_state.squeeze(1) 

            # Temporal Effect
            # (bs, h_dim)
            next_self_state = self.dropout(self.Temporal_effect(torch.cat((q_embed_i, r_i, cl_embed_i), dim=1), self_state))
            new_states = states.clone()
            new_states[torch.arange(batch_size), kc_i] = next_self_state
            states = new_states
            # print("tempro_effect", states)

            # Successor Influence
            successor_graph = torch.tensor(self.graph['array1']).to(device)
            states = self.GAT_layer1(states, kc_i, successor_graph)
            # print("successor influence", states)

            # Similarity Influence
            similarity_graph = torch.tensor(self.graph['array2']).to(device)
            states = self.GAT_layer2(states, kc_i, similarity_graph)
            # print("similarity influence", states)

            # Prediction Module
            next_state = states.gather(1, kc[:, i+1].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.h_dim))
            next_state = next_state.squeeze(1)
            
            y = self.sig(self.output_layer(torch.cat((q_embed_data[:, i+1], qd_embed_data[:, i+1], kb_embed_data[:, i+1], next_state), 1))).sum(1) / self.h_dim
            pred[:, i + 1] = y

        return pred


