import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2, alpha=0.2):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, states, concept_ids, adjacency):
        h = self.proj(states)
        batch_size, n_concepts, hidden_dim = h.shape
        src = h.unsqueeze(2).expand(batch_size, n_concepts, n_concepts, hidden_dim)
        dst = h.unsqueeze(1).expand(batch_size, n_concepts, n_concepts, hidden_dim)
        score = self.leaky_relu(self.attn(torch.cat([src, dst], dim=-1)).squeeze(-1))
        mask = adjacency.unsqueeze(0).expand(batch_size, -1, -1) > 0
        score = score.masked_fill(~mask, -1e9)
        weights = F.softmax(score, dim=-1)
        weights = self.dropout(weights)
        propagated = torch.matmul(weights, h)
        row_mask = adjacency[concept_ids].unsqueeze(-1) > 0
        updated = torch.where(row_mask, propagated, states)
        return F.elu(updated)


class LTMKT(nn.Module):
    def __init__(
        self,
        n_questions,
        n_concepts,
        n_question_difficulty,
        n_domain_transition,
        n_domain_coverage,
        question_dim=100,
        hidden_dim=90,
        response_dim=50,
        load_dim=10,
        dropout=0.2,
        question_text_embeddings=None,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.response_dim = response_dim
        if question_text_embeddings is not None:
            question_dim = question_text_embeddings.shape[1]
            self.question_embed = nn.Embedding.from_pretrained(question_text_embeddings, freeze=False, padding_idx=0)
        else:
            self.question_embed = nn.Embedding(n_questions + 1, question_dim, padding_idx=0)
        self.question_difficulty_embed = nn.Embedding(n_question_difficulty + 1, load_dim, padding_idx=0)
        self.domain_transition_embed = nn.Embedding(n_domain_transition + 1, load_dim, padding_idx=0)
        self.domain_coverage_embed = nn.Embedding(n_domain_coverage + 1, load_dim, padding_idx=0)
        self.state_evolution = nn.GRUCell(question_dim + response_dim + load_dim * 3, hidden_dim)
        self.intra_graph_layer = GraphAttentionLayer(hidden_dim, dropout=dropout)
        self.inter_graph_layer = GraphAttentionLayer(hidden_dim, dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(question_dim + hidden_dim + load_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        concept_ids,
        responses,
        question_ids,
        question_difficulty,
        domain_transition,
        domain_coverage,
        intra_graph,
        inter_graph,
        question_concepts=None,
        states=None,
    ):
        device = question_ids.device
        batch_size, seq_len = question_ids.shape
        if states is None:
            states = torch.zeros(batch_size, self.n_concepts + 1, self.hidden_dim, device=device)
            nn.init.xavier_uniform_(states[:, 1:])
        question_emb = self.question_embed(question_ids)
        difficulty_emb = self.question_difficulty_embed(question_difficulty)
        transition_emb = self.domain_transition_embed(domain_transition)
        coverage_emb = self.domain_coverage_embed(domain_coverage)
        load_emb = torch.cat([difficulty_emb, transition_emb, coverage_emb], dim=-1)
        response_emb = responses.unsqueeze(-1).repeat(1, 1, self.response_dim)
        predictions = torch.zeros(batch_size, seq_len, device=device)
        intra_graph = intra_graph.to(device)
        inter_graph = inter_graph.to(device)
        batch_index = torch.arange(batch_size, device=device)
        for step in range(seq_len - 1):
            current_concepts = concept_ids[:, step].long()
            current_state = states[batch_index, current_concepts]
            current_input = torch.cat(
                [question_emb[:, step], response_emb[:, step], load_emb[:, step]],
                dim=-1,
            )
            next_state = self.state_evolution(current_input, current_state)
            states = states.clone()
            states[batch_index, current_concepts] = next_state
            states = self.intra_graph_layer(states, current_concepts, intra_graph)
            states = self.inter_graph_layer(states, current_concepts, inter_graph)
            fused_state = self.fuse_question_state(states, concept_ids[:, step + 1].long(), question_ids[:, step + 1].long(), question_concepts)
            next_input = torch.cat([question_emb[:, step + 1], load_emb[:, step + 1], fused_state], dim=-1)
            predictions[:, step + 1] = torch.sigmoid(self.predictor(next_input)).squeeze(-1)
        return predictions

    def fuse_question_state(self, states, primary_concepts, question_ids, question_concepts):
        batch_index = torch.arange(states.shape[0], device=states.device)
        if question_concepts is None:
            return states[batch_index, primary_concepts]
        related = question_concepts.to(states.device)[question_ids]
        fallback = primary_concepts.unsqueeze(1)
        related = torch.where(related > 0, related, fallback)
        gathered = states[batch_index.unsqueeze(1), related]
        mask = (related > 0).unsqueeze(-1)
        summed = (gathered * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
