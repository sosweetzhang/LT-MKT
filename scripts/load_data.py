# Code reused from https://github.com/arghosh/AKT.git


import numpy as np
import math

class Data(object):
    def __init__(self, n_question, seqlen, separate_char):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen # seqlen = 200

        # 9
        # 57,55,10,73,79,84,85,96,95
        # 1,1,1,0,0,0,1,0,1
        # 60,58,71,81,88,98,99,119,117
        # 0,28,3,44,48,23,8,49,14
        # 0,9,14,5,5,5,10,7,8
    def load_data(self, path):
        f_data = open(path , 'r')
        q_data = []
        qa_data = []
        p_data = []
        qd_data = []
        kb_data = []
        for line_id, line in enumerate(f_data):
            line = line.strip( )
            # line_id starts from 0
            if line_id % 6 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print(len(Q))
            elif line_id % 6 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print(len(A),A)
            elif line_id % 6 == 3:
                P = line.split(self.separate_char)
                if len( P[len(P)-1] ) == 0:
                    P = P[:-1]
            elif line_id % 6 == 4:
                QD = line.split(self.separate_char)
                if len( QD[len(QD)-1] ) == 0:
                    QD = QD[:-1]           
            elif line_id % 6 == 5:
                KB = line.split(self.separate_char)
                if len( KB[len(KB)-1] ) == 0:
                    KB = KB[:-1]

                # start split the data
                n_split = 1
                #print('len(Q):',len(Q))
                if len(Q) > self.seqlen:
                    n_split = math.floor(len(Q) / self.seqlen)
                    if len(Q) % self.seqlen:
                        n_split = n_split + 1
                #print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    problem_sequence = []
                    question_difficulty = []
                    knowledge_burden_sequence = []
                    if k == n_split - 1:
                        end_index  = len(A)
                    else:
                        end_index = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, end_index):
                        if len(Q[i]) > 0 :
                            # int(A[i]) is in {0,1}
                            Xindex = int(A[i])
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(Xindex)
                            problem_sequence.append(int(P[i]))
                            question_difficulty.append(int(QD[i]))
                            knowledge_burden_sequence.append(int(KB[i]))

                        else:
                            print(Q[i])
                    #print('instance:-->', len(instance),instance)
                    q_data.append(question_sequence)
                    qa_data.append(answer_sequence)
                    p_data.append(problem_sequence)
                    qd_data.append(question_difficulty)
                    kb_data.append(knowledge_burden_sequence)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_data_array = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_data_array[j, :len(dat)] = dat

        qa_data_array = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_data_array[j, :len(dat)] = dat

        p_data_array = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_data_array[j, :len(dat)] = dat

        qd_data_array = np.zeros((len(qd_data), self.seqlen))
        for j in range(len(qd_data)):
            dat = qd_data[j]
            qd_data_array[j, :len(dat)] = dat

        kb_data_array = np.zeros((len(kb_data), self.seqlen))
        for j in range(len(kb_data)):
            dat = kb_data[j]
            kb_data_array[j, :len(dat)] = dat
        # data_array: [ array([[],[],..])] Shape: (3633, 200)

        ans = []
        for i in range(len(q_data)):
            ans.append((q_data_array[i, :], qa_data_array[i, :], p_data_array[i, :], qd_data_array[i, :], kb_data_array[i, :]))
        return ans
