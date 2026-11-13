# coding: utf-8
# 2024/11/25 @ shucun wang

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "../")
import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import json
import numpy as np

from load_data import Data
from lt_mkt.model import MKT
from data.dataset import MKTDataset
from train_mkt import Ktm

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 

def run(args):

    dat = Data(n_question=args.n_kc, seqlen=args.seq_len, separate_char=',') 
    train_data_path = "../data/" + args.dataset_name + "/train.txt"
    test_data_path = "../data/" + args.dataset_name + "/test.txt"

    # dataformat: tuple (q_data_array, qa_data_array)
    train_data = dat.load_data(train_data_path)
    test_data = dat.load_data(test_data_path)

    train = MKTDataset(train_data, args.n_kc)
    test = MKTDataset(test_data, args.n_kc)
    train_loader = DataLoader(train, args.bs, shuffle=True)
    test_loader = DataLoader(test, args.bs, shuffle=False)

    # read graph and q matrix
    graph = np.load(os.path.join("../data", args.dataset_name, args.dataset_name + "_graph.npz"))
    
    # log config
    logging.basicConfig(filename='../log/training_mkt_senior.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # model config
    model = MKT(args.n_q, args.n_kc, args.n_kb, args.q_dim, args.lambda_p, graph)
    model.to(device)
    loss_fc = F.binary_cross_entropy_with_logits
    optim = torch.optim.Adam(params = model.parameters(), 
                             lr=args.lr, eps=1e-8, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8)
    ktm = Ktm(model, args.n_kc)

    # start training
    best_valid_auc = 0.0
    for idx in range(args.epoch):
        loss, auc, acc, rmse = ktm.train_epoch(idx+1, optim, loss_fc, train_loader)
        print('Training --- loss : %3.5f, auc : %3.5f, accuracy : %3.5f, rmse : %3.5f' 
        % (loss, auc, acc, rmse))
        logging.info(f"Training--epoch[{str(idx+1)}]--loss: {loss}, auc: {auc}, acc: {acc}, rmse: {rmse}")

        loss, auc, acc, rmse = ktm.eval_epoch(idx+1, loss_fc, test_loader)
        print('Evaluating --- loss : %3.5f, auc : %3.5f, accuracy : %3.5f, rmse : %3.5f' 
        % (loss, auc, acc, rmse))
        logging.info(f"Evaluating--epoch[{str(idx+1)}]--loss: {loss}, auc: {auc}, acc: {acc}, rmse: {rmse}")

        if auc > best_valid_auc:
            print('valid auc improve: %3.4f to %3.4f' % (best_valid_auc, auc))
            best_valid_auc = auc

        # scheduler.step()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train mkt model")

    # model args
    parser.add_argument("--n_q", type=int, default=170)
    parser.add_argument("--n_kc", type=int, default=139)
    parser.add_argument("--n_kb", type=int, default=50) # senior 50
    parser.add_argument("--q_dim", type=int, default=100)
    parser.add_argument("--h_dim", type=int, default=90)
    parser.add_argument("-p", "--lambda_p", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)

    # train args
    parser.add_argument("--dataset_name", type=str, default="junior")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=25)

    args = parser.parse_args()
    run(args)