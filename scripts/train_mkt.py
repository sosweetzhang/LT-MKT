# coding: utf-8
# 2024/11/25 @ shucunwang

import logging
import torch
import torch.nn as nn

import numpy as np
from sklearn import metrics
from tqdm import tqdm

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class Ktm():
    def __init__(self, model, n_q):
        super().__init__()
        self.model = model
        self.n_q = n_q

    def train_epoch(self, epoch, optim, loss_fc, train_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.train()

        for (kc, r, q, qd, kb) in tqdm(train_loader, "Epoch %s" % epoch):
            kc, r, q, qd, kb = (
                kc.to(device),
                r.to(device),
                q.to(device),
                qd.to(device),
                kb.to(device),
            )
            mask = r[:, 1:] >= 0
            mask.to(device)

            pred = self.model(kc, r, q, qd, kb) # (bs, seq_len)
            f_pred = torch.masked_select(pred[:, 1:], mask)
            f_target = torch.masked_select(r[:, 1:], mask)
            loss = loss_fc(f_pred, f_target)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss.append(loss.item())

            pred = f_pred.detach().cpu().numpy()
            target = f_target.detach().cpu().numpy()
            
            pred_list.append(pred)
            target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = metrics.roc_auc_score(all_target, all_pred)
        rmse = metrics.mean_squared_error(all_target, all_pred, squared=False)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy, rmse

    def eval_epoch(self, epoch, loss_fc, test_loader):

        pred_list = []
        target_list = []
        epoch_loss = []

        self.model.eval()

        with torch.no_grad():
            for (kc, r, q, qd, kb) in tqdm(test_loader, "Epoch %s" % epoch):
                kc, r, q, qd, kb = (
                    kc.to(device),
                    r.to(device),
                    q.to(device),
                    qd.to(device),
                    kb.to(device),
                )
                mask = r[:, 1:] >= 0
                mask.to(device)

                pred = self.model(kc, r, q, qd, kb) # (bs, 200)
                f_pred = torch.masked_select(pred[:, 1:], mask)
                f_target = torch.masked_select(r[:, 1:], mask)
                loss = loss_fc(f_pred, f_target)

                epoch_loss.append(loss.item())

                pred = f_pred.detach().cpu().numpy()
                target = f_target.detach().cpu().numpy()
                
                pred_list.append(pred)
                target_list.append(target)

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        auc = metrics.roc_auc_score(all_target, all_pred)
        rmse = metrics.mean_squared_error(all_target, all_pred, squared=False)
        all_pred[all_pred >= 0.5] = 1.0
        all_pred[all_pred < 0.5] = 0.0
        accuracy = metrics.accuracy_score(all_target, all_pred)

        return np.mean(epoch_loss), auc, accuracy, rmse

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)