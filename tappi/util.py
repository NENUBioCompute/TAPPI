import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

from typing import Union
from functools import partial
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# Configuration
# =========================================================

device = 'cuda'
LEN_RANGE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GHMC_Loss(nn.Module):

    def __init__(self, device, bins=8, momentum=0.3):
        super().__init__()

        self.device = device
        self.bins = bins
        self.momentum = momentum

        self.edges = torch.linspace(0, 1, bins + 1)

        if momentum > 0:
            self.acc_sum = torch.zeros(bins)

    def forward(self, targets, logits, *_):
        """
        Parameters
        ----------
        targets : Tensor or array-like
            Class labels (N,)
        logits : Tensor
            Model logits (N, C)

        Returns
        -------
        loss : Tensor
        """

        if not torch.is_tensor(targets):
            print(targets)
            targets = torch.tensor(targets)

        targets = targets.to(self.device).long()

        num_classes = logits.shape[1]

        targets_onehot = F.one_hot(
            targets,
            num_classes=num_classes
        ).float()

        # Gradient magnitude
        g = torch.abs(
            logits.softmax(dim=1).detach() - targets_onehot
        )

        weights = torch.zeros_like(logits)

        total_elements = logits.numel()
        valid_bins = 0

        for i in range(self.bins):

            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])

            num_in_bin = inds.sum().item()

            if num_in_bin > 0:

                if self.momentum > 0:
                    self.acc_sum[i] = (
                        self.momentum * self.acc_sum[i]
                        + (1 - self.momentum) * num_in_bin
                    )
                    weights[inds] = total_elements / self.acc_sum[i]
                else:
                    weights[inds] = total_elements / num_in_bin

                valid_bins += 1

        if valid_bins > 0:
            weights = weights / valid_bins

        # Cross entropy
        loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        # Sample weights
        sample_weights = weights.max(dim=1)[0]

        loss = (loss * sample_weights).sum() / total_elements

        return loss

# =========================================================
# Utility Functions
# =========================================================

def print_progress_bar(iteration, total, length=40):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r|{bar}| {percent:.2f}% Complete")
    sys.stdout.flush()


def positional_encoding(max_len, d_model):

    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        -(np.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_len, d_model)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe.requires_grad = False

    return pe


# =========================================================
# ESM Feature Encoder
# =========================================================

class ESMModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch_tokens, repr_layers=[33], return_contacts=False):

        return self.model(
            batch_tokens,
            repr_layers=repr_layers,
            return_contacts=return_contacts
        )


class ESMFeatureEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.device = DEVICE

        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = self.alphabet.padding_idx

        self.model = ESMModelWrapper(self.model)

    def encode(self, sequences):

        batch_labels, batch_strs, batch_tokens = self.batch_converter(
            [(str(0), sequence) for sequence in sequences]
        )

        batch_tokens = batch_tokens.to(self.device)

        batch_mask = batch_tokens.eq(self.padding_idx)

        with torch.no_grad():

            results = self.model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False
            )

        token_representations = results["representations"][33]

        return token_representations, batch_mask


# =========================================================
# Pandas Data Reader
# =========================================================

class PandasDataReader:

    def __init__(self, df, batch_size=1, shuffle=False):

        self.df = df.sample(frac=1).reset_index(drop=True) if shuffle else df

        self.batch_size = batch_size
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.current_index >= len(self.df):
            raise StopIteration()

        batch = self.df.iloc[
            self.current_index:self.current_index + self.batch_size
        ]

        self.current_index += self.batch_size

        return batch.iloc[0] if self.batch_size == 1 else batch


# =========================================================
# TAPPI Prediction
# =========================================================

esm_model = ESMFeatureEncoder()

position_embedding = positional_encoding(
    4 * LEN_RANGE + 2,
    1280
).to(DEVICE)


def tappi_forward(positions, mut0, mut1, par0, model, device = DEVICE, esm_model = esm_model, position_embedding = position_embedding, len_range = LEN_RANGE):

    mut0,mut0_mask = esm_model.encode(mut0)
    # print(mut0_mask)
    mut1,mut1_mask = esm_model.encode(mut1)
    par0,par0_mask = esm_model.encode(par0)

    if mut0.shape[1] < 2*len_range:
        mut0 = torch.cat((mut0.mean(dim=1, keepdim=True),mut0),dim=1)
        mut0_mask = mut0_mask.cpu().numpy()
        mut0_mask = np.concatenate([np.full((mut0_mask.shape[0], 1), False), mut0_mask], axis=1)
        mut1 = torch.cat((mut1.mean(dim=1, keepdim=True),mut1),dim=1)
        mut1_mask = mut1_mask.cpu().numpy()
        mut1_mask = np.concatenate([np.full((mut1_mask.shape[0], 1), False), mut1_mask], axis=1)
    else:
        result = torch.randn(mut0.shape[0],2*len_range,mut0.shape[2])
        result_padding = (np.random.randint(0, 2, size=(mut0.shape[0], 2*len_range)) == 1)

        for ia in range(len(positions)):
            position = int(positions[ia])
            if position - len_range < 0 :
                result[ia, :, :] = mut0[ia,:2 * len_range,:]
                result_padding[ia, :] = mut0_mask[ia, : 2 * len_range].cpu()
            elif position + len_range > mut0.shape[1] :
                result[ia, :, :] = mut0[ia,-2 * len_range:,:]
                result_padding[ia, :] = mut0_mask[ia, -2 * len_range].cpu()
            else:
                result[ia, :, :] = mut0[ia,position - len_range : position + len_range,:]
                result_padding[ia, :] = mut0_mask[ia, position - len_range : position + len_range].cpu()

        mut0 = torch.cat((mut0.mean(dim=1, keepdim=True).cpu(), result),dim=1)
        mut0_mask = result_padding
        mut0_mask = np.concatenate([np.full((mut0_mask.shape[0], 1), False), mut0_mask], axis=1)

        result = torch.randn(mut1.shape[0],2 * len_range,mut1.shape[2])
        result_padding = (np.random.randint(0, 2, size=(mut1.shape[0], 2 * len_range)) == 1)
        for ia in range(len(positions)):
            position = int(positions[ia])
            if position - len_range < 0 :
                result[ia, :, :] = mut1[ia,:2 * len_range,:]
                result_padding[ia, :] = mut1_mask[ia , : 2 * len_range].cpu()
            elif position + len_range > mut1.shape[1] :
                result[ia, :, :] = mut1[ia,-2 * len_range : ,:]
                result_padding[ia, :] = mut1_mask[ia , -2 * len_range].cpu()
            else:

                result[ia, :, :] = mut1[ia,position - len_range : position + len_range , :]
                result_padding[ia, :] = mut1_mask[ia, position - len_range : position + len_range].cpu()
        mut1 = torch.cat((mut1.mean(dim=1, keepdim=True).cpu(), result),dim=1)
        mut1_mask = result_padding
        mut1_mask = np.concatenate([np.full((mut1_mask.shape[0], 1), False), mut1_mask], axis=1)
        
        
    mut0_mask = torch.from_numpy(mut0_mask)
    mut0_mask = mut0_mask.to(device)
    mut1_mask = torch.from_numpy(mut1_mask)
    mut1_mask = mut1_mask.to(device)
    mut0 = mut0.to(device)
    mut1 = mut1.to(device)
    par0 = par0.to(device)
    mut0 = mut0 + position_embedding[:mut0.shape[1] , : ]
    mut1 = mut1 + position_embedding[2 * len_range + 1 : 2 * len_range + 1 + mut1.shape[1] , : ]
    x, _, _, _ = model(mut0, mut1, par0,torch.cat((mut0_mask,mut1_mask),dim=1),par0_mask, None, None)

    return x

def tappi_predict(positions, mut0, mut1, par0, model, device = device, esm_model = esm_model, position_embedding = position_embedding, len_range = LEN_RANGE):

    mut0,mut0_mask = esm_model.encode(mut0)
    # print(mut0_mask)
    mut1,mut1_mask = esm_model.encode(mut1)
    par0,par0_mask = esm_model.encode(par0)

    if mut0.shape[1] < 2*len_range:
        mut0 = torch.cat((mut0.mean(dim=1, keepdim=True),mut0),dim=1)
        mut0_mask = mut0_mask.cpu().numpy()
        mut0_mask = np.concatenate([np.full((mut0_mask.shape[0], 1), False), mut0_mask], axis=1)
        mut1 = torch.cat((mut1.mean(dim=1, keepdim=True),mut1),dim=1)
        mut1_mask = mut1_mask.cpu().numpy()
        mut1_mask = np.concatenate([np.full((mut1_mask.shape[0], 1), False), mut1_mask], axis=1)
    else:
        result = torch.randn(mut0.shape[0],2*len_range,mut0.shape[2])
        result_padding = (np.random.randint(0, 2, size=(mut0.shape[0], 2*len_range)) == 1)

        for ia in range(len(positions)):
            position = int(positions[ia])
            if position - len_range < 0 :
                result[ia, :, :] = mut0[ia,:2 * len_range,:]
                result_padding[ia, :] = mut0_mask[ia, : 2 * len_range].cpu()
            elif position + len_range > mut0.shape[1] :
                result[ia, :, :] = mut0[ia,-2 * len_range:,:]
                result_padding[ia, :] = mut0_mask[ia, -2 * len_range].cpu()
            else:
                result[ia, :, :] = mut0[ia,position - len_range : position + len_range,:]
                result_padding[ia, :] = mut0_mask[ia, position - len_range : position + len_range].cpu()

        mut0 = torch.cat((mut0.mean(dim=1, keepdim=True).cpu(), result),dim=1)
        mut0_mask = result_padding
        mut0_mask = np.concatenate([np.full((mut0_mask.shape[0], 1), False), mut0_mask], axis=1)

        result = torch.randn(mut1.shape[0],2 * len_range,mut1.shape[2])
        result_padding = (np.random.randint(0, 2, size=(mut1.shape[0], 2 * len_range)) == 1)
        for ia in range(len(positions)):
            position = int(positions[ia])
            if position - len_range < 0 :
                result[ia, :, :] = mut1[ia,:2 * len_range,:]
                result_padding[ia, :] = mut1_mask[ia , : 2 * len_range].cpu()
            elif position + len_range > mut1.shape[1] :
                result[ia, :, :] = mut1[ia,-2 * len_range : ,:]
                result_padding[ia, :] = mut1_mask[ia , -2 * len_range].cpu()
            else:

                result[ia, :, :] = mut1[ia,position - len_range : position + len_range , :]
                result_padding[ia, :] = mut1_mask[ia, position - len_range : position + len_range].cpu()
        mut1 = torch.cat((mut1.mean(dim=1, keepdim=True).cpu(), result),dim=1)
        mut1_mask = result_padding
        mut1_mask = np.concatenate([np.full((mut1_mask.shape[0], 1), False), mut1_mask], axis=1)
        
        
    mut0_mask = torch.from_numpy(mut0_mask)
    mut0_mask = mut0_mask.to(device)
    mut1_mask = torch.from_numpy(mut1_mask)
    mut1_mask = mut1_mask.to(device)
    mut0 = mut0.to(device)
    mut1 = mut1.to(device)
    par0 = par0.to(device)
    mut0 = mut0 + position_embedding[:mut0.shape[1] , : ]
    mut1 = mut1 + position_embedding[2 * len_range + 1 : 2 * len_range + 1 + mut1.shape[1] , : ]
    x, tensors, attn_backbone, attn_neck = model(mut0, mut1, par0,torch.cat((mut0_mask,mut1_mask),dim=1),par0_mask, None, None)
    _, predicted = torch.max(x, 1)
    return predicted, x, tensors, attn_backbone, attn_neck


