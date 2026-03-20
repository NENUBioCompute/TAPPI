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
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    
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

# position_embedding = positional_encoding(
#     4 * LEN_RANGE + 2,
#     1280
# ).to(DEVICE)

len_range = 10
position_embedding = positional_encoding(2 * (2 * len_range + 1) + 2 , 1280).numpy()

def pad_features(feature_list):
    max_len = max(f.shape[0] for f in feature_list)
    dim = feature_list[0].shape[1]
    padded, mask = [], []

    for f in feature_list:
        pad_len = max_len - f.shape[0]
        padded_f = np.pad(f, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        padded.append(padded_f)

        m = np.concatenate([np.zeros(f.shape[0]), np.ones(pad_len)]).astype(bool)
        mask.append(m)

    padded_tensor = torch.tensor(np.stack(padded), dtype=torch.float32)
    mask_tensor = torch.tensor(np.stack(mask), dtype=torch.bool)
    return padded_tensor, mask_tensor

def model_predict(mut0, mut1, par0, position_0, position_1, model, esm_model = esm_model, position_embedding = position_embedding, device = DEVICE):
    embedding, mask = esm_model.encode([mut0])
    mut0_feature = embedding.detach().cpu().squeeze(0).numpy()
    embedding, mask = esm_model.encode([mut1])
    mut1_feature = embedding.detach().cpu().squeeze(0).numpy()
    embedding, mask = esm_model.encode([par0])
    par0_feature = embedding.detach().cpu().squeeze(0).numpy()

    parts_mut0 = []
    parts_mut1 = []
    parts_mut0.append(np.expand_dims(mut0_feature.mean(axis=0), axis = 0) + position_embedding[:1, :])
    parts_mut1.append(np.expand_dims(mut1_feature.mean(axis=0), axis = 0) + position_embedding[2 * len_range + 2 : 2 * len_range + 3, :])
    for j in range(len(position_0)):
        # if sample['mut1'][sample['positions_mut1'][j][0] : sample['positions_mut1'][j][1]] != sample['Resulting sequence'][j]
        
        center = int((position_0[j][0] + position_0[j][1]) / 2)

        # start = max(center - len_range, 0)
        start = max(center - len_range + 1, 0)
        part_mut0 = mut0_feature[start: center + len_range + 1 + 1, :]
        if start == 0:
            positions_ed = position_embedding[1 : 2 * len_range + 1 + 1, : ][-part_mut0.shape[0] : , :]
        else:
            positions_ed = position_embedding[1 : 2 * len_range + 1 + 1, : ][ : part_mut0.shape[0] , :]

        parts_mut0.append(part_mut0 + positions_ed)

        center = int((position_1[j][0] + position_1[j][1]) / 2)

        # start = max(center - len_range, 0)
        start = max(center - len_range + 1, 0)
        part_mut1 = mut1_feature[start: center + len_range + 1 + 1, :]
        if start == 0:
            positions_ed = position_embedding[3 + 2 * len_range : , : ][-part_mut1.shape[0] : , :]
        else:
            positions_ed = position_embedding[3 + 2 * len_range : , : ][ : part_mut1.shape[0] , :]
        parts_mut1.append(part_mut1 + positions_ed)

    # result = np.concatenate(all_arrays, axis=0)
    # mut0_f_list.append(np.concatenate(parts_mut0, axis=0))
    # mut1_f_list.append(np.concatenate(parts_mut1, axis=0))
    # par0_f_list.append(par0_feature)
    mut0_feature = np.concatenate(parts_mut0, axis=0)
    mut1_feature = np.concatenate(parts_mut1, axis=0)
    par0_feature = par0_feature
    mut0, mut0_mask = pad_features([mut0_feature])
    mut1, mut1_mask = pad_features([mut1_feature])
    par0, par0_mask = pad_features([par0_feature])
    mut0 = mut0.to(device)
    mut0_mask = mut0_mask.to(device)

    mut1 = mut1.to(device)
    mut1_mask = mut1_mask.to(device)

    par0 = par0.to(device)
    par0_mask = par0_mask.to(device)
    # print(mut0.shape, mut1.shape, par0.shape, torch.cat([mut0_mask, mut1_mask], dim=1).shape, torch.cat([mut0_mask, mut1_mask], dim=1).dtype)
    preds, _, _, _ = model(mut0, mut1, par0, torch.cat([mut0_mask, mut1_mask], dim=1), par0_mask, None, None)
    return preds.detach().cpu().numpy()



