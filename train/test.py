import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)

from tappi.model import TAPPI
from tappi.util import tappi_predict, PandasDataReader, print_progress_bar

# ==============================
# Configuration
# ==============================

BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DATA_PATH = "tappi_test_fold.csv"
MODEL_PATH = "model_params_tappi.pth"


# ==============================
# Load Model
# ==============================

print(f"Using device: {DEVICE}")

model = TAPPI()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()


# ==============================
# Load Dataset
# ==============================

x_test_fold = pd.read_csv(TEST_DATA_PATH)

data_reader = PandasDataReader(
    x_test_fold,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==============================
# Inference
# ==============================

labels = []
predictions = []
logits_list = []
tensor_outputs = []

for batch in data_reader:

    predicted_dif, e, ts, attn_backbone, attn_neck = tappi_predict(
        [batch["position"]],
        [batch["mut0"]],
        [batch["mut1"]],
        [batch["par0"]],
        model
    )

    labels.append(batch["label"])

    predictions.extend(predicted_dif.tolist())

    logits_list.append(e.detach().cpu())

    tensor_outputs.append(ts.detach().cpu())

    print_progress_bar(len(predictions), x_test_fold.shape[0])


# ==============================
# Prepare Evaluation Data
# ==============================

y_true = np.array(labels)
y_pred = np.array(predictions)

y_logits = torch.cat(logits_list, dim=0).numpy()
tensors = torch.cat(tensor_outputs, dim=0)

num_classes = y_logits.shape[1]

y_true_onehot = np.eye(num_classes)[y_true]


# ==============================
# Metrics
# ==============================

acc = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

per_class_precision = precision_score(
    y_true,
    y_pred,
    average=None,
    labels=range(num_classes)
)

per_class_recall = recall_score(
    y_true,
    y_pred,
    average=None,
    labels=range(num_classes)
)

per_class_f1 = f1_score(
    y_true,
    y_pred,
    average=None,
    labels=range(num_classes)
)


# ==============================
# AUC / AUPR
# ==============================

per_class_auc = []
per_class_aupr = []

for i in range(num_classes):

    auc = roc_auc_score(
        y_true_onehot[:, i],
        y_logits[:, i]
    )

    aupr = average_precision_score(
        y_true_onehot[:, i],
        y_logits[:, i]
    )

    per_class_auc.append(auc)
    per_class_aupr.append(aupr)

per_class_auc = np.array(per_class_auc)
per_class_aupr = np.array(per_class_aupr)

macro_aupr = per_class_aupr.mean()

weighted_aupr = np.average(
    per_class_aupr,
    weights=np.bincount(y_true, minlength=num_classes)
)


# ==============================
# Print Results
# ==============================

print("\n===== Evaluation Results =====")

print("Accuracy:", acc)
print("MCC:", mcc)

print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)

print("Per-class Precision:", per_class_precision.tolist())
print("Per-class Recall:", per_class_recall.tolist())
print("Per-class F1:", per_class_f1.tolist())

print("Per-class AUC:", per_class_auc.tolist())
print("Per-class AUPR:", per_class_aupr.tolist())

print("Macro AUPR:", macro_aupr)
print("Weighted AUPR:", weighted_aupr)