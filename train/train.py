import math
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from tappi.model import TAPPI
from tappi.util import tappi_forward, PandasDataReader, GHMC_Loss


DATA_PATH = "data/processed/mutations_correct.dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 7
EPOCHS = 20
LEN_HALF_PROTEIN = 10

MODEL_SAVE_PATH = "model_params_mirror_multi.pth"
BEST_MODEL_PATH = "model_best_val.pth"

print("Using device:", DEVICE)


df = pd.read_pickle(DATA_PATH)


new_df1 = df[df["label"] == 1].copy()
new_df1["mut0"], new_df1["mut1"] = new_df1["mut1"], new_df1["mut0"]
new_df1["label"] = 3

new_df2 = df[df["label"] == 3].copy()
new_df2["mut0"], new_df2["mut1"] = new_df2["mut1"], new_df2["mut0"]
new_df2["label"] = 1

new_df3 = df[df["label"] == 2].copy()
new_df3["mut0"], new_df3["mut1"] = new_df3["mut1"], new_df3["mut0"]

df = pd.concat([df, new_df1, new_df2, new_df3], ignore_index=True)


df["position_total"] = df["Feature range(s)"].apply(
    lambda x: sorted(
        set(
            [int(y.split("-")[0]) for y in x]
            + [int(y.split("-")[1]) for y in x]
        )
    )
)

df["position"] = df["position_total"].apply(
    lambda x: math.ceil((min(x) + max(x)) / 2)
)

df = df[df["mut0"] != df["mut1"]]

print("Dataset size:", df.shape)


df = df[df["position_total"].apply(
    lambda x: max(x) - min(x) < 2 * LEN_HALF_PROTEIN
)]

df = df[df["mut0"].str.len() <= 1500]
df = df[df["par0"].str.len() <= 1000]
df = df[df["label"] != 4]

print("Filtered dataset:", df.shape)


train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=np.ravel(train_df["label"])
)


model = TAPPI().to(DEVICE)

criterion = GHMC_Loss(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-6
)


train_acc_history = []
val_acc_history = []
test_acc_history = []

best_val_acc = 0


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    data_reader = PandasDataReader(
        train_df,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    label_true = []
    label_pred = []

    for batch in tqdm(data_reader):

        logits = tappi_forward(
            batch["position"].tolist(),
            batch["mut0"].tolist(),
            batch["mut1"].tolist(),
            batch["par0"].tolist(),
            model
        )

        loss = criterion(
            batch["label"].tolist(),
            logits,
            class_weights
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits, 1)

        label_true += batch["label"].tolist()
        label_pred += predicted.tolist()

    train_acc = accuracy_score(label_true, label_pred)

    train_acc_history.append(train_acc)

    print("Train Accuracy:", train_acc)


    model.eval()

    label_true = []
    label_pred = []

    data_reader = PandasDataReader(
        val_df,
        batch_size=10,
        shuffle=False
    )

    with torch.no_grad():

        for batch in data_reader:

            logits = tappi_forward(
                batch["position"].tolist(),
                batch["mut0"].tolist(),
                batch["mut1"].tolist(),
                batch["par0"].tolist(),
                model
            )

            _, predicted = torch.max(logits, 1)

            label_pred += predicted.tolist()
            label_true += batch["label"].tolist()

    val_acc = accuracy_score(label_true, label_pred)

    val_acc_history.append(val_acc)

    print("Val Accuracy:", val_acc)


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print("Saved best model")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    with open("train_result.json", "w") as f:
        json.dump(train_acc_history, f)

    with open("val_result.json", "w") as f:
        json.dump(val_acc_history, f)


print("\nLoading best model for test")

model.load_state_dict(torch.load(BEST_MODEL_PATH))

model.eval()

label_true = []
label_pred = []

data_reader = PandasDataReader(
    test_df,
    batch_size=10,
    shuffle=False
)

with torch.no_grad():

    for batch in data_reader:

        logits = tappi_forward(
            batch["position"].tolist(),
            batch["mut0"].tolist(),
            batch["mut1"].tolist(),
            batch["par0"].tolist(),
            model
        )

        _, predicted = torch.max(logits, 1)

        label_pred += predicted.tolist()
        label_true += batch["label"].tolist()

test_acc = accuracy_score(label_true, label_pred)

print("Final Test Accuracy:", test_acc)

with open("test_result.json", "w") as f:
    json.dump({"test_acc": test_acc}, f)

