import argparse
import torch
import numpy as np

from tappi.model import TAPPI
from tappi.util import model_predict

# ==============================
# Config
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_params_tappi_loss.pth"


# ==============================
# Build mutation + region
# ==============================
def build_input_from_mutations(wt_seq, mutation_list):
    seq = list(wt_seq)
    positions = []

    for mut in mutation_list:
        wt_aa = mut[0]
        pos = int(mut[1:-1]) - 1
        mut_aa = mut[-1]

        assert seq[pos] == wt_aa, f"Mismatch at position {pos+1}"

        seq[pos] = mut_aa
        positions.append(pos)

    min_pos = min(positions)
    max_pos = max(positions)

    if max_pos - min_pos > 20:
        raise ValueError(
            f"Mutation span too large: {min_pos+1}-{max_pos+1}"
        )

    region = [(min_pos, max_pos + 1)]

    mut_seq = "".join(seq)

    return mut_seq, region


# ==============================
# Load model
# ==============================
def load_model():
    model = TAPPI()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAPPI prediction (model_predict version)")

    parser.add_argument("--seq", required=True, help="Wild-type sequence")

    parser.add_argument(
        "--mutation",
        nargs="+",
        required=True,
        help="Mutations, e.g., Y5R I6A G8Y"
    )

    parser.add_argument("--partner", required=True, help="Partner sequence")

    args = parser.parse_args()

    wt_seq = args.seq
    mutation_list = args.mutation
    partner_seq = args.partner

    mut_seq, region = build_input_from_mutations(wt_seq, mutation_list)

    mut0 = wt_seq
    mut1 = mut_seq
    par0 = partner_seq

    positions_mut0 = region
    positions_mut1 = region

    # ===== load model =====
    model = load_model()
    print(f"\nUsing device: {DEVICE}")

    # ===== prediction =====
    with torch.no_grad():
        logits = model_predict(
            mut0,
            mut1,
            par0,
            positions_mut0,
            positions_mut1,
            model=model
        )

    label_map = ['Disrupting', 'Decreasing', 'No Effect', 'Increasing']

    logits_tensor = torch.tensor(logits)

    if logits_tensor.dim() > 1:
        logits_tensor = logits_tensor.squeeze(0)

    probs = torch.softmax(logits_tensor, dim=0)
    pred_idx = torch.argmax(probs).item()

    print("\n===== OUTPUT =====")
    print("Logits       :", logits_tensor.tolist())
    print("Prediction   :", label_map[pred_idx])
