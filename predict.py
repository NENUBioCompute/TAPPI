import argparse
import torch
from tappi.model import TAPPI
from tappi.util import tappi_predict

# ==============================
# Configuration
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_params_tappi.pth"


# ==============================
# Helper: build mutation sequence
# ==============================
def build_mutation_sequence(wt_seq, mutation_str):
    """
    mutation_str format: e.g., Y5R
    Y: WT amino acid
    5: position (1-based)
    R: mutated amino acid
    """
    wt_aa = mutation_str[0]
    pos = int(mutation_str[1:-1]) - 1  # convert to 0-based index
    mut_aa = mutation_str[-1]

    # Sanity check
    assert wt_seq[pos] == wt_aa, f"WT amino acid mismatch at position {pos+1}"

    # Build mutated sequence
    mut_seq = wt_seq[:pos] + mut_aa + wt_seq[pos+1:]
    return mut_seq, pos+1  # return 1-based position for TAPPI


# ==============================
# Parse command-line arguments
# ==============================
parser = argparse.ArgumentParser(description="TAPPI prediction CLI")
parser.add_argument("--seq", required=True, help="Wild-type sequence")
parser.add_argument("--mutation", required=True, help="Mutation in format Y5R")
parser.add_argument("--partner", required=True, help="Partner protein sequence")
args = parser.parse_args()

wt_sequence = args.seq
mutation_str = args.mutation
partner_sequence = args.partner

# ==============================
# Prepare mutation
# ==============================
mut_sequence, mut_position = build_mutation_sequence(wt_sequence, mutation_str)

# ==============================
# Load model
# ==============================
model = TAPPI()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Using device: {DEVICE}")

# ==============================
# Run prediction
# ==============================
predicted, logits, tensors, attn_backbone, attn_neck = tappi_predict(
    [mut_position],
    [wt_sequence],
    [mut_sequence],
    [partner_sequence],
    model
)

# ==============================
# Output
# ==============================
print("\nWild-type sequence:", wt_sequence)
print("Mutated sequence  :", mut_sequence)
print("Mutation position :", mut_position)
print("\nPrediction:", ['Disrupting', 'Decreasing', 'No Effect', 'Increasing'][predicted.item()])
print("Logits:", logits)