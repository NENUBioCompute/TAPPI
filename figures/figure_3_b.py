import requests
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Step 1: Download the JSON file
url = "https://figshare.com/ndownloader/files/63423897"
local_file = "connection_intermediate.json"

response = requests.get(url)
response.raise_for_status()  # Raise an error if download fails
with open(local_file, "wb") as f:
    f.write(response.content)

print(f"Downloaded {local_file} from Figshare.")

# Step 2: Load the saved data
with open(local_file, "r") as f:
    data = json.load(f)

pred_attns = data["pred_attns"]
labels_connect = data["labels_connect"]
connect = data["connect"]
unconnect = data["unconnect"]

# Step 3: Compute list_connection and list_unconnection
list_connection = []
list_unconnection = []

for part_weight in [2, 4, 8, 16, 32, 64]:
    connect_b = []
    unconnect_b = []
    for num in range(len(pred_attns)):
        attn = pred_attns[num]
        label = labels_connect[num]
        max_attn = max(attn)
        for i in range(len(attn)):
            if attn[i] < max_attn / part_weight:
                continue
            if i in label:
                connect_b.append(attn[i])
            else:
                unconnect_b.append(attn[i])
    list_connection.append(len(connect_b) / len(connect))
    list_unconnection.append(len(unconnect_b) / len(unconnect))

print("list_connection:", list_connection)
print("list_unconnection:", list_unconnection)

# Step 4: Plot the results
def to_percent(y, position):
    return f'{y * 100:.0f}%'

x = range(len(list_connection))
plt.figure(figsize=(4, 4))
plt.plot(x, list_connection, marker='o', color='#0E8585', label='In connect area')
plt.plot(x, list_unconnection, marker='s', color='#7E4909', label='In unconnect area')

plt.xlabel('Magnitude Threshold of Candidate Positions')
plt.ylabel('Proportion of Important Amino Acids')
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.legend()
plt.tight_layout()
plt.show()
