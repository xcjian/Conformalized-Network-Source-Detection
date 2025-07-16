# This is a draft showing how to train and test a Fully-connected NN.
# Please replace this Fully-connected NN with reasonable architecture.
# Please put your NN model in a folder parallel to SD-STGCN, e.g., SD-STGCN-torch.
# Also, store the output probs and ground truth on the test set in your NN folder.
# Format: a dictionary
"""
{
    'inputs': a list of # batch_size x n_nodes input data. This is [X[i][:, 0] for i in test_idx], but arranged in batches. The first available snapshot of the time series (after skipping), before converting to features.
'predictions': a list of # batch_size x n_nodes x 2 output probs.  This is the model output, arranged in batches.
'ground_truth': a list of # batch_size x n_frame x n_nodes input data. This is the ground truth y, arranged in batches.
}
Please convert these quantities to numpy array if they are torch tensors.
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import networkx as nx

import pickle
import torch

from torchmetrics import Precision, Recall, F1Score


from base_models import GCN, GAT, GraphSAGE, GIN, MLP
from data.utils import convert_timeseries_to_features

sys.path.append(os.path.join(os.path.dirname(__file__), '../SD-STGCN/utils'))
from math_graph import weight_matrix


def multi_label_loss(y, logits, weight):
    """
    Computes a weighted binary cross-entropy loss for multi-label classification.

    Parameters:
    y (torch.Tensor): Ground truth labels of shape (batch_size, n_vertex), containing 0 or 1 values.
    logits (torch.Tensor): Predicted logits of shape (batch_size, n_vertex, 2).
    weight (float): Weight applied to positive class (y == 1).

    Returns:
    torch.Tensor: Scalar loss value averaged over batch and vertices.
    """
    # y: (batch_size, n_vertex), logits: (batch_size, n_vertex, 2)
    if y.shape[:2] != logits.shape[:2] or logits.shape[2] != 2:
        raise ValueError(f"Invalid shapes: y.shape={y.shape}, logits.shape={logits.shape}")
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1 values")
    
    # Extract positive class logits (class 1)
    true_logits = logits[:, :, 1]  # (batch_size, n_vertex)
    
    # Compute binary cross-entropy loss
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        true_logits, y.float(), reduction='none'
    )  # (batch_size, n_vertex)
    
    # Apply weights: weight for y == 1, 1.0 for y == 0
    weights = torch.where(
        y == 1,
        torch.tensor(weight, device=y.device),
        torch.tensor(1.0, device=y.device)
    )  # (batch_size, n_vertex)
    
    # Weighted loss
    weighted_loss = bce_loss * weights  # (batch_size, n_vertex)
    
    # Average over batch and vertices
    return weighted_loss.mean()

# Simple GCN model [Replace this with your model]
# Model
class SimpleGCN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16):
        super(SimpleGCN, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index):
        # x: (batch_size, n_vertex, input_dim)
        batch_size, n_vertex, feat_dim = x.size()
        x = x.reshape(-1, feat_dim)  # (batch_size * n_vertex, input_dim)
        h = self.gcn(x, edge_index)  # (batch_size * n_vertex, hidden_dim)
        h = self.relu(h)
        logits = self.fc(h)  # (batch_size * n_vertex, 2)
        probs = self.softmax(logits)  # (batch_size * n_vertex, 2), [p_0, p_1]
        probs = probs.view(batch_size, n_vertex, 2)  # (batch_size, n_vertex, 2)
        return probs
    
# Read-in the data

data_path = '../SD-STGCN/dataset/highSchool/data/SIR/'

## SI models
# data_file = 'SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_torchentire.pickle'
# data_file = 'SIR_nsrc7_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_torchentire.pickle'
# data_file = 'SIR_nsrc10_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_torchentire.pickle'
## SIR models
data_file = 'SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_torchentire.pickle'
# data_file = 'SIR_nsrc7_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_torchentire.pickle'

graph_path = '../SD-STGCN/dataset/highSchool/data/graph/'
graph_file = 'highSchool.edgelist'
print(f"Loading graph from {graph_path + graph_file}")

with open(data_path + data_file, 'rb') as f:
    data = pickle.load(f)

g = nx.read_edgelist(graph_path + graph_file)
# Map node IDs to consecutive integers (0 to 773)
node_mapping = {node: idx for idx, node in enumerate(g.nodes())}
num_nodes = len(node_mapping)  # Should be 774
assert num_nodes == 774, f"Expected 774 nodes, got {num_nodes}"
# Extract edges and weights
edges = [(node_mapping[src], node_mapping[dst]) for src, dst in g.edges()]
edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape: (2, num_edges)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add (dst, src)

# Data preparation
n_frame = 16
ts = data[0]  # List of 21200 tensors, each (3, 774)
print(f"Shape of X time series: {ts.shape}")  # Should be [21200, 30, 774]

skip = data[2]
first_snapshot = torch.stack([ts[i][skip[i][0], :].clone().detach().float() for i in range(len(ts))])
seen_snapshots = torch.stack([ts[i][skip[i][0]:skip[i][0] + n_frame, :].clone().detach().float() for i in range(len(ts))])
print(f"Shape of first snapshot: {first_snapshot.shape}")  # Should be [21200, 774]
print(f"Shape of seen snapshots: {seen_snapshots.shape}")  # Should be [21200, 16, 774]


# X = convert_timeseries_to_features(data, nf=16, normalize=True)  # Convert to features
# X = torch.stack([x.T.clone().detach().float() for x in X])  # Batch tensor conversion
X = torch.stack([x.T.clone().detach().float() for x in seen_snapshots])  # Batch tensor conversion
print(f"Shape of X after conversion: {X.shape}")  

y = data[1]  # Shape: (21200,774)
y = y.clone().detach().float()

# Train-test split
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=False)
# test set size: 21200 * 0.2 = 4240
# train set size: 21200 * 0.8 = 16960
X_train = [X[i] for i in train_idx]
y_train = y[train_idx]
X_test = [X[i] for i in test_idx]
first_T_test = first_snapshot[test_idx]  # First snapshot for test set
y_test = y[test_idx]
print(f"Shape of X_train: {torch.stack(X_train).shape}, X_test: {torch.stack(X_test).shape}")
print(f"Shape of y_train: {y_train.shape}, y_test: {y_test.shape}")
print(f"Shape of first snapshot test: {first_T_test.shape}")  # Should be [4240, 774]

# Custom dataset
class SIRDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def neighbor_penalty(edge_index, probs):
    """
    Compute a penalty to discourage neighboring nodes from both having high probabilities.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges), representing the graph.
        probs (torch.Tensor): Predicted probabilities for the positive class, shape (batch_size, n_vertex).

    Returns:
        torch.Tensor: The neighbor penalty term.
    """
    # Extract probabilities for the positive class
    probs = probs.reshape(-1)  # Flatten to (batch_size * n_vertex,)

    # Get probabilities for source and destination nodes of each edge
    src_probs = probs[edge_index[0]]  # Probabilities of source nodes
    dst_probs = probs[edge_index[1]]  # Probabilities of destination nodes

    # Compute penalty as the product of probabilities for each edge
    penalty = (src_probs * dst_probs).sum()  # Sum over all edges

    return penalty

def compute_graph_laplacian(edge_index, num_nodes):
    """
    Compute the graph Laplacian.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.Tensor: Graph Laplacian of shape (num_nodes, num_nodes).
    """
    # Create adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1

    # Compute degree matrix
    degree = torch.diag(adj.sum(dim=1))

    # Compute Laplacian
    laplacian = degree - adj
    return laplacian

# Data loaders
train_dataset = SIRDataset(X_train, y_train)
test_dataset = SIRDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SimpleGCN(input_dim=3, hidden_dim=64).to(device)
from torch_geometric.nn import aggr

model = GraphSAGE(
    in_dim=X.shape[-1],  # Input feature dimension based on X
    hidden_dim=128,  # Hidden layer dimension
    out_dim=2,  # Output feature dimension (number of classes)
    num_layers=4,  # Number of layers
    dropout=0.5,  # Dropout rate
    aggr='mean',  # Aggregation methods for each layer
    use_batchnorm=True,  # Use Batch Normalization
    #use_bias=True,  # Use bias in linear layers
    use_skip=True,  # Use skip connections
).to(device)
edge_index = edge_index.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
weight = 700.0 *7 # Weight for y_i = 1
num_epochs = 20

filtering = False  # Set to True to filter out predictions with 0. probabilities
laplacian = compute_graph_laplacian(edge_index, num_nodes)
# Training
print("Starting training...")
model.train()
pos_weight = torch.tensor(weight, device=device)  # Move pos_weight creation outside the loop
for epoch in range(num_epochs):
    total_loss = 0
    total_bce_loss = 0
    total_focal_loss = 0
    total_penalty = 0
    total_samples = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        #print(f"Input shape: {X_batch.shape}") # Should be [batch_size, 774, 3]
        #print(f"Edge index shape: {edge_index.shape}")# Should be [2, num_edges]
        
        # Forward pass
        logits = model(X_batch.reshape(-1, X_batch.size(-1)), edge_index)  # Reshape input
        logits = logits.view(X_batch.size(0), X_batch.size(1), -1)  # Reshape output to (batch_size, n_vertex, 2)
        
        # Extract positive class logits (class 1)
        true_logits = logits[:, :, 1]  # (batch_size, n_vertex)
        # add a high-pass fileter to the logits with graph laplacian
        true_logits = torch.matmul(laplacian, true_logits.T).T  # Apply Laplacian (high-pass filter)

        
        if filtering:
            # Apply filtering: set logits to a very negative value where first_snapshot == 0
            first_snapshot_batch = X_batch[:, :, 0]  # Extract first snapshot
            true_logits = true_logits * (first_snapshot_batch != 0).float()

        # Use filtered logits for loss computation
        probs = torch.sigmoid(true_logits) 

        # Compute weighted BCE loss using torch's built-in function
        wbce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            true_logits, y_batch.float(), pos_weight=pos_weight
        )
        
        # Compute Focal loss
        alpha = 0.90  # Adjusted weighting factor for the extremely rare class
        gamma = 2.0   # Focusing parameter to reduce the loss for well-classified examples
        # Compute focal loss
        focal_loss = -alpha * (1 - probs) ** gamma * y_batch * torch.log(probs + 1e-8) - \
                 (1 - alpha) * probs ** gamma * (1 - y_batch) * torch.log(1 - probs + 1e-8)
        focal_loss = focal_loss.mean()  # Average over batch and vertices

        neighbor_penalty_term = neighbor_penalty(edge_index, probs)  # Compute neighbor penalty

        # Combine losses
        k1 = 20.0 # weight for focal loss
        k2 = 0.05  # weight for neighbor penalty term
        loss = wbce_loss + focal_loss * k1
        # loss = wbce_loss + neighbor_penalty_term * k2  # Use only BCE loss and neighbor penalty term
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        total_bce_loss += wbce_loss.item() * X_batch.size(0)
        total_focal_loss += focal_loss.item() * X_batch.size(0)
        total_penalty += neighbor_penalty_term.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        
    avg_loss = total_loss / total_samples
    
    avg_bce_loss = total_bce_loss / total_samples
    avg_focal_loss = total_focal_loss / total_samples
    avg_penalty = total_penalty / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, BCE Loss: {avg_bce_loss:.4f}, Focal Loss: {avg_focal_loss:.4f}, Penalty: {avg_penalty:.4f}')

# Testing with precision, recall, F1

print("Starting testing...")
model.eval()
precision = Precision(task='binary', average='macro').to(device)
recall = Recall(task='binary', average='macro').to(device)
f1 = F1Score(task='binary', average='macro').to(device)

# Initialize counters for TP, FP, TN, FN
total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
predictions_dict = {'inputs': [], 'predictions': [], 'ground_truth': []}  # Dictionary to store results

with torch.no_grad():
    for idx, (X_batch, y_batch) in enumerate(test_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # X_batch: (32, 774, 3), y_batch: (32, 774)

        # During testing
        logits = model(X_batch.reshape(-1, X_batch.size(-1)), edge_index)
        logits = logits.view(X_batch.size(0), X_batch.size(1), -1)

        true_logits = logits[:, :, 1]  # (batch_size, n_vertex)
        true_logits = torch.matmul(laplacian, true_logits.T).T  # Apply Laplacian (high-pass filter)

        # Use filtered logits for loss computation
        probs = torch.sigmoid(true_logits) 

        #probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        
        if filtering:
            # Apply filtering criteria: if first_snapshot == 0, set probability of being the source to 0
            first_snapshot_batch = X_batch[:, :, 0]  # Extract first snapshot (shape: [batch_size, n_vertex])
            probs[:, :, 1] = probs[:, :, 1] * (first_snapshot_batch != 0).float()  # Zero out probabilities where first_snapshot == 0

    
        preds = probs.argmax(dim=-1)  # Get predicted class

        # Save predictions, and ground truth
        predictions_dict['inputs'].append(X_batch.cpu().numpy()[:, :, 0])
        predictions_dict['ground_truth'].append(y_batch.cpu().numpy())
        predictions_dict['predictions'].append(probs.cpu().numpy())

        # Calculate TP, FP, TN, FN for the batch
        tp = ((preds == 1) & (y_batch == 1)).sum().item()
        fp = ((preds == 1) & (y_batch == 0)).sum().item()
        tn = ((preds == 0) & (y_batch == 0)).sum().item()
        fn = ((preds == 0) & (y_batch == 1)).sum().item()

        # Accumulate batch results
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        precision.update(preds, y_batch)
        recall.update(preds, y_batch)
        f1.update(preds, y_batch)

# Save predictions dictionary to a pickle file
output_path = f'./predictions/{os.path.splitext(data_file)[0]}_predictions.pickle'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(predictions_dict, f)
print(f"Predictions saved to {output_path}")

# Print results for the entire test set
print(f'Total True Positives (TP): {total_tp}')
print(f'Total False Positives (FP): {total_fp}')
print(f'Total True Negatives (TN): {total_tn}')
print(f'Total False Negatives (FN): {total_fn}')


# Compute final metrics
final_precision = precision.compute().item()
final_recall = recall.compute().item()
final_f1 = f1.compute().item()
print(f'Test Precision: {final_precision:.4f}')
print(f'Test Recall: {final_recall:.4f}')
print(f'Test F1 Score: {final_f1:.4f}')

