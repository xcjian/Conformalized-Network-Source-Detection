# This is a draft showing how to train and test a Fully-connected NN.
# Please replace this Fully-connected NN with reasonable architecture.
# Please put your NN model in a folder parallel to SD-STGCN, e.g., SD-STGCN-torch.
# Also, store the output probs and ground truth on the test set in your NN folder.
# Format: a dictionary
"""
{
    'inputs': a list of # batch_size x n_nodes input data. This is [X[i][:, 0] for i in test_idx], but arranged in batches.
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

sys.path.append('SD-STGCN/utils')  # Adjust to actual path
from math_graph import weight_matrix


def multi_label_loss(y, logits, weight):
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

data_path = 'SD-STGCN/dataset/highSchool/data/SIR/'
#data_file = 'SIR_nsrc1_Rzero2.5_beta0.25_gamma0_T30_ls21200_nf16_entire.pickle'
data_file = 'SIR_nsrc1_Rzero43.44_beta0.25_gamma0.15_T30_ls21200_nf16_torchentire_with_node_features.pickle'
graph_path = 'SD-STGCN/dataset/highSchool/data/graph/'
graph_file = 'highSchool.edgelist'

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
X = data[0]  # List of 21200 tensors, each (30, 774)
y = data[1]  # Shape: (21200,)
skip = data[2]
X = [torch.tensor(x[skip[idx][0]:skip[idx][0] + n_frame, :].T, dtype=torch.float32) for idx, x in enumerate(X)]  # Convert to tensors
y = torch.tensor(y, dtype=torch.int64)

# Train-test split
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=False)
X_train = [X[i] for i in train_idx]
y_train = y[train_idx]
X_test = [X[i] for i in test_idx]
y_test = y[test_idx]

# Custom dataset
class SIRDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Data loaders
train_dataset = SIRDataset(X_train, y_train)
test_dataset = SIRDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(input_dim=16, hidden_dim=16).to(device)
edge_index = edge_index.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
weight = 5.0  # Weight for y_i = 1
num_epochs = 10

# Training
print("Starting training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        logits = model(X_batch, edge_index)
        loss = multi_label_loss(y_batch, logits, weight)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    
    avg_loss = total_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Testing with precision, recall, F1
print("Starting testing...")
model.eval()
precision = Precision(task='binary', average='macro').to(device)
recall = Recall(task='binary', average='macro').to(device)
f1 = F1Score(task='binary', average='macro').to(device)

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # X_batch: (32, 774, 3), y_batch: (32, 774)
        probs = model(X_batch, edge_index)  # (32, 774, 2)
        preds = probs.argmax(dim=-1)  # Shape: (32, 774)
        precision.update(preds, y_batch)
        recall.update(preds, y_batch)
        f1.update(preds, y_batch)

# Compute final metrics
final_precision = precision.compute().item()
final_recall = recall.compute().item()
final_f1 = f1.compute().item()
print(f'Test Precision: {final_precision:.4f}')
print(f'Test Recall: {final_recall:.4f}')
print(f'Test F1 Score: {final_f1:.4f}')
