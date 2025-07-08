# from dgl.data.utils import extract_archive
# extract_archive("data/GIN/GINDataset.zip", "data/GIN")    #手动下载并解压缩

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = dgl.data.GINDataset("PROTEINS", self_loop=True, raw_dir="data/GIN")

print("Number of graphs:", len(dataset))
print("Node feature dimension:", dataset.dim_nfeats)
print("Number of graph categories:", dataset.gclasses)

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(range(num_train))
test_sampler = SubsetRandomSampler(range(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=32, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=32, drop_last=False)

# 获取dataloader的一个batch的方式：
# it = iter(train_dataloader)
# batch = next(it)
# print(batch)

# 查看batch的内容
# batched_graph, labels = batch
# print(
#     "Number of nodes for each graph element in the batch:",
#     batched_graph.batch_num_nodes(),
# )
# print(
#     "Number of edges for each graph element in the batch:",
#     batched_graph.batch_num_edges(),
# )

# # Recover the original graph elements from the minibatch
# graphs = dgl.unbatch(batched_graph)
# print("The original graphs in the minibatch:")
# print(graphs)

from dgl.nn import GraphConv
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")
    


# Create the model with given dimensions
model = GCN(dataset.dim_nfeats, 16, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["attr"].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["attr"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)
print("Test accuracy:", num_correct / num_tests)