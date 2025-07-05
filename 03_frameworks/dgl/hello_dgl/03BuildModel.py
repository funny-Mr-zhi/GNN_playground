import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGEConv(nn.Module):
    """Graph Convlution module used by GraphSAGE.

    parameters
    ----------
    in_feats : int
        Size of each input feature vector.
    out_feats : int
        Size of each output feature vector.
    """
    def __init__(self, in_feats, out_feats):
        super(SAGEConv, self).__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)

    def forward(self, g, h):
        """Forward computation.
        
        Parameters
        ----------
        g : DGLGraph
            The input graph.
        h : torch.Tensor
            The input feature matrix of shape (N, in_feats), where N is the number of
        """
        with g.local_scope():   # 使用local_scope()可以在函数结束后清除图的局部数据
            g.ndata['h'] = h
            # update_all is a message passing API
            g.update_all(   # 定义和执行message和reduce函数
                message_func = fn.copy_u('h', 'm'), # 从邻居节点复制特征到消息
                reduce_func = fn.mean('m', 'h_N') # 对消息进行平均，得到邻居节点的特征
            )
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            h_out = self.linear(h_total)
            return h_out
    
# 深层次的定制
class WeightedSAGEConv(nn.Module):
    """Graph Convlution module used by GraphSAGE with edge weights.

    parameters
    ----------
    in_feats : int
        Size of each input feature vector.
    out_feats : int
        Size of each output feature vector.
    """
    def __init__(self, in_feats, out_feats):
        super(WeightedSAGEConv, self).__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)

    def forward(self, g, h, w):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(
                fn.u_mul_e('h', 'w', 'm'),  # 从邻居节点复制特征到消息，并乘以边权重
                fn.mean('m', 'h_N')
            )
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            h_out = self.linear(h_total)
            return h_out

class Model(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_size)
        self.conv2 = SAGEConv(hidden_size, out_feats)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = self.conv2(g, h)
        return h
    
class WeightedModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(WeightedModel, self).__init__()
        self.conv1 = WeightedSAGEConv(in_feats, hidden_size)
        self.conv2 = WeightedSAGEConv(hidden_size, out_feats)

    def forward(self, g, h):
        h = F.relu(self.conv1(g, h, torch.zeros(g.number_of_edges(), 1).to(g.device)))
        h = self.conv2(g, h, torch.zeros(g.number_of_edges(), 1).to(g.device))
        return h

class MLP(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_feats)

    def forward(self, g,  h):
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h
    
class MLP_doubel_zeros(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(MLP_doubel_zeros, self).__init__()
        self.linear1 = nn.Linear(in_feats * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size * 2, out_feats)

    def forward(self, g, h):
        h = torch.cat([h, torch.zeros_like(h)], dim=1)  # 模拟邻居节点特征
        h = F.relu(self.linear1(h))
        h = torch.cat([h, torch.zeros_like(h)], dim=1)  # 模拟邻居节点特征
        h = self.linear2(h)
        return h

import dgl.data

dataset = dgl.data.CoraGraphDataset(raw_dir='data/cora')
g = dataset[0]

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(300):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that we should only compute the losses of the nodes in the training set,
        # i.e. with train_mask 1.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())

        if e % 5 == 0:
            print(f"Epoch {e:03d} | Loss {loss.item():.4f} | "
                  f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f}")


# model = Model(g.ndata["feat"].shape[1], 16, dataset.num_classes)
model = WeightedModel(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)
model = MLP_doubel_zeros(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)
model = MLP(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)
