import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载Cora数据集
dataset = dgl.data.CoraGraphDataset(raw_dir='./data/cora')    # 下载的数据集存储在指定目录
print("\n=====数据集加载完成=====\n")
# 整个数据集只有一个图
g = dataset[0]  # 获取图数据
# 查看图的基本信息
print(g)
print("节点信息", g.ndata)
print("边信息", g.edata)

# 定义网络
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

# 训练
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for epoch in range(100):
        # 前向传播
        logits = model(g, features)
        # 计算预测值
        pred = logits.argmax(dim=1)
        # 计算训练集上的损失
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # 计算三种准确率
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # 保存最佳验证准确率
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                  f"Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f}")

g = g.to("cuda")
model.to("cuda")  # 将图和模型移动到GPU上, 注意数据直接to是新建了一个g，需要替换
train(g, model)

