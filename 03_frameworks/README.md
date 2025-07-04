# 🛠️ GNN框架学习 (Frameworks)

掌握主流图神经网络框架的使用，从基础API到高级功能的全面学习。

## 🎯 学习目标

- 熟练使用PyTorch Geometric进行GNN开发
- 掌握DGL框架的核心功能
- 了解其他GNN框架的特点
- 能够选择合适的框架解决实际问题

## 📁 目录结构

### 🔥 pytorch_geometric/
PyTorch Geometric框架学习，包括：
- **基础使用**：安装配置、数据结构、基本API
- **模型构建**：预定义层、自定义层、模型组合
- **数据处理**：图数据加载、批处理、数据变换
- **训练优化**：训练循环、验证、模型保存
- **高级功能**：大图训练、分布式、自定义算子

### ⚡ dgl/
Deep Graph Library学习，包括：
- **核心概念**：DGLGraph、消息传递、边和节点特征
- **异构图**：异构图构建、RGCN、HeteroGraphConv
- **大规模训练**：邻居采样、子图采样、分布式训练
- **生态系统**：与PyTorch/TensorFlow集成

### 🌟 spektral/
Spektral (TensorFlow)框架，包括：
- **基础使用**：Keras风格的API
- **模型层**：GCN、GAT、GIN等层的实现
- **数据格式**：Spektral的图数据表示

## 📖 学习计划

### 第15-17周：PyTorch Geometric深度学习

#### 第15周：PyG基础
- [ ] 安装和环境配置
- [ ] 理解Data和Batch对象
- [ ] 学习基础的消息传递API
- [ ] 实现第一个GNN模型

**核心概念：**
```python
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing

# 创建图数据
edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# 自定义消息传递层
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.linear = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_j):
        return self.linear(x_j)
```

#### 第16周：PyG进阶功能
- [ ] 学习预定义的GNN层
- [ ] 掌握图池化操作
- [ ] 实现图分类模型
- [ ] 使用transform进行数据增强

#### 第17周：PyG高级特性
- [ ] 大图的邻居采样训练
- [ ] 异构图处理
- [ ] 自定义CUDA算子
- [ ] 模型解释和可视化

### 第18-20周：DGL框架学习

#### 第18周：DGL基础
- [ ] 理解DGLGraph数据结构
- [ ] 学习消息传递范式
- [ ] 实现基础GNN模型
- [ ] 掌握边和节点特征操作

**DGL核心API：**
```python
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建图
g = dgl.graph(([0, 1, 2, 0, 3], [1, 2, 3, 3, 0]))
g.ndata['feat'] = torch.randn(4, 10)  # 节点特征
g.edata['weight'] = torch.randn(5, 1)  # 边特征

# 消息传递
def message_func(edges):
    return {'m': edges.src['feat'] * edges.data['weight']}

def reduce_func(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

g.update_all(message_func, reduce_func)
```

#### 第19周：DGL异构图
- [ ] 学习异构图的概念和表示
- [ ] 使用HeteroGraphConv构建模型
- [ ] 实现异构图上的节点分类
- [ ] 处理不同类型的节点和边

#### 第20周：DGL大规模训练
- [ ] 学习邻居采样技术
- [ ] 掌握NodeDataLoader的使用
- [ ] 实现大图上的分布式训练
- [ ] 优化内存使用和训练速度

### 第21-22周：框架对比和选择

#### 框架特点对比

| 特性 | PyTorch Geometric | DGL | Spektral |
|------|------------------|-----|----------|
| 后端 | PyTorch | PyTorch/TensorFlow/MXNet | TensorFlow |
| 易用性 | 中等 | 较高 | 高 |
| 性能 | 高 | 高 | 中等 |
| 异构图 | 支持 | 强支持 | 基础支持 |
| 大图训练 | 支持 | 强支持 | 有限支持 |
| 社区活跃度 | 高 | 高 | 中等 |

## 🛠️ 实践项目

### 项目1：PyG实现节点分类
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 训练代码
dataset = Planetoid(root='/tmp/Cora', name='Cora')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### 项目2：DGL实现图分类
```python
import dgl
import torch
import torch.nn as nn
from dgl.nn import GraphConv, GlobalMeanPool

class GraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.pool = GlobalMeanPool()
        self.classify = nn.Linear(hidden_size, num_classes)
        
    def forward(self, g, h):
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.pool(g, h)
        return self.classify(h)

# 使用TUDataset进行图分类
from dgl.data import TUDataset
dataset = TUDataset('MUTAG')
dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=32, shuffle=True)
```

### 项目3：异构图实现
```python
import dgl
import torch
import torch.nn as nn
from dgl.nn import HeteroGraphConv, GraphConv

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
            
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
```

## 📚 实用技巧和最佳实践

### PyTorch Geometric技巧

#### 1. 内存优化
```python
# 使用数据并行
model = torch.nn.DataParallel(model)

# 梯度累积
accumulate_grad_batches = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulate_grad_batches
    loss.backward()
    if (i + 1) % accumulate_grad_batches == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. 自定义数据变换
```python
from torch_geometric.transforms import Compose, RandomNodeSplit, NormalizeFeatures

transform = Compose([
    RandomNodeSplit(num_val=0.1, num_test=0.2),
    NormalizeFeatures(),
])
dataset = MyDataset(transform=transform)
```

#### 3. 模型解释
```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
)
explanation = explainer(data.x, data.edge_index, index=0)
```

### DGL技巧

#### 1. 高效的批处理
```python
# 使用NodeDataLoader进行大图训练
sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
dataloader = dgl.dataloading.NodeDataLoader(
    g, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4
)
```

#### 2. 自定义聚合函数
```python
import dgl.function as fn

# 使用内置函数
g.update_all(fn.copy_e('weight', 'm'), fn.sum('m', 'h'))

# 自定义函数
def custom_reduce(nodes):
    # 加权平均
    weights = nodes.mailbox['w']
    messages = nodes.mailbox['m']
    return {'h': torch.sum(messages * weights, dim=1) / torch.sum(weights, dim=1)}
```

## 🔧 调试和性能优化

### 常见问题和解决方案

#### 1. 内存不足
- 使用邻居采样减少计算图大小
- 降低批大小
- 使用梯度检查点
- 升级到更大内存的GPU

#### 2. 训练速度慢
- 使用更高效的数据加载器
- 启用混合精度训练
- 优化图的存储格式
- 使用分布式训练

#### 3. 数值不稳定
- 添加批归一化层
- 使用残差连接
- 调整学习率和权重衰减
- 检查梯度裁剪

### 性能基准测试
```python
import time
import torch.profiler

# 简单计时
start_time = time.time()
output = model(data)
end_time = time.time()
print(f"Forward pass time: {end_time - start_time:.4f} seconds")

# 详细性能分析
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    output = model(data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 🎯 学习检查点

### PyTorch Geometric掌握度
- [ ] 能够创建和操作图数据对象
- [ ] 熟练使用预定义的GNN层
- [ ] 可以实现自定义的消息传递层
- [ ] 掌握图分类和节点分类的完整流程
- [ ] 了解大图训练的采样策略

### DGL掌握度
- [ ] 理解DGLGraph的数据结构
- [ ] 能够处理异构图数据
- [ ] 掌握高效的消息传递API
- [ ] 可以实现分布式图神经网络训练
- [ ] 了解DGL的生态系统

### 综合能力
- [ ] 能够根据任务需求选择合适的框架
- [ ] 具备调试和优化GNN模型的能力
- [ ] 可以将模型部署到生产环境
- [ ] 了解各框架的优缺点和适用场景

## 🔗 学习资源

### 官方文档
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [DGL Documentation](https://docs.dgl.ai/)
- [Spektral Documentation](https://graphneural.network/)

### 教程和示例
- [PyG Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [DGL Tutorials](https://docs.dgl.ai/tutorials/blitz/index.html)
- [Graph ML in 2023](https://github.com/graphdeeplearning/graphml-2023)

### 社区资源
- [PyG Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- [DGL Examples](https://github.com/dmlc/dgl/tree/master/examples)
- [Awesome Graph Neural Networks](https://github.com/thunlp/GNNPapers)

完成框架学习后，你将具备使用主流GNN框架进行实际开发的能力，为后续的项目实践做好准备！
