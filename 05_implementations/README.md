# 💻 代码实现 (Implementations)

通过从零开始的代码实现加深对GNN算法的理解，提升编程能力和工程实践水平。

## 🎯 学习目标

- 从零实现经典GNN算法，深入理解其原理
- 掌握高效的编程技巧和最佳实践
- 学会性能优化和模型调试
- 建立完整的实验评估体系

## 📁 目录结构

### 🔧 from_scratch/
从零开始实现，包括：
- **基础组件**：图数据结构、消息传递框架
- **经典模型**：GCN、GAT、GraphSAGE、GIN等
- **训练工具**：优化器、损失函数、评估指标
- **完整示例**：端到端的训练和测试流程

### 📚 tutorials/
教程式代码，包括：
- **入门教程**：循序渐进的学习材料
- **进阶教程**：复杂模型和技术的实现
- **案例研究**：具体问题的完整解决方案
- **最佳实践**：代码规范和优化技巧

### 📊 benchmarks/
性能基准测试，包括：
- **标准数据集**：在公开数据集上的测试结果
- **模型对比**：不同GNN模型的性能比较
- **效率分析**：训练时间和内存使用分析
- **可重现实验**：完整的实验配置和结果

## 📖 实现计划

### 第23-25周：基础组件实现

#### 第23周：图数据结构和基础操作
```python
# graph.py - 基础图数据结构实现
import numpy as np
import torch
from typing import List, Tuple, Optional, Union

class Graph:
    """基础图数据结构"""
    
    def __init__(self, num_nodes: int, edge_index: torch.Tensor, 
                 node_features: Optional[torch.Tensor] = None,
                 edge_features: Optional[torch.Tensor] = None):
        self.num_nodes = num_nodes
        self.edge_index = edge_index  # [2, num_edges]
        self.num_edges = edge_index.size(1)
        self.node_features = node_features
        self.edge_features = edge_features
        
        # 计算度数
        self.degrees = self._compute_degrees()
        # 构建邻接矩阵
        self.adj_matrix = self._build_adjacency_matrix()
        
    def _compute_degrees(self) -> torch.Tensor:
        """计算节点度数"""
        degrees = torch.zeros(self.num_nodes, dtype=torch.long)
        degrees.scatter_add_(0, self.edge_index[0], torch.ones_like(self.edge_index[0]))
        return degrees
        
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """构建邻接矩阵"""
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        adj[self.edge_index[0], self.edge_index[1]] = 1
        return adj
        
    def add_self_loops(self):
        """添加自环"""
        self_loop_index = torch.arange(self.num_nodes).unsqueeze(0).repeat(2, 1)
        self.edge_index = torch.cat([self.edge_index, self_loop_index], dim=1)
        self.num_edges = self.edge_index.size(1)
        
    def normalize_adjacency(self, method: str = 'symmetric'):
        """归一化邻接矩阵"""
        if method == 'symmetric':
            # D^(-1/2) * A * D^(-1/2)
            deg = self.degrees.float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            norm = deg_inv_sqrt[self.edge_index[0]] * deg_inv_sqrt[self.edge_index[1]]
            return norm
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    def get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居"""
        mask = self.edge_index[0] == node_id
        neighbors = self.edge_index[1][mask].tolist()
        return neighbors
```

#### 第24周：消息传递框架
```python
# message_passing.py - 通用消息传递框架
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any

class MessagePassing(nn.Module, ABC):
    """通用消息传递基类"""
    
    def __init__(self, aggr: str = 'add', flow: str = 'source_to_target'):
        super().__init__()
        self.aggr = aggr
        self.flow = flow
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # 1. 计算消息
        messages = self.message(x, edge_index, edge_attr)
        
        # 2. 聚合消息
        aggregated = self.aggregate(messages, edge_index)
        
        # 3. 更新节点特征
        updated = self.update(x, aggregated)
        
        return updated
        
    @abstractmethod
    def message(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算边上的消息"""
        pass
        
    def aggregate(self, messages: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """聚合消息"""
        num_nodes = edge_index.max().item() + 1
        
        if self.aggr == 'add':
            out = torch.zeros(num_nodes, messages.size(1), device=messages.device)
            out.scatter_add_(0, edge_index[1].unsqueeze(1).expand_as(messages), messages)
        elif self.aggr == 'mean':
            out = torch.zeros(num_nodes, messages.size(1), device=messages.device)
            count = torch.zeros(num_nodes, 1, device=messages.device)
            out.scatter_add_(0, edge_index[1].unsqueeze(1).expand_as(messages), messages)
            count.scatter_add_(0, edge_index[1].unsqueeze(1), torch.ones_like(messages[:, :1]))
            out = out / (count + 1e-8)
        elif self.aggr == 'max':
            out = torch.zeros(num_nodes, messages.size(1), device=messages.device)
            out.scatter_reduce_(0, edge_index[1].unsqueeze(1).expand_as(messages), 
                              messages, reduce='amax', include_self=False)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")
            
        return out
        
    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """更新节点特征"""
        return aggregated
```

#### 第25周：训练工具和评估指标
```python
# trainer.py - 训练工具
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time
from typing import Dict, List, Tuple

class GNNTrainer:
    """GNN训练器"""
    
    def __init__(self, model: nn.Module, lr: float = 0.01, 
                 weight_decay: float = 5e-4, device: str = 'cpu'):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, data: Graph, train_mask: torch.Tensor) -> float:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def evaluate(self, data: Graph, mask: torch.Tensor) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data)
            pred = out.argmax(dim=1)
            
            loss = F.cross_entropy(out[mask], data.y[mask]).item()
            acc = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
            f1 = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro')
            
        return {'loss': loss, 'accuracy': acc, 'f1_score': f1}
        
    def train(self, data: Graph, train_mask: torch.Tensor, 
              val_mask: torch.Tensor, epochs: int = 200, 
              early_stopping: int = None, verbose: bool = True) -> Dict[str, List]:
        """完整训练流程"""
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(data, train_mask)
            
            # 验证
            val_metrics = self.evaluate(data, val_mask)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # 早停检查
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if early_stopping and patience_counter >= early_stopping:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
                
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
                      
        return self.history
```

### 第26-28周：经典模型实现

#### 第26周：GCN实现
```python
# gcn.py - Graph Convolutional Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from message_passing import MessagePassing

class GCNConv(MessagePassing):
    """GCN卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 bias: bool = True, normalize: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 添加自环
        num_nodes = x.size(0)
        self_loop_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        
        # 计算归一化系数
        if self.normalize:
            edge_weight = self._compute_edge_weight(edge_index, num_nodes)
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            
        return super().forward(x, edge_index, edge_weight)
        
    def message(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        # 线性变换
        x = self.linear(x)
        # 发送消息时加权
        return x[edge_index[0]] * edge_attr.unsqueeze(1)
        
    def _compute_edge_weight(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """计算边的归一化权重 D^(-1/2) * A * D^(-1/2)"""
        # 计算度数
        degree = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float)
        degree.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        
        # 计算 D^(-1/2)
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 计算边权重
        edge_weight = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        return edge_weight

class GCN(nn.Module):
    """完整的GCN模型"""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        
    def forward(self, data: Graph) -> torch.Tensor:
        x, edge_index = data.node_features, data.edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.convs[-1](x, edge_index)
        return x
```

#### 第27周：GAT实现
```python
# gat.py - Graph Attention Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from message_passing import MessagePassing

class GATConv(MessagePassing):
    """GAT卷积层"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 dropout: float = 0.0, concat: bool = True, bias: bool = True):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        
        # 线性变换
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # 注意力参数
        self.att_src = nn.Parameter(torch.randn(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.randn(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.randn(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 线性变换
        x = self.linear(x).view(-1, self.heads, self.out_channels)
        
        return super().forward(x, edge_index)
        
    def message(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 计算注意力系数
        alpha = self._compute_attention(x, edge_index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        return x[edge_index[0]] * alpha.unsqueeze(-1)
        
    def _compute_attention(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """计算注意力系数"""
        # 计算源节点和目标节点的注意力得分
        alpha_src = (x * self.att_src).sum(dim=-1)[edge_index[0]]  # [num_edges, heads]
        alpha_dst = (x * self.att_dst).sum(dim=-1)[edge_index[1]]  # [num_edges, heads]
        
        # 计算注意力权重
        alpha = alpha_src + alpha_dst
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # 在每个节点的邻居上进行softmax
        alpha = self._softmax(alpha, edge_index[1])
        
        return alpha
        
    def _softmax(self, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """在指定维度上进行softmax"""
        num_nodes = index.max().item() + 1
        out = src - src.max()  # 数值稳定性
        out = out.exp()
        
        # 计算分母
        out_sum = torch.zeros(num_nodes, src.size(1), device=src.device)
        out_sum.scatter_add_(0, index.unsqueeze(1).expand_as(out), out)
        
        # 归一化
        out = out / (out_sum[index] + 1e-16)
        
        return out
        
    def update(self, x: torch.Tensor, aggregated: torch.Tensor) -> torch.Tensor:
        """更新节点特征"""
        if self.concat:
            out = aggregated.view(-1, self.heads * self.out_channels)
        else:
            out = aggregated.mean(dim=1)
            
        if self.bias is not None:
            out = out + self.bias
            
        return out

class GAT(nn.Module):
    """完整的GAT模型"""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, 
                            dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, 
                            heads=1, dropout=dropout, concat=False)
        
        self.dropout = dropout
        
    def forward(self, data: Graph) -> torch.Tensor:
        x, edge_index = data.node_features, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return x
```

## 🧪 实验和调试

### 模型调试技巧

#### 1. 梯度检查
```python
def check_gradients(model, data, loss_fn):
    """检查梯度是否正常"""
    model.train()
    output = model(data)
    loss = loss_fn(output, data.y)
    loss.backward()
    
    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for {name}")
        elif torch.isnan(param.grad).any():
            print(f"NaN gradient for {name}")
        elif (param.grad == 0).all():
            print(f"Zero gradient for {name}")
```

#### 2. 激活值监控
```python
class ActivationMonitor:
    """监控激活值的统计信息"""
    
    def __init__(self):
        self.activations = {}
        
    def register_hooks(self, model):
        """注册钩子函数"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.ELU, GCNConv, GATConv)):
                module.register_forward_hook(self._make_hook(name))
                
    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'zeros': (output == 0).float().mean().item()
                }
        return hook
        
    def print_stats(self):
        """打印激活值统计"""
        for name, stats in self.activations.items():
            print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
                  f"zeros={stats['zeros']:.2%}")
```

### 性能优化技巧

#### 1. 内存优化
```python
def optimize_memory_usage(model, data):
    """内存使用优化"""
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # 清理缓存
    torch.cuda.empty_cache()
```

#### 2. 计算优化
```python
def sparse_matrix_multiply(adj_matrix, features):
    """稀疏矩阵乘法优化"""
    if adj_matrix.is_sparse:
        return torch.sparse.mm(adj_matrix, features)
    else:
        return torch.mm(adj_matrix, features)
```

## 📊 基准测试

### 标准数据集测试
```python
# benchmark.py - 基准测试
import time
import torch
from torch_geometric.datasets import Planetoid, TUDataset
from sklearn.model_selection import StratifiedKFold

class GNNBenchmark:
    """GNN模型基准测试"""
    
    def __init__(self, models: Dict[str, nn.Module], datasets: List[str]):
        self.models = models
        self.datasets = datasets
        self.results = {}
        
    def run_node_classification(self, dataset_name: str) -> Dict:
        """节点分类基准测试"""
        dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
        data = dataset[0]
        
        results = {}
        for model_name, model_class in self.models.items():
            print(f"Testing {model_name} on {dataset_name}")
            
            # 初始化模型
            model = model_class(
                in_channels=dataset.num_features,
                hidden_channels=64,
                out_channels=dataset.num_classes
            )
            
            # 训练和评估
            trainer = GNNTrainer(model)
            history = trainer.train(data, data.train_mask, data.val_mask, epochs=200)
            
            # 测试集评估
            test_metrics = trainer.evaluate(data, data.test_mask)
            
            results[model_name] = {
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1_score'],
                'training_time': history['training_time']
            }
            
        return results
        
    def run_graph_classification(self, dataset_name: str, cv_folds: int = 10) -> Dict:
        """图分类基准测试"""
        dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name)
        
        results = {}
        for model_name, model_class in self.models.items():
            print(f"Testing {model_name} on {dataset_name} with {cv_folds}-fold CV")
            
            cv_scores = []
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), 
                                                                   [data.y.item() for data in dataset])):
                # 创建数据加载器
                train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=True)
                test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)
                
                # 初始化模型
                model = model_class(
                    in_channels=dataset.num_features,
                    hidden_channels=64,
                    out_channels=dataset.num_classes
                )
                
                # 训练和评估
                fold_acc = self._train_graph_classification(model, train_loader, test_loader)
                cv_scores.append(fold_acc)
                
            results[model_name] = {
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'cv_scores': cv_scores
            }
            
        return results
```

## 🎯 学习检查点

### 实现能力检查
- [ ] 能够从零实现消息传递框架
- [ ] 掌握经典GNN模型的核心算法
- [ ] 具备调试和优化模型的能力
- [ ] 能够进行完整的实验评估

### 工程实践检查
- [ ] 遵循良好的代码规范和结构
- [ ] 具备性能优化的意识和技能
- [ ] 能够进行系统的基准测试
- [ ] 掌握模型部署的基本技能

### 理论理解检查
- [ ] 深入理解各种GNN算法的原理
- [ ] 能够分析算法的时间和空间复杂度
- [ ] 理解不同设计选择的影响
- [ ] 具备改进现有算法的能力

通过从零开始的代码实现，你将对GNN算法有最深入的理解，为后续的创新研究和实际应用奠定坚实的技术基础！
