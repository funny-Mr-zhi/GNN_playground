# 🚀 项目实践 (Projects)

通过实际项目应用图神经网络解决真实世界问题，提升实战能力和项目经验。

## 🎯 学习目标

- 掌握GNN在不同任务上的应用方法
- 学会完整的项目开发流程
- 积累解决实际问题的经验
- 建立端到端的系统开发能力

## 📁 目录结构

### 🎯 node_classification/
节点分类项目，包括：
- **社交网络分析**：用户画像、社区发现
- **知识图谱**：实体分类、关系预测
- **生物网络**：蛋白质功能预测、基因分析
- **学术网络**：论文分类、作者识别

### 📊 graph_classification/
图分类项目，包括：
- **分子性质预测**：药物发现、毒性预测
- **社交网络分析**：网络类型识别
- **代码分析**：程序漏洞检测
- **脑网络分析**：疾病诊断

### 🔗 link_prediction/
链接预测项目，包括：
- **推荐系统**：商品推荐、好友推荐
- **知识图谱补全**：缺失关系预测
- **生物网络**：蛋白质相互作用预测
- **交通网络**：路径规划优化

### 🌍 real_world_applications/
真实世界应用，包括：
- **智能推荐系统**：电商、内容推荐
- **金融风控**：反欺诈、信用评估
- **智慧城市**：交通优化、资源配置
- **医疗健康**：疾病诊断、药物发现

## 📖 项目开发计划

### 第29-31周：基础任务项目

#### 第29周：节点分类 - 社交网络用户画像
**项目目标**: 基于社交网络数据预测用户的兴趣标签

**数据集**: Facebook、Reddit、Twitter等社交网络数据

**技术栈**:
- 数据处理: NetworkX, Pandas
- 模型: GCN, GAT, GraphSAGE
- 评估: Accuracy, F1-score, AUC

```python
# user_profiling.py - 用户画像项目主文件
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import classification_report
import pandas as pd
import networkx as nx

class UserProfilingGNN(torch.nn.Module):
    """用户画像GNN模型"""
    
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        # 第一层图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层图卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 分类层
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class SocialNetworkDataProcessor:
    """社交网络数据处理器"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_facebook_data(self):
        """加载Facebook数据集"""
        # 加载边列表
        edges = pd.read_csv(f"{self.data_path}/facebook_edges.csv")
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        
        # 加载节点特征
        features = pd.read_csv(f"{self.data_path}/facebook_features.csv")
        x = torch.tensor(features.drop('user_id', axis=1).values, dtype=torch.float)
        
        # 加载标签
        labels = pd.read_csv(f"{self.data_path}/facebook_labels.csv")
        y = torch.tensor(labels['interest_category'].values, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
        
    def create_train_test_split(self, data, train_ratio=0.6, val_ratio=0.2):
        """创建训练验证测试集划分"""
        num_nodes = data.x.size(0)
        indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data

def train_user_profiling_model():
    """训练用户画像模型"""
    # 数据加载和预处理
    processor = SocialNetworkDataProcessor("data/facebook")
    data = processor.load_facebook_data()
    data = processor.create_train_test_split(data)
    
    # 模型初始化
    model = UserProfilingGNN(
        num_features=data.x.size(1),
        num_classes=len(torch.unique(data.y)),
        hidden_dim=64
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练循环
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # 验证
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
            model.train()
    
    # 最终测试
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f'Test Accuracy: {test_acc:.4f}')
        
        # 详细评估报告
        print(classification_report(
            data.y[data.test_mask].cpu().numpy(),
            pred[data.test_mask].cpu().numpy()
        ))

if __name__ == "__main__":
    train_user_profiling_model()
```

#### 第30周：图分类 - 分子性质预测
**项目目标**: 预测分子的生物活性和毒性

**数据集**: MUTAG, PROTEINS, IMDB-BINARY等

```python
# molecular_property_prediction.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

class MolecularGNN(torch.nn.Module):
    """分子性质预测GNN模型"""
    
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 图级表示
        self.pool = global_mean_pool
        
        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # 图级池化
        x = self.pool(x, batch)
        
        # 分类
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def molecular_property_experiment():
    """分子性质预测实验"""
    # 加载数据集
    dataset = TUDataset(root='data/MUTAG', name='MUTAG')
    
    # 交叉验证
    cv_scores = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), 
                                                           [data.y.item() for data in dataset])):
        print(f"Fold {fold + 1}/10")
        
        # 创建数据加载器
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 初始化模型
        model = MolecularGNN(
            num_features=dataset.num_node_features,
            num_classes=dataset.num_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 训练
        model.train()
        for epoch in range(100):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total
        cv_scores.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    
    print(f"Average Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

if __name__ == "__main__":
    molecular_property_experiment()
```

#### 第31周：链接预测 - 推荐系统
**项目目标**: 基于用户-商品交互图进行商品推荐

```python
# recommendation_system.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class RecommendationGNN(torch.nn.Module):
    """推荐系统GNN模型"""
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # 用户和商品嵌入
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        
        # 图卷积层
        self.conv1 = SAGEConv(embedding_dim, embedding_dim)
        self.conv2 = SAGEConv(embedding_dim, embedding_dim)
        
        # 预测层
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, user_item_edge_index, user_indices, item_indices):
        # 获取节点嵌入
        num_users = self.user_embedding.num_embeddings
        num_items = self.item_embedding.num_embeddings
        
        # 创建节点特征矩阵
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # 图卷积
        x = F.relu(self.conv1(x, user_item_edge_index))
        x = F.relu(self.conv2(x, user_item_edge_index))
        
        # 获取用户和商品表示
        user_repr = x[user_indices]
        item_repr = x[item_indices + num_users]  # 商品索引偏移
        
        # 预测交互概率
        edge_repr = torch.cat([user_repr, item_repr], dim=1)
        return self.predictor(edge_repr).squeeze()

class RecommendationDataset:
    """推荐系统数据集"""
    
    def __init__(self, ratings_file):
        self.ratings = pd.read_csv(ratings_file)
        self.prepare_data()
        
    def prepare_data(self):
        """准备数据"""
        # 创建用户和商品ID映射
        self.user_to_idx = {user: idx for idx, user in enumerate(self.ratings['user_id'].unique())}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.ratings['item_id'].unique())}
        
        self.num_users = len(self.user_to_idx)
        self.num_items = len(self.item_to_idx)
        
        # 转换为索引
        self.ratings['user_idx'] = self.ratings['user_id'].map(self.user_to_idx)
        self.ratings['item_idx'] = self.ratings['item_id'].map(self.item_to_idx)
        
        # 创建边索引（用户-商品交互）
        user_indices = self.ratings['user_idx'].values
        item_indices = self.ratings['item_idx'].values + self.num_users  # 商品索引偏移
        
        self.edge_index = torch.tensor([
            np.concatenate([user_indices, item_indices]),
            np.concatenate([item_indices, user_indices])
        ], dtype=torch.long)
        
        # 正样本边
        self.pos_edge_index = torch.tensor([user_indices, self.ratings['item_idx'].values], dtype=torch.long)
        
    def get_train_test_split(self, test_ratio=0.2):
        """划分训练测试集"""
        num_edges = self.pos_edge_index.size(1)
        indices = torch.randperm(num_edges)
        
        test_size = int(num_edges * test_ratio)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        train_edge_index = self.pos_edge_index[:, train_indices]
        test_edge_index = self.pos_edge_index[:, test_indices]
        
        return train_edge_index, test_edge_index

def train_recommendation_model():
    """训练推荐模型"""
    # 加载数据
    dataset = RecommendationDataset('data/ratings.csv')
    train_edge_index, test_edge_index = dataset.get_train_test_split()
    
    # 初始化模型
    model = RecommendationGNN(dataset.num_users, dataset.num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练循环
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        
        # 正样本
        pos_user_indices = train_edge_index[0]
        pos_item_indices = train_edge_index[1]
        pos_pred = model(dataset.edge_index, pos_user_indices, pos_item_indices)
        
        # 负样本
        neg_edge_index = negative_sampling(
            train_edge_index, num_nodes=(dataset.num_users + dataset.num_items),
            num_neg_samples=train_edge_index.size(1)
        )
        neg_user_indices = neg_edge_index[0]
        neg_item_indices = neg_edge_index[1] - dataset.num_users  # 移除偏移
        neg_pred = model(dataset.edge_index, neg_user_indices, neg_item_indices)
        
        # 损失计算
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    
    # 测试评估
    model.eval()
    with torch.no_grad():
        # 正样本预测
        test_pos_pred = model(dataset.edge_index, test_edge_index[0], test_edge_index[1])
        
        # 负样本预测
        test_neg_edge_index = negative_sampling(
            test_edge_index, num_nodes=(dataset.num_users + dataset.num_items),
            num_neg_samples=test_edge_index.size(1)
        )
        test_neg_pred = model(
            dataset.edge_index, 
            test_neg_edge_index[0], 
            test_neg_edge_index[1] - dataset.num_users
        )
        
        # 评估指标
        y_true = torch.cat([torch.ones_like(test_pos_pred), torch.zeros_like(test_neg_pred)])
        y_score = torch.cat([test_pos_pred, test_neg_pred])
        
        auc = roc_auc_score(y_true.cpu(), y_score.cpu())
        ap = average_precision_score(y_true.cpu(), y_score.cpu())
        
        print(f'Test AUC: {auc:.4f}, Test AP: {ap:.4f}')

if __name__ == "__main__":
    train_recommendation_model()
```

### 第32-34周：真实世界应用项目

#### 智能推荐系统项目
**项目描述**: 构建基于图神经网络的多模态推荐系统

**技术特点**:
- 异构图建模（用户-商品-类别-品牌）
- 多模态特征融合（文本、图像、行为）
- 实时推荐服务部署
- A/B测试框架

#### 金融风控项目
**项目描述**: 基于交易网络的反欺诈检测系统

**技术特点**:
- 动态图建模（时序交易网络）
- 异常检测算法
- 可解释性分析
- 实时风险评估

#### 智慧城市项目
**项目描述**: 城市交通流量预测和优化

**技术特点**:
- 时空图神经网络
- 多源数据融合
- 实时预测系统
- 可视化展示

## 📊 项目评估标准

### 技术评估
- **模型性能**: 准确率、召回率、F1分数
- **计算效率**: 训练时间、推理速度、内存使用
- **可扩展性**: 大规模数据处理能力
- **鲁棒性**: 对噪声和异常数据的处理能力

### 工程评估
- **代码质量**: 可读性、可维护性、测试覆盖率
- **系统设计**: 架构合理性、模块化程度
- **部署能力**: 生产环境适应性
- **文档完整性**: 使用说明、API文档、技术报告

### 业务评估
- **问题解决**: 实际问题的解决程度
- **用户体验**: 系统易用性、响应速度
- **商业价值**: 成本效益、ROI分析
- **创新性**: 技术创新和业务创新

## 🛠️ 项目开发工具

### 开发环境
```bash
# 创建项目环境
conda create -n gnn-projects python=3.9
conda activate gnn-projects

# 安装依赖
pip install torch torchvision torchaudio
pip install torch-geometric
pip install dgl
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install jupyter notebook
pip install flask fastapi
pip install docker
```

### 项目模板
```
project_name/
├── data/                   # 数据文件
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── external/          # 外部数据
├── src/                   # 源代码
│   ├── models/            # 模型定义
│   ├── utils/             # 工具函数
│   ├── preprocessing/     # 数据预处理
│   └── evaluation/        # 评估脚本
├── notebooks/             # Jupyter笔记本
├── config/               # 配置文件
├── tests/                # 测试代码
├── docker/               # Docker文件
├── docs/                 # 文档
├── requirements.txt      # 依赖列表
├── setup.py             # 安装脚本
└── README.md            # 项目说明
```

## 🎯 学习检查点

### 项目开发能力
- [ ] 能够独立完成端到端的GNN项目
- [ ] 掌握数据预处理和特征工程技能
- [ ] 具备模型调优和性能优化能力
- [ ] 能够进行全面的实验评估

### 工程实践能力
- [ ] 遵循软件开发最佳实践
- [ ] 具备系统设计和架构能力
- [ ] 掌握模型部署和服务化技能
- [ ] 能够进行项目管理和协作开发

### 问题解决能力
- [ ] 能够分析和定义实际问题
- [ ] 具备选择合适技术方案的能力
- [ ] 能够处理项目中的各种挑战
- [ ] 具备持续改进和创新的意识

通过这些项目实践，你将积累丰富的实战经验，具备独立开发GNN应用系统的能力，为未来的职业发展奠定坚实基础！
