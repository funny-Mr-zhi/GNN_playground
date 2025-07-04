# 📚 学习资源 (Resources)

汇集图神经网络学习过程中需要的各种资源，包括数据集、工具、书籍课程等。

## 🎯 资源目标

- 提供高质量的学习数据集
- 整理实用的开发工具
- 推荐优秀的书籍和课程
- 建立完整的资源库

## 📁 目录结构

### 📊 datasets/
常用数据集，包括：
- **节点分类数据集**：Cora, CiteSeer, PubMed等
- **图分类数据集**：MUTAG, PROTEINS, IMDB等
- **链接预测数据集**：Facebook, Twitter, Amazon等
- **真实世界数据集**：交通、生物、社交网络数据

### 🛠️ tools/
开发工具集，包括：
- **可视化工具**：网络可视化、结果展示
- **数据处理工具**：数据清洗、格式转换
- **实验工具**：超参数调优、结果管理
- **部署工具**：模型服务化、容器化

### 📖 books_courses/
书籍课程推荐，包括：
- **经典教材**：图论、机器学习、深度学习
- **专业书籍**：图神经网络专著
- **在线课程**：大学课程、在线教程
- **学习路径**：系统学习规划

## 📊 数据集资源

### 节点分类数据集

#### 引文网络数据集
```python
# citation_networks.py - 引文网络数据集加载
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

class CitationDatasets:
    """引文网络数据集管理"""
    
    @staticmethod
    def load_cora():
        """加载Cora数据集"""
        dataset = Planetoid(root='data/Cora', name='Cora', 
                          transform=T.NormalizeFeatures())
        return dataset
        
    @staticmethod
    def load_citeseer():
        """加载CiteSeer数据集"""
        dataset = Planetoid(root='data/CiteSeer', name='CiteSeer',
                          transform=T.NormalizeFeatures())
        return dataset
        
    @staticmethod
    def load_pubmed():
        """加载PubMed数据集"""
        dataset = Planetoid(root='data/PubMed', name='PubMed',
                          transform=T.NormalizeFeatures())
        return dataset
    
    @staticmethod
    def get_dataset_info():
        """获取数据集信息"""
        datasets = ['Cora', 'CiteSeer', 'PubMed']
        info = {}
        
        for name in datasets:
            dataset = Planetoid(root=f'data/{name}', name=name)
            data = dataset[0]
            
            info[name] = {
                'num_nodes': data.x.size(0),
                'num_edges': data.edge_index.size(1) // 2,
                'num_features': data.x.size(1),
                'num_classes': len(torch.unique(data.y)),
                'train_nodes': data.train_mask.sum().item(),
                'val_nodes': data.val_mask.sum().item(),
                'test_nodes': data.test_mask.sum().item()
            }
        
        return info

# 使用示例
if __name__ == "__main__":
    info = CitationDatasets.get_dataset_info()
    for name, stats in info.items():
        print(f"{name}: {stats}")
```

#### 社交网络数据集
```python
# social_networks.py - 社交网络数据集
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

class SocialNetworkDatasets:
    """社交网络数据集管理"""
    
    @staticmethod
    def load_karate_club():
        """加载空手道俱乐部数据集"""
        G = nx.karate_club_graph()
        # 添加节点特征（度数作为特征）
        for node in G.nodes():
            G.nodes[node]['x'] = G.degree(node)
        
        data = from_networkx(G)
        return data
    
    @staticmethod
    def load_facebook_pages():
        """加载Facebook页面数据集"""
        # 这里需要下载Facebook页面数据集
        edges = pd.read_csv('data/facebook_pages/edges.csv')
        features = pd.read_csv('data/facebook_pages/features.csv')
        labels = pd.read_csv('data/facebook_pages/labels.csv')
        
        edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)
        x = torch.tensor(features.drop('page_id', axis=1).values, dtype=torch.float)
        y = torch.tensor(labels['page_type'].values, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    @staticmethod
    def create_synthetic_social_network(num_users=1000, num_features=50):
        """创建合成社交网络数据"""
        # 使用随机图模型
        G = nx.barabasi_albert_graph(num_users, 3)
        
        # 生成用户特征
        features = torch.randn(num_users, num_features)
        
        # 生成标签（基于社区结构）
        communities = nx.community.greedy_modularity_communities(G)
        labels = torch.zeros(num_users, dtype=torch.long)
        for i, community in enumerate(communities):
            for node in community:
                labels[node] = i
        
        edge_index = torch.tensor(list(G.edges())).T
        
        return Data(x=features, edge_index=edge_index, y=labels)
```

### 图分类数据集

#### 生物化学数据集
```python
# biochemical_datasets.py - 生物化学数据集
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

class BiochemicalDatasets:
    """生物化学数据集管理"""
    
    @staticmethod
    def load_mutag():
        """加载MUTAG数据集（分子致突变性）"""
        dataset = TUDataset(root='data/MUTAG', name='MUTAG')
        return dataset
    
    @staticmethod
    def load_proteins():
        """加载PROTEINS数据集（蛋白质功能）"""
        dataset = TUDataset(root='data/PROTEINS', name='PROTEINS')
        return dataset
    
    @staticmethod
    def load_nci1():
        """加载NCI1数据集（抗癌活性）"""
        dataset = TUDataset(root='data/NCI1', name='NCI1')
        return dataset
    
    @staticmethod
    def load_tox21():
        """加载Tox21数据集（毒性预测）"""
        # 需要单独下载Tox21数据集
        pass
    
    @staticmethod
    def get_molecular_datasets_info():
        """获取分子数据集信息"""
        dataset_names = ['MUTAG', 'PROTEINS', 'NCI1', 'PTC_MR']
        info = {}
        
        for name in dataset_names:
            try:
                dataset = TUDataset(root=f'data/{name}', name=name)
                info[name] = {
                    'num_graphs': len(dataset),
                    'num_classes': dataset.num_classes,
                    'num_node_features': dataset.num_node_features,
                    'num_edge_features': dataset.num_edge_features,
                    'avg_nodes': sum([data.x.size(0) for data in dataset]) / len(dataset),
                    'avg_edges': sum([data.edge_index.size(1) for data in dataset]) / len(dataset)
                }
            except:
                print(f"Failed to load dataset: {name}")
        
        return info
```

### 链接预测数据集

```python
# link_prediction_datasets.py - 链接预测数据集
import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, train_test_split_edges

class LinkPredictionDatasets:
    """链接预测数据集管理"""
    
    @staticmethod
    def load_facebook_social_network():
        """加载Facebook社交网络"""
        # 下载Facebook社交网络数据
        edges = pd.read_csv('data/facebook_social/edges.txt', sep=' ', header=None)
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        
        num_nodes = edge_index.max().item() + 1
        x = torch.eye(num_nodes)  # 使用单位矩阵作为节点特征
        
        data = Data(x=x, edge_index=edge_index)
        return data
    
    @staticmethod
    def load_protein_interaction():
        """加载蛋白质相互作用网络"""
        # 加载蛋白质相互作用数据
        ppi_data = pd.read_csv('data/ppi/interactions.csv')
        
        # 构建节点ID映射
        proteins = set(ppi_data['protein1'].unique()) | set(ppi_data['protein2'].unique())
        protein_to_id = {protein: i for i, protein in enumerate(proteins)}
        
        # 构建边索引
        edges = []
        for _, row in ppi_data.iterrows():
            p1_id = protein_to_id[row['protein1']]
            p2_id = protein_to_id[row['protein2']]
            edges.append([p1_id, p2_id])
            edges.append([p2_id, p1_id])  # 无向图
        
        edge_index = torch.tensor(edges).T
        
        # 生成节点特征（可以使用蛋白质序列特征等）
        num_nodes = len(proteins)
        x = torch.randn(num_nodes, 128)  # 假设使用128维特征
        
        return Data(x=x, edge_index=edge_index)
    
    @staticmethod
    def create_train_test_split(data, val_ratio=0.05, test_ratio=0.1):
        """创建训练验证测试集划分"""
        transform = train_test_split_edges(val_ratio=val_ratio, test_ratio=test_ratio)
        return transform(data)
    
    @staticmethod
    def load_amazon_product_network():
        """加载Amazon商品网络"""
        # 加载Amazon商品共同购买网络
        edges = pd.read_csv('data/amazon/edges.txt', sep='\t', header=None)
        edge_index = torch.tensor(edges.values.T, dtype=torch.long)
        
        # 加载商品特征
        features = pd.read_csv('data/amazon/features.csv')
        x = torch.tensor(features.drop('product_id', axis=1).values, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)
```

## 🛠️ 开发工具

### 可视化工具

```python
# visualization_tools.py - 可视化工具
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
import plotly.express as px

class GraphVisualizer:
    """图可视化工具"""
    
    @staticmethod
    def plot_graph_2d(data, node_labels=None, title="Graph Visualization"):
        """2D图可视化"""
        G = to_networkx(data, to_undirected=True)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray')
        
        # 绘制节点
        if node_labels is not None and hasattr(data, 'y'):
            unique_labels = torch.unique(data.y)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = data.y == label
                node_list = [j for j, m in enumerate(mask) if m]
                nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                                     node_color=[colors[i]], 
                                     label=f'Class {label.item()}')
            plt.legend()
        else:
            nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_graph_3d_interactive(data, node_labels=None):
        """3D交互式图可视化"""
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G, dim=3)
        
        # 提取节点坐标
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        
        # 提取边坐标
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # 创建边的轨迹
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none'
        )
        
        # 创建节点的轨迹
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=8,
                color=data.y if hasattr(data, 'y') else 'lightblue',
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'Node {i}' for i in range(len(node_x))],
            hoverinfo='text'
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Graph Visualization",
            showlegend=False,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        fig.show()
    
    @staticmethod
    def plot_attention_weights(data, attention_weights, layer_idx=0, head_idx=0):
        """可视化注意力权重"""
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制边，边的粗细表示注意力权重
        edges = list(G.edges())
        edge_weights = attention_weights[layer_idx][head_idx].detach().cpu().numpy()
        
        for i, (u, v) in enumerate(edges):
            if i < len(edge_weights):
                plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                        'gray', alpha=0.6, linewidth=edge_weights[i] * 5)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"Attention Weights - Layer {layer_idx}, Head {head_idx}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class LearningCurveVisualizer:
    """学习曲线可视化"""
    
    @staticmethod
    def plot_training_curves(history):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        if 'train_acc' in history:
            ax2.plot(history['train_acc'], label='Train Accuracy')
        if 'val_acc' in history:
            ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_hyperparameter_search(results):
        """可视化超参数搜索结果"""
        # 假设results是包含超参数和性能的字典列表
        df = pd.DataFrame(results)
        
        # 创建超参数vs性能的散点图
        fig = px.scatter_matrix(df, dimensions=df.columns[:-1], 
                               color=df.columns[-1],
                               title="Hyperparameter Search Results")
        fig.show()
```

### 实验管理工具

```python
# experiment_manager.py - 实验管理工具
import json
import os
import pickle
import time
from datetime import datetime
import torch
import numpy as np
from typing import Dict, Any

class ExperimentManager:
    """实验管理器"""
    
    def __init__(self, experiment_dir="experiments"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        
    def create_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """创建新实验"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return experiment_id
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float], step: int):
        """记录指标"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        metrics_path = os.path.join(experiment_path, "metrics.jsonl")
        
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_model(self, experiment_id: str, model: torch.nn.Module, epoch: int):
        """保存模型"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        model_path = os.path.join(experiment_path, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
    
    def save_results(self, experiment_id: str, results: Dict[str, Any]):
        """保存实验结果"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        results_path = os.path.join(experiment_path, "results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """加载实验"""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        
        # 加载配置
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载指标
        metrics_path = os.path.join(experiment_path, "metrics.jsonl")
        metrics = []
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        
        # 加载结果
        results_path = os.path.join(experiment_path, "results.json")
        results = {}
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        return {
            "config": config,
            "metrics": metrics,
            "results": results
        }
    
    def compare_experiments(self, experiment_ids: list) -> pd.DataFrame:
        """比较多个实验"""
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_data = self.load_experiment(exp_id)
            
            row = {"experiment_id": exp_id}
            row.update(exp_data["config"])
            
            # 添加最终结果
            if exp_data["results"]:
                row.update(exp_data["results"])
            
            # 添加最佳指标
            if exp_data["metrics"]:
                metrics_df = pd.DataFrame(exp_data["metrics"])
                for col in metrics_df.columns:
                    if col not in ["step", "timestamp"]:
                        row[f"best_{col}"] = metrics_df[col].max()
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

# 使用示例
if __name__ == "__main__":
    # 创建实验管理器
    exp_manager = ExperimentManager()
    
    # 创建实验
    config = {
        "model": "GCN",
        "hidden_dim": 64,
        "learning_rate": 0.01,
        "epochs": 200
    }
    exp_id = exp_manager.create_experiment("node_classification", config)
    
    # 模拟训练过程
    for epoch in range(10):
        metrics = {
            "train_loss": np.random.random(),
            "val_acc": np.random.random()
        }
        exp_manager.log_metrics(exp_id, metrics, epoch)
    
    # 保存最终结果
    results = {"test_accuracy": 0.85, "test_f1": 0.83}
    exp_manager.save_results(exp_id, results)
```

## 📖 推荐书籍和课程

### 经典教材
1. **《深度学习》** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
   - 深度学习的经典教材
   - 系统介绍了深度学习的理论基础

2. **《图论导引》** - Douglas B. West
   - 图论的入门教材
   - 涵盖图论的基本概念和算法

3. **《统计学习方法》** - 李航
   - 机器学习的经典中文教材
   - 理论严谨，适合深入学习

### 专业书籍
1. **《Graph Representation Learning》** - William L. Hamilton
   - 图表示学习的专门教材
   - 从基础到前沿的全面介绍

2. **《Networks, Crowds, and Markets》** - David Easley, Jon Kleinberg
   - 网络分析的经典教材
   - 结合经济学和社会学视角

### 在线课程
1. **CS224W: Machine Learning with Graphs** (Stanford)
   - Jure Leskovec教授主讲
   - 最权威的图机器学习课程

2. **Deep Learning Specialization** (Coursera)
   - Andrew Ng教授主讲
   - 深度学习的入门课程

3. **CS231n: Convolutional Neural Networks** (Stanford)
   - 计算机视觉和深度学习基础

### 论文和综述
1. **Graph Neural Networks: A Review of Methods and Applications**
   - 图神经网络的全面综述
   - 适合入门和参考

2. **A Comprehensive Survey on Graph Neural Networks**
   - 另一篇重要的GNN综述
   - 更新更及时

## 🎯 学习路径规划

### 初学者路径 (3-6个月)
1. **数学基础** (4周)
   - 线性代数复习
   - 概率论基础
   - 微积分基础

2. **图论基础** (4周)
   - 图的基本概念
   - 图算法
   - NetworkX实践

3. **深度学习基础** (6周)
   - 神经网络原理
   - PyTorch基础
   - 经典深度学习模型

4. **GNN入门** (6周)
   - GNN基本概念
   - 经典模型实现
   - 框架使用

### 进阶路径 (6-12个月)
1. **理论深入** (8周)
   - GNN理论分析
   - 高级架构
   - 最新研究

2. **应用实践** (12周)
   - 项目开发
   - 真实数据处理
   - 系统部署

3. **研究方向** (持续)
   - 论文阅读
   - 创新研究
   - 学术写作

### 专家路径 (12个月以上)
1. **深度研究** 
   - 原创性研究
   - 学术论文发表
   - 开源贡献

2. **技术领导**
   - 团队协作
   - 技术方案设计
   - 行业应用

## 🔗 在线资源链接

### 数据集下载
- [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/)
- [Graph Classification Datasets](https://chrsmrrs.github.io/datasets/)
- [Open Graph Benchmark](https://ogb.stanford.edu/)

### 工具和框架
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Deep Graph Library](https://www.dgl.ai/)
- [NetworkX](https://networkx.org/)

### 学术资源
- [Papers with Code - Graph Neural Networks](https://paperswithcode.com/methods/category/graph-neural-networks)
- [Awesome Graph Neural Networks](https://github.com/thunlp/GNNPapers)
- [Graph Deep Learning](https://github.com/gordicaleksa/pytorch-GAT)

建立这个完整的资源库将为你的GNN学习之旅提供强有力的支持！
