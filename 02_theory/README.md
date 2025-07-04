# 🧠 图神经网络理论 (GNN Theory)

深入学习图神经网络的核心理论，掌握各种GNN架构的原理和应用。

## 🎯 学习目标

- 理解GNN的基本原理和动机
- 掌握主流GNN架构的设计思想
- 学习高级的GNN理论和技术
- 能够根据问题特点选择合适的GNN模型

## 📁 目录结构

### 🔰 gnn_basics/
GNN基础概念，包括：
- **消息传递框架**：消息函数、聚合函数、更新函数
- **图卷积原理**：谱域和空域的图卷积
- **GNN的表达能力**：Weisfeiler-Lehman测试、GNN的局限性
- **训练技巧**：过拟合、过平滑、梯度消失

### 🏗️ architectures/
经典GNN架构，包括：
- **Graph Convolutional Networks (GCN)**
- **GraphSAGE**
- **Graph Attention Networks (GAT)**
- **Graph Isomorphism Networks (GIN)**
- **其他重要架构**：ChebNet、FastGCN、GraphSaint等

### 🚀 advanced_topics/
高级主题，包括：
- **图池化**：DiffPool、SAGPool、Graph U-Net
- **异构图神经网络**：HGT、RGCN、HAN
- **动态图神经网络**：CTDNE、DynGEM、EvolveGCN
- **图生成模型**：GraphVAE、GraphRNN、GraphGAN
- **图对抗学习**：图对抗攻击与防御

## 📖 学习计划

### 第5-6周：GNN基础概念

#### 第5周：消息传递框架
- [ ] 学习消息传递的基本概念
- [ ] 理解GNN的通用框架
- [ ] 掌握图卷积的数学原理
- [ ] 实现简单的消息传递网络

**重点内容：**
```python
# 消息传递的基本框架
def message_passing(node_features, edge_index, edge_attr):
    # 1. Message: 计算消息
    messages = message_function(node_features, edge_index, edge_attr)
    
    # 2. Aggregate: 聚合消息
    aggregated = aggregate_function(messages, edge_index)
    
    # 3. Update: 更新节点特征
    updated_features = update_function(node_features, aggregated)
    
    return updated_features
```

#### 第6周：GNN理论基础
- [ ] 学习谱图理论基础
- [ ] 理解拉普拉斯矩阵的性质
- [ ] 掌握Weisfeiler-Lehman算法
- [ ] 了解GNN的表达能力限制

### 第7-10周：经典GNN架构

#### 第7周：Graph Convolutional Networks (GCN)
- [ ] 理解GCN的动机和设计思想
- [ ] 掌握GCN的数学公式推导
- [ ] 实现GCN的前向传播
- [ ] 在节点分类任务上测试GCN

**GCN核心公式：**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

#### 第8周：GraphSAGE
- [ ] 学习GraphSAGE的归纳学习思想
- [ ] 掌握不同的聚合函数
- [ ] 理解邻居采样策略
- [ ] 实现GraphSAGE算法

#### 第9周：Graph Attention Networks (GAT)
- [ ] 理解注意力机制在图上的应用
- [ ] 掌握GAT的注意力计算方法
- [ ] 学习多头注意力机制
- [ ] 实现GAT模型

#### 第10周：Graph Isomorphism Networks (GIN)
- [ ] 理解GIN的理论基础
- [ ] 学习GIN如何达到最大表达能力
- [ ] 掌握GIN的网络结构
- [ ] 比较不同GNN的表达能力

### 第11-14周：高级主题

#### 第11周：图池化方法
- [ ] 学习图级任务的挑战
- [ ] 掌握不同池化策略：全局池化、层次池化
- [ ] 实现DiffPool算法
- [ ] 了解可学习的池化方法

#### 第12周：异构图神经网络
- [ ] 理解异构图的特点和挑战
- [ ] 学习关系图卷积网络(RGCN)
- [ ] 掌握异构图注意力网络(HAN)
- [ ] 实现异构图上的任务

#### 第13周：动态图神经网络
- [ ] 了解动态图的建模方法
- [ ] 学习时序图神经网络
- [ ] 掌握连续时间动态网络嵌入
- [ ] 实现动态链接预测任务

#### 第14周：图生成模型
- [ ] 学习图生成的方法和应用
- [ ] 掌握变分图自编码器(GraphVAE)
- [ ] 了解图生成对抗网络
- [ ] 实现分子生成任务

## 🛠️ 实践项目

### 项目1：从零实现GCN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # 实现GCN的前向传播
        support = self.linear(x)
        output = torch.mm(adj, support)
        return F.relu(output)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.dropout(self.gc1(x, adj), self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
```

### 项目2：注意力机制可视化
- 实现GAT模型
- 可视化注意力权重
- 分析注意力模式
- 理解模型的决策过程

### 项目3：图分类基准测试
- 在TUDataset上比较不同GNN
- 实现图级特征提取
- 分析模型性能差异
- 总结最佳实践

## 📚 重要论文列表

### 基础论文（必读）
1. **Spectral Networks and Locally Connected Networks on Graphs** (Bruna et al., 2014)
   - 第一个现代GNN模型
   - 建立了谱域图卷积的理论基础

2. **Semi-Supervised Classification with Graph Convolutional Networks** (Kipf & Welling, 2017)
   - GCN模型，简化了谱图卷积
   - 奠定了现代GNN的基础

3. **Inductive Representation Learning on Large Graphs** (Hamilton et al., 2017)
   - GraphSAGE，解决了归纳学习问题
   - 引入了邻居采样的概念

4. **Graph Attention Networks** (Veličković et al., 2018)
   - 将注意力机制引入图神经网络
   - 提高了模型的可解释性

5. **How Powerful are Graph Neural Networks?** (Xu et al., 2019)
   - 分析了GNN的表达能力
   - 提出了GIN模型

### 进阶论文（选读）
1. **Hierarchical Graph Representation Learning with Differentiable Pooling** (Ying et al., 2018)
2. **Relational inductive biases, deep learning, and graph networks** (Battaglia et al., 2018)
3. **Graph Neural Networks: A Review of Methods and Applications** (Zhou et al., 2018)
4. **Heterogeneous Graph Attention Network** (Wang et al., 2019)
5. **Dynamic Graph Neural Networks** (Skarding et al., 2021)

## 🧪 理论分析

### GNN的表达能力分析
- **1-WL测试等价性**：大多数GNN的表达能力等价于1-阶Weisfeiler-Lehman算法
- **局限性**：无法区分某些图结构，如star图和triangle图的某些组合
- **改进方向**：高阶GNN、图结构特征、位置编码

### 常见问题及解决方案

#### 过平滑问题
- **问题**：随着层数增加，节点表示趋于相同
- **解决方案**：残差连接、跳跃连接、自适应层数

#### 过度挤压问题
- **问题**：图瓶颈导致信息损失
- **解决方案**：注意力机制、门控机制、多尺度聚合

#### 可扩展性问题
- **问题**：大图训练内存和计算开销大
- **解决方案**：采样方法、批处理、分布式训练

## 📊 学习评估

### 理论掌握检查点
- [ ] 能够清晰解释消息传递框架
- [ ] 理解不同GNN架构的优缺点
- [ ] 掌握GNN的理论分析方法
- [ ] 了解当前研究的前沿方向

### 实践能力检查点
- [ ] 能够从零实现经典GNN模型
- [ ] 可以根据任务选择合适的架构
- [ ] 具备调试和优化GNN模型的能力
- [ ] 能够阅读和理解最新论文

## 🔗 学习资源

### 在线课程
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- [Graph Neural Networks (Coursera)](https://www.coursera.org/learn/graph-neural-networks)

### 书籍推荐
- 《Graph Representation Learning》 - William L. Hamilton
- 《Graph Neural Networks: Foundations, Frontiers, and Applications》

### 代码资源
- [PyTorch Geometric Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- [DGL Examples](https://github.com/dmlc/dgl/tree/master/examples)

### 论文跟踪
- [Papers with Code - Graph Neural Networks](https://paperswithcode.com/methods/category/graph-neural-networks)
- [Awesome Graph Neural Networks](https://github.com/thunlp/GNNPapers)

完成这个模块后，你将对图神经网络有深入的理论理解，为后续的框架学习和实际应用打下坚实基础！
