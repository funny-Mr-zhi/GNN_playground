# 🧠 图神经网络学习仓库 (GNN Playground)

> 一个系统性学习图神经网络(Graph Neural Networks)的综合仓库，从基础理论到实际应用的完整学习路径。

## 🎯 学习目标

- **理论基础**：掌握图论、深度学习和图神经网络的核心理论
- **框架应用**：熟练使用主流GNN框架进行开发
- **论文研读**：跟踪GNN领域的最新研究进展
- **项目实践**：通过实际项目应用GNN解决真实问题

## 📚 学习路径

### 阶段1：基础理论 (1-2个月)
- 数学基础：线性代数、概率论、图论
- 深度学习基础：神经网络、反向传播、优化方法
- 图表示学习：传统图表示方法、图嵌入

### 阶段2：GNN理论 (2-3个月)
- GNN基本概念和原理
- 经典GNN架构：GCN、GraphSAGE、GAT等
- 高级主题：图池化、异构图、动态图

### 阶段3：框架实践 (1-2个月)
- PyTorch Geometric实践
- DGL框架学习
- 其他GNN框架探索

### 阶段4：论文研读 (持续进行)
- 经典论文精读
- 最新研究跟踪
- 不同应用领域的GNN研究

### 阶段5：项目实践 (2-3个月)
- 节点分类、图分类、链接预测
- 真实世界应用项目
- 开源贡献

## 📁 项目结构

```
GNN_playground/
├── 01_foundations/                    # 基础理论
│   ├── mathematics/                   # 数学基础
│   ├── graph_theory/                  # 图论基础
│   └── deep_learning/                 # 深度学习基础
├── 02_theory/                         # GNN理论
│   ├── gnn_basics/                    # GNN基础概念
│   ├── architectures/                 # 经典架构
│   └── advanced_topics/               # 高级主题
├── 03_frameworks/                     # 框架学习
│   ├── pytorch_geometric/             # PyTorch Geometric
│   ├── dgl/                          # Deep Graph Library
│   └── spektral/                     # Spektral (TensorFlow)
├── 04_papers/                         # 论文研读
│   ├── classic_papers/                # 经典论文
│   ├── recent_advances/               # 最新进展
│   └── applications/                  # 应用领域
├── 05_implementations/                # 代码实现
│   ├── from_scratch/                  # 从零实现
│   ├── tutorials/                     # 教程代码
│   └── benchmarks/                    # 性能测试
├── 06_projects/                       # 项目实践
│   ├── node_classification/           # 节点分类
│   ├── graph_classification/          # 图分类
│   ├── link_prediction/               # 链接预测
│   └── real_world_applications/       # 真实应用
└── 07_resources/                      # 学习资源
    ├── datasets/                      # 数据集
    ├── tools/                         # 工具集
    └── books_courses/                 # 书籍课程
```

## 🚀 快速开始

### 环境配置
```bash
# 创建虚拟环境
conda create -n gnn-learning python=3.9
conda activate gnn-learning

# 安装基础依赖
pip install torch torchvision torchaudio
pip install torch-geometric
pip install dgl-cu118  # 根据CUDA版本选择
pip install networkx matplotlib seaborn
pip install jupyter notebook
```

### 学习建议

1. **循序渐进**：按照学习路径逐步推进，不要跳过基础部分
2. **理论结合实践**：每学完一个理论概念，立即通过代码实现加深理解
3. **记录总结**：在每个目录下创建学习笔记，记录重要概念和心得
4. **定期复习**：定期回顾之前学过的内容，构建知识体系

## 📖 学习计划详解

### 第一阶段：基础理论 (预计6-8周)

#### 第1-2周：数学基础
- [ ] 线性代数：矩阵运算、特征值分解、SVD
- [ ] 概率论：概率分布、贝叶斯定理、期望与方差
- [ ] 优化理论：梯度下降、随机梯度下降、Adam优化器
- [ ] 图论基础：图的基本概念、图的表示、图的遍历

#### 第3-4周：深度学习基础
- [ ] 神经网络基础：前向传播、反向传播
- [ ] 激活函数、损失函数、正则化
- [ ] CNN、RNN基础概念
- [ ] 深度学习框架使用（PyTorch）

#### 第5-6周：传统图方法
- [ ] 图的基本算法：BFS、DFS、最短路径
- [ ] 图的中心性度量：度中心性、介数中心性、特征向量中心性
- [ ] 传统图嵌入：Node2Vec、DeepWalk、LINE
- [ ] 图的谱理论基础

### 第二阶段：GNN理论 (预计8-10周)

#### 第7-9周：GNN基础
- [ ] GNN的基本思想和动机
- [ ] 消息传递范式
- [ ] 图卷积的数学原理
- [ ] GNN与CNN的关系

#### 第10-12周：经典GNN架构
- [ ] Graph Convolutional Networks (GCN)
- [ ] GraphSAGE
- [ ] Graph Attention Networks (GAT)
- [ ] Graph Isomorphism Networks (GIN)

#### 第13-16周：高级主题
- [ ] 图池化方法
- [ ] 异构图神经网络
- [ ] 动态图神经网络
- [ ] 图生成模型

### 第三阶段：框架实践 (预计4-6周)

#### 第17-19周：PyTorch Geometric
- [ ] 环境搭建和基本使用
- [ ] 数据处理和批处理
- [ ] 实现经典GNN模型
- [ ] 自定义图神经网络层

#### 第20-22周：DGL框架
- [ ] DGL基础概念和API
- [ ] 异构图处理
- [ ] 大规模图训练
- [ ] 分布式训练

### 第四阶段：论文研读 (持续进行)

#### 经典论文 (必读)
- [ ] Spectral Networks and Locally Connected Networks on Graphs (2014)
- [ ] Semi-Supervised Classification with Graph Convolutional Networks (2017)
- [ ] Inductive Representation Learning on Large Graphs (2017)
- [ ] Graph Attention Networks (2018)
- [ ] How Powerful are Graph Neural Networks? (2019)

#### 最新进展 (选读)
- [ ] GraphTransformer相关论文
- [ ] 大规模图神经网络
- [ ] 图对抗学习
- [ ] 可解释性图神经网络

### 第五阶段：项目实践 (预计8-10周)

#### 第23-25周：基础任务
- [ ] 节点分类：Cora、CiteSeer、PubMed数据集
- [ ] 图分类：TUDataset分子数据集
- [ ] 链接预测：社交网络、知识图谱

#### 第26-30周：高级应用
- [ ] 推荐系统中的GNN应用
- [ ] 化学分子性质预测
- [ ] 社交网络分析
- [ ] 知识图谱推理

## 📊 进度跟踪

创建一个学习进度表格，定期更新学习状态：

| 阶段 | 内容 | 状态 | 完成时间 | 备注 |
|------|------|------|----------|------|
| 基础理论 | 数学基础 | 🔄 进行中 | - | - |
| 基础理论 | 深度学习 | ⏳ 待开始 | - | - |
| GNN理论 | 基础概念 | ⏳ 待开始 | - | - |
| ... | ... | ... | ... | ... |

## 🤝 贡献指南

欢迎为这个学习仓库贡献内容：

1. 添加学习笔记和总结
2. 补充代码实现和示例
3. 推荐优质的学习资源
4. 分享学习心得和经验
5. 报告错误和改进建议

## 📝 学习记录

建议为每个学习阶段创建详细的学习记录：

- **理论笔记**：概念理解、数学推导
- **代码实现**：关键算法的实现
- **实验结果**：模型性能对比
- **心得体会**：学习过程中的思考

## 🔗 相关资源

- [PyTorch Geometric官方文档](https://pytorch-geometric.readthedocs.io/)
- [DGL官方文档](https://docs.dgl.ai/)
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)

## 📧 联系方式

如果在学习过程中遇到问题，欢迎通过以下方式交流：

- 在Issues中提出问题
- 参与Discussions讨论
- 提交Pull Request贡献内容

---

**开始你的图神经网络学习之旅吧！** 🚀
