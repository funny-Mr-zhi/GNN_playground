> DGL是一个python包，在现有DL框架基础上轻松实现GNN模型族。

* 支持的框架
    * Pytorch
    * TensorFlow
    * MXNet
* 优势
    * 对message passing的灵活控制
    * 通过自动分批和微调矩阵核心优化加速
    * 多CPU/GPU训练，支持大规模图训练

## A Blitz Introduction to DGL 

### Node Classification with DGL
> **节点分类**是图数据上常见的任务。GNN能够有效地结合拓扑结构信息和节点特征信息学习**节点表示**

以半监督的节点分类任务为例，使用Cora数据集(论文引用网络)

代码：`hello_dgl/01NodeCla.py`。熟悉了图数据的加载`dlg.data`加载经典数据集、图模型的定义：从`dgl.nn.xxxConv`获取图卷积层

暂存问题：训练100轮，train很快到了100%的准确率，后续val和test不会再有提升。
该问题如何解决？
* 重新划分数据集？(待验证)

### How Dose DGL Represent A Graph

代码：`hello_dgl/02DglGraph.py`。
了解dlg图的基本构成、特征读写、图上的操作、图的读写
* 图基本构成
    * dgl.graph()构图,必要的信息只需要连边情况
* 图的基本查询
    * g.edges()
    * g.num_nodes()
    * g.num_edges()
    * g.in_degrees(4) #获取4节点的入度
* 图的节点、边特征
    * dgl.ndata["feat"] = torch.rand(num_nodes, x, y)
    * dgl.edata["feat"] = torch.rand(num_edges, z, k)
    * 主要是`ndata`和`edata`两个属性，具体属性key可以自己定义
* 图的操作
    * g.subgraph([x, y, z]) 用x,y,z节点构建子图
    * g.edge_subgraph([x, y, z]) 用x,y,z边构建子图
    * dgl.add_reverse_edges(g) 添加反向边，有向图变无向图
* 读写
    * dgl.save("path", g)
    * dgl.load_graphs("path")

### Write your own GNN module

### Link Prediction using Graph Neural Networks

### Training a GNN fro Graph Classification

### Make Your Own Dataset



