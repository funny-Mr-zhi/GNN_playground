import os

os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import numpy as np
import torch

# 基础构图,三种方法等效，边用两组一维向量表示
g1 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]), num_nodes=5)
g2 = dgl.graph(
    (torch.LongTensor([0, 1, 2, 3]), torch.LongTensor([1, 2, 3, 4])),
    num_nodes=5
)
g3 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))

print("打印边信息", g1.edges())
# (tensor([0, 1, 2, 3]), tensor([1, 2, 3, 4])) 两个一维tensor

# 为节点和边特征赋值，g有两个属性ndata和edata
g1.ndata['x'] = torch.randn(5, 3)     # 节点特征，5个节点，每个节点3维特征
g1.ndata['y'] = torch.randn(5, 2, 3)  # 节点特征，5个节点，每个节点2维特征，每个特征3维
g1.edata['a'] = torch.randn(4, 3)     # 边特征，4条边，每条边3维特征

print("节点特征打印", g1.ndata['x'])

# 查询图属性
print("图的节点数", g1.num_nodes())
print("图的边数", g1.num_edges())
print("入度查询", g1.in_degrees(5))  # 查询节点5的入度

# 图上的变换操作
sg1 = g1.subgraph([0, 2, 3])  # 子图，用节点索引创建子图
sg2 = g1.edge_subgraph([0, 2])  # 边子图,用边索引创建子图

print("子图原节点DI", sg1.ndata[dgl.NID])
print("子图原边DI", sg1.edata[dgl.EID])
print("子图原节点DI", sg2.ndata[dgl.NID])
print("子图原边DI", sg2.edata[dgl.EID])

undir_g = dgl.add_reverse_edges(g1)  # 无向图，添加反向边
print("无向图的边信息", undir_g.edges())

# 保存和读取图
dgl.save_graphs('data/dgl_dataset/graph_g1.dgl', g1)  # 保存图到文件
print("保存图到文件完成")
dgl.save_graphs('data/dgl_dataset/graph_g_list.dgl', [g1, g2, undir_g])  # 保存图到文件
print("保存图列表到文件完成")

(g,), _ = dgl.load_graphs('data/dgl_dataset/graph_g1.dgl')  # 从文件读取图
print(g)
(g1_, g2_, undir_g_), _ = dgl.load_graphs('data/dgl_dataset/graph_g_list.dgl')  # 从文件读取图
print(g1_)
print(g2_)
print(undir_g_)