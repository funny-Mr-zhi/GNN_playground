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

通用GNN模型
 $$m^{(l)}_{u \rightarrow v} = M^{(l)}(h_v^{l-1}, h_u^{l-1}, e_{u \rightarrow v}^{l-1}) Message function$$
$$m^{(l)} = \sum_{u \in N(v)} m^{(l)}_{u \rightarrow v} Reduce function$$
$$h_v^{(l)} = U^{(l)} (h_v^{(l-1)}, m_v^{(l)}) Update function$$
> 可以理解为从邻居搜集消息，将消息进行规约，利用规约后的消息更新现有节点表示。

以具体的GraphSAGE为例进行手动实现
$$h_{N(v)^k} \leftarrow Average({h_u^{k-1}, \forall u \in N(v)})$$
$$ h_v^k = Relu(W^k · Concat(h_v^{k-1}, h_{N(v)}^k))$$

代码：`hello_dgl/03BuildModel.py`中手动实现了`SAGEConv`

重点理解三种函数如何在代码中体现
```python
def forward(self, g, h):
    with g.local_scope():   # 函数结束后清除图的局部数据
        g.ndata['h'] = h    # 当前特征值记录
        g.update_all(       # 执行message和reduce部分
            message_func = fn.copy_u('h', 'm')  #邻居搜集信息message func
            reduce_func = fn.mean('m', 'h_N')   #对信息进行聚合
        )
        h_total = torch.cat([h, g.ndata['h_N']], dim=1)
        return slef.linear(h_total)     # 完成update函数的计算过程
```
在`g.update_all`里面定义`message_func`和`reduce_func`, 最后的`update_func`自己在外面单独实现即可
> 深入理解可能需要解读一下update_all函数的具体实现
> 消息传递函数fn.copy_u('h', 'm'), fn.u_mul_e('h', 'w', 'm')

```python 
# 自定义等价message和reduce function
def u_mul_e_udf(edges): #消息传递函数应该是遍历应用于所有边，结果存放到node.mailbox里面
    return {"m": edges.src["h"] * edges.data["w"]}

def mean_udf(nodes):    # reduce函数遍历应用于所有节点
    return {"h_N": nodes.mailbox["m"].mean(1)}
```

>Q：后续发现将`weighted_SAGE`的边权重全部设置为0时，训练训练效果中trainAcc可以到1，但验证和测试效果不佳。此时`WeightModel`应该就类似于两个`Linear`组成的`MLP`，只不过每层h拼接了一个同样维度的`torch.zeros`后过`Linear`于是考虑拿MLP也训练一次，发现TrainAcc也能到1，但是需要的轮次更多，且验证和测试效果更差尝试分析原因: SAGE第二层忘记换成`zeros`了，换完以后完全一样，MLP在中间层拼接zeros后于SAGE完全一样。当然MLP是否拼接一个维度完全相同的zero对结果影响不大

更深层次定制：自定义message,reduce函数
```
def u_mul_e_udf(edges):
    return {"m": edges.src["h"] * edges.data["w"]}

def mean_udf(nodes):
    return {"h_N": nodes.mailbox["m"].mean(1)}
```
> 定制消息传递函数可以后续深入了解一下 TODO

### Link Prediction using Graph Neural Networks

这里学习用GNN做链路预测的工作流
* 将一定存在的边当作正例，一定不存在的边当作负例
* 采样正例和负例后分配到训练集和测试集当中
* 用二元分类度量(mttirc)评估模型

代码: `hello_dgl/04LinkPre.py`

* model没有什么特殊的，就是普通二层GraphSAGE
* Predictor，接在模型后面计算边表示，两种
    * DotPredictor：直接用g.apply_edge(fn.u_dot_v("h", "h", "source"))计算端点点积
    * MLPPredictor：自定义fn，类似MLP利用两侧节点属性计算边预测结果（效果更好）
* 图处理：
    * 将一张图拆分为四个部分，正反例的训练、测试图（边，重点在于操作方式，如何分离）
* 训练与验证
    * model用移除了test部分边的图作为输入，学习输出所有节点h
    * pred用train_pos_g和train_neg_g作为输入，获取正反例预测结果
    * loss用F.binary_cross_entropy计算
    * 验证用test_pos_g和test_neg_g作为输入
> 对于GNN模型，使用移除test边的图进行训练，输出所有节点h`train_g = dgl.remove_edges(g, eids[:test_size])`
>
> 对于pred模型，输入是特定类型的图(边)和所有节点h，利用边两端点节点表示预测边结果

> 定制节点表示计算apply_edge深入了解 TODO

> 对于如何获取反例并分解为四个图的操作，后续可以作为技能锻炼进行练习 TODO

> 理解链接预测的核心思路，在节点表示学习后接一个预测器，对于边的结果预测主要依赖边两端节点的表示，此时边上的apply_edge效果类似于节点上的update_all。

### Training a GNN fro Graph Classification

### Make Your Own Dataset


