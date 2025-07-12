# 论文标题

GraphMAE: Self-Supervised Masked Graph Autoencoders

## 基本信息
- **作者**: 
- **期刊/会议**: KDD2
- **年份**: 2022
- **引用数**: 334 (25.7.10)
- **代码**: https://github.com/THUDM/GraphMAE

## 背景与痛点

针对Graph Autoencoder(MAE-Maked autoencoder是SSL的一种)的缺陷进行改进，通过精心设计过的：重构目标、训练过程、损失函数、模型架构来尝试改进。主要专注于掩码策略和放缩余弦误差增强Graph的训练鲁棒性

对抗式SSL痛点：自监督学习(SSL)主要有两类：对抗式学习和生成式学习。图领域主要是对抗式学习，但是该方法依赖于设计精妙的训练策略和高质量的数据增强(通常依赖于启发式方法，效果因图而已)

现有生成式图SSL：任务类型有，预测缺失边、恢复节点数据、迭代地重建节点和边、链路预测和图聚类。

困境在于：
1. 过度强调链路重建，在边预测和节点聚类任务好，但节点和图预测不好
2. 没有扰动的特征重构可能不够强健
3. MSE作为损失并不够好，它本身有一些缺陷
4. 对于解码器，图嵌入信息可能并不丰富，MLP解码可能不够用

补充--相关工作：
* 对抗式SSL：主要包括负样本采样、架构的精妙设计和数据增强。当然图上并不像CV有系统化的增强理论，多靠手动增强，效果参差不齐。
* 生成式SSL：可以细分为自回归autoregressive和自编码autoencoder两个类型
    * 自回归方法因为图的内部无序性没什么道理可言（自回归本质类似于预测L1L2后，在此基础上预测L1L2L3的概率，需要一个固有顺序，但图不具备这种特点）
    * 自编码方法不要求解码时遵循固定的顺序

## 核心贡献
1. 主要贡献点1：Masked featrue reconstruction, 只专注于被掩码的特征的重建。(传统生成式SSl还关注完整数据和数据分布的重建， 当然这主要是MAE的创新点)
2. 主要贡献点2：Scaled cosine error. 用于替代原有的MSE损失，具体优势有待阅读
3. 主要贡献点3：Re-mask decoding，有助于encoder生成一个额外的、富含预测信息的预测目标。

## 方法概述
### 问题定义
- 输入：$G = (V, A, X)$
- 输出：$H = f_E(A, X), G' = f_D(A, H)$
- 目标：自监督学习图表示

是一个Graph autoencoder问题

### 核心方法
- 关键思想：识别和纠正现有GAE方法的不足
- 技术细节：
> 预测掩码的特征：实验证明将结构相似性作为目标对于下游任务没有什么帮助
$$\tilde x_i = \left\{\begin{array}{ll} x_{[M]} \quad v_i \in \tilde{V} \\ x_i \quad v_i \notin \tilde{V} \end{array} \right. \quad 采样节点子集\tilde V \subset V $$
> 对code重新掩码：code的维度很高，在CV领域没什么影响，因为图片本身就包含高维度信息，但是图用高维度code表示可能存在冗余。这里对部分节点的特征进行掩码
$$\tilde h_i = \left\{\begin{array}{ll} h_{[M]} \quad v_i \in \tilde{V} \\ h_i \quad v_i \notin \tilde{V} \end{array} \right. \quad 采样节点子集\tilde V \subset V $$

> 用GNN替代MLP作为decoder


> 使用scaled-cosine-error.(中间具体使 用哪种GNN没有固定)

$$L_{SCE} = \frac{1}{\tilde V} \sum_{v_i \in \tilde V} (1 - \frac{x_i^T z_i}{||x_i||  · ||z_i||})  \quad Z = f_D(A, \tilde H)$$

- 算法流程：
    * 输入图时，随机选择节点划分，用mask-token替代选中的部分节点子集的特征
    * 用encoder生成节点表示
    * 节点表示用相同方式re-mask后输入decoder
    * decoder重建原始节点特征，以scaled cosine error为指标


## 实验分析
### 数据集
- 数据集名称和特点
    * 节点分类：遵循GraphSage的归纳设置，`PPI`和`Reddit`数据集在没有见过的节点和图上进行测试，`Cora, Citeseer, PubMed, Ogbn-arxiv`采用直推式学习
    * 图分类：`MUTAG, PROTEINS, NCI1`使用节点特征作为输入,`IMDB-B, IMDB-M, REDDIT-B, COLLAB`使用节点度作为输入。
    * 迁移学习：学习分子属性预测。在从`ZINC15`中抽样出的两百万无标记分子上训练，在8个分类基准数据集上微调
- 实验设置
    * 节点分类：先训练Encoder，然后冻结Encoder参数，训练下游任务分类器。编解码器均采用GAT
    * 图分类：编解码器采用GIN，下游任务分类器采用LIBSV，评估采用10折叠的交叉验证，5次运行统计均值和方差
    * 迁移学习：下游任务用脚手架分割来模拟真实世界用例，输入节点特征有原子数、手性标记，边特征有连接类型和方向。以五层GIN为Encoder，1层GIN作为Decoder，进行10次实验统计均值和ROC-AUC的标准差

### 主要结果
- 性能对比表格：略
- 消融实验结果：
    * 重构标准：SCE v.s. MSE，MSE使节点级任务劣化更大
        * $\gamma$:不同数据集最优值不同，最优取值集中在2, 3
    * 掩码和重掩码：去除掩码劣化更多，重掩码也有一定劣化
    * 掩码率：cora和MUTAG是越高越好，PubMed是0.5最好。
    * 解码器类型：GAT再节点级任务表现最好，GIN在图级别任务表现最好
- 可视化分析：略

## 优缺点分析
### 优点
- 方法的创新性
- 实验的充分性
- 结果的显著性

### 缺点
- 方法的局限性
- 实验的不足
- 可能的改进方向

## 启发和思考
- 对自己研究的启发
- 可能的扩展方向
- 实现的难点

## 相关工作
- 引用的重要论文
    * 《Masked Autoencoders Are Scalable Vision Learners》2021 Facebook 何凯明
- 后续的相关工作
- 研究脉络梳理