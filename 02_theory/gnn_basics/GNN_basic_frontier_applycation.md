> 参考书目《图神经网络--基础、前沿与应用》第二部分：基础
* 4章：用于节点分类到图神经网络
* 5章图神经网路地表达能力
* 6章图神经网络的可扩展性
* 7章图神经网络的可解释性
* 8章图神经网络的对抗鲁棒性

# 图神经网络的表达能力

## 背景：深度学习模型的表达能力

**模型的表达能力**:机器学习问题可以被抽象为学习从某个特征空间到某个目标特征空间的映射$f^*$。这个问题的解决方案通常是由一个模型$f_\theta$给出的，该模型通过优化一些参数 $\theta$ 来近似 $f^*$。实践中真实的$f^*$通常是先验未知的。因此人们希望能够在一个相当广泛的范围内近似于$f^*$。对这个范围的估计称为该模型的`表达能力`。

> 在深度学习中`特征空间`是一个核心且基础的概念，它贯穿了模型设计、数据表示和任务优化的全过程。
>
> 简单来说，特征空间是一个由"特征维度"构成的数学空间，其中每个样本(数据点)，被表示为一个向量(或张量)，向量的每个维度对应样本的一个"特征"，而向量的数值则描述了该特征值的具体值。(学习线性代数的向量空间有助于理解这一部分)
>
> 深度学习的核心是**自动学习特征空间**，深度学习模型通过多层非线性变换，将原始数据(像素、文本符号、图节点属性)从"原始特征空间"逐步映射到"抽象特征空间"。每一层网络都对应中间特征空间，且层次越高，特征越抽象，越接近目标任务。（以图像分类为例，空间映射过程：`输入层`原始的像素空间-->`卷积层1`边缘，颜色特征空间-->`卷积层2`纹理，局部形状特征空间-->`全连接层`物体部件，语义特征空间-->`输出层`类别概率空间）
>
> 特征空间的核心作用体现在两个方面：一是度量样本相似性，在特征空间中，向量的距离被用来度量样本的"语义相似性"；二是简化任务复杂度，学习一个"友好的特征空间"，让任务的决策边界更简单。不同深度学习的特征空间有其特殊性，但核心逻辑一致。CNN聚焦"空间局部相关性"，通过卷积捕获局部特征，逐步聚合为全局特征；RNN聚焦"时序相关性"，通过记忆机制将序列信息编码为固定维度的向量；GNN聚焦"图结构相关性"，通过消息传递将节点属性于拓朴结构融合为节点/图的向量表示。
>
> 特征空间是深度学习中"数据的数学化身"，原始数据通过多层变换，从"杂乱的原始空间"进入"有序的抽象空间"，这个空间的好坏直接决定模型的能力；深度学习的"智能"本质就是自动学习出这样的优质特征空间，替代人工设计的局限。理解特征空间能帮助我们更清晰地分析模型行为，也为调优模型(如设计更有效的特征转换层)提供了方向。


**归纳偏置**: 在实践中，对参数的约束通常是从我们对于数据的先验知识中获得的，这些先验知识被称为归纳偏置

> 神经网络`NN`以其强大的表达能力而出名。只有一个隐藏层且由sigmoid函数激活的神经网络理论上以均匀逼近在紧空间上定义的任何连续函数。(这一证明后来被泛化到任意挤压激活函数, 1989)。
>
> 深度神经网络`DNN`架构通过堆叠多个隐藏层，用明显少于浅层神经网络的参数实现足够好的近似，通常**基于数据具有分层结构的事实**。
>
> 然而深度神经网络并不局限于某种特定的数据类型，用于支持特定类型数据的专用神经网络已经被开发出来，如循环神经网络`RNN`基于**数据在时间上保持平移不变性的有效模式**，而卷积神经网络`CNN`基于**数据在空间上保持平移不变性的有效模式**。
>
> 以上`DNN, RNN, CNN`都已经建立在针对特殊数据类型的归纳偏置上, 通过在模型的无限解空暗金中引入合理的假设和约束，可以缩小求解空间，提高模型在目标领域的泛化性。而图通常用于模拟多个元素之间的复杂关系和相互作用，这是另一种重要的数据类型，需要探寻独属于图/网络的归纳偏置。

与时间序列和图像相比，图是不规则的，图机器学习背后的一个基本假设是：预测的目标应该与图的节点顺序无关。为了与这一假设相匹配，GNN持有被称为排列不变量的一般归纳偏置。GNN给出的输出应该独立于图节点的索引分配方式，从而独立于它们被处理的顺序，要求其参数与节点排列无关，并在整个图共享。

> 图的**排列不变性**：时间序列数据的顺序直接反应时间依赖关系，图像数据的空间位置直接对应语义。他们天然具有顺序(一维或二维)。但图节点的索引是人为赋予的，并不是图天然具有，不对应任何图的特征(这里通过同构图可以加深理解，图中每个节点并不天然具有可排序的特点，想要特指某个节点或以特定顺序存储时需要认为指定一个排序，但该排序不包含任何图信息，同构图不考虑特定的节点排序，GNN要求输入同构图时输出相同， 结果与节点顺序无关)。
>
> 本质上，这是有关样本之间关系的归纳偏置：时间序列和图形天然具有固定的相关关系，二者天然具有固定顺序,排列上相邻就表示样本之间存在依赖，可以交换一些信息。但图节点的依赖关系不是固定的，它不具有可排序的特点，样本之间的关系通过额外引入的连接关系单独指定，即样本之间的关系不是固定不变的。
> 
> 进一步理解，对于具有天然顺序的数据，可以理解为一种遵循特殊规则的一类同构图。比如时间序列可以理解为路径，图像可以视为二维网格图(都是规则图)，这种数据中，样本之间的信息交互模式是固定不变的，只需要知道样本的顺序就可以实现。而对于更一般的图数据，样本之间的信息交互模式还需要依赖于额外指定的连接关系。这就导致图的归纳假设是需要满足排列不变性，每个图在样本间的信息交互方式上更具特殊性，该约束的引入也就使得图神经网络的解空间限制更强，更加聚焦于图问题的求解。

对于图表达能力的研究：
* 标准的`消息传递GNN`能够达到`一维Weisfeiler-Lemhan`测试的极限
* 克服`1-WL`的策略：注入随机属性、注入确定性距离属性、建立高阶GNN

## 图表征学习和问题的提出

以下从数学定义的角度完善图表征学习的相关定义。

**图结构数据**：设$\mathcal{G} = (\mathcal{V}, \Epsilon, X)$表示一个属性图。包括节点集合，边结合和节点属性矩阵$X \in R^{|\mathcal V| \times F}$。引入$A \in \{0, 1\}^{|\mathcal V|\times|\mathcal V|}$表示图$\mathcal G$的邻接矩阵，则可以将图表示为$\mathcal G = (A, X)$

**图表征学习**：给定一个图$\mathcal G \in \Gamma$ ，定义特征空间为$\mathcal X := \Gamma \times \mathcal S$，其中$\Gamma$表示图结构数据的空间，$\mathcal S$为所有感兴趣的节点子集。$\mathcal X$中的一个点可以表示为$(\mathcal G, S)$，其中S为一个感兴趣的节点子集。我们称$(\mathcal G, S)$为图表征学习(GRL)的一个实例，每个实例与目标空间$\mathcal Y$中的一个y相关联。假设特征和目标之间的标注关联函数可以表示为$f^*：\mathcal X \rightarrow\mathcal Y, f^*(\mathcal G, S) = y$，给定一组训练实列$\Theta$和一组测试实例$\psi $，图标表征学习的目标就是学习一个基于$\Theta$的函数f，使得f在$\psi$上近似于$f^*$

问题类型| $\mathcal G$ | $\mathcal S$
--|--|--
图分类问题|包含多个子图|默认为整个节点结合
节点分类问题|取决于节点之间联系跨越的跳数,可以是$\mathcal S$周围的局部诱导子图，也可以是整个图|一个实例中的S对应单个节点
链接预测问题| 可以是$\mathcal S$周围的局部诱导子图，也可以是整个图|一个实例中的S对应一对节点

**基本假设**：对于一个图表征学习问题，挑选任意两个GRL实例$(\mathcal G^{(1)}, S^{(1)}),(\mathcal G^{(2)}, S^{(2)}) \in \mathcal X$，基本的假设是如果$(\mathcal G^{(1)}, S^{(1)}) \cong (\mathcal G^{(2)}, S^{(2)})$ （即通过映射函数$\pi$映射后，二者忽略节点命名的情况下完全相同）那么他们在$\mathcal Y$中对应的目标也是相同的

> 基于这个基本假设，可以很自然地将引入相应的排列不变性作为归纳偏置。所有的图表征学习模型都应该满足这种归纳偏置

**排列不变性**：对于任意$(\mathcal G^{(1)}, S^{(1)}) \cong (\mathcal G^{(2)}, S^{(2)})$，都有$f(\mathcal G^{(1)}, S^{(1)}) = f(\mathcal G^{(2)}, S^{(2)})$，那么模型$f$满足排列不变性

**表达能力**：考虑一个图表征学习问题的特征空间$\mathcal X$和一个定义在$\mathcal X$上的模型$f$。定义另一个子空间$\mathcal X(f)$是商空间$\mathcal X / \cong$的一个子空间使得对于任意两个GRL实例$(\mathcal G^{(1)}, S^{(1)}),(\mathcal G^{(2)}, S^{(2)}) \in \mathcal X(f)$,都有$f(\mathcal G^{(1)}, S^{(1)}) \neq f(\mathcal G^{(2)}, S^{(2)})$，则$\mathcal X$的大小体现了模型f的表达能力。对于两个模型$f^{(1)}$和$f^{(2)}$，如果$\mathcal X(f^{(1)} \supset X(f^{(2)})$我们就说$f^{(1)}$比$f^{(2)}$更具表达能力。

> 该定义中的表达能力着眼于模型如何区分非同构的GRL实例，因而与传统意义上着眼于函数逼近意义上的神经网络的表达能力不完全一致。如果模型$f$不满足排列不变性，那么为图表征学习提供的$f$的飙到能力是意义不大的，如果没有这样的约束，图神经网络就可以近似所有的连续函数，其中包括了区分任意非同构的GRL实例的连续函数(通过引入排列不变性的归纳偏置， 将模型$f$的目标空间范围缩小， 剔除问题解不可能出现的部分空间)

**由此，关键问题就在于如何为图表征学习问题建立最具表达能力的排列不变性模型，特别是GNN**

## 强大的消息传递神经网络


# 图神经网络的可解释性




