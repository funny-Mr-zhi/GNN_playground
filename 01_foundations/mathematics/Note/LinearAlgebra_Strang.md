# Class 1
> **Class 1:Explanation of the geometic equation system** 方程组的几何解释

**基本问题**：N linear equations , n unknowns. 求解N个线性等式的n个未知数

**三种视角**
* Row picture
* column picture
* matrix from

以二元一次方程组为例
$$\begin{align} 2x - y = 0\\
-x+2y = 3 \end{align}$$
用线性代数表示
$$\left[
\begin{array}{ccc}
2 & -1 \\
-1 & 2 
\end{array}
\right] 
\left[
\begin{array}{ccc}
x\\y
\end{array}
\right] = \left[ \begin{array}{ccc} 0\\3 \end{array} \right]$$
* 从`Row picture`看，是二维空间的两条直线，交点即为解
* 从`column picture`看，是二维空间的两个向量，通过`linear combination`线性组合合成`(0, 3)`
* 从`Matrix form`看，计算左侧结构，可以当成矩阵乘法，也可以分解为`column picture`求解

具体解法中，图解只能在低维度使用，高维度不够直观，会用系统性的消元法解。这里重点关注`线性组合`，也就是理解列视角的看问题的方式。

一个常见的基本问题：
* Can I solve Ax = b for every b
* or say: Do the linear combinations of the columns fill N-D space
* 这个问题的答案与奇异\非奇异，可逆\非可逆有关

# Class 2
> elimination消元法解方程组

以三元一次方程组为例，一个自然的想法是利用消元法求解。

对应到矩阵上$Ax = b$，对角线上的元素称为主元，消元最终目的是主元下方的元素全部为0，且主元不为零。满足条件后，说明求解成功，通过回代可以求出所有变量的解。
* 从第一行开始，若当前主元行的主元不为零，则讲该行整体叠加到下面的行，确保主元下方元素都为0
* 若当前行主元为零，则可以交换主元行和其它行的位置，再次尝试第一步
* 逐行处理，若对角线主元全部非零且主元下方全部为零（上三角矩阵），则求解成功
    * 回代：在所有过程中间矩阵右侧补充b列，称为argumented matrix增广矩阵
    * b同步叠加操作，结果从最后一行开始向上逐行解出变量值
* 若无法满足上一步的要求，求解失败
> 该处理步骤可以和直接在方程组上进行消元的操作一一对应，直接目的就是消元。

**从另一个角度理解**：如何看待矩阵运算（一种行列变换）
```
|a b c| |x|      |a|     |b|     |c|
|d e f| |y| =  x |d| + y |e| + z |f|    //右侧列分解列叠加
|g h i| |z|      |g|     |h|     |i|

        |a b c|    
|x y z| |d e f| = x |a b c| + y |d e f| + z |g h i|  //左侧行分解行叠加
        |g h i|
 
|a b|   |0 1| 
|c d|   |1 0|   等效于对左侧矩阵进行列置换，右侧变换矩阵进行列变换

|0 1|   |a b|   
|1 0|   |c d|   等效于对右侧矩阵进行行置换，左侧变换矩阵进行行变换

|1 0|   |a   b|    |a    b|
|2 1|   |-2a d|  = |0 d+2b|
等效于对右侧矩阵进行行变换，第一行不变，第二行为二倍第一行叠加过来
```
<font color="red" size = 4> 深入理解矩阵乘与行列变换的对应关系！ </font>

有了上面的理解，消元无非是行变换，也就是在A左侧不断乘上新的矩阵达到消元的目的。即消元过程可表示为$E_{x}\dots(E_{1}A) = U$ 其中U为上三角矩阵。(这里注意，矩阵乘法有结合律，但没有交换律)

进一步可以思考逆矩阵的概念，从上面的理解角度，就是通过逆变换消除原来变换的影响
```
|1  0| |1   0| |a b|
|3  1| |-3  1| |c d|
以上两个左侧矩阵为逆变换，先是R2 = R2 - 3R1再是R2 = R2 + 3R1，还原回A
```
