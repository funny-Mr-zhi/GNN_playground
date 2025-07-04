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