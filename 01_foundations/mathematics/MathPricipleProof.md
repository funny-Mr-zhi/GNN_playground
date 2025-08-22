

# 概率论

## 常用函数性质

### Logistic Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

选择其产生伯努利参数$\phi$的原因：
* 值域在(0, 1)，符合伯努利分布参数约束
* 

**值域**
> 证明：`logistic sigmoid`函数值域为(0, 1)。( 即：对于任意实数$x$，都有$0 < \sigma(x) < 1$，且对于区间内任意值$y$,都存在对应$x$使得$\sigma(x)=y$ )

Logistic Sigmoid函数定义为：
\[
\sigma(x) = \frac{1}{1 + e^{-x}} \quad (x \in \mathbb{R})
\]


步骤1：证明\(\sigma(x) \in (0, 1)\)
1. 对任意\(x \in \mathbb{R}\)，指数函数性质：
   \[
   e^{-x} > 0 \quad (\forall x \in \mathbb{R})
   \]
2. 分母不等式推导：
   \[
   1 + e^{-x} > 1 \implies \frac{1}{1 + e^{-x}} < 1
   \]
3. 分子分母均为正数，故：
   \[
   \frac{1}{1 + e^{-x}} > 0
   \]
   综上：\(0 < \sigma(x) < 1\)。


步骤2：证明\((0, 1)\)内任意值均可取到（反函数法）
1. **求反函数**：  
   设\(y = \sigma(x) = \frac{1}{1 + e^{-x}}\)，解方程得：
   \[
   1 + e^{-x} = \frac{1}{y} \implies e^{-x} = \frac{1 - y}{y} \implies x = \ln\left(\frac{y}{1 - y}\right)
   \]
   反函数为：
   \[
   \sigma^{-1}(y) = \ln\left(\frac{y}{1 - y}\right) \quad (0 < y < 1)
   \]

2. **反函数定义域与值域**：  
   - 反函数定义域为\(y \in (0, 1)\)（保证表达式有意义）。  
   - 当\(y \to 0^+\)时，\(\sigma^{-1}(y) \to -\infty\)；当\(y \to 1^-\)时，\(\sigma^{-1}(y) \to +\infty\)。  
   - 反函数在\((0, 1)\)上连续且单调递增，故值域为\(\mathbb{R}\)。

3. **结论**：  
   对任意\(y \in (0, 1)\)，存在\(x = \sigma^{-1}(y) \in \mathbb{R}\)使得\(\sigma(x) = y\)。


**综上**：Logistic Sigmoid函数的值域为\((0, 1)\)。





