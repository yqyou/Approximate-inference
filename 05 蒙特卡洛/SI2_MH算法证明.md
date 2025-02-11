前面知道, 根据Markov采样定理, 我们要$p(x)$从中采样, 但是$p(x)$比较复杂。 一种方法是借助与该分布对应的马尔科夫链状态转移函数来进行采样。

那么怎么找到该分布的马尔科夫链状态转移函数呢？我们随便假定一个转移函数$Q$显然是不行的。即不满足细致平稳条件，即
$$
p(x_i)*Q(x_j | x_i) \not= p(x_j)*Q(x_i | x_j)
$$
但是我们可以对上面的式子做一个改造, 使细致平稳分布条件成立。方法是引入一个$a(x_j | x_i)$, 使得上面的式子可以取等号, 即
$$
p(x_i)*Q(x_j | x_i)*a(x_j | x_i) = p(x_j)*Q(x_i | x_j)*a(x_i | x_j)
$$
问题是什么样子的$a(x_j | x_i)$可以满足该要求呢？其实很简单,只要满足下面条件:
$$
a(x_j | x_i) = min(1, \frac{p(x_j)*Q(x_i | x_j)}{p(x_i)*Q(x_j | x_i)})
$$
把公式3带入公式2, 
$$
\begin{align*}

p(x_i)*Q(x_j | x_i)*a(x_j | x_i) 

&= p(x_i)*Q(x_j | x_i)*min(1, \frac{p(x_j)*Q(x_i | x_j)}{p(x_i)*Q(x_j | x_i)}) \\

&= min(p(x_i)*Q(x_j | x_i), p(x_j)*Q(x_i | x_j)) \\

&= p(x_j)*Q(x_i | x_j) * min(\frac{p(x_i)*Q(x_j | x_i)}{p(x_j)*Q(x_i | x_j)}, 1) \\

&= p(x_j)*Q(x_i | x_j)*a(x_i | x_j)

\end{align*} \tag{4}
$$
可以发现，如果另一个新的转换函数$M(x_j | x_i)=Q(x_j | x_i)*a(x_j | x_i)$

那么上式就可以写成
$$
p(x_i)*M(x_j | x_i)= p(x_j)*M(x_i | x_j)
$$
我们发现就满足了细致平稳分布条件。

注意，这里的$Q(x_j | x_i)$可以是任意的转换函数(e.g., 高斯函数)，上面的式子恒成立。