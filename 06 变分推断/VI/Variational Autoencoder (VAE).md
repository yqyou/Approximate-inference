# Variational Autoencoder (VAE)

变分自动编码器（VAE）是一种非线性潜变量模型，具有基于变分原理的高效梯度训练程序。在潜变量模型中，我们假定观察到的 $x$ 是由一些潜变量（未观察到的）$z$ 生成的；这些潜变量捕捉到了观察数据中的一些 "有趣 "结构，而这些结构并不能从观察数据本身立即看出。例如，一种称为独立成分分析的潜变量模型可用于将人们同时说话的录音中的单独语音信号分离出来。更正式地说，我们可以把潜变量模型看作是一个概率分布 $p(x|z)$，它描述了生成过程（即 $x$ 是如何从 $z$ 生成的）以及关于潜变量 $z$ 的先验值 $p(z)$。这相当于以下简单的图形模型：
$$
z\rightarrow x
$$
潜变量模型中的学习 我们在这种模型中的目的是学习生成过程，即 $p(x|z)$（我们假设 $p(z)$是已知的）。一个好的$p(x|z)$ 会给观测到的$x$ 赋以高概率；因此，我们可以通过最大化观测数据的概率（即$p(x)$）来学习一个好的$p(x|z)$。假设 $p(x|z)$ 的参数为 $θ$，我们需要解决以下优化问题
$$
{\rm \mathop{max}\limits_{\theta}} p_\theta(x)
$$
其中，$p_\theta(x)=\int_zp(z)p_\theta(x|z)$。这是一个困难的优化问题，因为它涉及 $z$的积分，有可能积分没有解析解。



Posterior inference in a latent variable model For the moment, let us set aside this learning problem and focus on a different one: posterior inference of $p(z|x)$. As we will see shortly, this problem is closely related to the learning problem and in fact leads to a method for solving it. Given $p(z)$ and $p(x|z)$, we would like to infer the posterior distribution $p(z|x)$. This is usually rather difficult because it involves an integral over $z$, $p(z|x) = \frac{p(x,z)}{\int_zp(x,z)}$. For most latent variable models, this integral cannot be evaluated, and $p(z|x)$ needs to be approximated. For example, we can use Markov chain Monte Carlo techniques to sample from the posterior. However, here we will look at an alternative technique based on variational inference. Variational inference converts the posterior inference problem into the optimization problem of finding an approximate probability distribution $q(z|x)$ that is as close as possible to $p(z|x)$. This can be formalized as solving the following optimization problem: 
$$
{\rm \mathop{min}\limits_{\phi} KL} (q_\phi(z|x)||p(z|x))
$$
where $\phi$ parameterizes the approximation $q$ and $KL(q||p)$ denotes the Kullback-Leibler divergence between $q$ and $p$ and is given by ${\rm KL}(q||p) = \int_xq(x) {\rm log} \frac{q(x)}{p(x)} $. However, this optimization problem is no easier than our original problem because it still requires us to evaluate p(z|x). Let us see if we can get around that. Plugging in the definition of KL, we can write,
$$
\begin{align*}
{\rm KL}(q_\phi(z|x)||p(z|x)) &= \int_\phi q_\phi(z|x) {\rm log} \frac{q_\phi(z|x)}{p(z|x)} \\
&=\int_\phi q_\phi(z|x) {\rm log} \frac{q_\phi(z|x)p(x)}{p(x,z)} \\
&=\int_\phi q_\phi(z|x) {\rm log} \frac{q_\phi(z|x)}{p(x,z)}+\int_\phi q_\phi(z|x) {\rm log} p(x) \\
&= \int_\phi q_\phi(z|x) {\rm log} q_\phi(z|x)-\int_\phi q_\phi(z|x) {\rm log} p(x,z) + \int_\phi q_\phi(z|x) {\rm log} p(x)
\end{align*} \tag{4}
$$


我们定义$L(\phi)=\int_z q_\phi(z|x) {\rm log} p(x,z)-\int_z q_\phi(z|x) {\rm log} q_\phi(z|x)$. 然后积分$\int_\phi q_\phi(z|x) {\rm log} p(x)$中的被积项${\rm log} p(x)$不含$\phi$。所以$\int_\phi q_\phi(z|x) {\rm log} p(x)={\rm log}p(x)$。那么上式可以写成
$$
{\rm KL}(q_\phi(z|x)||p(z|x))=-L(\phi)+{\rm log} p(x) \tag{5}
$$
由于 $p(x)$ 与 $q_\phi(z|x)$ 无关，最小化 ${\rm KL}(q_\phi(z|x)||p(z|x))$ 等于最大化 $L(\phi)$。请注意，优化 $L(\phi)$ 要容易得多，因为它只涉及 $p(x,z) = p(x|z)p(z)$ ，不涉及任何难解的积分。因此，我们可以通过求解下面的优化问题，对潜变量模型的后验进行变分推断：
$$
{\rm \mathop{max}\limits_{\phi}} \ \ L(\phi)
$$

## 回到学习问题

上面的推导也提出了学习生成模型 $p(x|z)$ 的方法。我们可以看到，$L(\phi)$ 实际上是观测数据 $p(x)$ 的对数概率的下限：





其中我们使用了 KL 从不为负这一事实。现在，假设我们不做后验推断，而是固定 $q$，并学习生成模型 $p_\theta(x|z)$。那么 $L$ 现在是 $θ$ 的函数，$L(θ) = \int_z q(z|x) {\rm log} \frac{p_\theta (x|z)p(z)}{q(z|x)}$ 。由于 $L$ 是 $log p(x)$ 的下限，我们可以最大化 L 作为最大化 log p(x) 的近似。事实上，如果 $q(z|x) = p(z|x)$，则 KL 项为零，$L(θ) = log p(x)$，即最大化 L 等于最大化 $p(x)$。这表明我们可以同时改变$\phi$和$\theta$来最大化$L$, 以达到同时学习 $q_\phi(z|x)$ 和 $p_\theta(x|z)$的目的:
$$
{\rm \mathop{max} \limits_{\theta, \phi}} \ \ L(\theta, \phi)
$$
我们重新把ELBO写成
$$
\begin{align*}
L(\theta,\phi)&=\int q_\phi(z|x) {\rm log} p(x,z) dz-\int q_\phi(z|x) {\rm log} q_\phi(z|x)dz \\
&=\int q_\phi(z|x) {\rm log} p_\theta(x|z)p(z)dz - \int q_\phi(z|x) {\rm log} q_\phi(z|x) dz \\
&=E_q[{\rm log} \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}]
\end{align*}
$$

## Solving the maximization problem in Eqn. 6.

One can use various techniques to solve the above maximization problem. Here, we will focus on stochastic gradient ascent since the variational autoencoder uses this technique. In gradient-based approaches, we evaluate the gradient of our objective with respect to model parameters and take a small step in the direction of the gradient. Therefore, we need to estimate the gradient of $L(\theta, \phi)$. Assuming we have a set of samples $z(l)$, $l$ = 1 . . . $L$ from qφ(z|x), we can form the following Monte Carlo estimate of L:



