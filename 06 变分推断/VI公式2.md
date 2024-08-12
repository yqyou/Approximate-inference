$$
log(p(s)) = \int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi}+\int{log(\frac{q(\phi)}{p(\phi|s)})}q(\phi){d\phi} \tag{7}
$$

$$
L = \int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi}
$$

$$
log(p(s)) = -L + D_{KL}[q(\phi) || p(\phi|s)
$$

$$
D_{KL}[q(\phi) || p(\phi|s)] = \int{log(\frac{q(\phi)}{p(\phi|s)})}q(\phi){d\phi}
$$

$$
log(p(s)) - L = D_{KL}[q(\phi) || p(\phi|s)
$$


$$
\begin{align*}
\hat{q(\phi)}&=\mathop{argmax}_{q(\phi)}L \\
&= \mathop{argmax}_{q(\phi)} \int{log(\frac{p(\phi,s)}{q(\phi)})}q(\phi){d\phi} \\
&= \mathop{argmax}_{q(\phi)} \int{(logp(s|\phi)+logp(\phi)-logq(\phi))}q(\phi){d\phi}
\end{align*}
$$

$$
\begin{align*}
L &= \int{(logp(s|\phi)+logp(\phi)-logq(\phi))}q(\phi){d\phi} \\
&\approx \frac{1}{N}\sum_{l=1}^N logp(s|\phi)^{(l)}+logp(\phi)^{(l)}-logq(\phi)^{(l)}
\end{align*}
$$

$$
\begin{align*}
\hat{q(\phi)}&=\mathop{argmax}_{q(\phi)}L \\
&\approx \mathop{argmax}_{q(\phi)} \frac{1}{N}\sum_{l=1}^N logp(s|\phi)^{(l)}+logp(\phi)^{(l)}-logq(\phi)^{(l)} \\

&=\mathop{argmin}_{q(\phi)} \frac{1}{N}\sum_{l=1}^N logq(\phi)^{(l)}-logp(s|\phi)^{(l)}-logp(\phi)^{(l)}
\end{align*}
$$

