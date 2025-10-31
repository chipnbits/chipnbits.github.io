---
title: "Dimensionality Reduction with the Johnson-Lindenstrauss Lemma"
subtitle: "A randomized approach to preserving distances in high-dimensional data"
date: 2025-10-23
author: "Simon Ghyselincks"
description: >-
  Randomized algorithms provide an alternative approach to improving computational efficiency in high-dimensional data processing where full accuracy can be traded for speed.
categories:
    - Randomized Algorithms

bibliography: references.bib
biblatexoptions: "style=numeric,sorting=nyt"
bibliostyle: numeric-comp

image: imgs\parzen_density_animation_with_gradients.gif
draft: false

execute:
  jupyter: python3


format:
  html:
    code-fold: true
    code-summary: "Show the code"

---
## Dimensionality Reduction with Property Preservation

Modern day computing increasingly is relying upon data-driven methods to solve complex problems. Much of machine learning, optimization, and statistical analysis requires data that is high-dimensional, forming large and expensive matrix computations. However, often these algorithms are primarily interested in resulting coarse scale properties of the operations rather than the fine details. For example a least-squares solver for $\min_x \|Ax-b\|^2$ may only need to find an approximate solution for $x$, we are minimizing the norm $\|.\|$ rather than exact values of $Ax-b$. If we can find a way to reduce the dimensionality of $A$ and $b$ while preserving the norm, we can speed up the computation. This is the essence of dimensionality reduction via matrix sketching, a topic recently covered by SIAM news [@saibaba2025randomized].

At its core, we use randomized algorithms that trade accuracy for speed for cases where precision is of less importance. One such approach is dimensionality reduction with the Johnson-Lindenstrauss lemma, explored ahead.

## The Johnson-Lindenstrauss Theorem

Suppose that there are $n$ data points that are vectors of dimension $d$ that we want to reduce to a smaller dimension $t$ while preserving the pairwise distances:

| **Component**        | **Description**                   | **Dimensions**              |
|----------------------|-----------------------------------|-----------------------------|
| $x_1, \ldots, x_n$   | Original high-dimensional points  | $\mathbb{R}^d$              |
| $y_1, \ldots, y_n$   | Reduced-dimensional points        | $\mathbb{R}^t$              |
| $d$                  | Original dimension                |                |
| $t$                  | Reduced dimension ($t \ll d$)     |                |
| $n$                  | Number of points                  |                |
| $f: \mathbb{R}^d \rightarrow \mathbb{R}^t$ | Embedding function     | $\mathbb{R}^d \rightarrow \mathbb{R}^t$ |
| $\|x_i - x_j\|$      | Original pairwise distance        |            |
| $\|y_i - y_j\|$      | Reduced pairwise distance         |            |

### Random Vectors

For a vector $g \in \mathbb{R}^d$ whose entries are independent and identically distributed (i.i.d.) with mean zero (e.g., standard normal distribution), we have the following property:
$$ \mathbb{E}[g^Tx] = \sum_{i=1}^d \mathbb{E}[g_i] x_i = 0. $$
This is by linearity of expectation, and uses the fact that each $g_i$ has mean zero.

For the case of looking at the squared inner product, we have:
$$\begin{align*}\mathbb{E}[(g^Tx)^2] &= \mathbb{E}[(g^Tx)^2] - \underbrace{\mathbb{E}[g^Tx]^2}_{0} \\
  &= \text{Var}[g^Tx]\\
  &= \sum_{i=1}^d \text{Var}[g_i x_i] \\
  &= \sum_{i=1}^d x_i^2 \text{Var}[g_i]
\end{align*} $$

So that if we choose each $g_i$ to have variance 1, we have $\mathbb{E}[(g^Tx)^2] = \|x\|^2$. Thus the square of the inner product with a random vector with entries $\mu = 0$ and $\sigma^2 = 1$ is an unbiased estimator of the squared norm of $x$.

In the special case where the random vector has gaussian i.i.d. entries $g_i \sim \mathcal{N}(0,1)$, we have that $$g^Tx \sim \mathcal{N}(0, \|x\|^2),$$ since the sum of independent gaussian random variables is also gaussian with mean equal to the sum of the means and variance equal to the sum of the variance: 
$\mathcal{N}(0,1) \cdot x_i \sim \mathcal{N}(0, x_i^2)$ 
$\sum_{i=1}^d \mathcal{N}(0, x_i^2) \sim \mathcal{N}(0, \sum_{i=1}^d x_i^2) = \mathcal{N}(0, \|x\|^2)$.

So then the expected value of the squared inner product is:
$$ \mathbb{E}[(g^Tx)^2] = \text{Var}[g^Tx] = \|x\|^2. $$

But since this is a random variable, we can't simply take the square root to get an estimate of $\|x\|$.

### Random Gaussian Matrix
We can improve the estimate by using multiple random vectors and averaging the results. This is a matrix-vector product with a random matrix.

Suppose that we construct a random linear mapping $f(x) = R x$ where $R$ is a $t \times d$ random matrix with entries drawn from a suitable distribution. If this distribution is independent gaussian entries $R_{ij} \sim \mathcal{N}(0,1)$, then each entry of the reduced vector $y = f(x)$ is given by:
$$ y_i = R_i x, $$
This will give $t$ total $u_i$ entries in the vector, with independent expectations of $\|x\|^2$. The norm of $y$ is then given by:
$$ \mathbb{E}[\|y\|^2] = \mathbb{E}\left[ \sum_{i=1}^t (R_i x)^2  \right]= t \|x\|^2. $$
For simplicity in the analysis and without loss of generality, we can consider the case where $\|x\| = 1$ having been rescaled appropriately. Then we have $\mathbb{E}[\|y\|^2] = t$ and we can bound the probability that $\|y\|^2$ deviates from its mean using concentration inequalities. The distribution of $\|y\|^2$ is a chi-squared distribution [@wikipedia2025chisquared] with $t$ degrees of freedom, with known tail bounds. 

#### Tail Bound for Chi-Squared Distribution
Let $\|y\|^2 \sim \chi^2_t$, then for any $0 < \epsilon < 1$ we have the following concentration bound:

$$\Pr\left[|\chi^2_t - t| \geq \epsilon t \right] \leq 2 \exp\left(-\frac{t}{8}\epsilon^2\right).$$
We can show an even tighter bound using for example the Chernoff concentration bound.

Then we remark that 
$$ \begin{align*}
  \Pr \left[ \chi^2_t \notin (1 \pm \epsilon) \cdot t \right] = \Pr \left[ |\chi^2_t - t| \geq \epsilon t \right] \leq 2 \exp\left(-\frac{t}{8}\epsilon^2\right).
\end{align*} $$
Then if we define this quantity to be $\delta := 2 \exp\left(-\frac{t}{8}\epsilon^2\right)$, we can solve for $t$ to get:
$$ t = \frac{8}{\epsilon^2} \ln\left(\frac{2}{\delta}\right). $$

#### Bound on Distance Preservation

Now this can be used to bound the norm preservation of a single vector.
$$ \begin{align*}
  \Pr \left[ \frac{\|Rv\|}{\sqrt{t}} \notin (1\pm\epsilon)\right] &= \Pr \left[ \|Rv\|^2 \notin (1\pm\epsilon)^2 \cdot t \right] \\
  &\leq \Pr \left[ \|Rv\|^2 \notin (1\pm \epsilon) \cdot t \right] \quad \text{(for $0 < \epsilon < 1$)} \\
  & \leq \delta = 2 \exp\left(-\frac{t}{8}\epsilon^2\right).
\end{align*} $$
The random Gaussian matrix $R$ thus preserves the relative norm of a single vector $v$ within some error tolerance $\epsilon$ with high probability $1-\delta$ for $\delta \in (0,1]$. This reduces to pick $2$ out of $3$ constraints on $t, \epsilon, \delta$ and solve for the third.

### Extending to Pairwise Distances

Given a set of $n$ vectors, there are a total of $n^2 - n$ pairwise distances and $n$ vector norms to preserve within some error tolerance. Then we can use the union bound on the previous result. Let $d_i$ be the set of all pairwise distances and vector norms to preserve, with $i$ ranging from $1$ to $n^2$. Then we have:

$$ \begin{align*}
  \Pr\left[ \text{any} \frac{1}{\sqrt{t}} \|R(d_{i})\| \notin (1\pm\epsilon)\|d_{i}\|\right] &\leq \sum_{i} \Pr\left[ \frac{1}{\sqrt{t}} \|R(d_{i})\| \notin (1\pm\epsilon)\|d_{i}\|\right] \\
  &\leq n^2 \cdot 2 \exp\left(-\frac{t}{8}\epsilon^2\right)\\
  &= 1/n \quad \text{for } \delta = \frac{1}{n^3}
\end{align*} $$

So the trick is to set $\delta = \frac{1}{n^3}$ which gives us the following bound on $t$ for some chosen $\epsilon$:
$$t = \frac{8}{\epsilon^2} \ln(2 n^3) = O(\log(n)/\epsilon^2)$$

Thus we have the Johnson-Lindenstrauss lemma:

---

### Johnson–Lindenstrauss Lemma
::: {.theorem #johnson-lindenstrauss name="Johnson–Lindenstrauss Lemma"}
Let $0 < \epsilon < 1$ and let $X = \{x_1, x_2, \ldots, x_n\} \subset \mathbb{R}^d$.  
There exists a linear mapping $f: \mathbb{R}^d \to \mathbb{R}^t$ with 
$$
t = O\!\left(\frac{\log n}{\epsilon^2}\right)
$$
such that for all $x_i, x_j \in X$,
$$
(1 - \epsilon)\|x_i - x_j\|^2 
\le 
\|f(x_i) - f(x_j)\|^2 
\le 
(1 + \epsilon)\|x_i - x_j\|^2.
$$

In particular, if $f(x) = \frac{1}{\sqrt{t}} R x$ where $R$ is a random matrix with entries $R_{ij} \sim \mathcal{N}(0,1)$, then with high probability this mapping satisfies the above inequality.
:::