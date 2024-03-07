- [Adaptive Metropolis Algorithm](#org7221fe3)
  - [Measure of effectiveness](#orgf624d31)
- [Example Target](#org6532649)
- [Scala implementation](#org4335bd1)
- [JAX implementation](#org9aeb6e9)
- [Results](#org84e6f3f)
  - [Scala](#orgd027b18)
  - [JAX](#org2cf8b96)
  - [Very high dimensions](#org3df5bf9)

This is my attempt at implementing Adaptive Metropolis in Scala, using the breeze library, and python, using JAX.

This is based on the example from the article "Examples of Adaptive MCMC" by Roberts and Rosenthal.


<a id="org7221fe3"></a>

# Adaptive Metropolis Algorithm

A AMRTH step is defined as follows;

-   If $j\leq 2d$, we do a MRTH step with proposal $q(x,\cdot)\sim \mathcal N(x,(0.1)^2I_d/d)$
-   If $j>2d$, we use the proposal $q(x,\cdot)\sim(1-\beta)\mathcal N(x,(2.38)^2\Sigma_j/d)+\beta\mathcal N(x,(0.1)^2I_d/d)$, where $\Sigma_j$ is the current empirical estimate of the covariance matrix so far.

We can compute the empirical covariance matrix at step $j$ by

$$\begin{aligned} \Sigma_j=\frac{{\sum_{i=0}^j} x_ix_i^{\intercal}}{j} - \frac{({\sum_{i=0}^j} x_i)({\sum_{i=0}^j} x_i)^{\intercal}}{j^2}. \end{aligned}$$

The logic I'm using is to carry forward $\sum x_ix_i^{\intercal}$ and $\sum x_i$ (as well as the current index, $j$) as part of our 'chain', in order to compute the empirical covariance matrix as we go along (I should possibly do a $\frac{n}{n-1}$ transormation to this matrix too), in order to sample from the proposal when $j>2d$ .


<a id="orgf624d31"></a>

## Measure of effectiveness

Roberts and Rosenthal also give the following measure of effectiveness;

$$\begin{aligned} b = d\frac{\sum \lambda_i^{-2}}{(\sum \lambda_i^{-1})^2 } \end{aligned}$$

where $\lambda_i$ are the eigenvalues of $\Sigma_p^{1/2}\Sigma^{-1/2}$ where $\Sigma_p$ is the empirical variance matrix at the pth iteration.

$b$ should approach 1 as the chain approaches the stationary distribution. Roughly, it measures the difference between the empirical and true variance matrices.


<a id="org6532649"></a>

# Example Target

We target the distribution $\pi(\cdot)\sim \mathcal N(0,\Sigma)$, where $\Sigma = M \in \mathbb R^{d\times d}$ is a matrix with random $\mathcal N[0,1]$ entries. In Scala, this can be found as below;

```scala
  import AdaptiveMetropolis._

  // dimension of the state space
  val d = 10

  // create a chaotic variance to target
  val data = Gaussian(0,1).sample(d*d).toArray.grouped(d).toArray
  val M = DenseMatrix(data: _*)
  val sigma = M.t * M
```

Note that Breeze's `DenseMatrix` and `DenseVector` are actually mutable in Scala, so we need to be careful not to mutate anything.


<a id="org4335bd1"></a>

# Scala implementation

My Scala implementation of this is found in `Main.scala` (it needs cleanup though). It is built around an object `Adaptive Metropolis`, with three methods:

-   `AM_step`, which takes the current state as well as the QR decomposition of the true variance, and outputs the next state of the chain.
-   `AM_iterator`, which iterates `AM_step` in order to create an infinite lazy list of samples.
-   `plotter`, which plots the 1st componant of the sample, and saves it to a file.

The `run` function then tests this, using `d=10`, `n=100000`, `burnin=100000` and `thinrate=10`. This function, once it finishes, prints out the true variance of $x_1$, the empirical estimate of it from the sample, the $b$ value, and the time the computation took. A trace plot of $x_1$ is also saved to `Figures/adaptive_trace_scala.png`.


<a id="org9aeb6e9"></a>

# JAX implementation

As you might imagine, the JAX implentation is very similar, even if it is a bit more fragmented. The `AM_step` function is split into four functions, `try_accept`, `init_step`, `adapt_step`, and `AM_step`. This is mainly due to the way JAX handles `if else` statements, making this seem like the convenient way to do it.

In the file `AM_in_JAX.org` (or `.md`), there is the source code as well as documentation for all the functions, but it is very similar to the scala version.


<a id="org84e6f3f"></a>

# Results

In both implementations, we run with d=10~, `n=100000`, `burnin=100000` and `thinrate=10`.


<a id="orgd027b18"></a>

## Scala

The Scala output can be found using the command `sbt run` in this project's root;

> The true variance of x<sub>1</sub> is 4.0203388245821 The Empirical sigma value is 4.041380732669074 The b value is 1.0000233300468546 The computation took 9.178699105 seconds

![img](/home/tate/MyProjects/AdaptiveMCMC/Figures/adaptive_trace_scala.png)


<a id="org2cf8b96"></a>

## JAX

The JAX output can be found by running `python AM_in_JAX.py` in this project's root;

> The true variance of x<sub>1</sub> is 9.967914581298828 The empirical sigma value is 9.99879264831543 The b value is 1.0000910758972168 The computation took 3.7597200870513916 seconds

Obviously, the numbers are different since the target variance is different, but this ran over twice as fast.

![img](/home/tate/MyProjects/AdaptiveMCMC/Figures/adaptive_trace_jax.png)


<a id="org3df5bf9"></a>

## Very high dimensions

The paper, in it's examples, get results for `d=100`. In Scala, using thinning and burn-in, the garbage collector does a good job and we can get high enough iteration counts the the program does very well.

For `d=100`, `n=10000`, `burnin = 1000000`, and `thinrate=100`, I get

> The true variance of x<sub>1</sub> is 87.24837703682367 The empirical sigma value is 85.1683515513272 The b value is 1.003561061350601

This was ran before implementing computation time, it took a long while though.

Unfortunately, as of now, the JAX version runs into a memory error with similar numbers (even though it is allocated around 8GB of memory). It seems that the garbage collector doesn't quite do the same magic as Scala's, and some more work will need to be done to get it to work.
