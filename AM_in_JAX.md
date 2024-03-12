- [Boilerplate](#org4225a84)
  - [Shabang](#orgecbf3a6)
  - [Imports](#org8aa5bc3)
- [`AM_step`](#orgb2eb533)
  - [`try_accept`](#org830fdfa)
    - [`test_try_accept`](#org2c7cd43)
  - [`init_step`](#org6e855bd)
    - [`test_init_step`](#orgfe0c3d5)
  - [`adap_step`](#org443af7b)
    - [`test_adapt_step`](#org77c79e4)
  - [`AM_step`](#orgd60fbaf)
    - [`test_AM_step`](#orgdec4dc5)
    - [Covariance function](#orgfebc6b0)
- [`effectiveness`](#org8cf22ae)
- [plotting](#org91f1af5)
- [`thinned_step`](#org93cee13)
  - [Testing](#orgde53d33)
- [High Dimensions](#orga8cca6a)
- [`main`](#org820b83a)
- [Scratch](#org489f808)
  - [Integer overflow](#org6c7e120)

This file <span class="underline">is</span> the source code; everything below gets 'tangled' into `AM_in_JAX.py`.


<a id="org4225a84"></a>

# Boilerplate


<a id="orgecbf3a6"></a>

## Shabang

```python
#!/usr/bin/env python3
```


<a id="org8aa5bc3"></a>

## Imports

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr, norm, eig, inv, cholesky
import jax
import time
```


<a id="orgb2eb533"></a>

# `AM_step`

AM<sub>step</sub> is fragmented into four functions, in contrast to the Scala version.

Let's assume that the `state` is a tuple with four elements, instead of it's own class. JAX reads this as a PyTree, which it is happy to preform operations on (which wouldn't be the case if I made `state` a class like in the Scala version)

More specificaly, `state = (j, x, x_sum, xxt_sum)`.


<a id="org830fdfa"></a>

## `try_accept`

This function takes a state, a proposed move, and a log probabilty, and returns the next state, using the probability as expected.

```python
def try_accept(state, prop, alpha, key):

  j       = state[0]
  x       = state[1]
  x_sum   = state[2]
  xxt_sum = state[3]
  d       = x.shape[0]
  
  log_prob = jnp.minimum(0.0, alpha)

  u = rand.uniform(key)

  #new_x = prop if (jnp.log(u) < log_prob) else x

  new_x, is_accepted = jl.cond((jnp.log(u) < log_prob),
                  0, lambda _: (prop, True),
                  0, lambda _: (x, False))
  
  return((j + 1,
          new_x,
          x_sum + new_x,
          xxt_sum + jnp.outer(new_x, new_x),
          is_accepted))
```


<a id="org2c7cd43"></a>

### `test_try_accept`

The below code block does a few tests on the `try_accept` function. If the tests pass, it will return `True`, otherwise it will throw an error.

```python
def test_try_accept():
    
    d = 10
    key = jax.random.PRNGKey(seed=2)
    keys = rand.split(key,10000)
    state0 = (0, jnp.zeros(10), jnp.zeros(10), jnp.identity(10), False)
    prop = jnp.ones(10)
    
    '''
    Test 1:
    if alpha=log(0.5), then the function should accept approx. 50% of the proposals
    '''
    assert jnp.abs(jnp.mean(jl.map(lambda x: try_accept(state0, prop, jnp.log(0.5), x), keys)[4]) - 0.5 < 0.1), "Accepting at unexpected rate"

    '''
    Test 1.5:
    if alpha=-0.33333333, then the function should accept approx. 0.7165 of the proposals
    '''
    assert jnp.abs(jnp.mean(jl.map(lambda x: try_accept(state0, prop, -0.3333333, x), keys)[4]) - 0.7165 < 0.1), "Accepting at unexpected rate"

    '''
    Test 2:
    if alpha=log(0)=-inf, then the function should never accept, and should return the
    proposed value
    '''
    assert jnp.all(try_accept(state0, prop, jnp.log(0), key)[1]==jnp.zeros(10)), "Not rejecting proposal"

    '''
    Test 3:
    if alpha=log(1)=0 then the function should always accept, and should return the
    proposed value
    '''
    assert jnp.all(try_accept(state0, prop, jnp.log(1), key)[1]==prop), "Not accepting proposal"

    '''
    Test 4:
    No matter what, j should increment by exactly 1
    '''
    assert jnp.all(jl.map(lambda x: try_accept(state0, prop, jnp.log(0.5), x), keys)[0]==1), "Index not correctly implemented"

    '''
    Test 5:
    When it accepts, the x_sum should increase accordingly
    '''
    assert jnp.all(try_accept(state0, prop, jnp.log(1), key)[2]==prop), "Not increased x_sum"
    assert jnp.all(try_accept(state0, prop, jnp.log(0), key)[2]==jnp.zeros(10)), "Not increased x_sum"

    '''
    Test 6:
    When it accepts, the xxt_sum should increase accordingly
    '''
    assert jnp.all(try_accept(state0, prop, jnp.log(1), key)[3]==jnp.identity(10) + jnp.outer(prop, prop)), "Not increased xxt_sum"
    assert jnp.all(try_accept(state0, prop, jnp.log(0), key)[3]==jnp.identity(10)), "Not increased xxt_sum"

    return True
```


<a id="org6e855bd"></a>

## `init_step`

The procedure for taking a step forward when $j\leq2d$. This is equivalent to a random walk metropolis step with proposal $\mathcal N(x,d^{-1}I)$.

```python
def init_step(state,q,r,key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    keys = rand.split(key,3)
    z = rand.normal(keys[0], shape=(d,))
    
    # The propasal distribution is N(x,1/d) for this first stage
    prop = z/d + x
    
    # Compute the log acceptance probability
    alpha = 0.5 * (x @ (solve(r, q.T @ x)) - (prop @ solve(r, q.T @ prop)))
    
    return(try_accept(state, prop, alpha, keys[1]))    
```


<a id="orgfe0c3d5"></a>

### `test_init_step`

```python
def test_init_step():

    # this doesn't take long, but I feel it still takes too long.
    # I don't want to get into the habit of writing tests with
    # this amount of computation.
    
    d = 2
    n = 100000
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    state0 = (0, jnp.zeros(2), jnp.zeros(2), jnp.identity(2), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
        
    '''
    Test 1:
    From state0, the result should be approximately distributed with a N(0,sigma) distribution;
    it should be a standard Random Walk metropolis
    '''
    def step(carry, _):
        nextstate = init_step(carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)
    
    assert norm(cov(jl.scan(step, state0, jnp.zeros(n))[1][1]) - sigma) < 0.2, "init_step not producing sample sufficiently close to the target distribution"

    return True
```


<a id="org443af7b"></a>

## `adap_step`

The actually adaptive part, implementing a step with proposal

$$\begin{aligned} q(x,\cdot)\sim(1-\beta)\mathcal N(x,(2.38)^2\Sigma_j/d)+\beta\mathcal N(x,(0.1)^2I_d/d) \end{aligned}$$

where $\Sigma_j$ is the current empirical covariance matrix.

```python
def adapt_step(state, q, r, key):

    j       = state[0] # this is an int32, not big enough when i square it below!
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    keys = rand.split(key,3)

    z = rand.normal(keys[0], shape=(d,))
    
    emp_var = (xxt_sum/j - jnp.outer(x_sum, x_sum)/(j**2))

    u = rand.uniform(keys[1])
    
    prop = jl.cond(u < 0.95,
                   x,
                   lambda y: rand.multivariate_normal(keys[2], y,
                                                 emp_var * (2.38**2/d)),
                   x,
                   lambda y:((rand.normal(keys[2], shape=(d,))/(100*d) + y)))
    
    # Compute the log acceptance probability
    alpha = 0.5 * (x.T @ (solve(r, q.T @ x))
                   - (prop.T @ solve(r, q.T @ prop)))
    
    return(try_accept(state, prop, alpha, keys[2]), emp_var)
```


<a id="org77c79e4"></a>

### `test_adapt_step`

```python
def test_adapt_step():

    d = 2
    n = 100000
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    # this state was chosen being close to an actual state of the adaptive chain
    state = (100, jnp.zeros(2), jnp.array([-80.0,-5.0]), jnp.array([[260.0,100.0],[100.0,150.0]]), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
    
    '''
    Test 1:
    From a (hypothetical) progressed point, the result should be approximately distributed with a N(0,sigma) distribution.
    '''
    def step(carry, _):
        nextstate = adapt_step(carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)
    
    assert norm(cov(jl.scan(step, state, jnp.zeros(n))[1][1]) - sigma) < 0.2, "adap_stepr not producing sample sufficiently close to the target distribution"

    
    return True
```


<a id="orgd60fbaf"></a>

## `AM_step`

Does one of the above two methods, depending on how far along the chain is.

```python
def AM_step(state, q, r, key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    return(jl.cond(j <= 2*d,
                   state,
                   lambda y: init_step(y, q, r, key),
                   state,
                   lambda y: adapt_step(y, q, r, key)[0]))
```


<a id="orgdec4dc5"></a>

### `test_AM_step`

```python
def test_AM_step():

    d = 2
    n = 100000
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    state0 = (0, jnp.zeros(2), jnp.zeros(2), jnp.identity(2), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
        
    '''
    Test 1:
    Similarily to the init_step test, from state0, the result should be approximately distributed with a N(0,sigma) distribution.
    '''
    def step(carry, _):
        nextstate = AM_step(carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)
    
    assert norm(cov(jl.scan(step, state0, jnp.zeros(n))[1][1]) - sigma) < 0.2, "init_step not producing sample sufficiently close to the target distribution"
    
    return True
```


<a id="orgfebc6b0"></a>

### Covariance function

Since there isn't one built-in anywhere as far as I can tell, this is a simple function to compute the covariance matrix of a sample.

```python
def cov(sample):
    
    means = jnp.mean(sample, axis=0)

    deviations = sample - means
    
    N = sample.shape[0]
    
    covariance = jnp.dot(deviations.T, deviations) / (N - 1)
    
    return covariance
```


<a id="org8cf22ae"></a>

# `effectiveness`

```python
def effectiveness(sigma, sigma_j):

    d = sigma.shape[0]
    
    sigma_j_decomp = eig(sigma_j)
    sigma_decomp = eig(sigma)
    
    rootsigmaj = sigma_j_decomp[1] @ jnp.diag(jnp.sqrt(sigma_j_decomp[0])) @ inv(sigma_j_decomp[1])
    rootsigmainv = inv(sigma_decomp[1] @ jnp.diag(jnp.sqrt(sigma_decomp[0])) @ inv(sigma_decomp[1]))
    
    lam = eig(rootsigmaj @ rootsigmainv)[0]
    lambdaminus2sum = sum(1/(lam*lam))
    lambdainvsum = sum(1/lam)

    b = (d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))).real

    return b
```


<a id="org91f1af5"></a>

# plotting

Exactly as in the Scala version, simply plots the trace of the first coordinate of the given sample, and saves it to a file.

```python
import matplotlib.pyplot as plt

def plotter(sample, file_path, d):
    
    first = sample[:,0]
    plt.figure(figsize=(590/96,370/96))
    plt.plot(first)
    plt.title(f'Trace plot of the first coordinate, d={d}')
    plt.xlabel('Step')
    plt.ylabel('First coordinate value')
    plt.grid(True)
    plt.savefig(file_path, dpi=96)

```


<a id="org93cee13"></a>

# `thinned_step`

Thinning as I've done it above is not memory efficient; it stores all `n` states and only thins right at the end. Instead, the function `thinned_step` uses a fori<sub>loop</sub> to 'jump' steps, which JAX knows how to garbage collect. This is especially important for high dimensional samples, as below.

```python
def thinned_step(thinrate, state, q, r, key):

    keys = rand.split(key,thinrate)
    
    return jl.fori_loop(0, thinrate, (lambda i, x: AM_step(x, q, r, keys[i])), state)
```


<a id="orgde53d33"></a>

## Testing

```python
def test_thinned_step():

    d = 2
    n = 1000
    thinrate = 10
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    # this state was chosen being close to an actual state of the adaptive chain
    state = (100, jnp.zeros(2), jnp.array([-80.0,-5.0]), jnp.array([[260.0,100.0],[100.0,150.0]]), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
    
    '''
    Test 1:
    the index of a state should increase by thinrate
    '''
    assert (thinned_step(thinrate, state, Q, R, keys[0])[0] == 100+thinrate), "thinned_step not correctly incrementing step count"

    return True
  
```


<a id="orga8cca6a"></a>

# High Dimensions

Due to memory constraints and garbage collection not being wuite as magical, we do burn-in seperately to the main sampling.

```python
def highd(d=10, n=10000, thinrate=10, burnin=1000):

    start_time = time.time()

    # the actual number of iterations is n*thin + burnin
    computed_size = n*thinrate + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=2)
    keys = rand.split(key, computed_size+1)
    
    # create a chaotic variance matrix to target
    M = rand.normal(keys[0], shape = (d,d))
    sigma = M.T @ M
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state before burn-in
    state0 = (1, jnp.zeros(d), jnp.zeros(d), jnp.identity(d), False)

    # JAX's ~scan~ isn't quite ~iterate~, so this is a 'dummy'
    # function with an unused argument to call thinned_step for the
    # actually used samples
    def step(carry, _):
        nextstate = thinned_step(thinrate, carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)

    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin, lambda i,x: AM_step(x, Q, R, keys[i]), state0)
    # this will take a while to run, but once it's done there is only 10000 more to compute;

    # the sample
    am_sample = jl.scan(step, start_state, jnp.zeros(n))[1]

    # the empirical covariance of the sample
    sigma_j = cov(am_sample[1])
    b = effectiveness(sigma,sigma_j)

    # the tiume of the computation in seconds
    end_time = time.time()
    duration = time.time()-start_time
    
    print(f"The true variance of x_1 is {sigma[0,0]}")
    print(f"The empirical sigma value is {sigma_j[0,0]}")
    print(f"The b value is {b}")
    print(f"The computation took {duration} seconds")

    plotter(am_sample[1], "Figures/adaptive_trace_jax_high_d.png", d)
    
    return am_sample

```


<a id="org820b83a"></a>

# `main`

The entry point for if the code is run in a console.

```python
def main():
    
    start_time = time.time()

    d = 10          # dimension of the state space
    n = 100000      # size of the desired sample
    thinrate = 10   # the thining rate
    burnin = 100000 # the number of iterations for burn-in

    # the actual number of iterations is n*thin + burnin
    computed_size = n*thinrate + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=2)
    keys = rand.split(key, computed_size+1)
    
    # create a chaotic variance matrix to target
    M = rand.normal(keys[0], shape = (d,d))
    sigma = M.T @ M
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state
    state0 = (1, jnp.zeros(d), jnp.zeros(d), jnp.identity(d), False)

    # JAX's ~scan~ isn't quite ~iterate~, so this is a 'dummy'
    # function with an unused argument to call AM_step
    def step(carry, _):
        nextstate = AM_step(carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)
    
    # the sample
    am_sample = jl.scan(step, state0, jnp.zeros(computed_size))[1][1][burnin:][::thinrate]

    # the empirical covariance of the sample
    sigma_j = cov(am_sample) 
    
    sigma_j_decomp = eig(sigma_j)
    sigma_decomp = eig(sigma)
    
    rootsigmaj = sigma_j_decomp[1] @ jnp.diag(jnp.sqrt(sigma_j_decomp[0])) @ inv(sigma_j_decomp[1])
    rootsigmainv = inv(sigma_decomp[1] @ jnp.diag(jnp.sqrt(sigma_decomp[0])) @ inv(sigma_decomp[1]))
    
    lam = eig(rootsigmaj @ rootsigmainv)[0]
    lambdaminus2sum = sum(1/(lam*lam))
    lambdainvsum = sum(1/lam)

    # According to Roberts and Rosenthal, this should go to
    # 1 at the stationary distribution
    b = (d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))).real

    # the tiume of the computation in seconds
    end_time = time.time()
    duration = time.time()-start_time
    
    print(f"The true variance of x_1 is {sigma[0,0]}")
    print(f"The empirical sigma value is {sigma_j[0,0]}")
    print(f"The b value is {b}")
    print(f"The computation took {duration} seconds")

    plotter(am_sample, "Figures/adaptive_trace_jax.png", d)
    plt.show()

if __name__ == "__main__":
    test_try_accept()
    test_init_step()
    test_adapt_step()
    test_AM_step()
    #main()
    highd(100,10000,100,1000000) # disable, unless you want the program to run for ages
```


<a id="org489f808"></a>

# Scratch

Here is some in-line python code that doesn't get tangled so i can get things to work properly


<a id="org6c7e120"></a>

## Integer overflow

Currently, due to the use of `j**2` in computing `emp_var` in the function `adapt_step`, we program hits an integer overflow very quickly.

To demonstrate this, here is a synthetic example. In `adapt_step`, I have (possibly temporarily) added `emp_var` as an output, so we can take a look.

```python
d = 2
n = 1000
key = jax.random.PRNGKey(seed=1)
keys = rand.split(key,n)
state0 = (2000000, jnp.zeros(2), jnp.ones(2), jnp.identity(2), False)
sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
Q, R = qr(sigma)

def step(carry, _):
    nextstate = adapt_step(carry, Q, R, keys[carry[0]])
    return(nextstate[0], nextstate[1])

results = jl.scan(step, state0, jnp.zeros(1))
emp_var = results[1]
print(emp_var)
```

```python
print(cholesky(emp_var))
```

Actually, this seems fine&#x2026; seems like JAX properly converts the type, so I'm back to square one (pun intended).

indeed, in this case it actually accepted, which seems rare looking at actual runs.

```python
print(results[0])
```

we see that it does accept in this case

```python
staten = (2000000, jnp.zeros(2), jnp.ones(2), jnp.array([[1e8, 0],[0,1e8]]), False)
```
