- [Boilerplate](#orge8fe329)
  - [Shabang](#org6e7881c)
  - [Imports](#org1a25898)
- [Reimplementation without classes](#org7bb2a80)
  - [`accept` (non-class)](#orge7443aa)
  - [`oneStep` (non-class)](#orgc97e46f)
    - [Non-adaptive part](#orgbf87985)
    - [Adaptive step](#orgd71a240)
  - [Testing 2 electric boogaloo](#org702fdfc)



<a id="orge8fe329"></a>

# Boilerplate


<a id="org6e7881c"></a>

## Shabang

```python
#!/usr/bin/env python3
```


<a id="org1a25898"></a>

## Imports

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr
import jax
```


<a id="org7bb2a80"></a>

# Reimplementation without classes

Instead of an object oriented approach, let's assume that the `state` is a tuple with four elements, instead of it's own class.

More specificaly, `state = (j, x, x_sum, xxt_sum)`,


<a id="orge7443aa"></a>

## `accept` (non-class)

```python
def accept(state, prop, alpha, key):

  j       = state[0]
  x       = state[1]
  x_sum   = state[2]
  xxt_sum = state[3]
  d       = x.shape[0]
  
  log_prob = jnp.minimum(0.0, alpha)

  u = rand.uniform(key)

  #new_x = prop if (jnp.log(u) < log_prob) else x

  new_x = jl.cond((jnp.log(u) < log_prob),
                  _,
                  lambda _: prop,
                  _,
                  lambda _: x)
  
  return((j + 1,
          new_x,
          x_sum + new_x,
          xxt_sum + jnp.outer(new_x, new_x)))
```


<a id="orgc97e46f"></a>

## `oneStep` (non-class)


<a id="orgbf87985"></a>

### Non-adaptive part

```python
def initStep(state,q,r,key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    keys = rand.split(key,3)
    z = rand.normal(keys[0], shape=(d,))
    
    # The propasal distribution is N(x,1/d) for this first stage
    prop = (z + x) * d
    
    # Compute the log acceptance probability
    alpha = 0.5 * (x @ (solve(r, q.T @ x))
                   - (prop @ solve(r, q.T @ prop)))
    
    return(accept(state, prop, alpha, keys[1]))
    
```


<a id="orgd71a240"></a>

### Adaptive step

```python
def adaptStep(state, q, r, key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    keys = rand.split(key,3)
    z = rand.normal(keys[0], shape=(d,))
    
    emp_var = xxt_sum/j - jnp.outer(x_sum, x_sum.T)/j**2

    u = rand.uniform(keys[0])

    prop = jl.cond(u < 0.95,
                   x,
                   lambda y: rand.multivariate_normal(keys[1], y,
                                                 emp_var * (2.38**2/d)),
                   x,
                   lambda y:((rand.normal(keys[1], shape=(d,)) + y) * 100 * d))
    
    # Compute the log acceptance probability
    alpha = 0.5 * (x @ (solve(r, q.T @ x))
                   - (prop @ solve(r, q.T @ prop)))
    
    return(accept(state, prop, alpha, keys[2]))
```

```python
def oneStep(state, q, r, key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    return(jl.cond(j <= 2*d,
                   _,
                   lambda _: initStep(state, q, r, key),
                   _,
                   lambda _: adaptStep(state, q, r, key)))
```


<a id="org702fdfc"></a>

## Testing 2 electric boogaloo

```python
x0 = (1,jnp.array([0.0,0.0]),jnp.array([0.0,0.0]), jnp.array([[1.0,0.0],[0.0,1.0]]))

sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
Q, R = qr(sigma)

n = 10000
thinrate = 10
burnin = 1000

key = jax.random.PRNGKey(seed=1)
keys = rand.split(key, n)

def step(carry, _):
    nextstate = oneStep(carry, Q, R, keys[carry[0]])
    return(nextstate, nextstate)

fin, results = jl.scan(step, x0, jnp.zeros(n))

#there is also jl.fori, which may be able to do the same thing

xxt_sum = results[3][n]
x_sum = results[2][n]

emp_var = xxt_sum/n - jnp.outer(x_sum, x_sum.T)/n**2

print(emp_var)
```
