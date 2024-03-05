- [Boilerplate](#orgc86c168)
  - [Shabang](#orgd4ecd29)
  - [Imports](#org199914f)
- [State Class definition](#org8b75c0b)
- [`accept` function](#org29292eb)
- [`oneStep` function](#orgf06e925)
- [Testing](#orge704d2a)



<a id="orgc86c168"></a>

# Boilerplate


<a id="orgd4ecd29"></a>

## Shabang

```python
#!/usr/bin/env python3
```


<a id="org199914f"></a>

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


<a id="org8b75c0b"></a>

# State Class definition

The Adaptive State class will contain a state of the chain as wll as a method to progress the state of the chain.

It has three attributes;

```python
class Adaptive_State:

    def __init__(self, j, x, x_sum, xxt_sum):

        self.j       = j
        self.x       = x  
        self.x_sum   = x_sum
        self.xxt_sum = xxt_sum

```


<a id="org29292eb"></a>

# `accept` function

The `accept` method decides whether to accept a given proposed move, given the log-probability and a prng key.

```python
    def accept(self, prop, alpha, key):

        log_prob = jnp.minimum(0.0, alpha)

        u = rand.uniform(key)

        new_x = prop if (jnp.log(u) < log_prob) else self.x
        
        return(Adaptive_State(
            self.j + 1,
            new_x,
            self.x_sum + new_x,
            self.xxt_sum + jnp.outer(new_x, new_x)))
    
```


<a id="orgf06e925"></a>

# `oneStep` function

The main chunk, using the algorithm from Roberts and Rosenthall to make a single step to the next state.

```python
    def oneStep(self, q, r, key):

        keys = rand.split(key,3)
        
        j       = self.j
        x       = self.x
        x_sum   = self.x_sum
        xxt_sum = self.xxt_sum
        d       = x.shape[0]

        if (j <= 2*d):

            z = rand.normal(keys[0], shape=(d,))

            # The propasal distribution is N(x,1/d) for this first stage
            prop = (z + x) * d

            # Compute the log acceptance probability
            alpha = 0.5 * (x @ (solve(r, q.T @ x))
                           - (prop @ solve(r, q.T @ prop)))
            
            return(self.accept(prop, alpha, keys[1]))
        
        else:
            
            emp_var = xxt_sum/j - jnp.outer(x_sum, x_sum.T)/j**2

            u = rand.uniform(keys[0])

            if (u < 0.95):
              prop = rand.multivariate_normal(keys[1], x, emp_var * (2.38**2/d))
            else:
              prop = ((rand.normal(keys[1], shape=(d,)) + x) * 100 * d)

            # Compute the log acceptance probability
            alpha = 0.5 * (x @ (solve(r, q.T @ x))
                           - (prop @ solve(r, q.T @ prop)))
            
            return(self.accept(prop, alpha, keys[2]))
            
```


<a id="orge704d2a"></a>

# Testing

```python
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr
import jax
import sys
sys.path.append('~/MyProjects/AdaptiveMCMC/')

from AM_in_JAX import Adaptive_State

x0 = Adaptive_State(1,jnp.array([0,0]),jnp.array([0,0]), jnp.array([[1,0],[0,1]]))

sigma = jnp.array([[2,1],[1,2]])
Q, R = qr(sigma)

key = jax.random.PRNGKey(seed=1)
keys = rand.split(key, n)

n = 1000
thinrate = 10
burnin = 1000

# Now i want to do an iterate, but I'm struggling to think of how to do this without for loops!

# I could use the scan operation 
'''
def step(carry, _):
    return(carry.oneStep(Q,R, keys[carry.j]), None)

_, results = jl.scan(step, x0, jnp.zeros(n))

results[-1].x
'''
# But Adaptive_State is not a valid JAX type. I could rewrite to not use a custom class, of course, but I'd rather not do that.

# Thinning and burnin can be done with [::thinrate] and [burnin:] I think?
```
