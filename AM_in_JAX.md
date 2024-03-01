- [Boilerplate](#org371815c)
  - [Shabang](#orgfafad14)
  - [Imports](#org5593541)
  - [Initialise key](#org1d6c7c6)
- [State Class definition](#org19736e8)
- [`accept` function](#org38512e2)
- [`oneStep` function](#org8e5fda3)
- [Testing](#orgc606726)



<a id="org371815c"></a>

# Boilerplate


<a id="orgfafad14"></a>

## Shabang

```python
#!/usr/bin/env python3
```


<a id="org5593541"></a>

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


<a id="org1d6c7c6"></a>

## Initialise key


<a id="org19736e8"></a>

# State Class definition

```python
class Adaptive_State:

    def __init__(self, j, x, x_sum, xxt_sum):

        self.j       = j
        self.x_sum   = x_sum
        self.xxt_sum = xxt_sum
        self.x       = x  
```


<a id="org38512e2"></a>

# `accept` function

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


<a id="org8e5fda3"></a>

# `oneStep` function

```python
    def oneStep(self, q, r, key):

        keys = rand.split(key,3)
        
        j       = self.j
        x_sum   = self.x_sum
        xxt_sum = self.xxt_sum
        x       = self.x
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
            
            prop = rand.multivariate_normal(keys[1], x, emp_var * (2.38**2/d)) if (u < 0.95)  else ((rand.normal(keys[1], shape=(d,)) + x) * 100 * d)

            # Compute the log acceptance probability
            alpha = 0.5 * (x @ (solve(r, q.T @ x))
                           - (prop @ solve(r, q.T @ prop)))
            
            return(self.accept(prop, alpha, keys[2]))
            
```


<a id="orgc606726"></a>

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

results = [x0]

sigma = jnp.array([[2,1],[1,2]])

Q, R = qr(sigma)

key = jax.random.PRNGKey(seed=1)

n = 1000

keys = rand.split(key, n)

for i in range(n): results.append(results[-1].oneStep(Q,R,keys[i]))

sample = [state.x for state in results]

emp_var = results[n-1].xxt_sum/n - jnp.outer(results[n-1].x_sum, results[n-1].x_sum.T)/n**2

print(emp_var)
```
