#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr
import jax

class Adaptive_State:

    def __init__(self, j, x, x_sum, xxt_sum):

        self.j       = j
        self.x_sum   = x_sum
        self.xxt_sum = xxt_sum
        self.x       = x

    def accept(self, prop, alpha, key):

        log_prob = jnp.minimum(0.0, alpha)

        u = rand.uniform(key)

        new_x = prop if (jnp.log(u) < log_prob) else self.x

        return(Adaptive_State(
            self.j + 1,
            new_x,
            self.x_sum + new_x,
            self.xxt_sum + jnp.outer(new_x, new_x)))

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
