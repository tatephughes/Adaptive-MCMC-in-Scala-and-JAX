#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr
import jax

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
