#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
#import jax.scipy.stats as stat
#from jax import vmap
from jax.numpy.linalg import solve, qr, norm, eig, inv, cholesky
import jax
import time
from AM_in_JAX_tests import *
import csv
import numpy as np

jax.config.update('jax_enable_x64', True)

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

@jit
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

def cov(sample):
    
    means = jnp.mean(sample, axis=0)

    deviations = sample - means
    
    N = sample.shape[0]
    
    covariance = jnp.dot(deviations.T, deviations) / (N - 1)
    
    return covariance

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

def plotter(sample, file_path, d):
    
    first = sample[:,0]
    plt.figure(figsize=(590/96,370/96))
    plt.plot(first)
    plt.title(f'Trace plot of the first coordinate, d={d}')
    plt.xlabel('Step')
    plt.ylabel('First coordinate value')
    plt.grid(True)
    plt.savefig(file_path, dpi=96)

def run_with_complexity(sigma_d, key):

    Q, R = qr(sigma_d) # take the QR decomposition of sigma

    # since I'm timing, this is not a pure function, so
    # it won't work completely through JAX.

    d = sigma_d.shape[0]
    
    # these numbers get good results up to d=100
    n = 10000
    thinrate = 10
    burnin = 1000000

    keys = rand.split(key, n + burnin)
    state0 = (0, jnp.zeros(d), jnp.zeros(d), jnp.identity(d), False)
    
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(0, burnin, lambda i,x: AM_step(x, Q, R, keys[i]), state0)
    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin:])[1]

    sigma_j = cov(am_sample[1])
    
    end_time = time.time()
    duration = time.time()-start_time
    
    b = effectiveness(sigma_d,sigma_j)

    return n, thinrate, burnin, duration, float(b) # making it into a normal float for readability

def compute_time_graph(sigma, csv_file):
    
    d = sigma.shape[0]

    key = rand.PRNGKey(seed=1)
    keys = rand.split(key, d)
    
    x = range(1, d+1)
    y = jnp.array([run_with_complexity(sigma[:i,:i], keys[i]) for i in x if print(i) or True])

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(np.array(y))

def thinned_step(thinrate, state, q, r, key):

    keys = rand.split(key,thinrate)
    
    return jl.fori_loop(0, thinrate, (lambda i, x: AM_step(x, q, r, keys[i])), state)

def main(d=10, n=100000, thinrate=10, burnin=10000, file="Figures/adaptive_trace_JAX.png"):

    start_time = time.time()

    # the actual number of iterations is n*thin + burnin
    computed_size = n*thinrate + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key, n + burnin)
    
    # create a chaotic variance matrix to target
    M = rand.normal(key, shape = (d,d))
    sigma = M.T @ M
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state before burn-in
    state0 = (0, jnp.zeros(d), jnp.zeros(d), jnp.identity(d), False)

    # JAX's ~scan~ isn't quite ~iterate~, so this is a 'dummy'
    # function with an unused argument to call thinned_step for the
    # actually used samples
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, key)
        return(nextstate, nextstate)

    # inital state, after burnin
    start_state = jl.fori_loop(0, burnin, lambda i,x: AM_step(x, Q, R, keys[i]), state0)

    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin:])[1]

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

    plotter(am_sample[1], file, d)
    
    return am_sample

if __name__ == "__main__":
    #test_try_accept()
    #test_init_step()
    #test_adapt_step()
    #test_AM_step()
    #test_thinned_step()
    main(d=10,n=10000, thinrate=10, burnin=10000)
    #or high dimensions
    #main(d=100, n=10000, thinrate=100, burnin=1000000, file ="Figures/adaptive_trace_JAX_high_d.png")
    #numpy_matrix = []
    #with open('chaotic_variance.csv', 'r', newline='') as file:
    #    reader = csv.reader(file)
    #    for row in reader:
    #        # Assuming the content is numeric, converts strings to floats.
    #        # This step might need adjustment based on the actual content of your CSV.
    #        numpy_matrix.append([float(item) for item in row])
    #sigma = jnp.array(numpy_matrix)
    #compute_time_graph(sigma, "data/JAX_compute_times.csv")
