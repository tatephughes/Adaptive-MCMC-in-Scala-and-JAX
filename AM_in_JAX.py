#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import solve, qr, norm, eig, eigh, inv, cholesky, det
import time
from AM_in_JAX_tests import *
import csv

jax.config.update('jax_enable_x64', False)

def try_accept(state, prop, alpha, key):

  """ Accepts a proposed move from ~state~ with probability ~exp(min(0,alpha))~
  
  state -- A tuple for the state of the chain, in the format ~(j, x, x_mean, prop_cov)~
  prop -- The proposed move, x
  alpha -- The pre-calculated log of the Hastings ratio
  key -- PRNG keys

  return -- The next state (tuple) of the chain with updated mean and covariance
  """

  j       = state[0]
  x       = state[1]
  x_mean  = state[2]
  prop_cov   = state[3]
  d       = x.shape[0]
  
  log_prob = jnp.minimum(0.0, alpha)

  u = rand.uniform(key)

  x_new, is_accepted = jl.cond((jnp.log(u) < log_prob),
                               0, lambda _: (prop, True),
                               0, lambda _: (x, False))

  x_mean_new = x_mean*(j-1)/j  + x_new/j

  # Implements the covariance update equation
  prop_cov_new = jl.cond(j <= 2*d,
                         j,
                         lambda t: prop_cov,
                         j,
                         lambda t: prop_cov*((t-1)/t) + (t*jnp.outer(x_mean,x_mean) - (t+1)*jnp.outer(x_mean_new,x_mean_new) + jnp.outer(x_new,x_new) + 0.01*jnp.identity(d))*5.6644/(t*d))
  
  # NOTE: seems inefficient to construct a diagonal identity matrix like this, I would imagine there is a better way to do this
  
  return((j + 1,
          x_new,
          x_mean_new,
          prop_cov_new,
          is_accepted))

def adapt_step(state, q, r, key):

    """ Samples from the current proposal distribution and computes the log Hastings Ratio, and returns the next state according to ~try_accept~

    state -- A tuple for the state of the chain, in the format ~(j, x, x_mean, prop_cov)~
    q,r -- The QR-decomposition of the target Covariance, for computing the inverse
    key -- PRNG key

    return -- The next state of the chain
    """
    
    j        = state[0]
    x        = state[1]
    prop_cov = state[3]
    d        = x.shape[0]

    keys = rand.split(key,2)
    
    prop = rand.multivariate_normal(keys[0], x, prop_cov)

    # Compute the log Hastings ratio
    alpha = 0.5 * (x.T @ (solve(r, q.T @ x))
                   - (prop.T @ solve(r, q.T @ prop)))

    return(try_accept(state, prop, alpha, keys[1]))

def cov(sample):
    
    means = jnp.mean(sample, axis=0)

    deviations = sample - means
    
    N = sample.shape[0]
    
    covariance = jnp.dot(deviations.T, deviations) / (N - 1)
    
    return covariance

def thinned_step(thinrate, state, q, r, key):

    """Performs ~thinrate~ iterations of adapt_step, withour saving the intermiade steps"""
    
    keys = rand.split(key,thinrate)

    # I think this should scan over the keys!
    return jl.fori_loop(0, thinrate, (lambda i, x: adapt_step(x, q, r, keys[i])), state)

def effectiveness(sigma, sigma_j):

    """Computes the sub-optimality factor between the true target covarinance ~sigma~ and the sampling covariance ~sigma_j~, from Roberts and Rosethal
    """

    d = sigma.shape[0]
    
    sigma_j_decomp = eigh(sigma_j)
    sigma_decomp = eigh(sigma)
    
    rootsigmaj = sigma_j_decomp[1] @ jnp.diag(jnp.sqrt(sigma_j_decomp[0])) @ inv(sigma_j_decomp[1])
    rootsigmainv = inv(sigma_decomp[1] @ jnp.diag(jnp.sqrt(sigma_decomp[0])) @ inv(sigma_decomp[1]))

    # the below line relies on the ~eig~ function which doesn't work on GPUs
    lam = eig(rootsigmaj @ rootsigmainv)[0]
    lambdaminus2sum = sum(1/(lam*lam))
    lambdainvsum = sum(1/lam)

    b = (d * (lambdaminus2sum / (lambdainvsum*lambdainvsum))).real

    return b

def plotter(sample, file_path, d):

    """Plots a trace plot of the dth coordinate of the given array of states, and saves the figure to ~file_path~"""
    
    first = sample[:,0]
    plt.figure(figsize=(590/96,370/96))
    plt.plot(first)
    plt.title(f'Trace plot of the first coordinate, d={d}')
    plt.xlabel('Step')
    plt.ylabel('First coordinate value')
    plt.grid(True)
    plt.savefig(file_path, dpi=96)

def run_with_complexity(sigma_d, key):

    """Runs the main loop on a given target Covariance, and gets the time the main loop took.

    sigma_d -- The target covariance to sample from, usually a submatrix of ~chaotic_variance.csv~
    key -- PRNG key

    return -- A tuple containing results of the test, including the duration and suboptimality factor
    """

    Q, R = qr(sigma_d) # take the QR decomposition of sigma

    d = sigma_d.shape[0]
    
    # these numbers get good results up to d=100
    n = 10000
    thinrate = 10
    burnin = 1000000

    keys = rand.split(key, n + burnin + 1)
    state0 = (1, jnp.zeros(d), jnp.zeros(d), jnp.identity(d)/d, False)
    
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, keys[i]), state0)
    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    end_time = time.time()
    duration = time.time()-start_time
    
    sigma_j = cov(am_sample[1])
    
    b = effectiveness(sigma_d,sigma_j)

    return n, thinrate, burnin, duration, float(b) # making it into a normal float for readability

def compute_time_graph(sigma, csv_file):

    """Loop through all the primary minors of ~sigma~ and runs the complexity test on each of them, saving the result to ~csv_file~
    """
    
    d = sigma.shape[0]

    key = rand.PRNGKey(seed=1)
    keys = rand.split(key, d)
    
    x = range(1, d+1)
    y = jnp.array([run_with_complexity(sigma[:i,:i], keys[i]) for i in x if print(i) or True])

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(y)

def main(d=10, n=100000, thinrate=10, burnin=10000, file="Figures/adaptive_trace_JAX.png"):

    """Runs the chain with a few diagnostics, mainly for testing. Returns a jax array containing the simulated sample.
    """

    # the actual number of iterations is n*thin + burnin
    # computed_size = n*thinrate + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key, n + burnin + 1)
    
    # create a chaotic variance matrix to target
    M = rand.normal(keys[0], shape = (d,d))
    sigma = M.T @ M
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state before burn-in
    state0 = (1, jnp.zeros(d), jnp.zeros(d), ((0.1)**2) * jnp.identity(d)/d, False)

    # JAX's ~scan~ isn't quite ~iterate~, so this is a 'dummy'
    # function with an unused argument to call thinned_step for the
    # actually used samples
    # NOTE: this comment may be out of date now that I am scanning over the keys
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, keys[i]), state0)

    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    # the tiume of the computation in seconds
    end_time = time.time()
    duration = time.time()-start_time
    
    # the empirical covariance of the sample
    #sigma_j = cov(am_sample[1])
    C_j = am_sample[3][-1]# / (5.6644/d)
    b = effectiveness(sigma,C_j)

    #print(f"The true variance of x_1 is {sigma[0,0]}")
    #print(f"The empirical sigma value is {C_j[0,0]}")
    print(f"The optimal sampling value is {sigma[0,0] * (5.6644/d)}")
    print(f"The actual sampling value is {C_j[0,0]}")
    print(f"The b value is {b}")
    print(f"The computation took {duration} seconds")

    plotter(am_sample[1], file, d)
    
    return am_sample

if __name__ == "__main__":
    #test_try_accept()
    #test_init_step()
    #test_adapt_step()
    #test_AM_hstep()
    #test_thinned_step()
    
    #main(file ="Figures/adaptive_trace_JAX_d_10.png")
    
    #or high dimensions
    
    #main(d=100, n=10000, thinrate=100, burnin=1000000, file ="Figures/adaptive_trace_JAX_d_100.png")

    # For computing the time graph
    
    matrix = []
    with open('./data/chaotic_variance.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(item) for item in row])
    sigma = jnp.array(matrix)
    compute_time_graph(sigma, "data/JAX_compute_times-laptop-2.csv")
