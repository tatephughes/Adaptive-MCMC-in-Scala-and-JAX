#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
from jax.numpy.linalg import solve, qr, norm, eig, eigh, inv, cholesky, det
from jax.scipy.linalg import solve_triangular
import numpy as np
import time
import csv
import os
import re

jax.config.update('jax_enable_x64', False)

def try_accept(state, prop, alpha, mix, eps, key):

  """ Accepts a proposed move from ~state~ with probability ~exp(min(0,alpha))~
  
  state -- A tuple for the state of the chain, in the format ~(j, x, x_mean, prop_cov)~
  prop -- The proposed move, x
  alpha -- The pre-calculated log of the Hastings ratio
  key -- PRNG keys
  
  return -- The next state (tuple) of the chain with updated mean and covariance
  """
  
  j        = state[0]
  x        = state[1]
  x_mean   = state[2]
  prop_cov = state[3]
  accept_count = state[4]
  d        = x.shape[0]
  
  log_prob = jnp.minimum(0.0, alpha)
  
  u = rand.uniform(key)

  x_new, is_accepted = jl.cond((jnp.log(u) < log_prob),
                               0, lambda _: (prop, 1),
                               0, lambda _: (x, 0))

  # update empirical mean
  x_mean_new = (x_mean*j + x_new)/(j+1)

  # update proposal covariance
  prop_cov_new = jnp.select(condlist   = [jnp.logical_or(mix, j<2*d), jnp.logical_and(not mix, j>=2*d)],
                            choicelist = [
                              prop_cov*((j-1)/j) +
                              (j*jnp.outer(x_mean-x_mean_new, x_mean-x_mean_new) +
                               jnp.outer(x_new - x_mean_new, x_new - x_mean_new)
                                 )*5.6644/(j*d),
                              prop_cov*((j-1)/j) +
                              (j*jnp.outer(x_mean-x_mean_new, x_mean-x_mean_new) +
                               jnp.outer(x_new - x_mean_new, x_new - x_mean_new) +
                               0.01*jnp.identity(d)
                                 )*5.6644/(j*d)],
                            default = 1)
  
  return((j + 1,
          x_new,
          x_mean_new,
          prop_cov_new,
          accept_count + is_accepted))

def adapt_step(state, q, r, mix, eps, key):

    """ Samples from the current proposal distribution and computes the log Hastings Ratio, and returns the next state according to ~try_accept~

    state -- A tuple for the state of the chain, in the format ~(j, x, x_mean, prop_cov)~
    q,r -- The QR-decomposition of the target Covariance, for computing the inverse
    key -- PRNG key

    return -- The next state of the chain
    """
    
    j = state[0]
    x = state[1]
    d = x.shape[0]
    prop_cov = state[3]
    
    keys = rand.split(key,3)

    prop = jl.cond(jnp.logical_or(j <= 2, jnp.logical_and(mix, rand.uniform(keys[0]) < eps)),
                   lambda key: rand.normal(key, shape=(d,))/(jnp.sqrt(100*d)) + x, # 'Safe' sampler
                   lambda key: rand.multivariate_normal(key, x, prop_cov), # 'Adaptive' sampler
                   keys[1])
    
    # Compute the log Hastings ratio
    alpha = 0.5 * (x.T @ (solve(r, q.T @ x)) - (prop.T @ solve(r, q.T @ prop)))
                   
    return(try_accept(state, prop, alpha, mix, eps, keys[2]))

def cov(sample):
    
    means = jnp.mean(sample, axis=1)
    
    deviations = sample.T - means
    
    N = sample.shape[0]
    
    covariance = (deviations.T @ deviations) / (N - 1)
    
    return covariance

def mhead(M, n=3):

    return M[0:n,0:n]

def thinned_step(thinrate, state, q, r, eps, mix, key):

    """Performs ~thinrate~ iterations of adapt_step, without saving the intermediate steps"""
    
    keys = rand.split(key,thinrate)

    # I think this should scan over the keys!
    return jl.fori_loop(0, thinrate, (lambda i, x: adapt_step(x, q, r, mix, eps, keys[i])), state)

def sub_optim_factor(sigma, sigma_j):

    """Computes the sub-optimality factor between the true target covariance ~sigma~ and the sampling covariance ~sigma_j~, from Roberts and Rosethal
    """
    
    d = sigma.shape[0]

    # looking at their code, this might be what was intended?
    lam = eig(sigma_j @ inv(sigma))[0]
    
    b = (d * sum(lam**-2) / sum(lam**-1)**2).real

    return b

def mat_sqrt(M):

    M_decomp = eig(M) # doesn't take advantage of the matrix properties!

    return M_decomp[1] @ jnp.diag(jnp.sqrt(M_decomp[0])) @ inv(M_decomp[1])

def plot_trace(sample, file_path, j=0):

    """Plots a trace plot of the jth coordinate of the given array of states,
    and saves the figure to ~file_path~"""
    
    first = sample[:,j]
    plt.figure(figsize=(590/96,370/96))
    plt.plot(first)
    plt.title(f'Trace plot of coordinate {j}')
    plt.xlabel('Step')
    plt.ylabel('First coordinate value')
    plt.grid(True)
    plt.savefig(file_path, dpi=96)

def run_with_complexity(sigma_d, mix, key):

    """Runs the main loop on a given target Covariance, and gets the time the main loop took.

    sigma_d -- The target covariance to sample from, usually a submatrix of ~chaotic_variance.csv~
    key -- PRNG key

    return -- A tuple containing results of the test, including the duration and suboptimality factor
    """

    Q, R = qr(sigma_d) # take the QR decomposition of sigma

    d = sigma_d.shape[0]
    
    # these numbers get good results up to d=100
    n = 1
    thinrate = 1
    burnin = 1000000

    keys = rand.split(key, n + burnin + 1)
    state0 = (2, jnp.zeros(d), jnp.zeros(d), ((0.1)**2) * jnp.identity(d)/d, 0)
    
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, mix, 0.01, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, mix, 0.01, keys[i]), state0)

    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    end_time = time.time()
    duration = time.time()-start_time
    
    sigma_j = am_sample[3][-1]

    b = sub_optim_factor(sigma_d,sigma_j)

    return n, thinrate, burnin, duration, float(b) # making it into a normal float for readability

def compute_time_graph(sigma, mix=False, csv_file="./data/JAX_compute_times_test.csv", is_64_bit=False):

    """Loop through all the primary minors of ~sigma~ and runs the complexity test on each of them, saving the result to ~csv_file~
    """

    jax.config.update('jax_enable_x64', is_64_bit)
    
    d = sigma.shape[0]

    key = rand.PRNGKey(seed=1)
    keys = rand.split(key, d)
    
    x = range(1, d+1)
    y = jnp.array([run_with_complexity(sigma[:i,:i], mix, keys[i]) for i in x if print(i) or True])

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(y)

def generate_sigma(d):

    key = jax.random.PRNGKey(seed=1)
    M = rand.normal(key, shape = (d,d))
    return inv(M @ M.T)

def read_sigma(d, file_path = './data/very_chaotic_variance.csv'):

    matrix = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(item) for item in row])
    return jnp.array(matrix)[0:d,0:d]

def main(d=10, n=1000, thinrate=1000, burnin=0,
         write_files = False,
         trace_file = "./Figures/adaptive_trace_JAX_test.png",
         sample_file = "./data/jax_sample",
         mix = False,
         eps = 0.01,
         get_sigma = read_sigma,
         use_64 = False):

    """Runs the chain with a few diagnostics, mainly for testing. Returns a jax array containing the simulated sample.I
    """

    jax.config.update('jax_enable_x64', use_64)
    
    # the actual number of iterations is n*thin + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key, n + burnin + 1)
    
    sigma = get_sigma(d=d)
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state before burn-in, j starts at "2" for safetys
    state0 = (2, jnp.zeros(d), jnp.zeros(d), ((0.1)**2) * jnp.identity(d)/d, 0)
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, mix, eps, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, mix, eps, keys[i]), state0)

    # the sample
    sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    # the time of the computation in seconds
    end_time = time.time()
    duration = time.time() - start_time
    
    # The final sampling covariance
    sigma_j = sample[3][-1] / (5.6644/d)
    acc_rate = sample[4][-1] / (n*thinrate+burnin)

    # According to Roberts and Rosethal, this value should go to 1.
    b1 = sub_optim_factor(sigma, jnp.identity(d))
    b2 = sub_optim_factor(sigma,sigma_j)

    print(f"The optimal sampling variance of x_1 is {sigma[0,0] * (5.6644/d)}")
    print(f"The actual sampling variance of x_1 is  {sigma_j[0,0] * (5.6644/d)}")
    print(f"The initial b value is {b1}")
    print(f"The final b value is {b2}")
    print(f"The acceptance rate is {acc_rate}")
    print(f"The computation took {duration} seconds")

    if write_files:

        # This mess writes out the sample into a format to be read by R with "source("<filename>")"

        eff_func = lambda M: sub_optim_factor(sigma, M)
        eff_vectorised = jax.vmap(eff_func)
        
        print("Computing the vector of b values...")
        # b_values = ', '.join([str(f) for f in eff_vectorised(sample[3])])
        b_values = ', '.join(map(str, eff_vectorised(sample[3])))
        print("Done!")
        
        print(f"Saving to the file {sample_file}...")
    
        if mix:
            if use_64:
                instance = "64_MD"
            else:
                instance = "32_MD"
        else:
            if use_64:
                instance = "64_IC"
            else:
                instance = "32_IC"

        results_func = ''.join(("output_results <- function(){",
                            f"chain_jax_{instance} <- mcmc(sample_jax_{instance}, thin={thinrate}, start=0); min_ess <- min(effectiveSize(chain_jax_{instance})); print(paste('The optimal sampling value of x_1 is', {sigma[0,0]} * (5.6644/{d}))); print(paste('The actual sampling value of x_1 is', {sigma_j[0,0]} * (5.6644/{d}))); print(paste('The initial b value is', b1_jax_{instance})); print(paste('The final b value is', b_vals_jax_{instance}[-1])); print(paste('The acceptance rate is', acc_rate_jax_{instance})); print(paste('The computation took', compute_time_jax_{instance}, 'seconds')); print(paste('The minimum Effective Sample Size is', min_ess)); print(paste('The minimum ESS per second is', min_ess/compute_time_jax_{instance}))",
                            "}"))

        lines = [
            "library(coda)",
            f"b1_jax_{instance} <- {b1}",
            f"acc_rate_jax_{instance} <- {acc_rate}",
            f"compute_time_jax_{instance} <- {duration}",
            f"sample_jax_{instance} <- matrix(c(" + ', '.join(map(str, sample[1].flatten())) + f"), ncol={d}, byrow=TRUE)",
            f"b_vals_jax_{instance} <- c(" + b_values + ")",
            results_func
        ]
                
        with open(sample_file, 'w') as f:
            for line in lines:
                    f.write(line + "\n\n")

        print("Done!")

        # Plotting has been moved over to be external, see diagnostics.org
        # plot the trace of the first coordinate
        #plot_trace(sample[1], trace_file, 0)
        
    return sample

if __name__ == "__main__":

    # This code checks wether the working  directory is correct, and if not, attemps
    # to change it.
    if not (re.search(r".*/Adaptive-MCMC-in-Scala-and-JAX$", os.getcwd())):
        os.chdir("../../../")
        if not (re.search(r".*/Adaptive-MCMC-in-Scala-and-JAX$", os.getcwd())):
            print("ERROR: Cannot find correct working directory")
        else:
            print("Succesfully found working directory")
    else:
        print("In correct working directory")

    sample = main(d=10, n=1000, thinrate=1000, burnin=0, mix=False)
