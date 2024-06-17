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
import os
import re

jax.config.update('jax_enable_x64', False)

def try_accept(state, prop, alpha, mix, key):

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

  x_mean_new = x_mean*(j-1)/j  + x_new/j

  prop_cov_new = jnp.select(condlist   = [mix, not mix],
                            choicelist = [
                              prop_cov*((j-1)/j) # this isn't good for float computation, maybe?
                              + (jnp.outer(x-x_mean, x-x_mean_new))*5.6644/(j*d),
                              prop_cov*((j-1)/j)
                              + (j*jnp.outer(x_mean,x_mean) -
                                 (j+1)*jnp.outer(x_mean_new,x_mean_new) +
                                 jnp.outer(x_new,x_new) +
                                 0.01*jnp.identity(d)
                                 )*5.6644/(j*d)],
                            default = 1)

  # NOTE: seems inefficient to construct a diagonal identity matrix like this, I would imagine there is a better way to do this
  
  return((j + 1,
          x_new,
          x_mean_new,
          prop_cov_new,
          accept_count+is_accepted))

def adapt_step(state, q, r, mix, key):

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

    prop = jl.cond((j <= 2*d) | (mix & (rand.uniform(keys[0]) < 0.01)),
                   lambda key: rand.normal(key, shape=(d,))/(100*d) + x, # 'Safe' sampler
                   lambda key: rand.multivariate_normal(key, x, prop_cov), # 'Adaptive' sampler
                   keys[1])
    
    # Compute the log Hastings ratio///
    alpha = 0.5 * (x.T @ (solve(r, q.T @ x))
                   - (prop.T @ solve(r, q.T @ prop)))

    return(try_accept(state, prop, alpha, mix, keys[2]))

def cov(sample):
    
    means = jnp.mean(sample, axis=1)
    
    deviations = sample.T - means
    
    N = sample.shape[0]
    
    covariance = (deviations.T @ deviations) / (N - 1)
    
    return covariance

def mhead(M, n=3):

    return M[0:n,0:n]

def thinned_step(thinrate, state, q, r, mix, key):

    """Performs ~thinrate~ iterations of adapt_step, withour saving the intermiade steps"""
    
    keys = rand.split(key,thinrate)

    # I think this should scan over the keys!
    return jl.fori_loop(0, thinrate, (lambda i, x: adapt_step(x, q, r, mix, keys[i])), state)

def sub_optim_factor(sigma, sigma_j):

    """Computes the sub-optimality factor between the true target covariance ~sigma~ and the sampling covariance ~sigma_j~, from Roberts and Rosethal
    """
    
    d = sigma.shape[0]
    
    """
    sigma_j_decomp = eigh(sigma_j)
    sigma_decomp = eigh(sigma)
    
    rootsigmaj = sigma_j_decomp[1] @ jnp.diag(jnp.sqrt(sigma_j_decomp[0])) @ inv(sigma_j_decomp[1])
    rootsigmainv = inv(sigma_decomp[1]) @ jnp.diag(1/jnp.sqrt(sigma_decomp[0])) @ sigma_decomp[1]

    # the below line relies on the ~eig~ function which doesn't work on GPUs
    lam = eig(rootsigmaj @ rootsigmainv)[0]
    """

    # maybe they meant cholesky?
    #lam = eig(cholesky(sigma_j) @ inv(cholesky(sigma)))[0]
    # the cleanest BUT NOT THE MOST EFFICIENT
    #lam = eig(mat_sqrt(sigma_j) @ inv(mat_sqrt(sigma)))[0]

    # without the square roots?
    #lam = eig(sigma_j @ inv(sigma))[0]

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
    mix = False

    keys = rand.split(key, n + burnin + 1)
    state0 = (1, jnp.zeros(d), jnp.zeros(d), ((0.1)**2) * jnp.identity(d)/d, 0)
    
    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, mix, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, mix, keys[i]), state0)

    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    end_time = time.time()
    duration = time.time()-start_time
    
    sigma_j = am_sample[3][-1]

    b = sub_optim_factor(sigma_d,sigma_j)

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

def generate_sigma(d):

    key = jax.random.PRNGKey(seed=1)
    M = rand.normal(key, shape = (d,d))
    return M @ M.T

def read_sigma(d, file_path = './data/chaotic_variance.csv'):

    matrix = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(item) for item in row])
    return jnp.array(matrix)[0:d,0:d]

def mixing_test(get_sigma = read_sigma, mix = False, csvfile = "./data/mixing_test.csv"):
    
    sigma = get_sigma(d=10)
    Q, R = qr(sigma) # take the QR decomposition of sigma
    d = sigma.shape[0]
    
    n = 100

    key = jax.random.PRNGKey(seed=1)

    sample = main(d=d, n=n, thinrate=20000, burnin=0,
                  file = "./Figures/adaptive_trace_JAX_mixing.png",
                  mix=mix, get_sigma=lambda d:sigma[0:d,0:d])

    print(sub_optim_factor(sigma, sample[3][-1]))
    
    eff_func = lambda M: sub_optim_factor(sigma, M)
    eff_vectorised = jax.vmap(eff_func)
    
    b_values = eff_vectorised(sample[3])

    y = jnp.column_stack((sample[0], b_values))
    
    with open(csvfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(y)

def main(d=10, n=1000, thinrate=10, burnin=10000,
         file="./Figures/adaptive_trace_JAX_test.png",
         mix = False,
         get_sigma = generate_sigma):

    """Runs the chain with a few diagnostics, mainly for testing. Returns a jax array containing the simulated sample.I
    """

    # the actual number of iterations is n*thin + burnin

    # keys for PRNG
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key, n + burnin + 1)
    
    sigma = get_sigma(d=d)
    Q, R = qr(sigma) # take the QR decomposition of sigma

    # initial state before burn-in
    state0 = (1, jnp.zeros(d), jnp.zeros(d), ((0.1)**2) * jnp.identity(d)/d, 0)

    def step(carry, key):
        nextstate = thinned_step(thinrate, carry, Q, R, mix, key)
        return(nextstate, nextstate)

    start_time = time.time()
    
    # inital state, after burnin
    start_state = jl.fori_loop(1, burnin+1, lambda i,x: adapt_step(x, Q, R, mix, keys[i]), state0)

    # the sample
    am_sample = jl.scan(step, start_state, keys[burnin+1:])[1]

    # the time of the computation in seconds
    end_time = time.time()
    duration = time.time() - start_time
    
    # The final sampling covariance
    sigma_j = am_sample[3][-1] / (5.6644/d)
    acc_rate = am_sample[4][-1] / (n*thinrate+burnin)

    # According to Roberts and Rosethal, this value should go to 1.
    b1 = sub_optim_factor(sigma, jnp.identity(d))
    b2 = sub_optim_factor(sigma,sigma_j)

    print(f"The optimal sampling value of x_1 is {sigma[0,0] * (5.6644/d)}")
    print(f"The actual sampling value of x_1 is  {sigma_j[0,0] * (5.6644/d)}")
    print(f"The initial b value is {b1}")
    print(f"The final b value is {b2}")
    print(f"The acceptance rate is {acc_rate}")
    print(f"The computation took {duration} seconds")

    # instead of this plotter function, i want it to write am_sample with all b values to a csv.
    plot_trace(am_sample[1], file, 1)
    
    return am_sample

if __name__ == "__main__":

    # This code checks wether the working directory is correct, and if not, attemps
    # to change it.
    if not (re.search(r".*/Adaptive-MCMC-in-Scala-and-JAX$", os.getcwd())):
        os.chdir("../../../")
        if not (re.search(r".*/Adaptive-MCMC-in-Scala-and-JAX$", os.getcwd())):
            print("ERROR: Cannot find correct working directory")
        else:
            print("Succesfully found working directory")
    else:
        print("In correct working directory")
    
    #sample = main(file = "./Figures/adaptive_trace_JAX_test.png", mix = True, get_sigma=read_sigma)

    compute_time_graph(read_sigma(d=10), "data/JAX_64bit_compute_times_laptop_test.csv")
    #mixing_test(read_sigma, mix=True,
    #            csvfile = "./data/so_factor_mixing.csv")
    #mixing_test(read_sigma, mix=False,
    #            csvfile = "./data/so_factor_not_mixing.csv")
