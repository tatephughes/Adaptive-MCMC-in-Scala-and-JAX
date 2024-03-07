#!/usr/bin/env python3

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr, norm, eig, inv
import jax
import time

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

def adapt_step(state, q, r, key):

    j       = state[0]
    x       = state[1]
    x_sum   = state[2]
    xxt_sum = state[3]
    d       = x.shape[0]

    keys = rand.split(key,3)

    z = rand.normal(keys[0], shape=(d,))
    
    emp_var = xxt_sum/j - jnp.outer(x_sum, x_sum.T)/j**2

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
    
    return(try_accept(state, prop, alpha, keys[2]))

def test_adapt_step():
    return True

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
                   lambda y: adapt_step(y, q, r, key)))

def test_AM_step():
    return True

def cov(sample):
    
    means = jnp.mean(sample, axis=0)

    deviations = sample - means
    
    N = sample.shape[0]
    
    covariance = jnp.dot(deviations.T, deviations) / (N - 1)
    
    return covariance

import matplotlib.pyplot as plt

def plotter(sample, file_path):
    
    first = sample[:,0]
    plt.figure(figsize=(590/96,370/96))
    plt.plot(first)
    plt.title('Trace plot of the first coordinate')
    plt.xlabel('Step')
    plt.ylabel('First coordinate value')
    plt.grid(True)
    plt.savefig(file_path, dpi=96)

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
    sigma_j = cov(am_sample   ) 
    
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

    plotter(am_sample, "Figures/adaptive_trace_jax.png")
    plt.show()

if __name__ == "__main__":
    test_try_accept()
    test_init_step()
    test_adapt_step()
    test_AM_step()
    main()
