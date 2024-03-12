import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.lax as jl
import jax.random as rand
import jax.scipy.stats as stat
from jax import vmap
from jax.numpy.linalg import solve, qr, norm, eig, inv, cholesky
import jax
import time
from AM_in_JAX import *

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

    return True

def test_adapt_step():

    d = 2
    n = 100000
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    # this state was chosen being close to an actual state of the adaptive chain
    state = (100, jnp.zeros(2), jnp.array([-80.0,-5.0]), jnp.array([[260.0,100.0],[100.0,150.0]]), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
    
    '''
    Test 1:
    From a (hypothetical) progressed point, the result should be approximately distributed with a N(0,sigma) distribution.
    '''
    def step(carry, _):
        nextstate = adapt_step(carry, Q, R, keys[carry[0]])[0]
        return(nextstate, nextstate)
    
    assert norm(cov(jl.scan(step, state, jnp.zeros(n))[1][1]) - sigma) < 0.2, "adap_stepr not producing sample sufficiently close to the target distribution"

    
    return True

def test_AM_step():

    d = 2
    n = 100000
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    state0 = (0, jnp.zeros(2), jnp.zeros(2), jnp.identity(2), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
        
    '''
    Test 1:
    Similarily to the init_step test, from state0, the result should be approximately distributed with a N(0,sigma) distribution.
    '''
    def step(carry, _):
        nextstate = AM_step(carry, Q, R, keys[carry[0]])
        return(nextstate, nextstate)
    
    assert norm(cov(jl.scan(step, state0, jnp.zeros(n))[1][1]) - sigma) < 0.2, "init_step not producing sample sufficiently close to the target distribution"
    
    return True

def test_thinned_step():

    d = 2
    n = 1000
    thinrate = 10
    key = jax.random.PRNGKey(seed=1)
    keys = rand.split(key,n)
    # this state was chosen being close to an actual state of the adaptive chain
    state = (100, jnp.zeros(2), jnp.array([-80.0,-5.0]), jnp.array([[260.0,100.0],[100.0,150.0]]), False)
    sigma = jnp.array([[2.0,1.0],[1.0,2.0]])
    Q, R = qr(sigma)
    
    '''
    Test 1:
    the index of a state should increase by thinrate
    '''
    assert (thinned_step(thinrate, state, Q, R, keys[0])[0] == 100+thinrate), "thinned_step not correctly incrementing step count"

    return True
