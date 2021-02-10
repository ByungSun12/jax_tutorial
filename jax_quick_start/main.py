# Load module
import timeit
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np

print("You need to run ipython not python.")
# Multiplying Matrices
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

def check_time(exec_code, setup_code):
    list_time = timeit.repeat(exec_code, setup=setup_code, repeat=10)
    print("\tRunning 10 -> Mean Seconds :", np.mean(list_time),"+-", np.std(list_time))

# Let's dive right in and multiply two big matrices.
print("Multiply two big matrices.")
exec_code = """
def test():
    jnp.dot(x, x.T).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import random
key = random.PRNGKey(0)
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
"""
check_time(exec_code, setup_code)
print()

# JAX NumPy functions work on regular NumPy arrays.
print("Multiply two big matrices on NumPy arrays.")
exec_code = """
def test():
    jnp.dot(x, x.T).block_until_ready()
"""
setup_code = """
import numpy as np
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)
size = 3000
x = np.random.normal(size=(size, size)).astype(np.float32)
"""
check_time(exec_code, setup_code)
print()
# That's slower because it has to transfer data to the GPU every time. 

# You can ensure that an NDArray is backed by device meory using `device_put`.
print("Multiply two big matrices on NumPy arrays with device_put.")
exec_code = """
def test():
    jnp.dot(x, x.T).block_until_ready()
"""
setup_code = """
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import device_put

key = random.PRNGKey(0)
size = 3000
x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
"""
check_time(exec_code, setup_code)
print()

# The output of `device_put` still acts like an NDArray,
# but it only copies values back to the CPU when they're needed for printing,
# plotting, saving to disk, branching, etc.
# The behavior of `device_put` is equivalent to the function `jit(lambda x: x)`, but it's faster.

# just numpy version
print("Just two big matrices with NumPy Module.")
exec_code = """
def test():
    np.dot(x, x.T)
"""
setup_code = """
import numpy as np

size = 3000
x = np.random.normal(size=(size, size)).astype(np.float32)
"""
check_time(exec_code, setup_code)
print()

# Usage of jit, grad, vmap
# `jit` : for speeding up code
# `grad` : for taking derivates
# `vmap` : for automatic vectorization or batching

# 1) `jit` to speed up functions
print("SELU function test before applying jit")
exec_code = """
def test():
    selu(x).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (1000000,))
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha*jnp.exp(x) - alpha)
"""
check_time(exec_code, setup_code)
print()

print("SELU function test after applying jit")
exec_code = """
def test():
    selu(x).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)
x = random.normal(key, (1000000,))
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha*jnp.exp(x) - alpha)
selu_jit = jit(selu)
"""
check_time(exec_code, setup_code)
print()

# 2) Taking derivations with `grad`
# Just like in Autograd, we can compute gradients with the `grad` function.
print("Getting gradient of sum_logistic function with x = [1, 2, 3]")
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))
print()

print("Checking gradients with first finite differences")
# Gradient checking function
def first_finite_differences(f, x):
    eps = 1e-3
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                    for v in jnp.eye(len(x))])
print(first_finite_differences(sum_logistic, x_small))
print()
# For more advanced autodiff, we can use `jax.vjp` for reverse-mode vector-Jacobian products
# and `jax.jvp` for forword-mode Jacobian-vector products.

# Efficiently computes full Hessian matrices:
from jax import jacfwd, jacrev
def hessian(fun):
    return jit(jacfwd(jacrev(fun)))

# 3) Auto-vectorization with `vmap`
print("Naively Batched Appky matrix : list comprehension")
exec_code = """
def test():
    naively_batched_apply_matrix(batched_x).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
    return jnp.dot(mat, v)

def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])
"""
check_time(exec_code, setup_code)
print()


print("Manually Batched Appky matrix")
exec_code = """
def test():
    batched_apply_matrix(batched_x).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
    return jnp.dot(mat, v)

@jit
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)
"""
check_time(exec_code, setup_code)
print()

print("Automatic Batched Appky matrix with vmap")
exec_code = """
def test():
    vmap_batched_apply_matrix(batched_x).block_until_ready()
"""
setup_code = """
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.PRNGKey(0)

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
    return jnp.dot(mat, v)

@jit
def vmap_batched_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)
"""
check_time(exec_code, setup_code)
print()