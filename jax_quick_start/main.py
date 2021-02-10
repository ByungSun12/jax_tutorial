# Load module
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# Multiplying Matrices
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

# Let's dive right in and multiply two big matrices.
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
# %timeit jnp.dot(x, x.T).block_until_ready() # asynchronous execution by default.

# JAX NumPy functions work on regular NumPy arrays.
import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
# %timeit jnp.dot(x, x.T).block_until_ready()
# That's slower because it has to transfer data to the GPU every time. 
# You can ensure that an NDArray is backed by device meory using `device_put`.