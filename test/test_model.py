import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import repeat


def test_moe():
    def moe_naive(
        x: jnp.ndarray,
        weights: jnp.ndarray,
        experts: jnp.ndarray,
        kernels: jnp.ndarray,
        biases: jnp.ndarray,
    ) -> jnp.ndarray:
        x_repeat = repeat(x, "b t d -> b t k d", k=2)

        act = (
            jnp.einsum("btkd,btkdf->btkf", x_repeat, kernels["up"][experts]) + biases["up"][experts]
        )
        gate = nn.swish(
            jnp.einsum("btkd,btkdf->btkf", x_repeat, kernels["gate"][experts])
            + biases["gate"][experts]
        )
        out = (
            jnp.einsum("btkf,btkfd->btkd", act * gate, kernels["down"][experts])
            + biases["down"][experts]
        )
        out = jnp.sum(out * weights[..., None], axis=2)

        return out

    def moe_ragged(
        x: jnp.ndarray,
        weights: jnp.ndarray,
        experts: jnp.ndarray,
        kernels: jnp.ndarray,
        biases: jnp.ndarray,
    ) -> jnp.ndarray:

        b, t, k = experts.shape
        d_model = x.shape[2]

        experts_flat = experts.reshape(-1)
        order = jnp.argsort(experts_flat)
        experts_sorted = experts_flat[order]
        expert_cnt = jnp.bincount(experts_flat, length=8)

        def ragged_dense(
            x: jnp.ndarray,
            kernel: jnp.ndarray,
            bias: jnp.ndarray,
            cnt: jnp.ndarray,
            sorted_idx: jnp.ndarray,
        ) -> jnp.ndarray:
            return jax.lax.ragged_dot(x, kernel, cnt) + bias[sorted_idx]

        buf = (
            jnp.zeros((b * t * k, d_model))
            .at[jnp.arange(b * t * k)]
            .set(x[order // (t * k), order // k % t])
        )
        act = ragged_dense(buf, kernels["up"], biases["up"], expert_cnt, experts_sorted)
        gate = nn.swish(
            ragged_dense(buf, kernels["gate"], biases["gate"], expert_cnt, experts_sorted)
        )
        out = ragged_dense(gate * act, kernels["down"], biases["down"], expert_cnt, experts_sorted)

        result = jnp.zeros_like(buf).at[order].set(out)
        result = result.reshape((b, t, k, d_model))
        result = jnp.sum(weights[..., None] * result, axis=2)

        return result

    rng = jax.random.key(42)

    x = jax.random.normal(rng, (4, 12, 32))
    weights = jax.random.normal(rng, (4, 12, 2))
    experts = jax.random.randint(rng, (4, 12, 2), 0, 8)
    kernels = {
        "up": jax.random.normal(rng, (8, 32, 64)),
        "gate": jax.random.normal(rng, (8, 32, 64)),
        "down": jax.random.normal(rng, (8, 64, 32)),
    }
    biases = {
        "up": jax.random.normal(rng, (8, 64)),
        "gate": jax.random.normal(rng, (8, 64)),
        "down": jax.random.normal(rng, (8, 32)),
    }

    out1 = moe_naive(x, weights, experts, kernels, biases)
    out2 = moe_ragged(x, weights, experts, kernels, biases)
    np.testing.assert_allclose(out1, out2)
