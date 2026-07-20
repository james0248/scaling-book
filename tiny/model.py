import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


class FeedForwardMLP(nn.Module):
    """SwiGLU (3 MLP gated layer)"""

    d_ffw: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d_model = x.shape[2]

        act = nn.Dense(self.d_ffw)(x)
        gate = nn.swish(nn.Dense(self.d_ffw)(x))
        out = nn.Dense(d_model)(gate * act)
        return out


def apply_rope(x: jnp.ndarray):
    """
    expects (b, t, n, h) shape

    x_m = [a1, a2, ..., a_d, b1, b2, ..., b_d]
    R_m @ x_m = [c1a1 - s1b1, c2a2 - s2b2, ..., s1a1 + c1b1, s2a2 + c2b2, ...]
    = [c1, c2, ...] * [a1, a2, ...] - [s1, s2, ...] * [b1, b2, ...] -> half
    = [s1, s2, ...] * [a1, a2, ...] + [c1, c2, ...] * [b1, b2, ...] -> half
    """
    _, t, _, d = x.shape

    theta = 10_000 ** (-2 * jnp.arange(d // 2) / d)
    thetas = jnp.outer(jnp.arange(t), theta)
    cos = jnp.expand_dims(jnp.cos(thetas), axis=1)
    sin = jnp.expand_dims(jnp.sin(thetas), axis=1)

    out1 = cos * x[..., : d // 2] - sin * x[..., d // 2 :]
    out2 = sin * x[..., : d // 2] + cos * x[..., d // 2 :]
    return jnp.concat((out1, out2), axis=-1)


def dot_product_attention(
    q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    d_head = q.shape[3]
    group_size = q.shape[2] // k.shape[2]

    q = rearrange(q, "b t (k g) h -> b t k g h", g=group_size)
    scores = jnp.einsum("btkgh,bskh->btskg", q, k)
    weights = nn.softmax(scores / jnp.sqrt(d_head), axis=2, where=mask[..., None, None])
    attn = jnp.einsum("btskg,bskh->btkgh", weights, v)
    attn = rearrange(attn, "b t k g h -> b t (k g h)")
    return attn


class AttentionBlock(nn.Module):
    """GQA with RoPE"""

    n_heads: int
    n_kv: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        d_model = x.shape[2]
        d_head = d_model // self.n_heads

        q = nn.Dense(d_head * self.n_heads)(x)
        k = nn.Dense(d_head * self.n_kv)(x)
        v = nn.Dense(d_head * self.n_kv)(x)

        q = rearrange(q, "b t (n h) -> b t n h", h=d_head)
        k = rearrange(k, "b s (k h) -> b s k h", h=d_head)
        v = rearrange(v, "b s (k h) -> b s k h", h=d_head)

        q = apply_rope(nn.RMSNorm()(q))
        k = apply_rope(nn.RMSNorm()(k))

        attn = dot_product_attention(q, k, v, mask)
        out = nn.Dense(d_model)(attn)
        return out


class Transformer(nn.Module):
    n_layers: int
    d_model: int
    d_ffw: int
    n_heads: int
    n_kv: int
    vocab_size: int

    @nn.compact
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        seq_len = token_ids.shape[1]

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(token_ids)
        mask = jnp.tril(jnp.ones((seq_len, seq_len))).astype(jnp.bool)
        for _ in range(self.n_layers):
            x = x + AttentionBlock(self.n_heads, self.n_kv)(nn.RMSNorm()(x), mask)
            x = x + FeedForwardMLP(self.d_ffw)(nn.RMSNorm()(x))

        x = nn.RMSNorm()(x)
        out = nn.Dense(self.vocab_size)(x)
        return out


if __name__ == "__main__":
    rng = jax.random.key(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    model = Transformer(n_layers=1, d_model=56, d_ffw=84, n_heads=4, n_kv=1, vocab_size=12)
    inp = jax.random.randint(inp_rng, (5, 13), 0, 12)
    params = model.init(init_rng, inp)

    b, t, n, h, k_ = 8, 4, 8, 16, 2
    q = jax.random.normal(rng, (b, t, n, h))
    k = jax.random.normal(rng, (b, t, k_, h))
    v = jax.random.normal(rng, (b, t, k_, h))

    # Official implementation
    out1 = jax.nn.dot_product_attention(q, k, v, is_causal=True)
    out1 = rearrange(out1, "b t n h -> b t (n h)")
    mask = jnp.tril(jnp.ones((t, t))).astype(jnp.bool)
    out2 = dot_product_attention(q, k, v, mask)

    print(f"is allclose: {jnp.allclose(out1, out2)}")
