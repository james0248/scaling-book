import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


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
    = [s1, s2, ...] * [a1, a2, ...] - [c1, c2, ...] * [b1, b2, ...] -> half
    """
    _, t, _, d = x.shape

    theta = 10_000 ** (-2 * jnp.arange(d // 2) / d)
    seq = jnp.arange(t)
    cos = jnp.expand_dims(jnp.outer(seq, jnp.cos(theta)), axis=1)
    sin = jnp.expand_dims(jnp.outer(seq, jnp.sin(theta)), axis=1)

    out1 = cos * x[..., : d // 2] - sin * x[..., d // 2 :]
    out2 = sin * x[..., : d // 2] + cos * x[..., d // 2 :]
    return jnp.concat((out1, out2), axis=-1)


class AttentionBlock(nn.Module):
    """GQA with RoPE"""

    n_heads: int
    n_kv: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        d_model = x.shape[2]
        d_head = d_model // self.n_heads
        group_size = self.n_heads // self.n_kv

        q = nn.Dense(d_head * self.n_heads)(x)
        k = nn.Dense(d_head * self.n_kv)(x)
        v = nn.Dense(d_head * self.n_kv)(x)

        q = rearrange(q, "b t (n h) -> b t n h", h=d_head)
        k = rearrange(k, "b s (k h) -> b s k h", h=d_head)
        v = rearrange(v, "b s (k h) -> b s k h", h=d_head)
        q = apply_rope(q)
        v = apply_rope(v)

        k_repeat = repeat(k, "b s k h -> b s (g k) h", g=group_size)
        v_repeat = repeat(v, "b s k h -> b s (g k) h", g=group_size)

        scores = jnp.einsum("btnh,bsnh->bts", q, k_repeat) / jnp.sqrt(d_head)
        weights = nn.softmax(scores, axis=2, where=mask[None, :])
        attn = jnp.einsum("bts,bsnh->btnh", weights, v_repeat)
        attn = rearrange(attn, "b t n h -> b t (n h)")
        return attn


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
            x = x + AttentionBlock(self.n_heads, self.n_kv)(nn.RMSNorm(self.d_model)(x), mask)
            x = x + FeedForwardMLP(self.d_ffw)(nn.RMSNorm(self.d_model)(x))

        x = nn.RMSNorm(self.d_model)(x)
        out = nn.Dense(self.vocab_size)(x)
        return out


if __name__ == "__main__":
    rng = jax.random.key(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    model = Transformer(n_layers=1, d_model=56, d_ffw=84, n_heads=4, n_kv=1, vocab_size=12)
    inp = jax.random.randint(inp_rng, (5, 13), 0, 12)
    params = model.init(init_rng, inp)
