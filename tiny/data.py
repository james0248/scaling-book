import numpy as np
import jax
import jax.numpy as jnp


def encode_batch(idxs: np.ndarray, max_digits: int) -> tuple[np.ndarray, np.ndarray]:
    size = 10**max_digits

    lhs, rhs = idxs // size, idxs % size
    answer = lhs * rhs

    def int2arr(x: int, max_digits: int):
        return (x[:, None] // 10 ** np.arange(max_digits)[::-1] % 10).astype(np.int8)

    token_ids = np.concat(
        (
            int2arr(lhs, max_digits),
            np.full((idxs.shape[0], 1), 10, dtype=np.int8),
            int2arr(rhs, max_digits),
            np.full((idxs.shape[0], 1), 11, dtype=np.int8),
            int2arr(answer, 2 * max_digits),
        ),
        axis=1,
    )
    mask = np.concat(
        (np.zeros(2 * max_digits + 1, dtype=bool), np.ones(2 * max_digits, dtype=bool))
    )

    return token_ids, mask


def decode_batch(token_ids: np.ndarray) -> str:
    """decodes a batch of tokens into readable string"""

    vocab_map = np.array(list("0123456789*="))
    chars = vocab_map[token_ids]
    return np.ascontiguousarray(chars).view(f"<U{chars.shape[1]}").ravel()


def generate_data(
    max_digits: int, num_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates training/eval dataset given the maximum number of digits.

    simply makes all possible combinations, shuffle them, split them, return
    """

    rng = np.random.default_rng(seed)
    size = 10 ** (2 * max_digits)

    num_samples = num_samples or size
    data = rng.choice(size, num_samples, replace=False)
    data, mask = encode_batch(data, max_digits)

    return data, mask


def get_batch_func(max_digits: int, batch_size: int):
    """function that generates multiply data on-the-fly using jax"""

    def get_batch(key):
        key1, key2 = jax.random.split(key, 2)

        # think as least-significant-first
        lhs = jax.random.randint(key1, (batch_size, max_digits), 0, 10)
        rhs = jax.random.randint(key2, (batch_size, max_digits), 0, 10)

        adder = (
            jnp.arange(max_digits)[:, None, None] + jnp.arange(max_digits)[None, :, None]
            == jnp.arange(2 * max_digits)[None, None, :]
        ).astype(jnp.int8)
        temp = jnp.einsum("bl,br,lri->bi", lhs, rhs, adder)

        def step(carry, x):
            cur = x + carry
            return cur // 10, cur % 10

        _, answer = jax.lax.scan(step, jnp.zeros((batch_size,), dtype=jnp.int32), temp.T)
        answer = answer.T

        lhs, rhs, answer = lhs[:, ::-1], rhs[:, ::-1], answer[:, ::-1]
        data = jnp.concat(
            (lhs, jnp.full((batch_size, 1), 10), rhs, jnp.full((batch_size, 1), 11), answer), axis=1
        ).astype(jnp.int8)
        mask = jnp.concat(
            (
                jnp.zeros(2 * max_digits + 1, dtype=jnp.bool),
                jnp.ones(2 * max_digits, dtype=jnp.bool),
            )
        )
        return data, mask

    return get_batch


if __name__ == "__main__":
    get_batch = get_batch_func(2, 10)
    key = jax.random.key(42)
    print(get_batch(key))
