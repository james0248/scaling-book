import numpy as np


def encode_batch(idxs: np.ndarray, max_digits: int) -> tuple[np.ndarray, np.ndarray]:
    size = 10**max_digits

    lhs, rhs = idxs // size, idxs % size
    answer = lhs + rhs

    def int2arr(x: int, max_digits: int):
        return (x[:, None] // 10 ** np.arange(max_digits)[::-1] % 10).astype(np.int8)

    token_ids = np.concat(
        (
            int2arr(lhs, max_digits),
            np.full((idxs.shape[0], 1), 10, dtype=np.int8),
            int2arr(rhs, max_digits),
            np.full((idxs.shape[0], 1), 11, dtype=np.int8),
            int2arr(answer, max_digits + 1),
        ),
        axis=1,
    )
    mask = np.concat(
        (np.zeros(2 * max_digits + 1, dtype=bool), np.ones(max_digits + 1, dtype=bool))
    )

    return token_ids, mask


def decode_batch(token_ids: np.ndarray) -> str:
    """decodes a batch of tokens into readable string"""

    vocab_map = np.array(list("0123456789+="))
    chars = vocab_map[token_ids]
    return np.ascontiguousarray(chars).view(f"<U{chars.shape[1]}").ravel()


def generate_data(
    max_digits: int, num_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates training/eval dataset given the maximum number of digits.

    simply makes all possible combinations, shuffle them, split them, return
    """

    rng = np.random.default_rng(seed)
    size = 10 ** (2 * max_digits)

    data = rng.choice(size, num_samples, replace=False)
    data, mask = encode_batch(data, max_digits)

    return data, mask
