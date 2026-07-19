import numpy as np


def encode_batch(idxs: np.ndarray, max_digits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    converts index to token_ids. zero pad numbers to max_digits for ease.

    we only accept 0-9, +, = (12 tokens)
    + -> 10, = -> 11
    """

    def _encode(idx: int, max_digits: int) -> tuple[np.ndarray, np.ndarray]:
        length = 10**max_digits
        lhs, rhs = idx // length, idx % length
        answer = lhs + rhs

        def int2arr(x: int, max_digits):
            return np.array(list(f"{x:0{max_digits}}"), dtype=int)

        out = np.concat(
            (
                int2arr(lhs, max_digits),
                np.array([10]),
                int2arr(rhs, max_digits),
                np.array([11]),
                int2arr(answer, max_digits + 1),
            )
        )
        mask = np.concat(
            (np.zeros(2 * max_digits + 2, dtype=bool), np.ones(max_digits + 1, dtype=bool))
        )
        return out, mask

    v_encode = np.vectorize(_encode, excluded={"max_digits"}, signature="(),()->(n),(n)")
    return v_encode(idxs, max_digits)


def decode_batch(token_ids: np.ndarray) -> str:
    """decodes a batch of tokens into readable string"""

    VOCAB = np.array(list("0123456789+="))
    chars = VOCAB[token_ids]
    return np.ascontiguousarray(chars).view(f"<U{chars.shape[1]}").ravel()


def generate_data(
    max_digits: int, split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates training/eval dataset given the maximum number of digits.

    simply makes all possible combinations, shuffle them, split them, return
    """

    rng = np.random.default_rng(seed)
    size = 10 ** (2 * max_digits)

    data = rng.permutation(np.arange(size))
    data, mask = encode_batch(data, max_digits)
    eval_data, train_data = data[: int(size * split)], data[int(size * split) :]
    eval_mask, train_mask = mask[: int(size * split)], mask[int(size * split) :]

    return eval_data, eval_mask, train_data, train_mask


if __name__ == "__main__":
    eval_data, eval_mask, train_data, train_mask = generate_data(max_digits=3, split=0.2, seed=42)

    print(eval_data[:5])
    print(eval_mask[:5])
    print(decode_batch(eval_data[:5]))
