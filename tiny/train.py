import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from tiny.data import generate_data
from tiny.model import Transformer


@jax.jit
def calculate_loss_acc(state: TrainState, params, batch: jnp.ndarray, mask: jnp.ndarray):
    batch_size = batch.shape[0]

    logits = state.apply_fn(params, batch)[:, :-1, :]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch[..., 1:])
    avg_loss = loss.sum(where=mask) / (mask.sum() * batch_size)

    acc = jnp.argmax(logits, axis=2) == batch[..., 1:]
    avg_acc = acc.sum(where=mask) / (mask.sum() * batch_size)

    return avg_loss, avg_acc


@jax.jit
def train_step(state: TrainState, batch: jnp.ndarray, mask: jnp.ndarray):
    grad_fn = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch, mask)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


def main():
    # Prepare data
    eval_data, train_data, mask = generate_data(max_digits=3, split=0.2, seed=42)

    # Init model
    rng = jax.random.key(42)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    inp = jax.random.randint(inp_rng, (2, 12), 0, 12)
    model = Transformer(n_layers=2, d_model=32, d_ffw=96, n_heads=4, n_kv=1, vocab_size=12)
    params = model.init(init_rng, inp)

    optimizer = optax.adam(learning_rate=3e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training loop
    batch_size = 128
    for i in range(0, len(train_data), batch_size):
        batch = jax.device_put(train_data[i : i + batch_size])
        state, loss, acc = train_step(state, batch, mask)
        print(f"[train step {i // batch_size + 1}]   loss: {loss:4f} | acc: {acc:4f}")

        if i // batch_size % 10 == 9:
            eval_batch = jax.device_put(eval_data)
            loss, acc = calculate_loss_acc(state, state.params, eval_batch, mask)
            print(f"  [eval step {i // batch_size + 1}]   loss: {loss:4f} | acc: {acc:4f}")


if __name__ == "__main__":
    main()
