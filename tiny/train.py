import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from hydra.utils import instantiate
from omegaconf import DictConfig

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


@hydra.main(version_base=None, config_path="config")
def main(cfg: DictConfig):
    # Prepare data
    eval_data, train_data, mask = generate_data(
        max_digits=cfg.data.max_digits, split=cfg.data.eval_split, seed=cfg.seed
    )
    train_cnt, eval_cnt = len(train_data), len(eval_data)
    eval_data, train_data, mask = (
        jax.device_put(eval_data),
        jax.device_put(train_data),
        jax.device_put(mask),
    )

    # Init model
    rng = jax.random.key(cfg.seed)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    model_cfg = cfg.model
    inp = jax.random.randint(inp_rng, (2, 2), 0, model_cfg.vocab_size)
    model = Transformer(
        n_layers=model_cfg.n_layers,
        d_model=model_cfg.d_model,
        d_ffw=model_cfg.d_ffw,
        n_heads=model_cfg.n_heads,
        n_kv=model_cfg.n_kv,
        vocab_size=model_cfg.vocab_size,
    )
    params = model.init(init_rng, inp)

    optimizer = instantiate(cfg.optimizer)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training loop
    train_history, eval_history = [], []
    for i in range(train_cnt // cfg.batch_size):
        batch = jax.lax.dynamic_slice_in_dim(train_data, i * cfg.batch_size, cfg.batch_size, axis=0)
        state, loss, acc = train_step(state, batch, mask)

        if (i + 1) % cfg.log_interval == 0:
            train_history.append({"loss": loss, "acc": acc})
            print(f"[train] step: {i} |  loss: {loss} | acc: {acc}")
        if (i + 1) % cfg.eval_interval == 0:
            total_loss, total_acc = 0, 0
            batch_cnt = eval_cnt // cfg.eval_batch_size
            for i in range(batch_cnt):
                eval_batch = jax.lax.dynamic_slice_in_dim(
                    batch, i * cfg.eval_batch_size, cfg.eval_batch_size, axis=0
                )
                loss, acc = calculate_loss_acc(state, state.params, eval_batch, mask)
                total_loss += loss
                total_acc += acc

            eval_loss, eval_acc = total_loss / batch_cnt, total_acc / batch_cnt
            eval_history.append({"loss": eval_loss, "acc": eval_acc})
            print(f"[eval] step: {i} |  loss: {eval_loss} | acc: {eval_acc}")


if __name__ == "__main__":
    main()
