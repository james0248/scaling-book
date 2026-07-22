import json

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from tiny.data import generate_data


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
    data, mask = generate_data(max_digits=cfg.data.max_digits, seed=cfg.seed)
    data, mask = jax.device_put(data), jax.device_put(mask)

    # Init model
    rng = jax.random.key(cfg.seed)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    model_cfg = cfg.model.config
    inp = jax.random.randint(inp_rng, (2, 2), 0, model_cfg.vocab_size)
    model = instantiate(model_cfg)
    params = model.init(init_rng, inp)

    optimizer = instantiate(cfg.optimizer)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training loop - log every step
    train_history = []
    for i in range(cfg.total_steps):
        batch = jax.lax.dynamic_slice_in_dim(data, i * cfg.batch_size, cfg.batch_size, axis=0)
        state, loss, _ = train_step(state, batch, mask)
        train_history.append({"loss": float(loss), "tokens": cfg.batch_size * (i + 1)})

    # Save full log
    with open(HydraConfig.get().runtime.output_dir + "/logs.jsonl", "w", encoding="utf-8") as f:
        for log in train_history:
            f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    main()
