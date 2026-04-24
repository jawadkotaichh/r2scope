# ICES in RODE

## Overview

This repository integrates an ICES-style exploration mechanism into RODE without changing the main RODE learning objective.

- RODE still learns from extrinsic reward only.
- ICES only changes how primitive actions are sampled during training.
- Evaluation stays purely exploitative and does not use ICES exploration.

The training-time action flow is:

```text
role = RODE role selector

if random() < ices_alpha:
    action = ICES explorer(action mask = env mask AND role mask)
else:
    action = greedy RODE action inside the same role mask
```

## What Was Added

### New modules

- `src/modules/ices/ices_scaffold.py`
  Learns a latent scaffold from `(state, joint action)` and `(state, joint action without agent i)` and uses the KL divergence between those Gaussian predictions as the intrinsic signal.

- `src/modules/ices/ices_explorer.py`
  Learns a policy/value pair that proposes primitive actions inside the selected role-restricted action space.

### Controller integration

`src/controllers/rode_controller.py` now:

- keeps the original RODE path unchanged when `use_ices: False`
- switches to ICES-controlled primitive exploration when `use_ices: True`
- samples only from actions that satisfy both the environment mask and the role mask
- stores an `ices_explore` flag in the episode batch so the explorer is trained only on actions it actually sampled

### Learner integration

`src/learners/rode_learner.py` now trains three independent pieces:

- original RODE action/value learning from extrinsic reward
- original RODE action encoder / role-action-space update
- ICES scaffold and explorer with separate optimizers

The ICES losses are separate from the RODE TD loss, so ICES gradients do not flow into the RODE Q-networks.

## ICES Signals

### Scaffold

The scaffold builds two Gaussian latent predictions for each transition:

- `p_full(z | s_t, u_t)`
- `p_minus_i(z | s_t, u_t^{-i})`

It reconstructs `s_{t+1}` with a shared decoder and uses:

- reconstruction loss for the next state
- KL-to-standard-normal regularization for the latent variables
- intrinsic scaffold `KL(p_full || p_minus_i)` for each agent

### Explorer

The explorer consumes:

- detached RODE agent hidden states
- the global state
- the selected role id
- the role action mask

It is trained with:

```text
policy loss = -(log pi(a) * advantage + ices_beta * entropy)
value loss = MSE(V, intrinsic_scaffold)
```

where the advantage is:

```text
advantage = intrinsic_scaffold - V
```

## Config

`src/config/algs/rode.yaml` now contains the ICES config surface with `use_ices: False`.

`src/config/algs/rode_ices.yaml` enables ICES by default.

Relevant options:

```yaml
use_ices: true
ices_alpha_start: 0.1
ices_alpha_finish: 0.05
ices_alpha_anneal_time: 500000
ices_beta: 0.1
ices_scaffold_lr: 0.0001
ices_explore_lr: 0.001
ices_grad_clip: 0.1
ices_latent_dim: 32
ices_action_embed_dim: 4
ices_train_interval: 1
ices_disable_eval: true
```

## Running Experiments

Use the ICES-enabled algorithm config:

```bash
python src/main.py --alg rode_ices --env sc2 --map sc2_27m_vs_30m
```

Example with overrides:

```bash
python src/main.py --alg rode_ices --env sc2 --map sc2_corridor with n_role_clusters=3 role_interval=5 t_max=5050000
```

To recover the original RODE behavior exactly, use:

```bash
python src/main.py --alg rode --env sc2 --map sc2_27m_vs_30m
```

## Logging

The learner now logs:

- `ices_alpha`
- `ices_intrinsic_mean`
- `ices_intrinsic_std`
- `ices_scaffold_loss`
- `ices_kl_mean`
- `ices_explorer_loss`
- `ices_entropy`
- `ices_value_loss`
- `ices_fraction_explore_actions`

## Notes

- ICES is disabled during evaluation in the controller path.
- If you load an older checkpoint without ICES files, RODE weights still load and the ICES modules keep their current initialization.
- The replay buffer still stores the original RODE data plus one extra field: `ices_explore`.
