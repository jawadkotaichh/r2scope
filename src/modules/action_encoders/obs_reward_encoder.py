import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch as th


class ObsRewardEncoder(nn.Module):
    def __init__(self, args):
        super(ObsRewardEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mixing_embed_dim = args.mixing_embed_dim
        self.action_latent_dim = args.action_latent_dim

        self.state_dim = int(np.prod(args.state_shape))
        self.obs_dim = int(np.prod(args.obs_shape))

        self.obs_encoder_avg = nn.Sequential(
            nn.Linear(self.obs_dim + self.n_actions * (self.n_agents - 1), args.state_latent_dim * 2),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim * 2, args.state_latent_dim))
        self.obs_decoder_avg = nn.Sequential(
            nn.Linear(args.state_latent_dim + args.action_latent_dim, args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim, self.obs_dim))

        self.action_encoder = nn.Sequential(nn.Linear(self.n_actions, args.state_latent_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(args.state_latent_dim * 2, args.action_latent_dim))

        self.reward_decoder_avg = nn.Sequential(
            nn.Linear(args.state_latent_dim + args.action_latent_dim, args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim, 1))

        # Precompute index for vectorized other_actions gather: [n_agents, n_agents-1]
        other_idx = th.tensor(
            [[j for j in range(self.n_agents) if j != i] for i in range(self.n_agents)],
            dtype=th.long,
        )
        self.register_buffer("_other_idx", other_idx, persistent=False)

    def predict(self, obs, actions):
        # used in learners (for training)
        other_actions = self.other_actions(actions)
        obs_reshaped = obs.contiguous().view(-1, self.obs_dim)
        inputs = th.cat([obs_reshaped, other_actions], dim=-1)

        # average
        obs_latent_avg = self.obs_encoder_avg(inputs)
        actions = actions.contiguous().view(-1, self.n_actions)
        action_latent_avg = self.action_encoder(actions)

        pred_avg_input = th.cat([obs_latent_avg, action_latent_avg], dim=-1)
        no_pred_avg = self.obs_decoder_avg(pred_avg_input)
        r_pred_avg = self.reward_decoder_avg(pred_avg_input)

        return no_pred_avg.view(-1, self.n_agents, self.obs_dim), r_pred_avg.view(-1, self.n_agents, 1)

    def forward(self):
        actions = th.eye(self.n_actions, dtype=th.float32, device=self._other_idx.device)
        actions_latent_avg = self.action_encoder(actions)
        return actions_latent_avg

    def other_actions(self, actions):
        # actions: [bs, n_agents, n_actions]
        assert actions.shape[1] == self.n_agents
        bs = actions.shape[0]
        # Gather with precomputed index [n_agents, n_agents-1]; order matches the
        # original python double loop (j in 0..n-1 skipping i).
        # actions[:, idx] -> [bs, n_agents, n_agents-1, n_actions]
        gathered = actions[:, self._other_idx]
        return gathered.reshape(bs * self.n_agents, (self.n_agents - 1) * self.n_actions)
