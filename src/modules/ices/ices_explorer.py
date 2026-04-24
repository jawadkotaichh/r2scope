import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ICESExplorer(nn.Module):
    def __init__(self, args):
        super(ICESExplorer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.max_roles = max(3, int(getattr(args, "n_role_clusters", 3)))

        hidden_dim = max(args.rnn_hidden_dim, args.state_latent_dim)
        policy_input_dim = args.rnn_hidden_dim + hidden_dim + self.max_roles + self.n_actions

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_body = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, self.n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def build_safe_action_mask(avail_actions, role_action_mask):
        role_mask = role_action_mask.float()
        if avail_actions is None:
            avail_mask = th.ones_like(role_mask)
        else:
            avail_mask = avail_actions.float()

        combined_mask = avail_mask * role_mask
        has_combined = combined_mask.sum(dim=-1, keepdim=True) > 0
        has_avail = avail_mask.sum(dim=-1, keepdim=True) > 0

        safe_mask = th.where(has_combined, combined_mask, avail_mask)
        if not has_avail.all():
            fallback = th.zeros_like(safe_mask)
            fallback[..., 0] = 1.0
            safe_mask = th.where(has_avail, safe_mask, fallback)
        return safe_mask

    def forward(self, agent_hidden, state, selected_roles, role_action_mask, avail_actions=None):
        state_features = self.state_encoder(state.float().reshape(-1, self.state_dim))
        state_features = state_features.view(*state.shape[:-1], -1)
        state_features = state_features.unsqueeze(-2).expand(*state.shape[:-1], self.n_agents, -1)

        role_features = F.one_hot(
            selected_roles.long().clamp(min=0, max=self.max_roles - 1),
            num_classes=self.max_roles,
        ).float()

        explorer_inputs = th.cat(
            [agent_hidden.float(), state_features, role_features, role_action_mask.float()],
            dim=-1,
        )
        explorer_hidden = self.policy_body(explorer_inputs)
        logits = self.policy_head(explorer_hidden)
        values = self.value_head(explorer_hidden)

        safe_mask = self.build_safe_action_mask(avail_actions, role_action_mask)
        masked_logits = logits.masked_fill(safe_mask <= 0, -1e10)
        return masked_logits, values, safe_mask

    def evaluate_actions(self, agent_hidden, state, selected_roles, role_action_mask, avail_actions, actions):
        masked_logits, values, safe_mask = self.forward(
            agent_hidden, state, selected_roles, role_action_mask, avail_actions=avail_actions
        )
        dist = Categorical(logits=masked_logits)
        action_ids = actions.squeeze(-1).long() if actions.dim() == masked_logits.dim() else actions.long()
        log_probs = dist.log_prob(action_ids)
        entropy = dist.entropy()
        return log_probs, entropy, values, safe_mask

    def sample_actions(self, agent_hidden, state, selected_roles, role_action_mask, avail_actions):
        masked_logits, values, safe_mask = self.forward(
            agent_hidden, state, selected_roles, role_action_mask, avail_actions=avail_actions
        )
        dist = Categorical(logits=masked_logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy(), values, safe_mask
