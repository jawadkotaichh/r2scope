import numpy as np
import torch as th
import torch.nn as nn


class ICESScaffold(nn.Module):
    def __init__(self, args):
        super(ICESScaffold, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.latent_dim = args.ices_latent_dim
        self.action_embed_dim = args.ices_action_embed_dim
        self.joint_action_dim = self.n_agents * self.action_embed_dim
        self.logvar_min = -10.0
        self.logvar_max = 10.0

        hidden_dim = max(args.state_latent_dim * 2, args.rnn_hidden_dim)
        encoder_input_dim = self.state_dim + self.joint_action_dim

        self.action_embed = nn.Linear(self.n_actions, self.action_embed_dim, bias=False)

        self.full_action_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim * 2),
        )
        self.missing_agent_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim * 2),
        )
        self.shared_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

        missing_mask = 1.0 - th.eye(self.n_agents, dtype=th.float32).view(1, self.n_agents, self.n_agents, 1)
        self.register_buffer("_missing_mask", missing_mask, persistent=False)

    def compute_intrinsic_scaffolds(self, batch):
        self.eval()
        with th.no_grad():
            stats = self._forward_batch(batch)
        return stats["intrinsic_scaffolds"]

    def train_scaffold(self, batch, optimiser, grad_clip):
        self.train()
        stats = self._forward_batch(batch)
        optimiser.zero_grad()
        stats["loss"].backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimiser.step()
        stats["grad_norm"] = grad_norm.detach()
        return {k: (v.detach() if isinstance(v, th.Tensor) else v) for k, v in stats.items()}

    def _forward_batch(self, batch):
        states = batch["state"][:, :-1].float()
        next_states = batch["state"][:, 1:].float()
        actions = batch["actions_onehot"][:, :-1].float()
        terminated = batch["terminated"][:, :-1].float()

        transition_mask = batch["filled"][:, :-1].float()
        transition_mask[:, 1:] = transition_mask[:, 1:] * (1 - terminated[:, :-1])

        batch_size, seq_len = states.shape[:2]
        flat_states = states.reshape(-1, self.state_dim)
        flat_next_states = next_states.reshape(-1, self.state_dim)
        flat_actions = actions.reshape(-1, self.n_agents, self.n_actions)
        embedded_actions = self.action_embed(flat_actions)
        flat_joint_actions = embedded_actions.reshape(-1, self.joint_action_dim)

        full_inputs = th.cat([flat_states, flat_joint_actions], dim=-1)
        full_mu, full_logvar = self._split_latent(self.full_action_encoder(full_inputs))
        full_latent = self._reparameterize(full_mu, full_logvar)
        full_recon = self.shared_decoder(full_latent)

        masked_actions = (embedded_actions.unsqueeze(1) * self._missing_mask).reshape(-1, self.joint_action_dim)
        repeated_states = flat_states.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.state_dim)
        repeated_next_states = flat_next_states.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.state_dim)

        missing_inputs = th.cat([repeated_states, masked_actions], dim=-1)
        missing_mu, missing_logvar = self._split_latent(self.missing_agent_encoder(missing_inputs))
        missing_latent = self._reparameterize(missing_mu, missing_logvar)
        missing_recon = self.shared_decoder(missing_latent)

        full_recon_loss = (full_recon - flat_next_states).pow(2).mean(dim=-1, keepdim=True)
        missing_recon_loss = (missing_recon - repeated_next_states).pow(2).mean(dim=-1, keepdim=True)

        full_kl = self._standard_normal_kl(full_mu, full_logvar).unsqueeze(-1)
        missing_kl = self._standard_normal_kl(missing_mu, missing_logvar).unsqueeze(-1)

        expanded_full_mu = full_mu.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.latent_dim)
        expanded_full_logvar = full_logvar.unsqueeze(1).expand(-1, self.n_agents, -1).reshape(-1, self.latent_dim)
        intrinsic_scaffolds = self._gaussian_kl(
            expanded_full_mu,
            expanded_full_logvar,
            missing_mu,
            missing_logvar,
        ).unsqueeze(-1)

        flat_mask = transition_mask.reshape(-1, 1)
        agent_mask = transition_mask.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(-1, 1)

        full_loss = ((full_recon_loss + full_kl) * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
        missing_loss = ((missing_recon_loss + missing_kl) * agent_mask).sum() / agent_mask.sum().clamp_min(1.0)
        loss = full_loss + missing_loss

        intrinsic = intrinsic_scaffolds.view(batch_size, seq_len, self.n_agents, 1)
        intrinsic_mean = (intrinsic * transition_mask.unsqueeze(2)).sum() / (
            transition_mask.unsqueeze(2).sum().clamp_min(1.0) * self.n_agents
        )
        intrinsic_centered = intrinsic - intrinsic_mean
        intrinsic_std = th.sqrt(
            ((intrinsic_centered * intrinsic_centered) * transition_mask.unsqueeze(2)).sum()
            / (transition_mask.unsqueeze(2).sum().clamp_min(1.0) * self.n_agents)
        )
        kl_mean = (
            (full_kl * flat_mask).sum() + (missing_kl * agent_mask).sum()
        ) / (flat_mask.sum() + agent_mask.sum()).clamp_min(1.0)

        return {
            "loss": loss,
            "kl_mean": kl_mean,
            "intrinsic_mean": intrinsic_mean,
            "intrinsic_std": intrinsic_std,
            "intrinsic_scaffolds": intrinsic,
        }

    def _split_latent(self, encoder_outputs):
        mu, logvar = th.chunk(encoder_outputs, 2, dim=-1)
        return mu, logvar.clamp(min=self.logvar_min, max=self.logvar_max)

    def _reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _standard_normal_kl(mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)

    @staticmethod
    def _gaussian_kl(mu_p, logvar_p, mu_q, logvar_q):
        var_p = logvar_p.exp().clamp_min(1e-6)
        var_q = logvar_q.exp().clamp_min(1e-6)
        mean_diff = mu_p - mu_q
        return 0.5 * (
            logvar_q
            - logvar_p
            + (var_p + mean_diff.pow(2)) / var_q
            - 1.0
        ).sum(dim=-1)
