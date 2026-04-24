from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
from modules.ices import ICESExplorer
from modules.roles import REGISTRY as role_REGISTRY
from modules.role_selectors import REGISTRY as role_selector_REGISTRY
import torch as th

from sklearn.cluster import KMeans
import numpy as np
import copy
import os


# This multi-agent controller shares parameters between agents
class RODEMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        self.role_interval = args.role_interval

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.n_roles = 3
        self._build_roles()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.role_selector = role_selector_REGISTRY[args.role_selector](input_shape, args)
        self.action_encoder = action_encoder_REGISTRY[args.action_encoder](args)

        self.hidden_states = None
        self.role_hidden_states = None
        self.selected_roles = None
        self.n_clusters = args.n_role_clusters
        self.role_action_spaces = th.ones(self.n_roles, self.n_actions).to(args.device)

        self.role_latent = th.ones(self.n_roles, self.args.action_latent_dim).to(args.device)
        self.action_repr = th.ones(self.n_actions, self.args.action_latent_dim).to(args.device)
        self.use_ices = getattr(args, "use_ices", False)
        self.ices_explorer = ICESExplorer(args) if self.use_ices else None
        self.last_ices_explore = None
        self.last_ices_alpha = 0.0

        self._agent_id_eye = None
        self._agent_id_eye_bs = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, role_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, t_env=t_env)

        # role_action_spaces [n_roles, n_actions]; selected_roles [bs*n_agents]
        role_avail_actions = self.role_action_spaces.index_select(
            0, self.selected_roles.long().view(-1)
        ).to(th.int32).view(ep_batch.batch_size, self.n_agents, -1)

        if self.use_ices:
            chosen_actions = self._select_actions_with_ices(
                ep_batch, t_ep, t_env, bs, agent_outputs, avail_actions, role_avail_actions, test_mode
            )
        else:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs],
                                                                role_avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, self.selected_roles, role_avail_actions

    def forward(self, ep_batch, t, test_mode=False, t_env=None):
        agent_inputs = self._build_inputs(ep_batch, t)

        # select roles
        self.role_hidden_states = self.role_agent(agent_inputs, self.role_hidden_states)
        role_outputs = None
        if t % self.role_interval == 0:
            role_outputs = self.role_selector(self.role_hidden_states, self.role_latent)
            self.selected_roles = self.role_selector.select_role(role_outputs, test_mode=test_mode, t_env=t_env).squeeze()
            # [bs * n_agents]

        # compute individual q-values
        self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        roles_q = []
        for role_i in range(self.n_roles):
            role_q = self.roles[role_i](self.hidden_states, self.action_repr)  # [bs * n_agents, n_actions]
            roles_q.append(role_q)
        roles_q = th.stack(roles_q, dim=1)  # [bs*n_agents, n_roles, n_actions]
        agent_outs = th.gather(roles_q, 1, self.selected_roles.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.n_actions))
        # [bs * n_agents, 1, n_actions]

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), \
            (None if role_outputs is None else role_outputs.view(ep_batch.batch_size, self.n_agents, -1))

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.role_hidden_states = self.role_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        if self.use_ices:
            self.last_ices_explore = th.zeros(batch_size, self.n_agents, 1, dtype=th.uint8, device=self.args.device)
            self.last_ices_alpha = 0.0

    def parameters(self):
        params = list(self.agent.parameters())
        params += list(self.role_agent.parameters())
        for role_i in range(self.n_roles):
            params += list(self.roles[role_i].parameters())
        params += list(self.role_selector.parameters())

        return params

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.role_agent.load_state_dict(other_mac.role_agent.state_dict())
        if other_mac.n_roles > self.n_roles:
            self.n_roles = other_mac.n_roles
            self.roles = copy.deepcopy(other_mac.roles)
        else:
            for role_i in range(self.n_roles):
                self.roles[role_i].load_state_dict(other_mac.roles[role_i].state_dict())
        self.role_selector.load_state_dict(other_mac.role_selector.state_dict())

        self.action_encoder.load_state_dict(other_mac.action_encoder.state_dict())
        self.role_action_spaces = copy.deepcopy(other_mac.role_action_spaces)
        self.role_latent = copy.deepcopy(other_mac.role_latent)
        self.action_repr = copy.deepcopy(other_mac.action_repr)
        if self.use_ices and other_mac.ices_explorer is not None:
            self.ices_explorer.load_state_dict(other_mac.ices_explorer.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.role_agent.cuda()
        for role_i in range(self.n_roles):
            self.roles[role_i].cuda()
        self.role_selector.cuda()
        self.action_encoder.cuda()
        if self.use_ices:
            self.ices_explorer.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.role_agent.state_dict(), "{}/role_agent.th".format(path))
        for role_i in range(self.n_roles):
            th.save(self.roles[role_i].state_dict(), "{}/role_{}.th".format(path, role_i))
        th.save(self.role_selector.state_dict(), "{}/role_selector.th".format(path))

        th.save(self.action_encoder.state_dict(), "{}/action_encoder.th".format(path))
        th.save(self.role_action_spaces, "{}/role_action_spaces.pt".format(path))
        th.save(self.role_latent, "{}/role_latent.pt".format(path))
        th.save(self.action_repr, "{}/action_repr.pt".format(path))
        if self.use_ices:
            th.save(self.ices_explorer.state_dict(), "{}/ices_explorer.th".format(path))

    def load_models(self, path):
        self.role_action_spaces = th.load("{}/role_action_spaces.pt".format(path),
                                          map_location=lambda storage, loc: storage).to(self.args.device)
        self.n_roles = self.role_action_spaces.shape[0]
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.role_agent.load_state_dict(th.load("{}/role_agent.th".format(path), map_location=lambda storage, loc: storage))
        for role_i in range(self.n_roles):
            try:
                self.roles[role_i].load_state_dict(th.load("{}/role_{}.th".format(path, role_i),
                                                   map_location=lambda storage, loc: storage))
            except:
                self.roles.append(role_REGISTRY[self.args.role](self.args))
            self.roles[role_i].update_action_space(self.role_action_spaces[role_i].detach().cpu().numpy())
            if self.args.use_cuda:
                self.roles[role_i].cuda()
        self.role_selector.load_state_dict(th.load("{}/role_selector.th".format(path),
                                           map_location=lambda storage, loc: storage))

        self.action_encoder.load_state_dict(th.load("{}/action_encoder.th".format(path),
                                                    map_location=lambda storage, loc:storage))
        self.role_latent = th.load("{}/role_latent.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        self.action_repr = th.load("{}/action_repr.pt".format(path),
                                   map_location=lambda storage, loc: storage).to(self.args.device)
        if self.use_ices:
            explorer_path = "{}/ices_explorer.th".format(path)
            if os.path.exists(explorer_path):
                self.ices_explorer.load_state_dict(
                    th.load(explorer_path, map_location=lambda storage, loc: storage)
                )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.role_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_roles(self):
        self.roles = [role_REGISTRY[self.args.role](self.args) for _ in range(self.n_roles)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            if self._agent_id_eye is None or self._agent_id_eye_bs != bs or self._agent_id_eye.device != batch.device:
                self._agent_id_eye = th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1).contiguous()
                self._agent_id_eye_bs = bs
            inputs.append(self._agent_id_eye)

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def update_role_action_spaces(self):
        action_repr = self.action_encoder()
        action_repr_array = action_repr.detach().cpu().numpy()  # [n_actions, action_latent_d]

        k_means = KMeans(n_clusters=self.n_clusters, random_state=0).fit(action_repr_array)

        spaces = []
        for cluster_i in range(self.n_clusters):
            spaces.append((k_means.labels_ == cluster_i).astype(np.float64))

        o_spaces = copy.deepcopy(spaces)
        spaces = []

        for space_i ,space in enumerate(o_spaces):
            _space = copy.deepcopy(space)
            _space[0] = 0.
            _space[1] = 0.

            if _space.sum() == 2.:
                spaces.append(o_spaces[space_i])
            if _space.sum() >= 3:
                _space[:6] = 1.
                spaces.append(_space)

        for space in spaces:
            space[0] = 1.

        if len(spaces) < 3:
            spaces.append(spaces[0])
            spaces.append(spaces[1])

        print('>>> Role Action Spaces', spaces)

        n_roles = len(spaces)
        if n_roles > self.n_roles:
            for _ in range(self.n_roles, n_roles):
                self.roles.append(role_REGISTRY[self.args.role](self.args))
                if self.args.use_cuda:
                    self.roles[-1].cuda()

        self.n_roles = n_roles

        for role_i, space in enumerate(spaces):
            self.roles[role_i].update_action_space(space)

        # for _ in range(self.n_roles, 10):
        #     del self.roles[self.n_roles]

        self.role_action_spaces = th.as_tensor(np.array(spaces), dtype=th.float32, device=self.args.device)  # [n_roles, n_actions]
        self.role_latent = th.matmul(self.role_action_spaces, action_repr) / self.role_action_spaces.sum(dim=-1,
                                                                                                         keepdim=True)
        self.role_latent = self.role_latent.detach().clone()
        self.action_repr = action_repr.detach().clone()

    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])

    def ices_alpha(self, t_env):
        if not self.use_ices:
            return 0.0

        anneal_time = int(getattr(self.args, "ices_alpha_anneal_time", 0))
        if anneal_time <= 0:
            return float(self.args.ices_alpha_finish)

        progress = min(max(float(t_env or 0), 0.0) / float(anneal_time), 1.0)
        return float(self.args.ices_alpha_start + progress * (self.args.ices_alpha_finish - self.args.ices_alpha_start))

    def _select_actions_with_ices(self, ep_batch, t_ep, t_env, bs, agent_outputs, avail_actions, role_avail_actions,
                                  test_mode):
        selected_roles = self.selected_roles.view(ep_batch.batch_size, self.n_agents)[bs]
        exploit_actions = self._select_greedy_actions(agent_outputs[bs], avail_actions[bs], role_avail_actions[bs])

        self.action_selector.epsilon = 0.0
        self.last_ices_alpha = 0.0
        self.last_ices_explore = th.zeros(
            exploit_actions.shape[0], self.n_agents, 1, dtype=th.uint8, device=exploit_actions.device
        )

        if test_mode:
            return exploit_actions

        alpha = self.ices_alpha(t_env)
        self.last_ices_alpha = alpha
        explore_mask = (th.rand_like(exploit_actions.float()) < alpha).long()

        with th.no_grad():
            explorer_actions, _, _, _, _ = self.ices_explorer.sample_actions(
                self.hidden_states.detach().view(ep_batch.batch_size, self.n_agents, -1)[bs],
                ep_batch["state"][:, t_ep][bs],
                selected_roles.long(),
                role_avail_actions[bs].float(),
                avail_actions[bs].float(),
            )

        chosen_actions = explore_mask * explorer_actions + (1 - explore_mask) * exploit_actions
        self.last_ices_explore = explore_mask.unsqueeze(-1).to(th.uint8)
        return chosen_actions

    def _select_greedy_actions(self, agent_outputs, avail_actions, role_avail_actions):
        safe_mask = ICESExplorer.build_safe_action_mask(avail_actions, role_avail_actions)
        masked_q_values = agent_outputs.clone()
        masked_q_values[safe_mask <= 0] = -float("inf")
        return masked_q_values.max(dim=2)[1]
