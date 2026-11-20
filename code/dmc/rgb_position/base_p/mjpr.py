import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import make_encoder
from transition_model import make_transition_model, transformer

LOG_FREQ = 10000

save_num1=[]
save_num2=[]
save_num3=[]
save_num4=[]
global save_i
save_i=0

class CURL(nn.Module):
    def __init__(self, fusion_dim):
        super(CURL, self).__init__()
        self.W = nn.Parameter(torch.rand(fusion_dim, fusion_dim), requires_grad=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.cpc_optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )

    def compute_logits(self, s_s, q):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """

        Wz = torch.matmul(self.W, q.T)  # (z_dim,B)
        logits = torch.matmul(s_s, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def update_cpc(self, s1, s2):
        logits = self.compute_logits(s1, s2)
        labels = torch.arange(logits.shape[0]).long().cuda()

        loss = self.cross_entropy_loss(logits, labels)

        return loss


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_categorical(value, limit=300):
    value = value.float()  # Avoid any fp16 shenanigans
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], (limit * 2 + 1),
                               device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1),
                              lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1),
                              upper_weight.unsqueeze(-1))
    return distribution


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomResizedCropAug(nn.Module):
    def __init__(self, scale=(0.1, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        super().__init__()
        self.scale = scale
        self.ratio = ratio

    def _sample_crop_params(self, batch_size, height, width, device, dtype):
        area = height * width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        crop_height = torch.full(
            (batch_size,),
            height,
            device=device,
            dtype=torch.long,
        )
        crop_width = torch.full(
            (batch_size,),
            width,
            device=device,
            dtype=torch.long,
        )
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)

        for _ in range(10):
            target_area = area * torch.empty(
                batch_size,
                device=device,
                dtype=dtype,
            ).uniform_(
                self.scale[0],
                self.scale[1],
            )
            aspect_ratio = torch.exp(
                torch.empty(
                    batch_size,
                    device=device,
                    dtype=dtype,
                ).uniform_(
                    log_ratio[0],
                    log_ratio[1],
                )
            )

            candidate_width = torch.round(
                torch.sqrt(target_area * aspect_ratio)
            ).long()
            candidate_height = torch.round(
                torch.sqrt(target_area / aspect_ratio)
            ).long()
            candidate_valid = (
                (candidate_height > 0)
                & (candidate_height <= height)
                & (candidate_width > 0)
                & (candidate_width <= width)
            )
            update = candidate_valid & ~valid
            crop_height = torch.where(update, candidate_height, crop_height)
            crop_width = torch.where(update, candidate_width, crop_width)
            valid = valid | candidate_valid

            if bool(valid.all()):
                break

        in_ratio = float(width) / float(height)
        if in_ratio < self.ratio[0]:
            fallback_width = width
            fallback_height = int(round(fallback_width / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            fallback_height = height
            fallback_width = int(round(fallback_height * self.ratio[1]))
        else:
            fallback_width = width
            fallback_height = height

        crop_height = torch.where(
            valid,
            crop_height,
            torch.full_like(crop_height, fallback_height),
        )
        crop_width = torch.where(
            valid,
            crop_width,
            torch.full_like(crop_width, fallback_width),
        )

        max_top = height - crop_height + 1
        max_left = width - crop_width + 1
        top = torch.floor(
            torch.rand(batch_size, device=device, dtype=dtype)
            * max_top.to(dtype)
        ).long()
        left = torch.floor(
            torch.rand(batch_size, device=device, dtype=dtype)
            * max_left.to(dtype)
        ).long()

        return top, left, crop_height, crop_width

    def forward(self, x):
        assert x.dim() == 4
        batch_size, channels, height, width = x.shape
        top, left, crop_height, crop_width = self._sample_crop_params(
            batch_size,
            height,
            width,
            x.device,
            x.dtype,
        )

        yy = torch.arange(
            height,
            device=x.device,
            dtype=x.dtype,
        ).view(1, height, 1)
        xx = torch.arange(
            width,
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, width)

        top = top.to(x.dtype).view(batch_size, 1, 1)
        left = left.to(x.dtype).view(batch_size, 1, 1)
        crop_height = crop_height.to(x.dtype).view(batch_size, 1, 1)
        crop_width = crop_width.to(x.dtype).view(batch_size, 1, 1)

        sample_y = top + (yy + 0.5) * crop_height / float(height) - 0.5
        sample_x = left + (xx + 0.5) * crop_width / float(width) - 0.5
        sample_y = sample_y.expand(batch_size, height, width)
        sample_x = sample_x.expand(batch_size, height, width)

        grid_y = (sample_y + 0.5) * (2.0 / float(height)) - 1.0
        grid_x = (sample_x + 0.5) * (2.0 / float(width)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        cropped = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        assert cropped.shape == (batch_size, channels, height, width)
        return cropped


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, log_std_min, log_std_max, num_layers,
                 num_filters):
        super().__init__()
        assert len(obs_shape) == 2
        image_obs_shape = obs_shape[0]
        position_obs_shape = obs_shape[1]
        assert len(image_obs_shape) == 3
        assert len(position_obs_shape) in (1, 2)

        self.image_encoder = make_encoder(encoder_type,
                                          image_obs_shape,
                                          encoder_feature_dim,
                                          num_layers,
                                          num_filters,
                                          output_logits=True)
        self.position_encoder = make_encoder('position',
                                          position_obs_shape,
                                          encoder_feature_dim,
                                          num_layers,
                                          num_filters,
                                          output_logits=True)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        fusion_dim = self.image_encoder.feature_dim + self.position_encoder.feature_dim
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0]),
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):
        out1 = self.image_encoder(obs[0], detach=detach_encoder)
        out2 = self.position_encoder(obs[1], detach=detach_encoder)
        obs = torch.cat([out1, out2], dim=-1)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max -
                                            self.log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim,
                                             hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()
        assert len(obs_shape) == 2
        image_obs_shape = obs_shape[0]
        position_obs_shape = obs_shape[1]
        assert len(image_obs_shape) == 3
        assert len(position_obs_shape) in (1, 2)

        self.image_encoder = make_encoder(encoder_type,
                                          image_obs_shape,
                                          encoder_feature_dim,
                                          num_layers,
                                          num_filters,
                                          output_logits=True)
        self.position_encoder = make_encoder('position',
                                          position_obs_shape,
                                          encoder_feature_dim,
                                          num_layers,
                                          num_filters,
                                          output_logits=True)
        fusion_dim = self.image_encoder.feature_dim + self.position_encoder.feature_dim
        self.Q1 = QFunction(fusion_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(fusion_dim, action_shape[0], hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def get_state(self, obs, detach_encoder=False):

        out1 = self.image_encoder(obs[0], detach=detach_encoder)
        out2 = self.position_encoder(obs[1], detach=detach_encoder)
        obs = torch.cat([out1, out2], dim=-1)
        return obs

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.get_state(obs, detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss.

    Args:
        num: Number of losses.

    Example:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        save_num1.append(x[0].detach().cpu().numpy())
        save_num2.append(x[1].detach().cpu().numpy())
        save_num3.append(self.params[0].detach().clone().cpu().numpy())
        save_num4.append(self.params[1].detach().clone().cpu().numpy())
        global save_i
        save_i+=1
        # if(save_i%100==0):
        #     np.save('loss1.npy',np.array(save_num1))
        #     np.save('loss2.npy',np.array(save_num2))
        #     np.save('a1.npy',np.array(save_num3))
        #     np.save('a2.npy',np.array(save_num4))
        
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class ActionPrefixEncoder(nn.Module):
    """Causally encode all actions between an anchor and each SPR target."""

    def __init__(
            self,
            action_dim,
            feature_dim,
            max_prefix_length,
            hidden_dim=512,
            num_layers=2):
        super().__init__()
        if max_prefix_length < 1:
            raise ValueError("max_prefix_length must be at least 1.")

        self.feature_dim = int(feature_dim)
        self.max_prefix_length = int(max_prefix_length)
        self.action_embedding = nn.Linear(action_dim, feature_dim)
        self.state_embedding = nn.Linear(2 * feature_dim, feature_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_prefix_length + 1, feature_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        num_heads = 2 if feature_dim % 2 == 0 else 1
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(sequence_length, device):
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def forward(self, fused_anchor_tokens, actions, prefix_lengths):
        assert fused_anchor_tokens.dim() == 3
        assert fused_anchor_tokens.shape[1] == 2
        assert fused_anchor_tokens.shape[2] == self.feature_dim
        assert actions.dim() == 3
        assert actions.shape[0] == fused_anchor_tokens.shape[0]

        prefix_lengths = torch.as_tensor(
            prefix_lengths,
            dtype=torch.long,
            device=actions.device,
        )
        assert prefix_lengths.dim() == 1
        assert prefix_lengths.numel() > 0
        assert int(prefix_lengths.min().item()) >= 1
        assert int(prefix_lengths.max().item()) <= actions.shape[1]
        assert actions.shape[1] <= self.max_prefix_length

        state_token = self.state_embedding(
            fused_anchor_tokens.flatten(start_dim=1)
        ).unsqueeze(1)
        action_tokens = self.action_embedding(actions)
        tokens = torch.cat([state_token, action_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, :tokens.shape[1], :]
        causal_mask = self._causal_mask(tokens.shape[1], tokens.device)
        encoded = self.encoder(tokens, mask=causal_mask)

        # Token index p contains the state token followed by actions [0, p).
        prefix_embeddings = encoded.index_select(1, prefix_lengths)
        assert prefix_embeddings.shape == (
            actions.shape[0],
            prefix_lengths.numel(),
            self.feature_dim,
        )
        return prefix_embeddings


class SPR(nn.Module):
    """Some Information about SPR"""

    def __init__(self,
                 action_shape,
                 critic,

                 transition_model_type='probabilistic',
                 transition_model_layer_width=512,
                 encoder_feature_dim=50,
                 spr_horizon=5,
                 spr_action_skip=1,
                 latent_dim=512,
                 time_offset=0,
                 momentum_tau=1.0,
                 aug_prob=1.0):
        super(SPR, self).__init__()
        self.spr_horizon = int(spr_horizon)
        self.spr_action_skip = int(spr_action_skip)
        if self.spr_horizon < 1:
            raise ValueError("spr_horizon must be at least 1.")
        if self.spr_action_skip < 1:
            raise ValueError("spr_action_skip must be at least 1.")
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.aug_prob = aug_prob
        self.encoder_feature_dim = encoder_feature_dim
        transition_input_dim = 3 * encoder_feature_dim
        self.transition_input_dim = transition_input_dim
        self.image_transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width, input_dim=transition_input_dim)
        self.position_transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape,
            transition_model_layer_width, input_dim=transition_input_dim)
        self.fusion_transition_model = transformer()
        self.action_prefix_encoder = ActionPrefixEncoder(
            action_dim=action_shape[0],
            feature_dim=encoder_feature_dim,
            max_prefix_length=self.spr_horizon * self.spr_action_skip,
            hidden_dim=transition_model_layer_width,
        )
        self.image_encoder = critic.image_encoder
        self.position_encoder = critic.position_encoder
        self.target_image_encoder = copy.deepcopy(critic.image_encoder)
        self.target_position_encoder = copy.deepcopy(critic.position_encoder)
        self.global_classifier_image = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier_image = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier_image = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_classifier_position = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier_position = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier_position = nn.Sequential(
            nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, encoder_feature_dim))

        self.action_net = nn.Linear(action_shape[0], 50)

        self.transforms = RandomShiftsAug(4)
        self.eval_transforms = RandomShiftsAug(4)
        self.position_mask_ratio = 0.1
        self.uses_augmentation = True

    def image_spr_loss(self, latents, target_latents, observation):
        global_latents = self.global_classifier_image(latents)  # proj
        global_latents = self.global_final_classifier_image(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier_image(target_latents)
        targets = global_targets.view(-1, observation.shape[1], self.spr_horizon + 1,
                                      global_targets.shape[-1]).transpose(
            1, 2)
        latents = global_latents.view(-1, observation.shape[1], self.spr_horizon + 1,
                                      global_latents.shape[-1]).transpose(
            1, 2)
        loss = self.spr_loss(latents, targets)
        return loss

    def position_spr_loss(self, latents, target_latents, observation):
        global_latents = self.global_classifier_position(latents)  # proj
        global_latents = self.global_final_classifier_position(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier_position(target_latents)
        targets = global_targets.view(-1, observation.shape[1], self.spr_horizon + 1,
                                      global_targets.shape[-1]).transpose(
            1, 2)
        latents = global_latents.view(-1, observation.shape[1], self.spr_horizon + 1,
                                      global_latents.shape[-1]).transpose(
            1, 2)
        loss = self.spr_loss(latents, targets)
        return loss

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def apply_transforms(self, image):
        image = self.transforms(image)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):

        flat_images = images.reshape(-1, *images.shape[-3:])

        if augment:
            processed_images = self.apply_transforms(flat_images)
        else:
            processed_images = flat_images
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    @torch.no_grad()
    def augment_position(self, position, mask_ratio=0.1):
        assert position.dim() in (2, 3)
        if mask_ratio > 0.0:
            keep_prob = 1.0 - mask_ratio
            keep_mask = (torch.rand_like(position) < keep_prob).float()
            return position * keep_mask
        return position

    def do_spr_loss(
            self,
            pred_latents_image,
            pred_latents_position,
            observation_image,
            observation_position,
            target_indices):

        pred_latents_image = torch.stack(pred_latents_image, 1)
        pred_latents_position = torch.stack(pred_latents_position, 1)
        image_latents = pred_latents_image[:observation_image.shape[1]].flatten(
            0, 1)  # batch*jumps, *
        position_latents = pred_latents_position[:observation_position.shape[1]].flatten(
            0, 1)  # batch*jumps, *
        neg_image_latents = pred_latents_image[observation_image.shape[1]:].flatten(0, 1)
        neg_position_latents = pred_latents_position[observation_position.shape[1]:].flatten(0, 1)
        image_latents = torch.cat([image_latents, neg_image_latents], 0)
        position_latents = torch.cat([position_latents, neg_position_latents], 0)

        target_indices = torch.as_tensor(
            target_indices,
            dtype=torch.long,
            device=observation_image.device,
        )
        assert target_indices.shape == (self.spr_horizon + 1,)
        assert int(target_indices.max().item()) < observation_image.shape[0]
        assert int(target_indices.max().item()) < observation_position.shape[0]
        observation_image = observation_image.index_select(0, target_indices)
        observation_position = observation_position.index_select(0, target_indices)
        target_images = observation_image.transpose(0, 1).flatten(2, 3)
        target_position = observation_position.transpose(0, 1)

        target_images = self.transform(target_images, True)  # Augment
        target_position_shape = target_position.shape
        target_position = self.augment_position(
            target_position.flatten(0, 1),
            mask_ratio=self.position_mask_ratio,
        ).reshape(target_position_shape)

        with torch.no_grad():
            target_image_latents = self.target_image_encoder(target_images.flatten(0, 1))
            target_position_latents = self.target_position_encoder(target_position.flatten(0, 1))

        global_loss_image = self.image_spr_loss(image_latents, target_image_latents,
                                                observation_image)
        global_loss_position = self.position_spr_loss(position_latents, target_position_latents,
                                                observation_position)
        # split to batch, jumps
        spr_loss_image = global_loss_image.view(-1, observation_image.shape[1])
        spr_loss_position = global_loss_position.view(-1, observation_position.shape[1])
        return spr_loss_image, spr_loss_position

class mjpr(object):
    """SPR representation learning with SAC."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            # from spr

            transition_model_type='probabilistic',
            transition_model_layer_width=512,
            spr_horizon=5,
            spr_action_skip=1,
            latent_dim=512,
            time_offset=0,
            momentum_tau=1.0,
            aug_prob=1.0,
            t0_spr_loss_weight=0.0,
            model_spr_weight=1.0,
            auxiliary_task_lr=1e-3,
            # from curl
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            curl_latent_dim=128):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        assert len(obs_shape) == 2
        image_obs_shape = obs_shape[0]
        position_obs_shape = obs_shape[1]
        assert len(image_obs_shape) == 3
        assert len(position_obs_shape) in (1, 2)
        self.image_size = image_obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.spr_horizon = int(spr_horizon)
        self.spr_action_skip = int(spr_action_skip)
        if self.spr_horizon < 1:
            raise ValueError("spr_horizon must be at least 1.")
        if self.spr_action_skip < 1:
            raise ValueError("spr_action_skip must be at least 1.")
        self.model_spr_weight = model_spr_weight
        self.t0_spr_loss_weight = t0_spr_loss_weight

        self.actor = Actor(obs_shape, action_shape, hidden_dim, encoder_type,
                           encoder_feature_dim, actor_log_std_min,
                           actor_log_std_max, num_layers,
                           num_filters).to(device)

        self.critic = Critic(obs_shape, action_shape, hidden_dim, encoder_type,
                             encoder_feature_dim, num_layers,
                             num_filters).to(device)

        self.critic_target = Critic(obs_shape, action_shape, hidden_dim,
                                    encoder_type, encoder_feature_dim,
                                    num_layers, num_filters).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.image_encoder.copy_conv_weights_from(self.critic.image_encoder)
        self.actor.position_encoder.load_state_dict(
            self.critic.position_encoder.state_dict()
        )
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        self.curl = CURL(50).to(device)
        self.awl=AutomaticWeightedLoss(num=2)
        self.awl_optimizer= torch.optim.Adam(self.awl.parameters(),
                                                lr=critic_lr,
                                                betas=(critic_beta, 0.999))
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=(actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=(critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=(alpha_beta, 0.999))

        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.SPR = SPR(action_shape,
                           self.critic,

                           transition_model_type,
                           transition_model_layer_width,
                           encoder_feature_dim,
                           self.spr_horizon,
                           self.spr_action_skip,
                           latent_dim,
                           time_offset,
                           momentum_tau,
                           aug_prob,
                           ).to(self.device)
            # optimizer for critic encoder and spr loss
            self.image_encoder_optimizer = torch.optim.Adam(
                self.critic.image_encoder.parameters(), lr=2 * encoder_lr)
            self.position_encoder_optimizer = torch.optim.Adam(
                self.critic.position_encoder.parameters(), lr=2 * encoder_lr)
            self.image_curl = torch.optim.Adam(
                self.critic.image_encoder.parameters(), lr= encoder_lr)
            self.position_curl = torch.optim.Adam(
                self.critic.position_encoder.parameters(), lr= encoder_lr)
            online_encoder_param_ids = {
                id(parameter)
                for parameter in (
                    list(self.SPR.image_encoder.parameters())
                    + list(self.SPR.position_encoder.parameters())
                )
            }
            spr_only_params = [
                parameter
                for parameter in self.SPR.parameters()
                if parameter.requires_grad
                and id(parameter) not in online_encoder_param_ids
            ]
            self.spr_optimizer = torch.optim.Adam(
                spr_only_params,
                lr=auxiliary_task_lr,
            )
            self.awl_optimizer = torch.optim.Adam( self.awl.parameters() ,
                                                  lr=auxiliary_task_lr,weight_decay=0)
          

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.SPR.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs[0] = torch.FloatTensor(obs[0]).to(self.device)
            obs[1] = torch.FloatTensor(obs[1]).to(self.device)
            obs[0] = obs[0].unsqueeze(0)
            obs[1] = obs[1].unsqueeze(0)
            mu, _, _, _ = self.actor(obs,
                                     compute_pi=False,
                                     compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):

        with torch.no_grad():
            image = torch.FloatTensor(obs[0]).to(self.device)
            position = torch.FloatTensor(obs[1]).to(self.device)
            image = image.unsqueeze(0)
            position = position.unsqueeze(0)
            mu, pi, _, _ = self.actor([image, position], compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        # stem forward

    def update_cpc(self,spr_kwargs):
        observation_image = spr_kwargs["image"]
        observation_position = spr_kwargs["position"]
        input_image = self.SPR.transform(observation_image[0].flatten(1, 2), augment=False)
        input_position = observation_position[0]
        # stem forward

        image_latent = self.SPR.image_encoder(input_image)  # Fold if T dimension.
        position_latent = self.SPR.position_encoder(input_position)
        cpc_loss = self.curl.update_cpc(image_latent, position_latent)
        self.image_curl.zero_grad()
        self.position_curl.zero_grad()
        self.curl.cpc_optimizer.zero_grad()
        cpc_loss.backward()
        self.curl.cpc_optimizer.step()
        self.position_curl.step()
        self.image_curl.step()
    def update_spr(self, spr_kwargs, L, step):
        # image: [T, B, frame_stack, C, H, W]
        # position: [T, B, frame_stack, position_dim]
        observation_image = spr_kwargs["image"]
        observation_position = spr_kwargs["position"]
        action = spr_kwargs["action"]  # [T, B, action_dim]

        pred_latents_position = []
        pred_latents_image = []

        rollout_start_idx = int(self.SPR.time_offset)
        rollout_final_idx = (
            rollout_start_idx
            + self.spr_horizon * self.spr_action_skip
        )
        assert rollout_start_idx >= 0
        assert rollout_final_idx < observation_image.shape[0]
        assert rollout_final_idx < observation_position.shape[0]
        assert rollout_final_idx - 1 < action.shape[0]

        # Keep the baseline online augmentations at the real anchor timestep.
        input_image = self.SPR.transform(
            observation_image[rollout_start_idx].flatten(1, 2).clone(),
            augment=True,
        )
        input_position = self.SPR.augment_position(
            observation_position[rollout_start_idx].clone(),
            mask_ratio=self.SPR.position_mask_ratio,
        )
        image_anchor = self.SPR.image_encoder(input_image)
        position_anchor = self.SPR.position_encoder(input_position)
        cpc_loss = self.curl.update_cpc(image_anchor, position_anchor)
        pred_latents_image.append(image_anchor)
        pred_latents_position.append(position_anchor)

        encoder_feature_dim = self.SPR.encoder_feature_dim
        assert image_anchor.shape[-1] == encoder_feature_dim
        assert position_anchor.shape[-1] == encoder_feature_dim
        anchor_tokens = torch.stack(
            [image_anchor, position_anchor],
            dim=1,
        )
        assert anchor_tokens.shape[1:] == (2, encoder_feature_dim)
        fused_anchor_tokens = self.SPR.fusion_transition_model(anchor_tokens)
        assert fused_anchor_tokens.shape == anchor_tokens.shape

        required_action_steps = self.spr_horizon * self.spr_action_skip
        action_prefix = action[
            rollout_start_idx:rollout_start_idx + required_action_steps
        ]
        assert action_prefix.shape[0] == required_action_steps
        action_prefix = action_prefix.transpose(0, 1)
        prefix_lengths = torch.arange(
            self.spr_action_skip,
            required_action_steps + 1,
            self.spr_action_skip,
            device=action.device,
            dtype=torch.long,
        )
        assert prefix_lengths.shape == (self.spr_horizon,)
        prefix_embeddings = self.SPR.action_prefix_encoder(
            fused_anchor_tokens,
            action_prefix,
            prefix_lengths,
        )
        assert prefix_embeddings.shape == (
            image_anchor.shape[0],
            self.spr_horizon,
            encoder_feature_dim,
        )

        # Predict all horizons directly from the same real online anchor.
        batch_size = image_anchor.shape[0]
        horizon_count = self.spr_horizon
        image_anchor_all = image_anchor.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        position_anchor_all = position_anchor.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        image_fusion_all = fused_anchor_tokens[:, 0, :].unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        position_fusion_all = fused_anchor_tokens[:, 1, :].unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        prefix_embeddings_all = prefix_embeddings.transpose(0, 1)

        image_transition_input = torch.cat(
            [image_anchor_all, image_fusion_all, prefix_embeddings_all],
            dim=-1,
        )
        position_transition_input = torch.cat(
            [position_anchor_all, position_fusion_all, prefix_embeddings_all],
            dim=-1,
        )
        expected_transition_shape = (
            horizon_count,
            batch_size,
            self.SPR.transition_input_dim,
        )
        assert image_transition_input.shape == expected_transition_shape
        assert position_transition_input.shape == expected_transition_shape

        predicted_image = self.SPR.image_transition_model.sample_prediction(
            image_transition_input.flatten(0, 1)
        ).view(horizon_count, batch_size, encoder_feature_dim)
        predicted_position = self.SPR.position_transition_model.sample_prediction(
            position_transition_input.flatten(0, 1)
        ).view(horizon_count, batch_size, encoder_feature_dim)
        pred_latents_image.extend(predicted_image.unbind(dim=0))
        pred_latents_position.extend(predicted_position.unbind(dim=0))

        # Preserve baseline CURL temporal weighting for every predicted horizon.
        for horizon_idx in range(horizon_count):
            cpc_loss = cpc_loss + 0.1 * self.curl.update_cpc(
                predicted_image[horizon_idx],
                predicted_position[horizon_idx],
            )

        target_indices = rollout_start_idx + torch.arange(
            self.spr_horizon + 1,
            device=observation_image.device,
            dtype=torch.long,
        ) * self.spr_action_skip
        assert target_indices[0].item() == rollout_start_idx
        assert target_indices[-1].item() == rollout_final_idx
        spr_loss_image, spr_loss_position = self.SPR.do_spr_loss(
            pred_latents_image,
            pred_latents_position,
            observation_image,
            observation_position,
            target_indices,
        )

        model_spr_loss_image = spr_loss_image[1:].mean(0)  # (jumps, bs) => (bs, )
        t0_spr_loss_image = spr_loss_image[0]
        model_spr_loss_position = spr_loss_position[1:].mean(0)  # (jumps, bs) => (bs, )
        t0_spr_loss_position = spr_loss_position[0]

        final_spr_loss_image = self.t0_spr_loss_weight * t0_spr_loss_image.mean() + \
                               self.model_spr_weight * model_spr_loss_image.mean()
        final_spr_loss_position = self.t0_spr_loss_weight * t0_spr_loss_position.mean() + \
                               self.model_spr_weight * model_spr_loss_position.mean()
        loss_sum = self.awl.forward(final_spr_loss_position, final_spr_loss_image)
        all_loss = loss_sum+0.05*cpc_loss
        self.image_encoder_optimizer.zero_grad()
        self.position_encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        self.curl.cpc_optimizer.zero_grad()
        self.awl_optimizer.zero_grad()
        all_loss.backward()
        self.awl_optimizer.step()
        self.position_encoder_optimizer.step()
        self.image_encoder_optimizer.step()
        self.curl.cpc_optimizer.step()
        self.spr_optimizer.step()

    def update(self, replay_buffer, L, step):

        obs, action, reward, next_obs, not_done, spr_kwargs = replay_buffer.sample_spr()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        #self.update_cpc(spr_kwargs)
        self.update_spr(spr_kwargs, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1,
                               self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2,
                               self.critic_tau)

            soft_update_params(self.critic.position_encoder,
                               self.critic_target.position_encoder,
                               self.encoder_tau)
            soft_update_params(self.critic.image_encoder,
                               self.critic_target.image_encoder,
                               self.encoder_tau)

            soft_update_params(self.SPR.image_encoder, self.SPR.target_image_encoder,
                               self.SPR.momentum_tau)
            soft_update_params(self.SPR.global_classifier_image,
                               self.SPR.global_target_classifier_image,
                               self.SPR.momentum_tau)

            soft_update_params(self.SPR.position_encoder, self.SPR.target_position_encoder,
                               self.SPR.momentum_tau)
            soft_update_params(self.SPR.global_classifier_position,
                               self.SPR.global_target_classifier_position,
                               self.SPR.momentum_tau)

    # def save(self, task_name, agent_name, seed ):
    #     torch.save(self.actor.state_dict(),
    #                'actor_%s.pt' % (task_name+agent_name+seed))
    #     torch.save(self.critic.state_dict(),
    #                'critic_%s.pt' % (task_name+agent_name+seed))

    # def save_spr(self, task_name, agent_name, seed ):
    #     torch.save(self.SPR.state_dict(), 'spr_%s.pt' % (task_name+agent_name+seed))


    def save(self, task_name, agent_name, seed, step, model_dir='.'):
        os.makedirs(model_dir, exist_ok=True)
        actor_name = os.path.join(model_dir, f'actor_{task_name}_{agent_name}_{seed}_step{step}.pt')
        critic_name = os.path.join(model_dir, f'critic_{task_name}_{agent_name}_{seed}_step{step}.pt')
        
        torch.save(self.actor.state_dict(), actor_name)
        torch.save(self.critic.state_dict(), critic_name)
        print(f"Actor and critic saved at step {step}")

    def save_spr(self, task_name, agent_name, seed, step, model_dir='.'):
        os.makedirs(model_dir, exist_ok=True)
        spr_name = os.path.join(model_dir, f'spr_{task_name}_{agent_name}_{seed}_step{step}.pt')
        torch.save(self.SPR.state_dict(), spr_name)
        print(f"SPR saved at step {step}")
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))
