import copy
import math
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import make_encoder
from transition_model import make_transition_model, transformer
LOG_FREQ = 10000

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def log_optimizer_grad_stats(L, tag, optimizer, step):
    total_sq_norm = 0.0
    has_grad = False
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is None:
                continue
            has_grad = True
            grad_norm = parameter.grad.detach().norm(2).item()
            total_sq_norm += grad_norm ** 2
    global_norm = math.sqrt(total_sq_norm) if has_grad else 0.0
    L.log(f'{tag}_grad_global_norm', global_norm, step)

def to_categorical(value, limit=300):
    value = value.float()
    value = value.clamp(-limit, limit)
    distribution = torch.zeros(value.shape[0], limit * 2 + 1, device=value.device)
    lower = value.floor().long() + limit
    upper = value.ceil().long() + limit
    upper_weight = value % 1
    lower_weight = 1 - upper_weight
    distribution.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
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
    return (lead_dim, T, B, shape)

class RandomResizedCropAug(nn.Module):
    """Batch RandomResizedCrop that preserves the input resolution."""

    def __init__(self, scale=(0.1, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        super().__init__()
        self.scale = scale
        self.ratio = ratio

    def _sample_crop_params(self, batch_size, height, width, device, dtype):
        area = height * width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        crop_height = torch.full((batch_size,), height, device=device, dtype=torch.long)
        crop_width = torch.full((batch_size,), width, device=device, dtype=torch.long)
        valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        for _ in range(10):
            target_area = area * torch.empty(batch_size, device=device, dtype=dtype).uniform_(self.scale[0], self.scale[1])
            aspect_ratio = torch.exp(torch.empty(batch_size, device=device, dtype=dtype).uniform_(log_ratio[0], log_ratio[1]))
            candidate_width = torch.round(torch.sqrt(target_area * aspect_ratio)).long()
            candidate_height = torch.round(torch.sqrt(target_area / aspect_ratio)).long()
            candidate_valid = (candidate_height > 0) & (candidate_height <= height) & (candidate_width > 0) & (candidate_width <= width)
            update = candidate_valid & ~valid
            crop_height = torch.where(update, candidate_height, crop_height)
            crop_width = torch.where(update, candidate_width, crop_width)
            valid = valid | candidate_valid
            if bool(valid.all()):
                break
        input_ratio = float(width) / float(height)
        if input_ratio < self.ratio[0]:
            fallback_width = width
            fallback_height = int(round(fallback_width / self.ratio[0]))
        elif input_ratio > self.ratio[1]:
            fallback_height = height
            fallback_width = int(round(fallback_height * self.ratio[1]))
        else:
            fallback_width = width
            fallback_height = height
        crop_height = torch.where(valid, crop_height, torch.full_like(crop_height, fallback_height))
        crop_width = torch.where(valid, crop_width, torch.full_like(crop_width, fallback_width))
        max_top = height - crop_height + 1
        max_left = width - crop_width + 1
        top = torch.floor(torch.rand(batch_size, device=device, dtype=dtype) * max_top.to(dtype)).long()
        left = torch.floor(torch.rand(batch_size, device=device, dtype=dtype) * max_left.to(dtype)).long()
        return (top, left, crop_height, crop_width)

    def forward(self, x):
        assert x.dim() == 4
        batch_size, channels, height, width = x.shape
        top, left, crop_height, crop_width = self._sample_crop_params(batch_size, height, width, x.device, x.dtype)
        yy = torch.arange(height, device=x.device, dtype=x.dtype).view(1, height, 1)
        xx = torch.arange(width, device=x.device, dtype=x.dtype).view(1, 1, width)
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
        cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        assert cropped.shape == (batch_size, channels, height, width)
        return cropped

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
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

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
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-06).sum(-1, keepdim=True)
    return (mu, pi, log_pi)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Actor(nn.Module):
    """MLP actor network."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters):
        super().__init__()
        self.image_encoder = make_encoder(encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=True)
        self.pos_encoder = make_encoder('pos', 5, encoder_feature_dim, num_layers=2, num_filters=128, output_logits=True)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        fusion_dim = 2 * encoder_feature_dim
        self.trunk = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * action_shape[0]))
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        out1 = self.image_encoder(obs[0], detach=detach_encoder)
        out3 = self.pos_encoder(obs[1], detach=detach_encoder)
        fused_obs = torch.cat([out1, out3], dim=-1)
        mu, log_std = self.trunk(fused_obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
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
        return (mu, pi, log_pi, log_std)

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
        self.trunk = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)

class AutomaticWeightedLoss(nn.Module):
    """Learnable uncertainty weighting for multiple task losses."""

    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses):
        if len(losses) != self.params.numel():
            raise ValueError(f'Expected {self.params.numel()} losses, got {len(losses)}.')
        loss_sum = 0.0
        for index, loss in enumerate(losses):
            weight = 0.5 / self.params[index].pow(2)
            regularizer = torch.log1p(self.params[index].pow(2))
            loss_sum = loss_sum + weight * loss + regularizer
        return loss_sum

class CoMMMaskedProjectionInfoNCELoss(nn.Module):
    """Two-modality adaptation of the reference masked CoMM objective."""

    def __init__(self, representation_dim=50, hidden_dim=512, projection_dim=256, temperature=0.1):
        super().__init__()
        assert temperature > 0
        self.representation_dim = representation_dim
        self.num_modalities = 2
        self.fusion_representation_dim = self.num_modalities * representation_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.fused_pair_weight = 1.0
        self.unimodal_pair_weight = 0.5
        self.num_unimodal_pairs = 2 * self.num_modalities
        self.num_pair_relations = 1 + self.num_unimodal_pairs
        self.loss_normalizer = self.fused_pair_weight + self.unimodal_pair_weight * self.num_unimodal_pairs
        self.image_mask_token = nn.Parameter(torch.zeros(1, 1, representation_dim))
        self.pos_mask_token = nn.Parameter(torch.zeros(1, 1, representation_dim))
        self.image_slot_embedding = nn.Parameter(torch.zeros(1, 1, representation_dim))
        self.pos_slot_embedding = nn.Parameter(torch.zeros(1, 1, representation_dim))
        self.projector = nn.Sequential(nn.Linear(self.fusion_representation_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, projection_dim, bias=False))
        for parameter in (self.image_mask_token, self.pos_mask_token, self.image_slot_embedding, self.pos_slot_embedding):
            nn.init.trunc_normal_(parameter, std=0.02)
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_sequence(self, tokens, fusion_module):
        assert tokens.shape[1:] == (self.num_modalities, self.representation_dim)
        slot_embeddings = torch.cat([self.image_slot_embedding, self.pos_slot_embedding], dim=1)
        output = fusion_module(tokens + slot_embeddings)
        assert output.shape == tokens.shape
        return output.flatten(start_dim=1)

    def info_nce(self, query, key):
        logits = torch.matmul(query, key.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean() * 100.0
        return (loss, accuracy)

    def forward(self, image_raw, pos_raw, image_view1, pos_view1, image_view2, pos_view2, fusion_module):
        inputs = [image_raw, pos_raw, image_view1, pos_view1, image_view2, pos_view2]
        for tensor in inputs:
            assert tensor.shape == image_raw.shape
        batch_size = image_raw.shape[0]
        assert batch_size > 1
        masks = [self.image_mask_token.expand(batch_size, -1, -1), self.pos_mask_token.expand(batch_size, -1, -1)]
        raw_modalities = [image_raw, pos_raw]
        unimodal_sequences = []
        for modality_index, latent in enumerate(raw_modalities):
            tokens = [mask for mask in masks]
            tokens[modality_index] = latent.unsqueeze(1)
            unimodal_sequences.append(torch.cat(tokens, dim=1))
        full_view1 = torch.stack([image_view1, pos_view1], dim=1)
        full_view2 = torch.stack([image_view2, pos_view2], dim=1)
        all_sequences = torch.cat(unimodal_sequences + [full_view1, full_view2], dim=0)
        all_representations = self.encode_sequence(all_sequences, fusion_module)
        all_projected = F.normalize(self.projector(all_representations).float(), p=2.0, dim=-1, eps=1e-08)
        z_image, z_pos, z_f1, z_f2 = torch.split(all_projected, batch_size, dim=0)
        loss_f1_f2, acc_f1_f2 = self.info_nce(z_f1, z_f2)
        unimodal_losses = []
        unimodal_accuracies = []
        for latent in (z_image, z_pos):
            loss_view1, acc_view1 = self.info_nce(latent, z_f1)
            loss_view2, acc_view2 = self.info_nce(latent, z_f2)
            unimodal_losses.extend([loss_view1, loss_view2])
            unimodal_accuracies.extend([acc_view1, acc_view2])
        loss = (self.fused_pair_weight * loss_f1_f2 + self.unimodal_pair_weight * sum(unimodal_losses)) / self.loss_normalizer
        accuracy = (acc_f1_f2 + sum(unimodal_accuracies)) / float(self.num_pair_relations)
        return {'loss': loss, 'f1_f2_loss': loss_f1_f2, 'image_full_loss': 0.5 * (unimodal_losses[0] + unimodal_losses[1]), 'pos_full_loss': 0.5 * (unimodal_losses[2] + unimodal_losses[3]), 'ssl_acc': accuracy}

class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(self, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__()
        self.image_encoder = make_encoder(encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=True)
        self.pos_encoder = make_encoder('pos', 5, encoder_feature_dim, num_layers=2, num_filters=128, output_logits=True)
        fusion_dim = 2 * encoder_feature_dim
        self.Q1 = QFunction(fusion_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(fusion_dim, action_shape[0], hidden_dim)
        self.outputs = dict()
        self.apply(weight_init)

    def get_state(self, obs, detach_encoder=False):
        out1 = self.image_encoder(obs[0], detach=detach_encoder)
        out3 = self.pos_encoder(obs[1], detach=detach_encoder)
        return torch.cat([out1, out3], dim=-1)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.get_state(obs, detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return (q1, q2)

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
        noise = 1.0 + self.scale * r.clamp(-2.0, 2.0)
        return x * noise

class ActionPrefixEncoder(nn.Module):
    """Causally encode all actions between an anchor and each SPR target."""

    def __init__(self, action_dim, feature_dim, num_modalities, max_prefix_length, hidden_dim=512, num_layers=2):
        super().__init__()
        if max_prefix_length < 1:
            raise ValueError('max_prefix_length must be at least 1.')
        self.feature_dim = int(feature_dim)
        self.num_modalities = int(num_modalities)
        self.max_prefix_length = int(max_prefix_length)
        self.action_embedding = nn.Linear(action_dim, feature_dim)
        self.state_embedding = nn.Linear(self.num_modalities * feature_dim, feature_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_prefix_length + 1, feature_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        num_heads = 2 if feature_dim % 2 == 0 else 1
        layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.0, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _causal_mask(sequence_length, device):
        return torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, fused_anchor_tokens, actions, prefix_lengths):
        assert fused_anchor_tokens.shape[1:] == (self.num_modalities, self.feature_dim)
        assert actions.dim() == 3
        prefix_lengths = torch.as_tensor(prefix_lengths, dtype=torch.long, device=actions.device)
        assert int(prefix_lengths.min().item()) >= 1
        assert int(prefix_lengths.max().item()) <= actions.shape[1]
        state_token = self.state_embedding(fused_anchor_tokens.flatten(start_dim=1)).unsqueeze(1)
        action_tokens = self.action_embedding(actions)
        tokens = torch.cat([state_token, action_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, :tokens.shape[1], :]
        encoded = self.encoder(tokens, mask=self._causal_mask(tokens.shape[1], tokens.device))
        prefix_embeddings = encoded.index_select(1, prefix_lengths)
        assert prefix_embeddings.shape == (actions.shape[0], prefix_lengths.numel(), self.feature_dim)
        return prefix_embeddings

class SPR(nn.Module):
    """Two-modality SPR with direct action-prefix horizon prediction."""

    def __init__(self, action_shape, critic, transition_model_type='probabilistic', transition_model_layer_width=512, encoder_feature_dim=50, spr_horizon=1, spr_action_skip=1, latent_dim=512, time_offset=0, momentum_tau=1.0, aug_prob=1.0):
        super(SPR, self).__init__()
        self.spr_horizon = int(spr_horizon)
        self.spr_action_skip = int(spr_action_skip)
        if self.spr_horizon < 1:
            raise ValueError('spr_horizon must be at least 1.')
        if self.spr_action_skip < 1:
            raise ValueError('spr_action_skip must be at least 1.')
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.aug_prob = aug_prob
        self.encoder_feature_dim = encoder_feature_dim
        self.num_spr_modalities = 2
        self.transition_input_dim = 3 * encoder_feature_dim
        self.image_transition_model = make_transition_model(transition_model_type, encoder_feature_dim, action_shape, transition_model_layer_width, input_dim=self.transition_input_dim)
        self.pos_transition_model = make_transition_model(transition_model_type, encoder_feature_dim, action_shape, transition_model_layer_width, input_dim=self.transition_input_dim)
        self.fusion_transition_model = transformer()
        self.action_prefix_encoder = ActionPrefixEncoder(action_dim=action_shape[0], feature_dim=encoder_feature_dim, num_modalities=self.num_spr_modalities, max_prefix_length=self.spr_horizon * self.spr_action_skip, hidden_dim=transition_model_layer_width)
        self.image_encoder = critic.image_encoder
        self.pos_encoder = critic.pos_encoder
        self.target_image_encoder = copy.deepcopy(critic.image_encoder)
        self.target_pos_encoder = copy.deepcopy(critic.pos_encoder)
        self.global_classifier_image = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier_image = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier_image = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.global_classifier_pos = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.global_target_classifier_pos = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.global_final_classifier_pos = nn.Sequential(nn.Linear(encoder_feature_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, encoder_feature_dim))
        self.transforms = RandomResizedCropAug(scale=(0.1, 1.0))
        self.spr_transforms = RandomShiftsAug(4)
        self.vector_mask_ratio = 0.1
        self.uses_augmentation = True

    def image_spr_loss(self, latents, target_latents):
        global_latents = self.global_final_classifier_image(self.global_classifier_image(latents))
        with torch.no_grad():
            global_targets = self.global_target_classifier_image(target_latents)
        return self.spr_loss(global_latents, global_targets)

    def pos_spr_loss(self, latents, target_latents):
        global_latents = self.global_final_classifier_pos(self.global_classifier_pos(latents))
        with torch.no_grad():
            global_targets = self.global_target_classifier_pos(target_latents)
        return self.spr_loss(global_latents, global_targets)

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2.0, dim=-1, eps=0.001)
        f_x2 = F.normalize(f_x2s.float(), p=2.0, dim=-1, eps=0.001)
        assert f_x1s.shape == f_x2s.shape
        assert f_x1s.dim() == 3
        return F.mse_loss(f_x1, f_x2, reduction='none').sum(-1).transpose(0, 1)

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
        processed_images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images

    def apply_spr_transforms(self, image):
        return self.spr_transforms(image)

    @torch.no_grad()
    def transform_spr(self, images, augment=False):
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_spr_transforms(flat_images)
        else:
            processed_images = flat_images
        return processed_images.view(*images.shape[:-3], *processed_images.shape[1:])

    @torch.no_grad()
    def augment_vector(self, vector):
        assert vector.dim() >= 2
        if self.vector_mask_ratio > 0.0:
            keep_prob = 1.0 - self.vector_mask_ratio
            keep_mask = (torch.rand_like(vector) < keep_prob).to(vector.dtype)
            return vector * keep_mask
        return vector

    def do_spr_loss(self, pred_latents_image, pred_latents_pos, observation_image, observation_pos, target_indices):
        image_latents = torch.stack(pred_latents_image, dim=1)
        pos_latents = torch.stack(pred_latents_pos, dim=1)
        target_indices = torch.as_tensor(target_indices, dtype=torch.long, device=observation_image.device)
        expected_terms = self.spr_horizon + 1
        assert target_indices.shape == (expected_terms,)
        assert image_latents.shape[1] == expected_terms
        selected_images = observation_image.index_select(0, target_indices)
        selected_pos = observation_pos.index_select(0, target_indices)
        target_images = selected_images.transpose(0, 1).flatten(2, 3)
        target_pos = selected_pos.transpose(0, 1)
        target_images = self.transform_spr(target_images, augment=True)
        with torch.no_grad():
            batch_size = target_images.shape[0]
            target_image_latents = self.target_image_encoder(target_images.flatten(0, 1)).view(batch_size, expected_terms, -1)
            target_pos_latents = self.target_pos_encoder(target_pos.flatten(0, 1)).view(batch_size, expected_terms, -1)
        return (self.image_spr_loss(image_latents, target_image_latents), self.pos_spr_loss(pos_latents, target_pos_latents))

class mjpr(object):
    """SPR representation learning with SAC."""

    def __init__(self, obs_shape, action_shape, device, transition_model_type='probabilistic', transition_model_layer_width=512, spr_horizon=5, spr_action_skip=2, latent_dim=512, time_offset=0, momentum_tau=1.0, aug_prob=1.0, t0_spr_loss_weight=0.0, model_spr_weight=1.0, spr_loss_weight=1.0, comm_loss_weight=0.1, auxiliary_loss_weight=1.0, auxiliary_task_lr=0.001, hidden_dim=256, discount=0.99, init_temperature=0.01, alpha_lr=0.001, alpha_beta=0.9, actor_lr=0.001, actor_beta=0.9, actor_log_std_min=-10, actor_log_std_max=2, actor_update_freq=2, critic_lr=0.001, critic_beta=0.9, critic_tau=0.005, critic_target_update_freq=2, encoder_type='pixel', encoder_feature_dim=50, encoder_lr=0.001, encoder_tau=0.005, num_layers=4, num_filters=32, cpc_update_freq=1, log_interval=100, detach_encoder=False, curl_latent_dim=128):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.spr_horizon = int(spr_horizon)
        self.spr_action_skip = int(spr_action_skip)
        if self.spr_horizon < 1:
            raise ValueError('spr_horizon must be at least 1.')
        if self.spr_action_skip < 1:
            raise ValueError('spr_action_skip must be at least 1.')
        self.model_spr_weight = model_spr_weight
        self.t0_spr_loss_weight = t0_spr_loss_weight
        self.spr_loss_weight = spr_loss_weight
        self.comm_loss_weight = comm_loss_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.transition_model_type = transition_model_type
        self.transition_model_layer_width = transition_model_layer_width
        self.latent_dim = latent_dim
        self.time_offset = time_offset
        self.momentum_tau = momentum_tau
        self.aug_prob = aug_prob
        self.hidden_dim = hidden_dim
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta
        self.actor_lr = actor_lr
        self.actor_beta = actor_beta
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.critic_lr = critic_lr
        self.critic_beta = critic_beta
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_lr = encoder_lr
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.auxiliary_task_lr = auxiliary_task_lr
        self.action_shape = action_shape
        self.actor = Actor(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, actor_log_std_min, actor_log_std_max, num_layers, num_filters).to(device)
        self.critic = Critic(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, num_layers, num_filters).to(device)
        self.critic_target = Critic(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, num_layers, num_filters).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.image_encoder.copy_conv_weights_from(self.critic.image_encoder)
        self.actor.pos_encoder.copy_conv_weights_from(self.critic.pos_encoder)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))
        if self.encoder_type == 'pixel':
            self.SPR = SPR(action_shape, self.critic, transition_model_type, transition_model_layer_width, encoder_feature_dim, self.spr_horizon, self.spr_action_skip, latent_dim, time_offset, momentum_tau, aug_prob).to(self.device)
            self.image_encoder_optimizer = torch.optim.Adam(self.critic.image_encoder.parameters(), lr=encoder_lr * 2)
            self.pos_encoder_optimizer = torch.optim.Adam(self.critic.pos_encoder.parameters(), lr=encoder_lr * 2)
            online_encoder_ids = {id(parameter) for module in (self.SPR.image_encoder, self.SPR.pos_encoder) for parameter in module.parameters()}
            spr_parameters = [parameter for parameter in self.SPR.parameters() if parameter.requires_grad and id(parameter) not in online_encoder_ids]
            self.spr_optimizer = torch.optim.Adam(spr_parameters, lr=auxiliary_task_lr)
            self.awl = AutomaticWeightedLoss(num=2).to(self.device)
            self.awl_optimizer = torch.optim.Adam(self.awl.parameters(), lr=auxiliary_task_lr, weight_decay=0)
            self.comm_loss_fn = CoMMMaskedProjectionInfoNCELoss(representation_dim=encoder_feature_dim, hidden_dim=512, projection_dim=256, temperature=0.1).to(self.device)
            self.comm_fusion_model = transformer().to(self.device)
            comm_parameters = list(self.comm_fusion_model.parameters()) + list(self.comm_loss_fn.projector.parameters()) + [self.comm_loss_fn.image_mask_token, self.comm_loss_fn.pos_mask_token, self.comm_loss_fn.image_slot_embedding, self.comm_loss_fn.pos_slot_embedding]
            self.comm_optimizer = torch.optim.Adam(comm_parameters, lr=auxiliary_task_lr)
            spr_ids = {id(parameter) for group in self.spr_optimizer.param_groups for parameter in group['params']}
            comm_ids = {id(parameter) for group in self.comm_optimizer.param_groups for parameter in group['params']}
            awl_ids = {id(parameter) for group in self.awl_optimizer.param_groups for parameter in group['params']}
            assert online_encoder_ids.isdisjoint(spr_ids)
            assert online_encoder_ids.isdisjoint(comm_ids)
            assert online_encoder_ids.isdisjoint(awl_ids)
            assert spr_ids.isdisjoint(comm_ids)
            assert spr_ids.isdisjoint(awl_ids)
            assert comm_ids.isdisjoint(awl_ids)
        self.train()
        self.critic_target.train()
        self.action_shape = action_shape
        self.frame_stack = obs_shape[0]
        self.image_size = obs_shape[-1]

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.SPR.train(training)
            self.awl.train(training)
            self.comm_loss_fn.train(training)
            self.comm_fusion_model.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs[0] = torch.FloatTensor(obs[0]).to(self.device)
            obs[1] = torch.FloatTensor(obs[1]).to(self.device)
            obs[0] = obs[0].unsqueeze(0)
            obs[1] = obs[1].unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            image = torch.FloatTensor(obs[0]).to(self.device)
            pos = torch.FloatTensor(obs[1]).to(self.device)
            image = image.unsqueeze(0)
            pos = pos.unsqueeze(0)
            mu, pi, _, _ = self.actor([image, pos], compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + not_done * self.discount * target_V
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(L, 'train/grad_critic', self.critic_optimizer, step)
        self.critic_optimizer.step()
        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(L, 'train/grad_actor', self.actor_optimizer, step)
        self.actor_optimizer.step()
        self.actor.log(L, step)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_spr(self, spr_kwargs, L, step):
        observation_image = spr_kwargs['image']
        observation_pos = spr_kwargs['pos']
        action = spr_kwargs['action']
        pred_latents_image = []
        pred_latents_pos = []
        rollout_start_idx = int(self.SPR.time_offset)
        rollout_final_idx = rollout_start_idx + self.spr_horizon * self.spr_action_skip
        assert rollout_start_idx >= 0
        assert rollout_final_idx < observation_image.shape[0]
        assert rollout_final_idx < observation_pos.shape[0]
        assert rollout_final_idx - 1 < action.shape[0]
        raw_obs = [observation_image[rollout_start_idx].flatten(1, 2), observation_pos[rollout_start_idx]]
        image_view1 = self.SPR.transform(raw_obs[0].clone(), augment=True)
        image_view2 = self.SPR.transform(raw_obs[0].clone(), augment=True)
        pos_view1 = self.SPR.augment_vector(raw_obs[1].clone())
        pos_view2 = self.SPR.augment_vector(raw_obs[1].clone())
        image_raw_latent = self.SPR.image_encoder(raw_obs[0])
        pos_raw_latent = self.SPR.pos_encoder(raw_obs[1])
        image_view1_latent = self.SPR.image_encoder(image_view1)
        pos_view1_latent = self.SPR.pos_encoder(pos_view1)
        image_view2_latent = self.SPR.image_encoder(image_view2)
        pos_view2_latent = self.SPR.pos_encoder(pos_view2)
        comm_outputs = self.comm_loss_fn(image_raw_latent, pos_raw_latent, image_view1_latent, pos_view1_latent, image_view2_latent, pos_view2_latent, fusion_module=self.comm_fusion_model)
        comm_loss = comm_outputs['loss']
        spr_image = self.SPR.transform_spr(raw_obs[0].clone(), augment=True)
        image_anchor = self.SPR.image_encoder(spr_image)
        pos_anchor = pos_raw_latent
        pred_latents_image.append(image_anchor)
        pred_latents_pos.append(pos_anchor)
        encoder_feature_dim = image_anchor.shape[-1]
        assert encoder_feature_dim == self.SPR.encoder_feature_dim
        anchor_tokens = torch.stack([image_anchor, pos_anchor], dim=1)
        assert anchor_tokens.shape[1:] == (2, encoder_feature_dim)
        fused_anchor_tokens = self.SPR.fusion_transition_model(anchor_tokens)
        assert fused_anchor_tokens.shape == anchor_tokens.shape
        required_action_steps = self.spr_horizon * self.spr_action_skip
        action_prefix = action[rollout_start_idx:rollout_start_idx + required_action_steps]
        assert action_prefix.shape[0] == required_action_steps
        action_prefix = action_prefix.transpose(0, 1)
        prefix_lengths = torch.arange(self.spr_action_skip, required_action_steps + 1, self.spr_action_skip, device=action.device, dtype=torch.long)
        prefix_embeddings = self.SPR.action_prefix_encoder(fused_anchor_tokens, action_prefix, prefix_lengths)
        for horizon_idx in range(self.spr_horizon):
            prefix_embedding = prefix_embeddings[:, horizon_idx, :]
            predicted_image = self.SPR.image_transition_model.sample_prediction(torch.cat([image_anchor, fused_anchor_tokens[:, 0, :], prefix_embedding], dim=-1))
            predicted_pos = self.SPR.pos_transition_model.sample_prediction(torch.cat([pos_anchor, fused_anchor_tokens[:, 1, :], prefix_embedding], dim=-1))
            pred_latents_image.append(predicted_image)
            pred_latents_pos.append(predicted_pos)
        target_indices = rollout_start_idx + torch.arange(self.spr_horizon + 1, device=observation_image.device, dtype=torch.long) * self.spr_action_skip
        spr_loss_image, spr_loss_pos = self.SPR.do_spr_loss(pred_latents_image, pred_latents_pos, observation_image, observation_pos, target_indices)
        model_spr_loss_image = spr_loss_image[1:].mean(0)
        t0_spr_loss_image = spr_loss_image[0]
        model_spr_loss_pos = spr_loss_pos[1:].mean(0)
        t0_spr_loss_pos = spr_loss_pos[0]
        final_spr_loss_image = self.t0_spr_loss_weight * t0_spr_loss_image.mean() + self.model_spr_weight * model_spr_loss_image.mean()
        final_spr_loss_pos = self.t0_spr_loss_weight * t0_spr_loss_pos.mean() + self.model_spr_weight * model_spr_loss_pos.mean()
        loss_sum = self.awl(final_spr_loss_image, final_spr_loss_pos)
        weighted_spr_branch_loss = self.spr_loss_weight * loss_sum
        weighted_comm_loss = self.comm_loss_weight * comm_loss
        auxiliary_loss = weighted_spr_branch_loss + weighted_comm_loss
        all_loss = self.auxiliary_loss_weight * auxiliary_loss
        weighted_spr_loss = self.auxiliary_loss_weight * weighted_spr_branch_loss
        weighted_auxiliary_comm_loss = self.auxiliary_loss_weight * weighted_comm_loss
        if step % self.log_interval == 0:
            L.log('train/spr_loss', loss_sum.item(), step)
            L.log('train/spr_image_loss', final_spr_loss_image.item(), step)
            L.log('train/spr_pos_loss', final_spr_loss_pos.item(), step)
            awl_weights = 0.5 / self.awl.params.detach().pow(2)
            for modality_index, modality_name in enumerate(('image', 'pos')):
                L.log(f'train/awl_{modality_name}_param', self.awl.params[modality_index].item(), step)
                L.log(f'train/awl_{modality_name}_weight', awl_weights[modality_index].item(), step)
            L.log('train/spr_loss_weighted', weighted_spr_branch_loss.item(), step)
            L.log('train/spr_loss_weighted_by_auxiliary', weighted_spr_loss.item(), step)
            L.log('train/comm_loss', comm_loss.item(), step)
            L.log('train/comm_loss_weighted', weighted_comm_loss.item(), step)
            L.log('train/comm_loss_weighted_by_auxiliary', weighted_auxiliary_comm_loss.item(), step)
            L.log('train/comm_fused_loss', comm_outputs['f1_f2_loss'].item(), step)
            L.log('train/comm_image_full_loss', comm_outputs['image_full_loss'].item(), step)
            L.log('train/comm_pos_full_loss', comm_outputs['pos_full_loss'].item(), step)
            L.log('train/comm_ssl_acc', comm_outputs['ssl_acc'].item(), step)
            L.log('train/all_auxiliary_loss', all_loss.item(), step)
            L.log('train/spr_horizon', float(self.spr_horizon), step)
            L.log('train/spr_action_skip', float(self.spr_action_skip), step)
        self.image_encoder_optimizer.zero_grad()
        self.pos_encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        self.awl_optimizer.zero_grad()
        self.comm_optimizer.zero_grad()
        all_loss.backward()
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(L, 'train/grad_spr', self.spr_optimizer, step)
            log_optimizer_grad_stats(L, 'train/grad_awl', self.awl_optimizer, step)
            log_optimizer_grad_stats(L, 'train/grad_comm', self.comm_optimizer, step)
        self.image_encoder_optimizer.step()
        self.pos_encoder_optimizer.step()
        self.spr_optimizer.step()
        self.awl_optimizer.step()
        self.comm_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, spr_kwargs = replay_buffer.sample_spr()
        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_spr(spr_kwargs, L, step)
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)
        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            soft_update_params(self.critic.pos_encoder, self.critic_target.pos_encoder, self.encoder_tau)
            soft_update_params(self.critic.image_encoder, self.critic_target.image_encoder, self.encoder_tau)
            soft_update_params(self.SPR.image_encoder, self.SPR.target_image_encoder, self.SPR.momentum_tau)
            soft_update_params(self.SPR.pos_encoder, self.SPR.target_pos_encoder, self.SPR.momentum_tau)
            soft_update_params(self.SPR.global_classifier_image, self.SPR.global_target_classifier_image, self.SPR.momentum_tau)
            soft_update_params(self.SPR.global_classifier_pos, self.SPR.global_target_classifier_pos, self.SPR.momentum_tau)

    def get_config(self):
        """Return agent configuration dictionary from saved instance variables."""
        return {'transition_model_type': self.transition_model_type, 'transition_model_layer_width': self.transition_model_layer_width, 'spr_horizon': self.spr_horizon, 'spr_action_skip': self.spr_action_skip, 'latent_dim': self.latent_dim, 'time_offset': self.time_offset, 'momentum_tau': self.momentum_tau, 'aug_prob': self.aug_prob, 'model_spr_weight': self.model_spr_weight, 't0_spr_loss_weight': self.t0_spr_loss_weight, 'spr_loss_weight': self.spr_loss_weight, 'comm_loss_weight': self.comm_loss_weight, 'auxiliary_loss_weight': self.auxiliary_loss_weight, 'auxiliary_task_lr': self.auxiliary_task_lr, 'hidden_dim': self.hidden_dim, 'discount': self.discount, 'init_temperature': self.init_temperature, 'alpha_lr': self.alpha_lr, 'alpha_beta': self.alpha_beta, 'actor_lr': self.actor_lr, 'actor_beta': self.actor_beta, 'actor_log_std_min': self.actor_log_std_min, 'actor_log_std_max': self.actor_log_std_max, 'actor_update_freq': self.actor_update_freq, 'critic_lr': self.critic_lr, 'critic_beta': self.critic_beta, 'critic_tau': self.critic_tau, 'critic_target_update_freq': self.critic_target_update_freq, 'encoder_type': self.encoder_type, 'encoder_feature_dim': self.encoder_feature_dim, 'encoder_lr': self.encoder_lr, 'encoder_tau': self.encoder_tau, 'num_layers': self.num_layers, 'num_filters': self.num_filters, 'detach_encoder': self.detach_encoder, 'curl_latent_dim': self.curl_latent_dim, 'image_size': self.image_size, 'action_shape': self.action_shape}

    def save(self, task_name, agent_name, seed, episode, model_dir='.'):
        os.makedirs(model_dir, exist_ok=True)
        actor_name = os.path.join(model_dir, f'actor_{task_name}{agent_name}{seed}.pt')
        critic_name = os.path.join(model_dir, f'critic_{task_name}{agent_name}{seed}.pt')
        for f in glob.glob(os.path.join(model_dir, 'actor_*.pt')):
            try:
                os.remove(f)
            except Exception:
                pass
        for f in glob.glob(os.path.join(model_dir, 'critic_*.pt')):
            try:
                os.remove(f)
            except Exception:
                pass
        config = self.get_config()
        torch.save({'state_dict': self.actor.state_dict(), 'config': config}, actor_name)
        torch.save({'state_dict': self.critic.state_dict(), 'config': config}, critic_name)

    def save_spr(self, task_name, agent_name, seed, episode, model_dir='.'):
        os.makedirs(model_dir, exist_ok=True)
        spr_name = os.path.join(model_dir, f'spr_{task_name}{agent_name}{seed}.pt')
        for f in glob.glob(os.path.join(model_dir, 'spr_*.pt')):
            try:
                os.remove(f)
            except Exception:
                pass
        config = self.get_config()
        torch.save({'spr': self.SPR.state_dict(), 'comm_loss_fn': self.comm_loss_fn.state_dict(), 'comm_fusion_model': self.comm_fusion_model.state_dict(), 'spr_optimizer': self.spr_optimizer.state_dict(), 'awl': self.awl.state_dict(), 'awl_optimizer': self.awl_optimizer.state_dict(), 'comm_optimizer': self.comm_optimizer.state_dict(), 'image_encoder_optimizer': self.image_encoder_optimizer.state_dict(), 'pos_encoder_optimizer': self.pos_encoder_optimizer.state_dict(), 'config': config}, spr_name)

    def load(self, model_dir, step):
        actor_path = '%s/actor_%s.pt' % (model_dir, step)
        critic_path = '%s/critic_%s.pt' % (model_dir, step)
        actor_checkpoint = torch.load(actor_path)
        if isinstance(actor_checkpoint, dict) and 'state_dict' in actor_checkpoint:
            self.actor.load_state_dict(actor_checkpoint['state_dict'])
        else:
            self.actor.load_state_dict(actor_checkpoint)
        critic_checkpoint = torch.load(critic_path)
        if isinstance(critic_checkpoint, dict) and 'state_dict' in critic_checkpoint:
            self.critic.load_state_dict(critic_checkpoint['state_dict'])
        else:
            self.critic.load_state_dict(critic_checkpoint)
