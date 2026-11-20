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


class CoMMMaskedProjectionInfoNCELoss(nn.Module):
    def __init__(
        self,
        representation_dim=50,
        hidden_dim=512,
        projection_dim=256,
        temperature=0.1,
    ):
        super().__init__()
        assert temperature > 0
        assert representation_dim == 50, (
            "transition_model.transformer currently expects 50-dimensional tokens."
        )
        self.representation_dim = representation_dim
        self.fusion_representation_dim = 2 * representation_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.fused_pair_weight = 1.0
        self.unimodal_pair_weight = 0.5
        self.num_unimodal_pairs = 4
        self.num_pair_relations = 1 + self.num_unimodal_pairs
        self.loss_normalizer = (
            self.fused_pair_weight
            + self.unimodal_pair_weight * self.num_unimodal_pairs
        )
        self.rgb_mask_token = nn.Parameter(
            torch.zeros(1, 1, representation_dim)
        )
        self.position_mask_token = nn.Parameter(
            torch.zeros(1, 1, representation_dim)
        )
        self.rgb_slot_embedding = nn.Parameter(
            torch.zeros(1, 1, representation_dim)
        )
        self.position_slot_embedding = nn.Parameter(
            torch.zeros(1, 1, representation_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(
                self.fusion_representation_dim,
                hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                projection_dim,
                bias=False,
            ),
        )
        nn.init.trunc_normal_(self.rgb_mask_token, std=0.02)
        nn.init.trunc_normal_(self.position_mask_token, std=0.02)
        nn.init.trunc_normal_(self.rgb_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.position_slot_embedding, std=0.02)
        self._init_projector()

    def _init_projector(self):
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_sequence(self, tokens, fusion_module):
        assert tokens.dim() == 3
        assert tokens.shape[1:] == (
            2,
            self.representation_dim,
        )
        slot_embeddings = torch.cat(
            [
                self.rgb_slot_embedding,
                self.position_slot_embedding,
            ],
            dim=1,
        )
        tokens = tokens + slot_embeddings
        output = fusion_module(tokens)
        assert output.shape == tokens.shape
        representation = output.reshape(
            output.shape[0],
            self.fusion_representation_dim,
        )
        assert representation.shape == (
            output.shape[0],
            self.fusion_representation_dim,
        )
        return representation, output

    def info_nce(self, query, key):
        assert query.dim() == 2
        assert key.dim() == 2
        assert query.shape == key.shape
        assert query.shape[1] == self.projection_dim
        batch_size = query.shape[0]

        logits = torch.matmul(
            query,
            key.transpose(0, 1),
        ) / self.temperature
        assert logits.shape == (batch_size, batch_size)
        labels = torch.arange(
            batch_size,
            device=logits.device,
            dtype=torch.long,
        )
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            accuracy = (
                logits.argmax(dim=1) == labels
            ).float().mean() * 100.0
        return loss, accuracy

    def forward(
        self,
        rgb_raw,
        position_raw,
        rgb_view1,
        position_view1,
        rgb_view2,
        position_view2,
        fusion_module,
    ):
        inputs = [
            rgb_raw,
            position_raw,
            rgb_view1,
            position_view1,
            rgb_view2,
            position_view2,
        ]
        for tensor in inputs:
            assert tensor.dim() == 2
            assert tensor.shape == rgb_raw.shape
        assert rgb_raw.shape[1] == self.representation_dim
        batch_size = rgb_raw.shape[0]
        assert batch_size > 1

        rgb_mask = self.rgb_mask_token.expand(batch_size, -1, -1)
        position_mask = self.position_mask_token.expand(batch_size, -1, -1)
        rgb_only_tokens = torch.cat(
            [
                rgb_raw.unsqueeze(1),
                position_mask,
            ],
            dim=1,
        )
        position_only_tokens = torch.cat(
            [
                rgb_mask,
                position_raw.unsqueeze(1),
            ],
            dim=1,
        )
        full_view1_tokens = torch.stack(
            [rgb_view1, position_view1],
            dim=1,
        )
        full_view2_tokens = torch.stack(
            [rgb_view2, position_view2],
            dim=1,
        )
        expected_token_shape = (
            batch_size,
            2,
            self.representation_dim,
        )
        assert rgb_only_tokens.shape == expected_token_shape
        assert position_only_tokens.shape == expected_token_shape
        assert full_view1_tokens.shape == expected_token_shape
        assert full_view2_tokens.shape == expected_token_shape

        all_sequences = torch.cat(
            [
                rgb_only_tokens,
                position_only_tokens,
                full_view1_tokens,
                full_view2_tokens,
            ],
            dim=0,
        )
        assert all_sequences.shape == (
            4 * batch_size,
            2,
            self.representation_dim,
        )
        all_representations, all_output_tokens = self.encode_sequence(
            all_sequences,
            fusion_module,
        )
        assert all_output_tokens.shape == all_sequences.shape
        assert all_representations.shape == (
            4 * batch_size,
            self.fusion_representation_dim,
        )
        h_r, h_p, h_f1, h_f2 = torch.split(
            all_representations,
            batch_size,
            dim=0,
        )
        assert h_r.shape == (batch_size, self.fusion_representation_dim)
        assert h_p.shape == h_r.shape
        assert h_f1.shape == h_r.shape
        assert h_f2.shape == h_r.shape

        all_projected = self.projector(all_representations)
        assert all_projected.shape == (
            4 * batch_size,
            self.projection_dim,
        )
        all_projected = F.normalize(
            all_projected.float(),
            p=2.0,
            dim=-1,
            eps=1e-8,
        )

        z_r, z_p, z_f1, z_f2 = torch.split(
            all_projected,
            batch_size,
            dim=0,
        )
        assert z_r.shape == (batch_size, self.projection_dim)
        assert z_p.shape == z_r.shape
        assert z_f1.shape == z_r.shape
        assert z_f2.shape == z_r.shape

        loss_f1_f2, acc_f1_f2 = self.info_nce(
            z_f1,
            z_f2,
        )
        loss_r_f1, acc_r_f1 = self.info_nce(
            z_r,
            z_f1,
        )
        loss_r_f2, acc_r_f2 = self.info_nce(
            z_r,
            z_f2,
        )
        loss_p_f1, acc_p_f1 = self.info_nce(
            z_p,
            z_f1,
        )
        loss_p_f2, acc_p_f2 = self.info_nce(
            z_p,
            z_f2,
        )

        comm_loss_t = (
            self.fused_pair_weight * loss_f1_f2
            + self.unimodal_pair_weight * (
                loss_r_f1
                + loss_r_f2
                + loss_p_f1
                + loss_p_f2
            )
        ) / self.loss_normalizer
        ssl_acc_t = (
            acc_f1_f2
            + acc_r_f1
            + acc_r_f2
            + acc_p_f1
            + acc_p_f2
        ) / float(self.num_pair_relations)
        rgb_full_loss = 0.5 * (loss_r_f1 + loss_r_f2)
        position_full_loss = 0.5 * (loss_p_f1 + loss_p_f2)
        rgb_full_acc = 0.5 * (acc_r_f1 + acc_r_f2)
        position_full_acc = 0.5 * (acc_p_f1 + acc_p_f2)

        with torch.no_grad():
            off_diag_mask = ~torch.eye(
                batch_size,
                device=z_f1.device,
                dtype=torch.bool,
            )
            preprojector_feature_std = all_representations.std(dim=0).mean()
            preprojector_norm_mean = all_representations.norm(
                p=2,
                dim=-1,
            ).mean()
            h_r_std = h_r.std(dim=0).mean()
            h_p_std = h_p.std(dim=0).mean()
            h_f1_std = h_f1.std(dim=0).mean()
            h_f2_std = h_f2.std(dim=0).mean()
            h_r_norm = h_r.norm(p=2, dim=-1).mean()
            h_p_norm = h_p.norm(p=2, dim=-1).mean()
            h_f1_norm = h_f1.norm(p=2, dim=-1).mean()
            h_f2_norm = h_f2.norm(p=2, dim=-1).mean()
            preprojector_fusion_view_distance = (
                h_f1 - h_f2
            ).pow(2).mean().sqrt()
            projected_feature_std = all_projected.std(dim=0).mean()
            projected_feature_norm_mean = all_projected.norm(
                p=2,
                dim=-1,
            ).mean()
            z_r_std = z_r.std(dim=0).mean()
            z_p_std = z_p.std(dim=0).mean()
            z_f1_std = z_f1.std(dim=0).mean()
            z_f2_std = z_f2.std(dim=0).mean()
            z_r_norm = z_r.norm(p=2, dim=-1).mean()
            z_p_norm = z_p.norm(p=2, dim=-1).mean()
            z_f1_norm = z_f1.norm(p=2, dim=-1).mean()
            z_f2_norm = z_f2.norm(p=2, dim=-1).mean()
            projected_fusion_view_distance = (
                z_f1 - z_f2
            ).pow(2).mean().sqrt()
            similarity = torch.matmul(
                z_f1,
                z_f2.transpose(0, 1),
            )
            positive_similarity = similarity.diag().mean()
            negative_similarity = similarity[off_diag_mask].mean()
            similarity_margin = positive_similarity - negative_similarity

        return {
            "loss": comm_loss_t,
            "f1_f2_loss": loss_f1_f2,
            "r_f1_loss": loss_r_f1,
            "r_f2_loss": loss_r_f2,
            "p_f1_loss": loss_p_f1,
            "p_f2_loss": loss_p_f2,
            "rgb_full_loss": rgb_full_loss,
            "position_full_loss": position_full_loss,
            "r_f_loss": rgb_full_loss,
            "p_f_loss": position_full_loss,
            "ssl_acc": ssl_acc_t,
            "f1_f2_acc": acc_f1_f2,
            "r_f1_acc": acc_r_f1,
            "r_f2_acc": acc_r_f2,
            "p_f1_acc": acc_p_f1,
            "p_f2_acc": acc_p_f2,
            "rgb_full_acc": rgb_full_acc,
            "position_full_acc": position_full_acc,
            "r_f_acc": rgb_full_acc,
            "p_f_acc": position_full_acc,
            "preprojector_feature_std": preprojector_feature_std,
            "preprojector_norm_mean": preprojector_norm_mean,
            "h_r_std": h_r_std,
            "h_p_std": h_p_std,
            "h_f1_std": h_f1_std,
            "h_f2_std": h_f2_std,
            "h_r_norm": h_r_norm,
            "h_p_norm": h_p_norm,
            "h_f1_norm": h_f1_norm,
            "h_f2_norm": h_f2_norm,
            "preprojector_fusion_view_distance": preprojector_fusion_view_distance,
            "projected_feature_std": projected_feature_std,
            "projected_feature_norm_mean": projected_feature_norm_mean,
            "positive_similarity_mean": positive_similarity,
            "negative_similarity_mean": negative_similarity,
            "similarity_margin": similarity_margin,
            "z_r_std": z_r_std,
            "z_p_std": z_p_std,
            "z_f1_std": z_f1_std,
            "z_f2_std": z_f2_std,
            "z_r_norm": z_r_norm,
            "z_p_norm": z_p_norm,
            "z_f1_norm": z_f1_norm,
            "z_f2_norm": z_f2_norm,
            "projected_fusion_view_distance": projected_fusion_view_distance,
            "fusion_view_distance": projected_fusion_view_distance,
            "fusion_input_token_count": rgb_raw.new_tensor(
                float(full_view1_tokens.shape[1])
            ),
            "fusion_token_dim": rgb_raw.new_tensor(
                float(full_view1_tokens.shape[-1])
            ),
            "fusion_output_token_count": rgb_raw.new_tensor(
                float(all_output_tokens.shape[1])
            ),
            "fusion_concat_output_dim": rgb_raw.new_tensor(
                float(all_representations.shape[-1])
            ),
        }


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def _parameter_ids(parameters):
    return {id(p) for p in parameters}


def load_matching_state_dict(module, state_dict):
    current = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    return skipped, missing, unexpected


def log_optimizer_grad_stats(L, tag, optimizer, step):
    total_sq_norm = 0.0
    has_grad = False
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            has_grad = True
            grad_norm = param.grad.detach().data.norm(2)
            total_sq_norm += grad_norm.item() ** 2
    global_norm = math.sqrt(total_sq_norm) if has_grad else 0.0
    L.log(f"{tag}_grad_global_norm", global_norm, step)


def log_module_grad_stats(L, tag, module, step):
    total_sq_norm = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is None:
            continue
        has_grad = True
        grad_norm = param.grad.detach().data.norm(2)
        total_sq_norm += grad_norm.item() ** 2
    global_norm = math.sqrt(total_sq_norm) if has_grad else 0.0
    L.log(f"{tag}_grad_global_norm", global_norm, step)


def log_parameter_grad_stats(L, tag, param, step):
    if param.grad is None:
        global_norm = 0.0
    else:
        global_norm = param.grad.detach().data.norm(2).item()
    L.log(f"{tag}_grad_global_norm", global_norm, step)


def get_loss_gradients(loss, parameters):
    params = [param for param in parameters if param.requires_grad]
    if not params:
        return [], []
    if not loss.requires_grad:
        return params, [None for _ in params]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    return params, grads


def get_grad_norm_from_loss(loss, parameters):
    _, grads = get_loss_gradients(loss, parameters)
    total_sq_norm = 0.0
    has_grad = False
    for grad in grads:
        if grad is None:
            continue
        has_grad = True
        total_sq_norm += grad.detach().pow(2).sum().item()
    return math.sqrt(total_sq_norm) if has_grad else 0.0


def get_grad_cosine_from_losses(loss1, loss2, parameters):
    params = [param for param in parameters if param.requires_grad]
    if not params or not loss1.requires_grad or not loss2.requires_grad:
        return 0.0

    grads1 = torch.autograd.grad(
        loss1,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grads2 = torch.autograd.grad(
        loss2,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for grad1, grad2 in zip(grads1, grads2):
        if grad1 is not None:
            grad1 = grad1.detach()
            norm1_sq += grad1.pow(2).sum().item()
        if grad2 is not None:
            grad2 = grad2.detach()
            norm2_sq += grad2.pow(2).sum().item()
        if grad1 is not None and grad2 is not None:
            dot += (grad1 * grad2).sum().item()

    denom = math.sqrt(norm1_sq) * math.sqrt(norm2_sq) + 1e-12
    return dot / denom


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


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses):
        if len(losses) != self.params.numel():
            raise ValueError(
                f"Expected {self.params.numel()} losses, got {len(losses)}."
            )
        loss_sum = losses[0].new_zeros(())
        for index, loss in enumerate(losses):
            parameter_squared = self.params[index].pow(2)
            loss_sum = loss_sum + (
                0.5 / parameter_squared * loss
                + torch.log1p(parameter_squared)
            )
        return loss_sum


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

    def _apply_crop_params(self, x, top, left, crop_height, crop_width):
        assert x.dim() == 4
        batch_size, channels, height, width = x.shape
        assert top.shape == (batch_size,)
        assert left.shape == (batch_size,)
        assert crop_height.shape == (batch_size,)
        assert crop_width.shape == (batch_size,)

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

    def forward(self, x, params=None, return_params=False):
        assert x.dim() == 4
        batch_size, channels, height, width = x.shape
        if params is None:
            params = self._sample_crop_params(
                batch_size,
                height,
                width,
                x.device,
                x.dtype,
            )
        cropped = self._apply_crop_params(x, *params)
        if return_params:
            return cropped, params
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
        image_latent = self.image_encoder(obs[0], detach=detach_encoder)
        position_latent = self.position_encoder(obs[1], detach=detach_encoder)
        fused_obs = torch.cat([image_latent, position_latent], dim=-1)

        mu, log_std = self.trunk(fused_obs).chunk(2, dim=-1)

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

        image_latent = self.image_encoder(obs[0], detach=detach_encoder)
        position_latent = self.position_encoder(obs[1], detach=detach_encoder)
        return torch.cat([image_latent, position_latent], dim=-1)

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
            transition_model_type,
            encoder_feature_dim,
            action_shape,
            transition_model_layer_width,
            input_dim=transition_input_dim,
        )
        self.position_transition_model = make_transition_model(
            transition_model_type,
            encoder_feature_dim,
            action_shape,
            transition_model_layer_width,
            input_dim=transition_input_dim,
        )
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

        self.transforms = RandomShiftsAug(4)
        self.eval_transforms = RandomShiftsAug(4)
        self.position_noise_std = 0.0
        self.position_mask_ratio = 0.1
        self.uses_augmentation = True

    def image_spr_loss(self, latents, target_latents):
        global_latents = self.global_classifier_image(latents)  # proj
        global_latents = self.global_final_classifier_image(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier_image(target_latents)
        loss = self.spr_loss(global_latents, global_targets)
        return loss

    def position_spr_loss(self, latents, target_latents):
        global_latents = self.global_classifier_position(latents)  # proj
        global_latents = self.global_final_classifier_position(global_latents)  # pred
        with torch.no_grad():
            global_targets = self.global_target_classifier_position(target_latents)
        loss = self.spr_loss(global_latents, global_targets)
        return loss

    def spr_loss(self, f_x1s, f_x2s):
        assert f_x1s.shape == f_x2s.shape
        assert f_x1s.dim() == 3  # [B, 1 + H, D]
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        # Return [1 + H, B] to preserve the existing t0/future loss split.
        return F.mse_loss(
            f_x1,
            f_x2,
            reduction="none",
        ).sum(-1).transpose(0, 1)

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
    def augment_position(self, position, noise_std=0.0, mask_ratio=0.1):
        assert position.dim() in (2, 3)
        assert noise_std >= 0.0
        assert 0.0 <= mask_ratio <= 1.0

        augmented = position
        if noise_std > 0.0:
            flat = position.detach().reshape(position.size(0), -1)
            scale = flat.std(
                dim=-1,
                keepdim=True,
                unbiased=False,
            ).clamp_min(1e-4)
            while scale.dim() < position.dim():
                scale = scale.unsqueeze(-1)
            augmented = (
                position
                + noise_std * scale * torch.randn_like(position)
            )

        if mask_ratio > 0.0:
            keep_prob = 1.0 - mask_ratio
            keep_mask = (torch.rand_like(position) < keep_prob).float()
            augmented = augmented * keep_mask
        return augmented

    def do_spr_loss(
            self,
            pred_latents_image,
            pred_latents_position,
            observation_image,
            observation_position,
            target_indices):
        image_latents = torch.stack(pred_latents_image, dim=1)
        position_latents = torch.stack(pred_latents_position, dim=1)
        target_indices = torch.as_tensor(
            target_indices,
            dtype=torch.long,
            device=observation_image.device,
        )
        expected_terms = self.spr_horizon + 1
        assert target_indices.shape == (expected_terms,)
        assert image_latents.shape[1] == expected_terms
        assert position_latents.shape[1] == expected_terms
        assert int(target_indices.max().item()) < observation_image.shape[0]
        assert int(target_indices.max().item()) < observation_position.shape[0]

        selected_images = observation_image.index_select(0, target_indices)
        selected_position = observation_position.index_select(0, target_indices)
        target_images = selected_images.transpose(0, 1).flatten(2, 3)
        target_position = selected_position.transpose(0, 1)

        target_images = self.transform(target_images, augment=True)
        target_position_shape = target_position.shape
        target_position = self.augment_position(
            target_position.flatten(0, 1),
            noise_std=self.position_noise_std,
            mask_ratio=self.position_mask_ratio,
        ).reshape(target_position_shape)

        with torch.no_grad():
            target_image_latents = self.target_image_encoder(
                target_images.flatten(0, 1)
            ).view(target_images.shape[0], expected_terms, -1)
            target_position_latents = self.target_position_encoder(
                target_position.flatten(0, 1)
            ).view(target_position.shape[0], expected_terms, -1)

        spr_loss_image = self.image_spr_loss(
            image_latents,
            target_image_latents,
        )
        spr_loss_position = self.position_spr_loss(
            position_latents,
            target_position_latents,
        )
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
            spr_loss_weight=1.0,
            auxiliary_task_lr=1e-3,
            # SAC parameters
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
            num_env_steps=None,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            comm_loss_weight=0.1,
            auxiliary_loss_weight=1,
            log_interval=100,
            detach_encoder=False,
            curl_latent_dim=128):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        # Deprecated compatibility arguments. They are intentionally not used.
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.total_steps = num_env_steps
        self._logged_static_auxiliary_config = False
        self.comm_temporal_aug_group_size = 1
        assert len(obs_shape) == 2
        image_obs_shape = obs_shape[0]
        position_obs_shape = obs_shape[1]
        assert len(image_obs_shape) == 3
        assert len(position_obs_shape) in (1, 2)
        self.image_size = image_obs_shape[-1]
        # Deprecated compatibility argument. No CURL module is created.
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
        self.spr_loss_weight = spr_loss_weight
        self.comm_loss_weight = comm_loss_weight
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Deprecated compatibility argument.

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

        # Tie actor encoders to critic encoders.
        self.actor.image_encoder.copy_conv_weights_from(self.critic.image_encoder)
        self.actor.position_encoder.load_state_dict(
            self.critic.position_encoder.state_dict()
        )
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
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
            self.SPR = SPR(
                action_shape=action_shape,
                critic=self.critic,
                transition_model_type=transition_model_type,
                transition_model_layer_width=transition_model_layer_width,
                encoder_feature_dim=encoder_feature_dim,
                spr_horizon=self.spr_horizon,
                spr_action_skip=self.spr_action_skip,
                latent_dim=latent_dim,
                time_offset=time_offset,
                momentum_tau=momentum_tau,
                aug_prob=aug_prob,
            ).to(self.device)
            self.image_encoder_optimizer = torch.optim.Adam(
                self.critic.image_encoder.parameters(), lr=2 * encoder_lr)
            self.position_encoder_optimizer = torch.optim.Adam(
                self.critic.position_encoder.parameters(), lr=2 * encoder_lr)
            online_encoder_param_ids = _parameter_ids(
                list(self.SPR.image_encoder.parameters())
                + list(self.SPR.position_encoder.parameters())
            )
            spr_only_params = [
                p for p in self.SPR.parameters()
                if p.requires_grad and id(p) not in online_encoder_param_ids
            ]
            self.spr_optimizer = torch.optim.Adam(
                spr_only_params,
                lr=auxiliary_task_lr,
            )
            self.awl = AutomaticWeightedLoss(num=2).to(self.device)
            self.awl_optimizer = torch.optim.Adam(
                self.awl.parameters(),
                lr=auxiliary_task_lr,
                weight_decay=0,
            )
            self.comm_loss_fn = CoMMMaskedProjectionInfoNCELoss(
                representation_dim=encoder_feature_dim,
                hidden_dim=512,
                projection_dim=256,
                temperature=0.1,
            ).to(self.device)
            self.comm_transforms = RandomResizedCropAug(
                scale=(0.5, 1.0)
            )
            assert not hasattr(self.comm_loss_fn, "fusion")
            self.comm_fusion_model = transformer().to(self.device)
            assert self.SPR.fusion_transition_model is not self.comm_fusion_model
            comm_params = (
                list(self.comm_fusion_model.parameters())
                + list(self.comm_loss_fn.projector.parameters())
                + [
                    self.comm_loss_fn.rgb_mask_token,
                    self.comm_loss_fn.position_mask_token,
                    self.comm_loss_fn.rgb_slot_embedding,
                    self.comm_loss_fn.position_slot_embedding,
                ]
            )
            self.comm_optimizer = torch.optim.Adam(
                comm_params,
                lr=auxiliary_task_lr,
            )
            image_encoder_ids = _parameter_ids(self.critic.image_encoder.parameters())
            position_encoder_ids = _parameter_ids(self.critic.position_encoder.parameters())
            all_trainable_spr_ids = _parameter_ids(
                p for p in self.SPR.parameters()
                if p.requires_grad
            )
            spr_ids = _parameter_ids(
                p
                for group in self.spr_optimizer.param_groups
                for p in group["params"]
            )
            comm_ids = _parameter_ids(
                p
                for group in self.comm_optimizer.param_groups
                for p in group["params"]
            )
            awl_ids = _parameter_ids(
                p
                for group in self.awl_optimizer.param_groups
                for p in group["params"]
            )
            expected_awl_ids = _parameter_ids(self.awl.parameters())
            critic_ids = _parameter_ids(
                p
                for group in self.critic_optimizer.param_groups
                for p in group["params"]
            )
            image_encoder_optimizer_ids = _parameter_ids(
                p
                for group in self.image_encoder_optimizer.param_groups
                for p in group["params"]
            )
            position_encoder_optimizer_ids = _parameter_ids(
                p
                for group in self.position_encoder_optimizer.param_groups
                for p in group["params"]
            )
            spr_fusion_ids = _parameter_ids(
                self.SPR.fusion_transition_model.parameters()
            )
            comm_fusion_ids = _parameter_ids(
                self.comm_fusion_model.parameters()
            )
            projector_ids = _parameter_ids(
                self.comm_loss_fn.projector.parameters()
            )
            mask_token_ids = _parameter_ids(
                [
                    self.comm_loss_fn.rgb_mask_token,
                    self.comm_loss_fn.position_mask_token,
                ]
            )
            slot_embedding_ids = _parameter_ids(
                [
                    self.comm_loss_fn.rgb_slot_embedding,
                    self.comm_loss_fn.position_slot_embedding,
                ]
            )
            spr_transition_ids = _parameter_ids(
                list(self.SPR.image_transition_model.parameters())
                + list(self.SPR.position_transition_model.parameters())
                + list(self.SPR.fusion_transition_model.parameters())
                + list(self.SPR.action_prefix_encoder.parameters())
            )
            action_prefix_ids = _parameter_ids(
                self.SPR.action_prefix_encoder.parameters()
            )
            assert image_encoder_ids.issubset(critic_ids), (
                "Baseline critic optimizer must contain image encoder parameters."
            )
            assert position_encoder_ids.issubset(critic_ids), (
                "Baseline critic optimizer must contain position encoder parameters."
            )
            assert image_encoder_optimizer_ids == image_encoder_ids, (
                "Image encoder optimizer must contain only image encoder parameters."
            )
            assert position_encoder_optimizer_ids == position_encoder_ids, (
                "Position encoder optimizer must contain only position encoder parameters."
            )
            assert image_encoder_ids.isdisjoint(spr_ids), (
                "SPR optimizer must not contain image encoder parameters."
            )
            assert position_encoder_ids.isdisjoint(spr_ids), (
                "SPR optimizer must not contain position encoder parameters."
            )
            expected_spr_optimizer_ids = (
                all_trainable_spr_ids
                - image_encoder_ids
                - position_encoder_ids
            )
            assert spr_ids == expected_spr_optimizer_ids, (
                "SPR optimizer must contain every trainable SPR parameter except "
                "the shared online encoders."
            )
            assert image_encoder_ids.isdisjoint(comm_ids), (
                "CoMM optimizer must not contain image encoder parameters."
            )
            assert position_encoder_ids.isdisjoint(comm_ids), (
                "CoMM optimizer must not contain position encoder parameters."
            )
            assert spr_transition_ids.issubset(spr_ids), (
                "SPR optimizer must contain transition and SPR fusion parameters."
            )
            assert spr_transition_ids.isdisjoint(comm_ids), (
                "CoMM optimizer must not contain SPR transition or fusion parameters."
            )
            assert action_prefix_ids.issubset(spr_ids), (
                "SPR optimizer must contain the action-prefix encoder."
            )
            assert action_prefix_ids.isdisjoint(comm_ids), (
                "CoMM optimizer must not contain action-prefix encoder parameters."
            )
            assert projector_ids, (
                "CoMM projection head must have trainable parameters."
            )
            assert spr_fusion_ids, (
                "SPR fusion Transformer must have trainable parameters."
            )
            assert comm_fusion_ids, (
                "CoMM fusion Transformer must have trainable parameters."
            )
            assert spr_fusion_ids.isdisjoint(comm_fusion_ids), (
                "SPR and CoMM fusion Transformers must have completely "
                "independent parameters."
            )
            assert spr_fusion_ids.issubset(spr_ids), (
                "SPR fusion Transformer must belong to SPR optimizer."
            )
            assert spr_fusion_ids.isdisjoint(comm_ids), (
                "SPR fusion Transformer must not belong to CoMM optimizer."
            )
            assert comm_fusion_ids.issubset(comm_ids), (
                "CoMM fusion Transformer must belong to CoMM optimizer."
            )
            assert comm_fusion_ids.isdisjoint(spr_ids), (
                "CoMM fusion Transformer must not belong to SPR optimizer."
            )
            assert spr_ids.isdisjoint(comm_ids), (
                "SPR and CoMM optimizers must manage disjoint parameter sets."
            )
            assert awl_ids == expected_awl_ids, (
                "AWL optimizer must contain only the SPR AWL parameters."
            )
            assert awl_ids.isdisjoint(spr_ids), (
                "AWL and SPR optimizers must manage disjoint parameter sets."
            )
            assert awl_ids.isdisjoint(comm_ids), (
                "AWL and CoMM optimizers must manage disjoint parameter sets."
            )
            assert awl_ids.isdisjoint(image_encoder_ids), (
                "AWL optimizer must not contain image encoder parameters."
            )
            assert awl_ids.isdisjoint(position_encoder_ids), (
                "AWL optimizer must not contain position encoder parameters."
            )
            assert projector_ids.issubset(comm_ids), (
                "CoMM optimizer must contain every projector parameter."
            )
            assert mask_token_ids.issubset(comm_ids), (
                "CoMM optimizer must contain missing-modality mask tokens."
            )
            assert slot_embedding_ids.issubset(comm_ids), (
                "CoMM optimizer must contain modality slot embeddings."
            )
            expected_comm_ids = (
                comm_fusion_ids
                | projector_ids
                | mask_token_ids
                | slot_embedding_ids
            )
            assert comm_ids == expected_comm_ids, (
                "CoMM optimizer must contain only CoMM fusion, projector, "
                "mask tokens, and slot embeddings."
            )
        self.train()
        self.critic_target.train()

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
        if step % self.log_interval == 0:
            L.log(
                "train/grad_rgb_encoder_from_critic",
                get_grad_norm_from_loss(
                    critic_loss,
                    self.critic.image_encoder.parameters(),
                ),
                step,
            )
            L.log(
                "train/grad_position_encoder_from_critic",
                get_grad_norm_from_loss(
                    critic_loss,
                    self.critic.position_encoder.parameters(),
                ),
                step,
            )
        critic_loss.backward()
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(L, "train/grad_critic", self.critic_optimizer, step)
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
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(L, "train/grad_actor", self.actor_optimizer, step)
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

    def update_spr(self, spr_kwargs, L, step):
        # Sequences cover time_offset + spr_horizon * spr_action_skip transitions.
        observation_image = spr_kwargs["image"]
        observation_position = spr_kwargs["position"]

        # [sequence_length, B, action_dim]
        action = spr_kwargs["action"]

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

        comm_outputs_by_timestep = []
        comm_num_timesteps = 1
        comm_aug_group_size = min(
            self.comm_temporal_aug_group_size,
            comm_num_timesteps,
        )
        assert comm_aug_group_size >= 1
        image_latent = None
        position_latent = None
        rgb_view1_crop_params = None
        rgb_view2_crop_params = None
        position_view1_aug_params = None
        position_view2_aug_params = None

        def apply_grouped_rgb_aug(raw_rgb, crop_params):
            flat_rgb = raw_rgb.reshape(-1, *raw_rgb.shape[-3:])
            with torch.no_grad():
                if crop_params is None:
                    flat_aug, crop_params = self.comm_transforms(
                        flat_rgb,
                        return_params=True,
                    )
                else:
                    flat_aug = self.comm_transforms(
                        flat_rgb,
                        params=crop_params,
                    )
            rgb_aug = flat_aug.view(
                *raw_rgb.shape[:-3],
                *flat_aug.shape[1:],
            )
            return rgb_aug, crop_params

        def sample_position_aug_params(position):
            noise = None
            if self.SPR.position_noise_std > 0.0:
                noise = torch.randn_like(position)
            keep_mask = None
            if self.SPR.position_mask_ratio > 0.0:
                keep_prob = 1.0 - self.SPR.position_mask_ratio
                keep_mask = (torch.rand_like(position) < keep_prob).float()
            return noise, keep_mask

        def apply_grouped_position_aug(position, aug_params):
            noise, keep_mask = aug_params
            position_aug = position
            if noise is not None:
                flat = position.detach().reshape(position.size(0), -1)
                scale = flat.std(
                    dim=-1,
                    keepdim=True,
                    unbiased=False,
                ).clamp_min(1e-4)
                while scale.dim() < position.dim():
                    scale = scale.unsqueeze(-1)
                position_aug = (
                    position
                    + self.SPR.position_noise_std * scale * noise
                )
            if keep_mask is not None:
                position_aug = position_aug * keep_mask
            return position_aug

        for timestep_idx in range(rollout_start_idx, rollout_start_idx + comm_num_timesteps):
            timestep_offset = timestep_idx - rollout_start_idx
            if timestep_offset % comm_aug_group_size == 0:
                rgb_view1_crop_params = None
                rgb_view2_crop_params = None
                position_view1_aug_params = None
                position_view2_aug_params = None

            raw_rgb_unflattened = observation_image[timestep_idx].clone()
            raw_position = observation_position[timestep_idx].clone()
            assert raw_rgb_unflattened.dim() == 5
            assert raw_position.dim() == 3

            raw_rgb = raw_rgb_unflattened.flatten(1, 2)
            rgb_aug_view1, rgb_view1_crop_params = apply_grouped_rgb_aug(
                raw_rgb.clone(),
                rgb_view1_crop_params,
            )
            rgb_aug_view2, rgb_view2_crop_params = apply_grouped_rgb_aug(
                raw_rgb.clone(),
                rgb_view2_crop_params,
            )
            if position_view1_aug_params is None:
                position_view1_aug_params = sample_position_aug_params(
                    raw_position,
                )
            if position_view2_aug_params is None:
                position_view2_aug_params = sample_position_aug_params(
                    raw_position,
                )
            position_aug_view1 = apply_grouped_position_aug(
                raw_position,
                position_view1_aug_params,
            )
            position_aug_view2 = apply_grouped_position_aug(
                raw_position,
                position_view2_aug_params,
            )

            assert rgb_aug_view1.shape == rgb_aug_view2.shape
            assert position_aug_view1.shape == position_aug_view2.shape
            assert rgb_aug_view1.shape == raw_rgb.shape
            assert position_aug_view1.shape == raw_position.shape

            R = self.SPR.image_encoder(raw_rgb)
            P = self.SPR.position_encoder(raw_position)
            R1 = self.SPR.image_encoder(rgb_aug_view1)
            P1 = self.SPR.position_encoder(position_aug_view1)
            R2 = self.SPR.image_encoder(rgb_aug_view2)
            P2 = self.SPR.position_encoder(position_aug_view2)

            assert R.shape == P.shape
            assert R.shape == R1.shape
            assert R.shape == P1.shape
            assert R.shape == R2.shape
            assert R.shape == P2.shape

            comm_outputs_t = self.comm_loss_fn(
                R,
                P,
                R1,
                P1,
                R2,
                P2,
                fusion_module=self.comm_fusion_model,
            )
            comm_outputs_by_timestep.append(comm_outputs_t)

            if timestep_idx == rollout_start_idx:
                raw_rgb_t0 = raw_rgb
                raw_position_t0 = raw_position
                rgb_view1 = rgb_aug_view1
                rgb_view2 = rgb_aug_view2
                position_view1 = position_aug_view1
                position_view2 = position_aug_view2
                spr_rgb_t0 = self.SPR.transform(
                    raw_rgb.clone(),
                    augment=True,
                )
                spr_position_t0 = self.SPR.augment_position(
                    raw_position.clone(),
                    noise_std=self.SPR.position_noise_std,
                    mask_ratio=self.SPR.position_mask_ratio,
                )
                image_latent = self.SPR.image_encoder(spr_rgb_t0)
                position_latent = self.SPR.position_encoder(spr_position_t0)

        assert len(comm_outputs_by_timestep) == comm_num_timesteps
        assert image_latent is not None
        assert position_latent is not None
        comm_outputs_t0 = comm_outputs_by_timestep[0]
        comm_timestep_weights = torch.full(
            (comm_num_timesteps,),
            1,
            device=comm_outputs_t0["loss"].device,
            dtype=comm_outputs_t0["loss"].dtype,
        )
        comm_timestep_weights[0] = 1.0
        comm_timestep_weight_sum = comm_timestep_weights.sum()
        comm_outputs = {
            key: (
                torch.stack(
                    [output[key] for output in comm_outputs_by_timestep],
                    dim=0,
                )
                * comm_timestep_weights.view(
                    comm_num_timesteps,
                    *([1] * comm_outputs_t0[key].dim()),
                )
            ).sum(dim=0) / comm_timestep_weight_sum
            for key in comm_outputs_t0.keys()
        }
        comm_loss_t0 = comm_outputs_t0["loss"]
        comm_loss = comm_outputs["loss"]

        image_anchor = image_latent
        position_anchor = position_latent
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
        image_anchor_fusion = fused_anchor_tokens[:, 0, :]
        position_anchor_fusion = fused_anchor_tokens[:, 1, :]

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

        # Every horizon is predicted directly from the same real online anchor.
        batch_size = image_anchor.shape[0]
        horizon_count = self.spr_horizon
        image_anchor_all = image_anchor.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        position_anchor_all = position_anchor.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        image_fusion_all = image_anchor_fusion.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        position_fusion_all = position_anchor_fusion.unsqueeze(0).expand(
            horizon_count, -1, -1
        )
        prefix_embeddings_all = prefix_embeddings.transpose(0, 1)

        image_transition_input = torch.cat(
            [
                image_anchor_all,
                image_fusion_all,
                prefix_embeddings_all,
            ],
            dim=-1,
        )
        position_transition_input = torch.cat(
            [
                position_anchor_all,
                position_fusion_all,
                prefix_embeddings_all,
            ],
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
        assert predicted_image.shape == (
            horizon_count,
            batch_size,
            encoder_feature_dim,
        )
        assert predicted_position.shape == predicted_image.shape
        pred_latents_image.extend(predicted_image.unbind(dim=0))
        pred_latents_position.extend(predicted_position.unbind(dim=0))

        comm_rgb_full_loss = comm_outputs["rgb_full_loss"]
        comm_position_full_loss = comm_outputs["position_full_loss"]
        comm_ssl_acc = comm_outputs["ssl_acc"]
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

        model_spr_loss_image = spr_loss_image[1:].mean(0)  # (H, B) -> (B,)
        t0_spr_loss_image = spr_loss_image[0]
        model_spr_loss_position = spr_loss_position[1:].mean(0)  # (H, B) -> (B,)
        t0_spr_loss_position = spr_loss_position[0]
        model_spr_loss_image_mean = model_spr_loss_image.mean()
        model_spr_loss_position_mean = model_spr_loss_position.mean()
        t0_spr_loss_image_mean = t0_spr_loss_image.mean()
        t0_spr_loss_position_mean = t0_spr_loss_position.mean()

        final_spr_loss_image = self.t0_spr_loss_weight * t0_spr_loss_image_mean + \
                               self.model_spr_weight * model_spr_loss_image_mean
        final_spr_loss_position = self.t0_spr_loss_weight * t0_spr_loss_position_mean + \
                               self.model_spr_weight * model_spr_loss_position_mean
        loss_sum = self.awl(
            final_spr_loss_position,
            final_spr_loss_image,
        )
        weighted_spr_branch_loss = self.spr_loss_weight * loss_sum
        weighted_comm_loss = self.comm_loss_weight * comm_loss
        auxiliary_loss = weighted_spr_branch_loss + weighted_comm_loss
        weighted_spr_loss = self.auxiliary_loss_weight * weighted_spr_branch_loss
        weighted_auxiliary_comm_loss = (
            self.auxiliary_loss_weight * weighted_comm_loss
        )
        all_loss = self.auxiliary_loss_weight * auxiliary_loss

        if step % self.log_interval == 0:
            L.log('train/spr_loss', loss_sum.item(), step)
            L.log('train/spr_image_loss', final_spr_loss_image.item(), step)
            L.log('train/spr_position_loss', final_spr_loss_position.item(), step)
            L.log('train/spr_t0_image_loss', t0_spr_loss_image_mean.item(), step)
            L.log('train/spr_t0_position_loss', t0_spr_loss_position_mean.item(), step)
            L.log('train/spr_t0_loss', (t0_spr_loss_image_mean + t0_spr_loss_position_mean).item(), step)
            L.log('train/spr_model_image_loss', model_spr_loss_image_mean.item(), step)
            L.log('train/spr_model_position_loss', model_spr_loss_position_mean.item(), step)
            L.log('train/spr_model_loss', (model_spr_loss_image_mean + model_spr_loss_position_mean).item(), step)
            L.log('train/spr_final_image_loss_weighted', final_spr_loss_image.item(), step)
            L.log('train/spr_final_position_loss_weighted', final_spr_loss_position.item(), step)
            awl_position_weight = 0.5 / self.awl.params[0].detach().pow(2)
            awl_image_weight = 0.5 / self.awl.params[1].detach().pow(2)
            L.log('train/awl_position_param', self.awl.params[0].item(), step)
            L.log('train/awl_image_param', self.awl.params[1].item(), step)
            L.log('train/awl_position_weight', awl_position_weight.item(), step)
            L.log('train/awl_image_weight', awl_image_weight.item(), step)
            L.log('train/auxiliary_loss_unweighted', auxiliary_loss.item(), step)
            L.log('train/all_auxiliary_loss', all_loss.item(), step)
            for horizon_idx in range(1, self.spr_horizon + 1):
                L.log(
                    f'train/spr_rgb_h{horizon_idx}',
                    spr_loss_image[horizon_idx].mean().item(),
                    step,
                )
                L.log(
                    f'train/spr_position_h{horizon_idx}',
                    spr_loss_position[horizon_idx].mean().item(),
                    step,
                )
            L.log('train/comm_loss', comm_loss.item(), step)
            L.log('train/comm_loss_weighted', weighted_comm_loss.item(), step)
            L.log('train/spr_loss_weighted',
                  weighted_spr_branch_loss.item(), step)
            L.log('train/spr_loss_weighted_by_auxiliary',
                  weighted_spr_loss.item(), step)
            L.log('train/comm_loss_weighted_by_auxiliary',
                  weighted_auxiliary_comm_loss.item(), step)
            L.log('train/comm_rgb_full_loss', comm_rgb_full_loss.item(), step)
            L.log('train/comm_position_full_loss', comm_position_full_loss.item(), step)
            L.log('train/comm_ssl_acc', comm_ssl_acc.item(), step)
            L.log('train/comm_t0_loss', comm_loss_t0.item(), step)
            L.log('train/comm_f1_f2_loss', comm_outputs["f1_f2_loss"].item(), step)
            L.log('train/comm_r_f1_loss', comm_outputs["r_f1_loss"].item(), step)
            L.log('train/comm_r_f2_loss', comm_outputs["r_f2_loss"].item(), step)
            L.log('train/comm_p_f1_loss', comm_outputs["p_f1_loss"].item(), step)
            L.log('train/comm_p_f2_loss', comm_outputs["p_f2_loss"].item(), step)
            L.log('train/comm_f1_f2_acc', comm_outputs["f1_f2_acc"].item(), step)
            L.log('train/comm_r_f1_acc', comm_outputs["r_f1_acc"].item(), step)
            L.log('train/comm_r_f2_acc', comm_outputs["r_f2_acc"].item(), step)
            L.log('train/comm_p_f1_acc', comm_outputs["p_f1_acc"].item(), step)
            L.log('train/comm_p_f2_acc', comm_outputs["p_f2_acc"].item(), step)
            L.log('train/comm_t0_ssl_acc', comm_outputs_t0["ssl_acc"].item(), step)
            L.log('train/comm_t0_r_f_acc', comm_outputs_t0["r_f_acc"].item(), step)
            L.log('train/comm_t0_p_f_acc', comm_outputs_t0["p_f_acc"].item(), step)
            L.log('train/comm_rgb_view_distance',
                  (rgb_view1 - rgb_view2).detach().pow(2).mean().sqrt().item(), step)
            L.log('train/comm_position_view_distance',
                  (position_view1 - position_view2).detach().pow(2).mean().sqrt().item(), step)
            L.log('train/comm_preprojector_norm_mean',
                  comm_outputs["preprojector_norm_mean"].item(), step)
            L.log('train/comm_h_r_norm', comm_outputs["h_r_norm"].item(), step)
            L.log('train/comm_h_p_norm', comm_outputs["h_p_norm"].item(), step)
            L.log('train/comm_h_f1_norm', comm_outputs["h_f1_norm"].item(), step)
            L.log('train/comm_h_f2_norm', comm_outputs["h_f2_norm"].item(), step)
            L.log('train/comm_projected_feature_norm_mean',
                  comm_outputs["projected_feature_norm_mean"].item(), step)
            L.log('train/comm_positive_similarity_mean',
                  comm_outputs["positive_similarity_mean"].item(), step)
            L.log('train/comm_negative_similarity_mean',
                  comm_outputs["negative_similarity_mean"].item(), step)
            L.log('train/comm_similarity_margin',
                  comm_outputs["similarity_margin"].item(), step)
            L.log('train/comm_z_r_norm', comm_outputs["z_r_norm"].item(), step)
            L.log('train/comm_z_p_norm', comm_outputs["z_p_norm"].item(), step)
            L.log('train/comm_z_f1_norm', comm_outputs["z_f1_norm"].item(), step)
            L.log('train/comm_z_f2_norm', comm_outputs["z_f2_norm"].item(), step)
            L.log('train/comm_preprojector_fusion_view_distance',
                  comm_outputs["preprojector_fusion_view_distance"].item(), step)
            L.log('train/comm_projected_fusion_view_distance',
                  comm_outputs["projected_fusion_view_distance"].item(), step)

            if not self._logged_static_auxiliary_config:
                L.log('train/model_spr_weight',
                      float(self.model_spr_weight), step)
                L.log('train/t0_spr_loss_weight',
                      float(self.t0_spr_loss_weight), step)
                L.log('train/spr_loss_weight',
                      float(self.spr_loss_weight), step)
                L.log('train/comm_loss_weight',
                      float(self.comm_loss_weight), step)
                L.log('train/auxiliary_loss_weight',
                      float(self.auxiliary_loss_weight), step)
                L.log('train/position_noise_std',
                      float(self.SPR.position_noise_std), step)
                L.log('train/position_mask_ratio',
                      float(self.SPR.position_mask_ratio), step)
                L.log('train/spr_horizon',
                      float(self.spr_horizon), step)
                L.log('train/spr_action_skip',
                      float(self.spr_action_skip), step)
                L.log('train/comm_num_timesteps',
                      float(comm_num_timesteps), step)
                L.log('train/comm_temporal_aug_group_size',
                      float(comm_aug_group_size), step)
                L.log('train/comm_t0_timestep_weight',
                      comm_timestep_weights[0].item(), step)
                L.log('train/comm_future_timestep_weight',
                      comm_timestep_weights[-1].item(), step)
                L.log('train/comm_timestep_weight_sum',
                      comm_timestep_weight_sum.item(), step)
                self._logged_static_auxiliary_config = True

        if step % self.log_interval == 0:
            image_encoder_params = list(self.critic.image_encoder.parameters())
            position_encoder_params = list(
                self.critic.position_encoder.parameters()
            )

            grad_rgb_from_spr = get_grad_norm_from_loss(
                weighted_spr_loss,
                image_encoder_params,
            )
            grad_rgb_from_comm = get_grad_norm_from_loss(
                weighted_auxiliary_comm_loss,
                image_encoder_params,
            )
            grad_rgb_from_comm_raw = get_grad_norm_from_loss(
                comm_loss,
                image_encoder_params,
            )
            grad_position_from_spr = get_grad_norm_from_loss(
                weighted_spr_loss,
                position_encoder_params,
            )
            grad_position_from_comm = get_grad_norm_from_loss(
                weighted_auxiliary_comm_loss,
                position_encoder_params,
            )
            grad_position_from_comm_raw = get_grad_norm_from_loss(
                comm_loss,
                position_encoder_params,
            )
            grad_rgb_spr_comm_cosine = get_grad_cosine_from_losses(
                weighted_spr_loss,
                weighted_auxiliary_comm_loss,
                image_encoder_params,
            )
            grad_position_spr_comm_cosine = get_grad_cosine_from_losses(
                weighted_spr_loss,
                weighted_auxiliary_comm_loss,
                position_encoder_params,
            )

            L.log(
                "train/grad_rgb_encoder_from_spr",
                grad_rgb_from_spr,
                step,
            )
            L.log(
                "train/grad_rgb_encoder_from_comm",
                grad_rgb_from_comm,
                step,
            )
            L.log(
                "train/grad_rgb_encoder_from_comm_raw",
                grad_rgb_from_comm_raw,
                step,
            )
            L.log(
                "train/grad_position_encoder_from_spr",
                grad_position_from_spr,
                step,
            )
            L.log(
                "train/grad_position_encoder_from_comm",
                grad_position_from_comm,
                step,
            )
            L.log(
                "train/grad_position_encoder_from_comm_raw",
                grad_position_from_comm_raw,
                step,
            )
            L.log(
                "train/grad_rgb_encoder_spr_comm_ratio",
                grad_rgb_from_spr / (grad_rgb_from_comm + 1e-12),
                step,
            )
            L.log(
                "train/grad_position_encoder_spr_comm_ratio",
                grad_position_from_spr / (grad_position_from_comm + 1e-12),
                step,
            )
            L.log(
                "train/grad_rgb_encoder_spr_comm_cosine",
                grad_rgb_spr_comm_cosine,
                step,
            )
            L.log(
                "train/grad_position_encoder_spr_comm_cosine",
                grad_position_spr_comm_cosine,
                step,
            )

        self.image_encoder_optimizer.zero_grad()
        self.position_encoder_optimizer.zero_grad()
        self.spr_optimizer.zero_grad()
        self.awl_optimizer.zero_grad()
        self.comm_optimizer.zero_grad()
        all_loss.backward()
        if step % self.log_interval == 0:
            log_optimizer_grad_stats(
                L,
                "train/grad_image_encoder",
                self.image_encoder_optimizer,
                step,
            )
            log_optimizer_grad_stats(
                L,
                "train/grad_position_encoder",
                self.position_encoder_optimizer,
                step,
            )
            log_optimizer_grad_stats(L, "train/grad_spr", self.spr_optimizer, step)
            log_optimizer_grad_stats(L, "train/grad_awl", self.awl_optimizer, step)
            log_optimizer_grad_stats(L, "train/grad_comm", self.comm_optimizer, step)
            log_module_grad_stats(
                L,
                "train/grad_spr_fusion_transformer",
                self.SPR.fusion_transition_model,
                step,
            )
            log_module_grad_stats(
                L,
                "train/grad_spr_action_prefix_encoder",
                self.SPR.action_prefix_encoder,
                step,
            )
            log_module_grad_stats(
                L,
                "train/grad_comm_fusion_transformer",
                self.comm_fusion_model,
                step,
            )
            log_module_grad_stats(
                L,
                "train/grad_comm_projector",
                self.comm_loss_fn.projector,
                step,
            )
            log_parameter_grad_stats(
                L,
                "train/grad_comm_rgb_mask_token",
                self.comm_loss_fn.rgb_mask_token,
                step,
            )
            log_parameter_grad_stats(
                L,
                "train/grad_comm_position_mask_token",
                self.comm_loss_fn.position_mask_token,
                step,
            )
            log_parameter_grad_stats(
                L,
                "train/grad_comm_rgb_slot_embedding",
                self.comm_loss_fn.rgb_slot_embedding,
                step,
            )
            log_parameter_grad_stats(
                L,
                "train/grad_comm_position_slot_embedding",
                self.comm_loss_fn.position_slot_embedding,
                step,
            )
        self.position_encoder_optimizer.step()
        self.image_encoder_optimizer.step()
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
        payload = {
            "spr": self.SPR.state_dict(),
            "comm_loss_fn": self.comm_loss_fn.state_dict(),
            "comm_fusion_model": self.comm_fusion_model.state_dict(),
            "awl": self.awl.state_dict(),
            "spr_optimizer": self.spr_optimizer.state_dict(),
            "awl_optimizer": self.awl_optimizer.state_dict(),
            "comm_optimizer": self.comm_optimizer.state_dict(),
            "image_encoder_optimizer": self.image_encoder_optimizer.state_dict(),
            "position_encoder_optimizer": self.position_encoder_optimizer.state_dict(),
            "spr_horizon": self.spr_horizon,
            "spr_action_skip": self.spr_action_skip,
            "step": step,
        }
        torch.save(payload, spr_name)
        print(f"Auxiliary modules saved at step {step}")

    def load_auxiliary(self, aux_path):
        payload = torch.load(aux_path, map_location=self.device)
        def load_optimizer_if_present(key, optimizer, label):
            if key not in payload:
                return
            try:
                optimizer.load_state_dict(payload[key])
            except (ValueError, RuntimeError) as error:
                if label == "CoMM":
                    print(
                        "Warning: incompatible legacy CoMM optimizer state; "
                        "the masked concat-readout projection-head CoMM optimizer "
                        "remains newly initialized."
                    )
                else:
                    print(f"Warning: incompatible {label} optimizer state: {error}")

        if isinstance(payload, dict) and "spr" in payload:
            skipped, missing, unexpected = load_matching_state_dict(
                self.SPR,
                payload["spr"],
            )
            transition_input_skipped = any(
                key in skipped
                for key in (
                    "image_transition_model.fc.weight",
                    "position_transition_model.fc.weight",
                )
            )
            if transition_input_skipped:
                print(
                    "Warning: skipped legacy recursive transition input-layer "
                    "weights because action-prefix transition inputs have a "
                    "different shape."
                )
            prefix_keys = tuple(
                key for key in self.SPR.state_dict()
                if key.startswith("action_prefix_encoder.")
            )
            if any(key in missing for key in prefix_keys):
                print(
                    "Warning: checkpoint has no compatible action-prefix encoder; "
                    "the new causal prefix encoder remains newly initialized."
                )
            if "comm" in payload:
                print("Warning: legacy CoMM module parameters were ignored.")
            if "comm_loss_fn" in payload:
                skipped_comm_loss, missing_comm_loss, _ = load_matching_state_dict(
                    self.comm_loss_fn,
                    payload["comm_loss_fn"],
                )
                legacy_bilinear = any(
                    key == "W" or key.startswith("representation_norm.")
                    for key in payload["comm_loss_fn"].keys()
                )
                legacy_mean_pooled_projector = any(
                    key.startswith("projector.")
                    for key in skipped_comm_loss
                )
                legacy_private_comm_fusion = any(
                    key.startswith("fusion.")
                    for key in skipped_comm_loss
                )
                if legacy_bilinear:
                    print(
                        "Warning: legacy masked concat-readout bilinear CoMM checkpoint is "
                        "incompatible with the shared three-layer projection-head CoMM. "
                        "Compatible mask-token and slot-embedding parameters were loaded "
                        "when possible. The projection head and CoMM optimizer remain newly "
                        "initialized."
                    )
                if legacy_private_comm_fusion:
                    print(
                        "Warning: skipped obsolete private CoMM fusion parameters. "
                        "The CoMM fusion Transformer remains separately initialized."
                    )
                if legacy_mean_pooled_projector:
                    print(
                        "Warning: legacy mean-pooled CoMM projector is incompatible because "
                        "its input dimension is 50 instead of 100. The new concat-readout "
                        "projector remains newly initialized."
                    )
                if skipped_comm_loss:
                    print(
                        "Warning: skipped incompatible CoMM "
                        f"parameters: {skipped_comm_loss}"
                    )
            else:
                print(
                    "Warning: checkpoint has no masked concat-readout projection-head "
                    "CoMM parameters; CoMM remains newly initialized."
                )
            if "comm_fusion_model" in payload:
                self.comm_fusion_model.load_state_dict(payload["comm_fusion_model"])
            else:
                print(
                    "Warning: checkpoint has no separate CoMM fusion Transformer. "
                    "The CoMM fusion Transformer remains separately initialized."
                )
            if "awl" in payload:
                self.awl.load_state_dict(payload["awl"])
            else:
                print(
                    "Warning: checkpoint has no SPR AWL parameters; "
                    "AWL remains newly initialized."
                )
            load_optimizer_if_present("spr_optimizer", self.spr_optimizer, "SPR")
            load_optimizer_if_present("awl_optimizer", self.awl_optimizer, "SPR AWL")
            load_optimizer_if_present("comm_optimizer", self.comm_optimizer, "CoMM")
            if "image_encoder_optimizer" in payload:
                load_optimizer_if_present(
                    "image_encoder_optimizer",
                    self.image_encoder_optimizer,
                    "image encoder",
                )
            if "position_encoder_optimizer" in payload:
                load_optimizer_if_present(
                    "position_encoder_optimizer",
                    self.position_encoder_optimizer,
                    "position encoder",
                )
            if (
                "encoder_optimizer" in payload
                and "image_encoder_optimizer" not in payload
                and "position_encoder_optimizer" not in payload
            ):
                print(
                    "Warning: unified encoder optimizer state was ignored; "
                    "baseline split encoder optimizers remain newly initialized."
                )
            legacy_keys = {"curl", "curl.W", "curl_optimizer"}
            if legacy_keys.intersection(payload.keys()):
                print("Warning: legacy CURL parameters were ignored; CoMM remains initialized.")
        else:
            load_matching_state_dict(self.SPR, payload)
            print("Warning: legacy SPR checkpoint has no CoMM parameters.")
        
    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))
