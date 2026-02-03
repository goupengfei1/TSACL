import math
import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def temporal_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    if reduction:
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    else:
        loss = (logits[:, t, T + t - 1].mean(dim=1) + logits[:, T + t, t].mean(dim=1)) / 2
    return loss#, a_loss


def instance_contrastive_loss(z1, z2, reduction=True):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    if reduction:
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    else:
        loss = (logits[:, i, B + i - 1].mean(dim=0) + logits[:, B + i, i].mean(dim=0)) / 2

    return loss#, a_loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0, reduction=True):
    B = z1.size(0)
    if reduction:
        loss = torch.tensor(0., device=z1.device)
    else:
        loss = torch.zeros(B, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2, reduction)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2, reduction)
        d += 1
    return loss / d


def relative_mask(distance_matrix):
    same_label_mask = (distance_matrix == 0.0)
    relative_matrix = distance_matrix.masked_fill(same_label_mask, np.inf) # remove same label
    min_vals, _ = torch.min(relative_matrix, dim=1, keepdim=True)
    pos_mask = (relative_matrix == min_vals).float()
    neg_mask = torch.ones_like(relative_matrix) - same_label_mask.float()
    return pos_mask, neg_mask


def get_circle_embedding(N):
    index = np.arange(N)
    interval = 2 * np.pi / N
    theta = index * interval
    x = np.cos(theta)
    y = np.sin(theta)
    embeds = np.stack([x, y], axis=1)
    return embeds


def auto_mask(distance_matrix, B):
    """
    distance_matrix: (T, B, B)
    """
    # 1. 创建对角线掩码 (Self Mask)
    # 使用 torch.eye 确保精准屏蔽，不受浮点误差影响
    eye_mask = torch.eye(B, device=distance_matrix.device).bool()  # (B, B)
    # 扩展到 T 维度: (T, B, B)
    eye_mask = eye_mask.unsqueeze(0).expand_as(distance_matrix)

    # 2. 移除自己 (填入负无穷，确保不会被 max 选中)
    distance_matrix_wo_self = distance_matrix.masked_fill(eye_mask, -1e9)

    # 3. 寻找每行的最大值 (Hard Max Positive)
    # max_vals: (T, B, 1)
    max_vals, _ = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)

    # 4. 生成正样本 Mask
    # 考虑到浮点误差，建议使用微小的 epsilon 比较，或者直接相等(如果之前数值没变动)
    # 这里使用 >= max_vals - epsilon 更稳健，防止有多个极值时漏选
    pos_mask = (distance_matrix_wo_self >= max_vals - 1e-6).float()

    return pos_mask

def auto_mask(distance_matrix, self_mask):
    """
        distance_matrix: (T, B, B)
    """
    distance_matrix_wo_self = distance_matrix.masked_fill(self_mask, -np.inf)
    #取出每行最大值，标记为正样本
    max_vals, pos_posi = torch.max(distance_matrix_wo_self, dim=2, keepdim=True)
    pos_mask = (distance_matrix_wo_self == max_vals).float()
    return pos_mask

    # return pos_mask

class SemanticCon(nn.Module):
    def __init__(self, batch_size, seq_len,  itr, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SemanticCon, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.itr_now = itr
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.seq_len = seq_len

    def avg_global_contrastive_loss_ci(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        BC, T, D = features.shape
        B, _, _ = labels.shape
        C = BC // B
        labels = labels.squeeze(1).repeat(1, C, 1, 1).reshape(BC, T, -1)
        features = features.permute(1, 0, 2) # feature:(BC, T, D) -> （BC， seq_len, D）
        labels_mat = labels.permute(1, 0, 2) # labels:(BC, T, 5) -> （BC， seq_len, 5）
        labels_mat = F.cosine_similarity(labels_mat.unsqueeze(2), labels_mat.unsqueeze(1),dim=-1)  # T, BC, BC

        # Compute global representation similarities
        global_features = features.clone()  # (T, BC, D)
        anchor_dot_contrast = torch.div(
            torch.matmul(global_features, global_features.permute(0, 2, 1)),
            # (T, BC, D) X (T, BC, D) > (T, BC, BC)
            self.temperature)
        # For numerical stability
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_min.detach()  # subtract most large value

        # (T, BC, BC)
        global_distmap = labels_mat.to(device)
        self_mask = (global_distmap == 1)
        pos_mask = auto_mask(global_distmap, self_mask)
        neg_mask = torch.ones_like(pos_mask)
        neg_mask.scatter_(dim=2, index=torch.arange(BC).reshape(1, -1, 1).repeat(T, 1, 1).to(features.get_device()),
                          value=0)

        exp_global_logits = torch.exp(global_logits) * neg_mask

        log_global_prob = global_logits - torch.log(
            exp_global_logits.sum(-1, keepdim=True))

        mean_log_global_prob_pos = (global_distmap * pos_mask * log_global_prob).sum(-1) \
                                   / pos_mask.sum(-1)

        global_loss = (- (self.temperature / self.base_temperature) * mean_log_global_prob_pos).mean()
        return global_loss


    def avg_global_contrastive_loss(self, features, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        B, T, D = features.shape
        features = features.permute(1, 0, 2) # feature:(B, T, D) -> （B， seq_len, D）
        labels_mat = labels.permute(1, 0, 2)  # labels:(B, T, 4) -> （B， seq_len, 4）

        # (T, BC, D)
        labels_mat = F.cosine_similarity(labels_mat.unsqueeze(2), labels_mat.unsqueeze(1),dim=-1)  # T, B, B
        if torch.isinf(labels_mat).any():
            print("global_features contains inf")
            exit()

        # Compute global representation similarities
        global_features = features.clone()  # (T, B, D)
        anchor_dot_contrast = torch.div(
            torch.matmul(global_features, global_features.permute(0, 2, 1)),
            # (T, B, D) X (T, B, D) > (T, B, B)
            self.temperature)
        # For numerical stability
        logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
        global_logits = anchor_dot_contrast - logits_min.detach()  # subtract most large value
        global_distmap = labels_mat.to(device)

        self_mask = (global_distmap == 1)
        pos_mask = auto_mask(global_distmap, self_mask)
        neg_mask = torch.ones_like(pos_mask)
        neg_mask.scatter_(dim=2, index=torch.arange(B).reshape(1, -1, 1).repeat(T, 1, 1).to(features.get_device()),
                          value=0)

        exp_global_logits = torch.exp(global_logits) * neg_mask
        log_global_prob = global_logits - torch.log(
            exp_global_logits.sum(-1, keepdim=True))  # (psize, p_num * BC, p_num * B) > (psize, p_num * BC, 1)
        mean_log_global_prob_pos = (pos_mask * log_global_prob).sum(-1) \
                                   / pos_mask.sum(-1)

        global_loss = (- (self.temperature / self.base_temperature) * mean_log_global_prob_pos).mean()

        return global_loss


    def forward(self, features, labels):

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            print('features shape > 3')

        BC, I, D = features.shape
        if labels.shape[0] == BC: # feature: S
            global_loss = self.avg_global_contrastive_loss(features, labels)
        else: # feature : M
            global_loss = self.avg_global_contrastive_loss_ci(features, labels)

        return global_loss



if __name__ == '__main__':
    batch_channels = 10
    batch = 5
    n_view = 2
    seq_len = 12
    dim = 2
    # features = torch.rand((batch_size, n_view, seq_len, dim)) # (Batch_size, N_view, T, Dim)
    features = torch.rand((batch, seq_len, dim)) # (Batch_size, T, Dim)
    labels = torch.rand((batch, seq_len, 5))
    distmap = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()

    supcreg = SemanticCon(batch_size=5, seq_len=seq_len, contrast_mode='all')
    local_loss, global_loss = supcreg(features.cuda(), labels.cuda())
    print(local_loss, global_loss.shape)