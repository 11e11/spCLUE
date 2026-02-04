import torch
from torch import nn
import math
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x, xbar, eps=1e-8):
        posScores = torch.exp((x * xbar).sum(dim=1) / self.temperature)
        negScores = torch.exp((x @ xbar.T) / self.temperature).sum(dim=1)
        return -torch.log(posScores / (negScores + eps)).mean()

class ClusterLoss(nn.Module):
    def __init__(
            self,
            n_classes,
            device=torch.device("cuda:0"),
            temperature=0.2,
    ):
        super(ClusterLoss, self).__init__()
        self.n_classes = n_classes
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(n_classes)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, n_classes):
        N = 2 * n_classes
        mask = torch.ones(N, N)
        mask.fill_diagonal_(0)
        for i in range(n_classes):
            mask[i, i + n_classes] = 0
            mask[i + n_classes, i] = 0
        mask = mask.bool()
        return mask

    def normalizeLabel(self, c_i, c_j):
        c_i = torch.square(c_i)
        c_j = torch.square(c_j)
        p_i = c_i.sum(dim=0).view(-1)
        c_i /= p_i
        p_i = c_i.sum(dim=1).view(-1)
        c_i /= p_i.unsqueeze(1)
        p_j = c_j.sum(dim=0).view(-1)
        c_j /= p_j
        p_j = c_j.sum(dim=1).view(-1)
        c_j /= p_j.unsqueeze(1)
        return c_i, c_j

    def forward(self, c_i, c_j):
        p_i = c_i.sum(dim=0).view(-1)
        p_i /= p_i.sum()
        neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        neg_entropy_loss = neg_entropy_i + neg_entropy_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.n_classes
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1),
                                c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.n_classes)
        sim_j_i = torch.diag(sim, -self.n_classes)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=-1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + 1. * neg_entropy_loss

class GraphConsis(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, emb, graphWeight):
        dist1 = torch.cdist(emb, emb, p=2)
        dist1 = torch.div(dist1, torch.max(dist1))
        return torch.mean((1 - dist1) * graphWeight)


class GraphRecLoss(nn.Module):
    def __init__(self, norm_val, pos_weight) -> None:
        super().__init__()
        self.norm_val = norm_val
        self.pos_weight = pos_weight

    def forward(self, emb, target):
        # emb = F.normalize(emb, p=2, dim=1)
        input = emb @ emb.T
        logits = F.binary_cross_entropy_with_logits(input,
                                                    target,
                                                    pos_weight=self.pos_weight)
        return self.norm_val * logits


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, xbar):
        return torch.square(x - xbar).mean(dim=1).mean()


class ZINBLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, mean, disp, pi=0, scale_factor=1.0, ridge_lambda=0.0):
        '''
        args: x, raw count, [N, hvgs]
              scale_factor, [n,]
        '''
        eps = 1e-10
        mean = (mean.T * scale_factor).T
        if pi == 0:
            pi = torch.tensor(0.0)

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(
            x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphGuidedContrastiveLoss(nn.Module):
    """
    Gated graph-guided contrastive loss with boundary filtering.
    Only compute loss for:
      1. High-confidence edges (top weight percentile)
      2. Non-boundary nodes (consistent predictions)
    """
    def __init__(self, temperature=0.2, k_pos=2, k_neg=256, n_anchors=1024):
        super().__init__()
        self.temperature = temperature
        self.k_pos = k_pos        # Reduced from 3 to 2 for stricter positives
        self.k_neg = k_neg        
        self.n_anchors = n_anchors
    
    def forward(self, z, neighbors, weights, pred_labels1, pred_labels2, 
                weight_threshold=0.0, eps=1e-8):
        """
        Args:
            z: fused embeddings [N, D]
            neighbors: [N, k_c] pre-computed neighbor indices
            weights: [N, k_c] edge weights
            pred_labels1: predictions from view 1 [N, n_clusters] (softmax output)
            pred_labels2: predictions from view 2 [N, n_clusters]
            weight_threshold: minimum edge weight to consider (for gating)
        Returns:
            gated graph-guided contrastive loss
        """
        N, D = z.shape
        device = z.device
        
        # === Gate 1: Identify non-boundary nodes (prediction consistency) ===
        # Use hard labels for consistency check
        hard_label1 = pred_labels1.argmax(dim=1)  # [N]
        hard_label2 = pred_labels2.argmax(dim=1)  # [N]
        consistent_mask = (hard_label1 == hard_label2)  # [N], True for non-boundary
        
        # Additionally, check prediction confidence (max prob > 0.6)
        max_prob1 = pred_labels1.max(dim=1)[0]
        max_prob2 = pred_labels2.max(dim=1)[0]
        # print("Max prob1 stats: min {:.4f}, max {:.4f}, mean {:.4f}".format(
        #     max_prob1.min().item(), max_prob1.max().item(), max_prob1.mean().item()))
        # print("Max prob2 stats: min {:.4f}, max {:.4f}, mean {:.4f}".format(
        #     max_prob2.min().item(), max_prob2.max().item(), max_prob2.mean().item()))
        confident_mask = (max_prob1 > 0) & (max_prob2 > 0)
        
        # Combined gate: consistent AND confident
        valid_anchor_mask = consistent_mask & confident_mask  # [N]
        valid_anchor_indices = torch.where(valid_anchor_mask)[0]
        
        if len(valid_anchor_indices) == 0:
            # No valid anchors, return zero loss
            return torch.tensor(0.0, device=device)
        
        # Normalize embeddings
        z_norm = F.normalize(z, p=2, dim=1)
        
        # === Sample anchors from valid (non-boundary) nodes ===
        n_valid = len(valid_anchor_indices)
        if self.n_anchors < n_valid:
            sample_idx = torch.randperm(n_valid, device=device)[:self.n_anchors]
            anchor_idx = valid_anchor_indices[sample_idx]
        else:
            anchor_idx = valid_anchor_indices
        
        n_actual_anchors = anchor_idx.size(0)
        
        # Get anchor embeddings
        z_anchor = z_norm[anchor_idx]  # [n_anchors, D]
        
        # === Gate 2: Filter neighbors by edge weight (only high-confidence) ===
        anchor_neighbors = neighbors[anchor_idx]  # [n_anchors, k_c]
        anchor_weights = weights[anchor_idx]      # [n_anchors, k_c]
        
        # Mask out low-weight neighbors
        high_conf_mask = anchor_weights >= weight_threshold  # [n_anchors, k_c]
        
        # For each anchor, sample k_pos positives from high-confidence neighbors
        pos_indices_list = []
        valid_anchor_list = []
        
        for i in range(n_actual_anchors):
            valid_neighbors = anchor_neighbors[i][high_conf_mask[i]]
            
            if len(valid_neighbors) < self.k_pos:
                # Not enough high-confidence neighbors, skip this anchor
                continue
            
            # Randomly sample k_pos from valid neighbors
            perm = torch.randperm(len(valid_neighbors), device=device)[:self.k_pos]
            pos_idx = valid_neighbors[perm]
            
            pos_indices_list.append(pos_idx)
            valid_anchor_list.append(i)
        
        if len(valid_anchor_list) == 0:
            # No anchors have enough high-confidence neighbors
            return torch.tensor(0.0, device=device)
        
        # Stack valid anchors and their positives
        valid_anchor_list = torch.tensor(valid_anchor_list, dtype=torch.long, device=device)
        z_anchor_valid = z_anchor[valid_anchor_list]  # [n_valid_anchors, D]
        pos_indices = torch.stack(pos_indices_list)    # [n_valid_anchors, k_pos]
        
        n_valid_anchors = len(valid_anchor_list)
        
        # Get positive embeddings
        z_pos = z_norm[pos_indices.flatten()].view(n_valid_anchors, self.k_pos, D)
        
        # === Sample negative samples (random, excluding positives) ===
        neg_idx = torch.randint(0, N, (n_valid_anchors, self.k_neg), device=device)
        z_neg = z_norm[neg_idx]  # [n_valid_anchors, k_neg, D]
        
        # === Compute similarities ===
        sim_pos = torch.sum(z_anchor_valid.unsqueeze(1) * z_pos, dim=2) / self.temperature
        sim_neg = torch.sum(z_anchor_valid.unsqueeze(1) * z_neg, dim=2) / self.temperature
        
        # InfoNCE loss
        pos_exp = torch.exp(sim_pos).sum(dim=1)
        neg_exp = torch.exp(sim_neg).sum(dim=1)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + eps))
        
        # Log gating statistics
        # if torch.rand(1).item() < 0.01:  # Log 1% of the time to avoid spam
        boundary_ratio = (~consistent_mask).float().mean().item()
        valid_ratio = n_valid_anchors / N
        print(f"   [Gating] Boundary: {boundary_ratio:.1%}, Valid anchors: {valid_ratio:.1%}")
        
        return loss.mean()