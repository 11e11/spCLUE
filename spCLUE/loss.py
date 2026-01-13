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

        return loss + 1.* neg_entropy_loss

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
class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss
    更适合稀疏计数数据
    
    SCE = α * H(p, q) + β * H(q, p)
    其中 H(p,q) = -Σ p*log(q) 是交叉熵
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=None):
        """
        Parameters:
        -----------
        alpha : float
            标准CE的权重 H(target, pred)
        beta : float
            反向CE的权重 H(pred, target)
        num_classes : int or None
            如果是分类任务需要指定
        """
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.eps = 1e-10  # 数值稳定性
    
    def forward(self, pred, target):
        """
        Parameters:
        -----------
        pred :  torch.Tensor
            预测值 [batch_size × n_features]
        target : torch.Tensor
            真实值 [batch_size × n_features]
        
        Returns:
        --------
        loss : torch.Tensor
            SCE损失
        """
        # 确保值在合理范围内
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        target = torch.clamp(target, min=self.eps, max=1.0)
        
        # 归一化到概率分布（沿特征维度）
        pred = pred / (pred.sum(dim=1, keepdim=True) + self.eps)
        target = target / (target.sum(dim=1, keepdim=True) + self.eps)
        
        # 标准交叉熵 H(target, pred)
        ce = -torch.sum(target * torch.log(pred + self.eps), dim=1)
        
        # 反向交叉熵 H(pred, target)
        rce = -torch.sum(pred * torch.log(target + self.eps), dim=1)
        
        # 对称交叉熵
        sce = self.alpha * ce + self.beta * rce
        
        return sce.mean()


class ReconstructionLoss(nn.Module):
    """
    重构损失的统一接口
    支持MSE和SCE
    """
    def __init__(self, loss_type='sce', **kwargs):
        """
        Parameters:
        -----------
        loss_type : str
            'mse' 或 'sce'
        **kwargs : dict
            传递给具体loss的参数
        """
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'sce':
            self.loss_fn = SCELoss(
                alpha=kwargs.get('alpha', 0.1),
                beta=kwargs.get('beta', 1.0)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, target):
        """
        Parameters: 
        -----------
        pred : torch.Tensor
            重构的基因表达
        target : torch.Tensor
            真实的基因表达
        
        Returns:
        --------
        loss : torch.Tensor
        """
        if self.loss_type == 'sce':
            # SCE需要非负值
            pred = F.softplus(pred)  # 确保非负
            # 或者用 sigmoid:  pred = torch.sigmoid(pred)
        
        return self.loss_fn(pred, target)
