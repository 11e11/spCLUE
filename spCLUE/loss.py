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

class CCRLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # CCR 损失不使用温度系数，它是基于距离/相似度偏差的硬约束 [cite: 256, 263]

    def forward(self, x, xbar):
        """
        x: 视图1的嵌入 (N, d), 例如空间图嵌入 E_s
        xbar: 视图2的嵌入 (N, d), 例如特征图嵌入 E_f
        """
        N = x.size(0)
        
        # 1. L2 归一化：确保内积等于余弦相似度 
        x_norm = F.normalize(x, p=2, dim=1)
        xbar_norm = F.normalize(xbar, p=2, dim=1)
        
        # 2. 计算跨视图亲和力矩阵 S (N x N)
        # S[i, j] 表示视图1的节点 i 与视图2的节点 j 之间的相似度 [cite: 258, 261]
        S = torch.mm(x_norm, xbar_norm.t())
        
        # 3. 计算对角线项 (Consistency)：同一节点在不同视图应高度一致
        # 目标是让 S[i, i] 趋近于 1 [cite: 263, 265]
        diag_sim = torch.diag(S)
        loss_diag = torch.mean((diag_sim - 1) ** 2)
        
        # 4. 计算非对角线项 (Discriminability)：不同节点之间应保持区分度
        # 目标是让 S[i, j] (i != j) 趋近于 0 [cite: 263, 265]
        # 技巧：先计算矩阵所有元素的平方和，减去对角线元素的平方和
        all_sq = S ** 2
        off_diag_sq_sum = all_sq.sum() - (diag_sim ** 2).sum()
        
        # 归一化系数为 N(N-1) [cite: 263]
        loss_off_diag = off_diag_sq_sum / (N * (N - 1))
        
        # 总损失 [cite: 263]
        return loss_diag + loss_off_diag

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
        disp = torch.clamp(disp, max=1e6)
        # mean = (mean.T * scale_factor).T
        # if pi == 0:
        #     pi = torch.tensor(0.0)

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
# class ZINBLoss(nn.Module):
#     def forward(self, x, mean, disp, pi, ridge_lambda=0.0):
#         eps = 1e-10
        
#         # 1. 对 disp 进行截断，防止 lgamma 爆炸 (MAFN 核心逻辑)
#         disp = torch.clamp(disp, max=1e6)
        
#         # 2. 负二项分布(NB)部分的似然计算
#         # t1 部分处理伽马函数
#         t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
#         # t2 部分处理均值和离散度的对数关系
#         t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
#             x * (torch.log(disp + eps) - torch.log(mean + eps)))
        
#         nb_final = t1 + t2
        
#         # 3. ZINB 分支逻辑
#         # 当 x > 0 时的损失 (仅来自非零部分)
#         nb_case = nb_final - torch.log(1.0 - pi + eps)
        
#         # 当 x = 0 时的损失 (来自 Dropout + NB 产生的 0)
#         # zero_nb 表示 NB 分布产生 0 的概率: (disp / (disp + mean))^disp
#         zero_nb = torch.pow(disp / (disp + mean + eps), disp)
#         zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
#         # 4. 根据 x 是否为 0 进行选择
#         result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

#         # 5. Pi 的正则化 (可选)
#         if ridge_lambda > 0:
#             ridge = ridge_lambda * torch.square(pi)
#             result += ridge

#         # 6. 最终稳定性处理 (MAFN 风格)
#         result = torch.where(torch.isnan(result), torch.full_like(result, float('inf')), result)
#         result = torch.where(torch.isinf(result), torch.full_like(result, 1e10), result) # 进一步防止 inf

#         return torch.mean(result)
