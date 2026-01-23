# import torch
# from torch import nn
# import math
# import torch.nn.functional as F


# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.2) -> None:
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, x, xbar, eps=1e-8):
#         posScores = torch.exp((x * xbar).sum(dim=1) / self.temperature)
#         negScores = torch.exp((x @ xbar.T) / self.temperature).sum(dim=1)
#         return -torch.log(posScores / (negScores + eps)).mean()

# class ClusterLoss(nn.Module):
#     def __init__(
#             self,
#             n_classes,
#             device=torch.device("cuda:0"),
#             temperature=0.2,
#     ):
#         super(ClusterLoss, self).__init__()
#         self.n_classes = n_classes
#         self.temperature = temperature
#         self.device = device

#         self.mask = self.mask_correlated_clusters(n_classes)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)

#     def mask_correlated_clusters(self, n_classes):
#         N = 2 * n_classes
#         mask = torch.ones(N, N)
#         mask.fill_diagonal_(0)
#         for i in range(n_classes):
#             mask[i, i + n_classes] = 0
#             mask[i + n_classes, i] = 0
#         mask = mask.bool()
#         return mask

#     def normalizeLabel(self, c_i, c_j):
#         c_i = torch.square(c_i)
#         c_j = torch.square(c_j)
#         p_i = c_i.sum(dim=0).view(-1)
#         c_i /= p_i
#         p_i = c_i.sum(dim=1).view(-1)
#         c_i /= p_i.unsqueeze(1)
#         p_j = c_j.sum(dim=0).view(-1)
#         c_j /= p_j
#         p_j = c_j.sum(dim=1).view(-1)
#         c_j /= p_j.unsqueeze(1)
#         return c_i, c_j

#     def forward(self, c_i, c_j):
#         p_i = c_i.sum(dim=0).view(-1)
#         p_i /= p_i.sum()
#         neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
#         p_j = c_j.sum(0).view(-1)
#         p_j /= p_j.sum()
#         neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
#         neg_entropy_loss = neg_entropy_i + neg_entropy_j

#         c_i = c_i.t()
#         c_j = c_j.t()
#         N = 2 * self.n_classes
#         c = torch.cat((c_i, c_j), dim=0)

#         sim = self.similarity_f(c.unsqueeze(1),
#                                 c.unsqueeze(0)) / self.temperature

#         sim_i_j = torch.diag(sim, self.n_classes)
#         sim_j_i = torch.diag(sim, -self.n_classes)

#         positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         negative_clusters = sim[self.mask].reshape(N, -1)

#         labels = torch.zeros(N).to(positive_clusters.device).long()
#         logits = torch.cat((positive_clusters, negative_clusters), dim=-1)
#         loss = self.criterion(logits, labels)
#         loss /= N

#         return loss + 1.* neg_entropy_loss

# class GraphConsis(nn.Module):
#     def __init__(self, ) -> None:
#         super().__init__()

#     def forward(self, emb, graphWeight):
#         dist1 = torch.cdist(emb, emb, p=2)
#         dist1 = torch.div(dist1, torch.max(dist1))
#         return torch.mean((1 - dist1) * graphWeight)


# class GraphRecLoss(nn.Module):
#     def __init__(self, norm_val, pos_weight) -> None:
#         super().__init__()
#         self.norm_val = norm_val
#         self.pos_weight = pos_weight

#     def forward(self, emb, target):
#         # emb = F.normalize(emb, p=2, dim=1)
#         input = emb @ emb.T
#         logits = F.binary_cross_entropy_with_logits(input,
#                                                     target,
#                                                     pos_weight=self.pos_weight)
#         return self.norm_val * logits


# class MSELoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x, xbar):
#         return torch.square(x - xbar).mean(dim=1).mean()


# class ZINBLoss(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

#     def forward(self, x, mean, disp, pi=0, scale_factor=1.0, ridge_lambda=0.0):
#         '''
#         args: x, raw count, [N, hvgs]
#               scale_factor, [n,]
#         '''
#         eps = 1e-10
#         mean = (mean.T * scale_factor).T
#         if pi == 0:
#             pi = torch.tensor(0.0)

#         t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(
#             x + disp + eps)
#         t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
#             x * (torch.log(disp + eps) - torch.log(mean + eps)))
#         nb_final = t1 + t2

#         nb_case = nb_final - torch.log(1.0 - pi + eps)
#         zero_nb = torch.pow(disp / (disp + mean + eps), disp)
#         zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
#         result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

#         if ridge_lambda > 0:
#             ridge = ridge_lambda * torch.square(pi)
#             result += ridge

#         result = torch.mean(result)
#         return result
# class SCELoss(nn.Module):
#     """
#     Symmetric Cross Entropy Loss
#     更适合稀疏计数数据
    
#     SCE = α * H(p, q) + β * H(q, p)
#     其中 H(p,q) = -Σ p*log(q) 是交叉熵
#     """
#     def __init__(self, alpha=0.1, beta=1.0, num_classes=None):
#         """
#         Parameters:
#         -----------
#         alpha : float
#             标准CE的权重 H(target, pred)
#         beta : float
#             反向CE的权重 H(pred, target)
#         num_classes : int or None
#             如果是分类任务需要指定
#         """
#         super(SCELoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.num_classes = num_classes
#         self.eps = 1e-10  # 数值稳定性
    
#     def forward(self, pred, target):
#         """
#         Parameters:
#         -----------
#         pred :  torch.Tensor
#             预测值 [batch_size × n_features]
#         target : torch.Tensor
#             真实值 [batch_size × n_features]
        
#         Returns:
#         --------
#         loss : torch.Tensor
#             SCE损失
#         """
#         # 确保值在合理范围内
#         pred = torch.clamp(pred, min=self.eps, max=1.0)
#         target = torch.clamp(target, min=self.eps, max=1.0)
        
#         # 归一化到概率分布（沿特征维度）
#         pred = pred / (pred.sum(dim=1, keepdim=True) + self.eps)
#         target = target / (target.sum(dim=1, keepdim=True) + self.eps)
        
#         # 标准交叉熵 H(target, pred)
#         ce = -torch.sum(target * torch.log(pred + self.eps), dim=1)
        
#         # 反向交叉熵 H(pred, target)
#         rce = -torch.sum(pred * torch.log(target + self.eps), dim=1)
        
#         # 对称交叉熵
#         sce = self.alpha * ce + self.beta * rce
        
#         return sce.mean()


# class ReconstructionLoss(nn.Module):
#     """
#     重构损失的统一接口
#     支持MSE和SCE
#     """
#     def __init__(self, loss_type='sce', **kwargs):
#         """
#         Parameters:
#         -----------
#         loss_type : str
#             'mse' 或 'sce'
#         **kwargs : dict
#             传递给具体loss的参数
#         """
#         super(ReconstructionLoss, self).__init__()
#         self.loss_type = loss_type
        
#         if loss_type == 'mse':
#             self.loss_fn = nn.MSELoss()
#         elif loss_type == 'sce':
#             self.loss_fn = SCELoss(
#                 alpha=kwargs.get('alpha', 0.1),
#                 beta=kwargs.get('beta', 1.0)
#             )
#         else:
#             raise ValueError(f"Unknown loss type: {loss_type}")
    
#     def forward(self, pred, target):
#         """
#         Parameters: 
#         -----------
#         pred : torch.Tensor
#             重构的基因表达
#         target : torch.Tensor
#             真实的基因表达
        
#         Returns:
#         --------
#         loss : torch.Tensor
#         """
#         if self.loss_type == 'sce':
#             # SCE需要非负值
#             pred = F.softplus(pred)  # 确保非负
#             # 或者用 sigmoid:  pred = torch.sigmoid(pred)
        
#         return self.loss_fn(pred, target)
# spCLUE/loss.py
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
        denominator = torch.clamp(negScores - posScores, min=eps)
    
        # return -torch.log(posScores / denominator).mean()
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

        return loss, 1.* neg_entropy_loss
    
class LocalAggregationContrastiveLoss(nn.Module):
    """
    局部聚合级对比学习损失
    
    拉近两个视图中同一Spot的AvgReadout（邻居平均）
    推远随机选择的Spot的AvgReadout
    """
    def __init__(self, temperature=0.2, n_negative_samples=256):
        super(LocalAggregationContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.n_negative_samples = n_negative_samples
    
    def compute_avg_readout(self, z, adj):
        """
        计算每个spot的局部聚合嵌入（邻居平均）
        
        Args:
            z: [n_spots × embed_dim] 嵌入
            adj: [n_spots × n_spots] 邻接矩阵（稀疏）
        
        Returns:
            avg_z: [n_spots × embed_dim] 平均聚合嵌入
        """
        # 计算度矩阵
        degrees = torch.sparse.sum(adj, dim=1).to_dense()  # [n_spots]
        degrees = degrees.unsqueeze(1)  # [n_spots × 1]
        
        # 聚合邻居
        neighbor_sum = torch.spmm(adj, z)  # [n_spots × embed_dim]
        
        # 平均（避免除以0）
        avg_z = neighbor_sum / (degrees + 1e-8)
        
        return avg_z
    
    def forward(self, z1, z2, adj1, adj2):
        """
        Args:
            z1: [n_spots × embed_dim] 空间视图嵌入
            z2: [n_spots × embed_dim] 表达视图嵌入
            adj1: 空间图邻接矩阵
            adj2: 表达图邻接矩阵
        
        Returns:
            loss: 标量
        """
        n_spots = z1.size(0)
        
        # 🔥 计算局部聚合嵌入（AvgReadout）
        avg_z1 = self.compute_avg_readout(z1, adj1)  # [n_spots × embed_dim]
        avg_z2 = self.compute_avg_readout(z2, adj2)
        
        # L2归一化
        avg_z1 = F.normalize(avg_z1, p=2, dim=1)
        avg_z2 = F.normalize(avg_z2, p=2, dim=1)
        
        # 🔥 对比学习：同一spot的avg_z1和avg_z2为正样本
        # 计算正样本相似度
        pos_sim = (avg_z1 * avg_z2).sum(dim=1) / self.temperature  # [n_spots]
        
        # 🔥 采样负样本：随机选择其他spot的avg_z2
        if self.n_negative_samples >= n_spots:
            # 使用所有spot作为负样本
            neg_sim = torch.mm(avg_z1, avg_z2.t()) / self.temperature  # [n_spots × n_spots]
        else:
            # 随机采样负样本
            neg_indices = torch.randint(0, n_spots, (n_spots, self.n_negative_samples), device=z1.device)
            neg_z2 = avg_z2[neg_indices]  # [n_spots × n_negative_samples × embed_dim]
            neg_sim = (avg_z1.unsqueeze(1) * neg_z2).sum(dim=2) / self.temperature  # [n_spots × n_negative_samples]
        
        # InfoNCE损失
        pos_exp = torch.exp(pos_sim)  # [n_spots]
        
        if self.n_negative_samples >= n_spots:
            neg_exp_sum = torch.exp(neg_sim).sum(dim=1)  # [n_spots]
        else:
            neg_exp_sum = torch.exp(neg_sim).sum(dim=1)  # [n_spots]
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8)).mean()
        
        return loss
    
# ==================== 🔥 新增：原型对比学习损失 ====================
from fast_pytorch_kmeans import KMeans
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, n_clusters, embed_dim, device, temperature=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.device = device
        self.temperature = temperature
        self.kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        self.prototypes = None # 只有一组共享的原型

    def forward(self, z_spatial, z_expr, epoch=None):
        # 1.归一化特征 (非常重要，否则 K-Means 基于欧氏距离会失效)
        z_spatial = F.normalize(z_spatial, dim=1)
        z_expr = F.normalize(z_expr, dim=1)

        # 2.只有在需要更新原型时才运行 K-Means
        # 为了保证一致性，我们把两个视图拼起来一起聚类，或者只对表达视图聚类（表达视图通常语义更强）
        if self.prototypes is None or (epoch % 10 == 0): # 简化逻辑：每10轮或未初始化时更新
            with torch.no_grad():
                # 策略：将两个视图的特征平均，或者拼接，找到共有的几何中心
                # 这里采用拼接策略，让原型存在于共享空间
                z_all = torch.cat([z_spatial, z_expr], dim=0)
                
                # 运行 K-Means
                self.kmeans.fit(z_all)
                self.prototypes = self.kmeans.centroids.to(self.device) # [K, D]
                
                # 归一化原型
                self.prototypes = F.normalize(self.prototypes, dim=1)

        # 3.计算 ProtoNCE Loss (InfoNCE)
        # 目标：z_spatial 应该属于它在 K-Means 中被分配的那个 Cluster
        # 但是！为了梯度回传，我们不能用 K-Means 的硬标签，我们用"另一个视图"作为正样本的引力
        
        # 计算 Logits: [N, K]
        logits_spa = torch.mm(z_spatial, self.prototypes.t()) / self.temperature
        logits_expr = torch.mm(z_expr, self.prototypes.t()) / self.temperature

        # 4.这里的关键是：谁是 Target？
        # PCL 的标准做法是：用 E-step 算出的聚类结果作为 Target
        with torch.no_grad():
            # 重新计算当前 batch 每个点属于哪个原型 (Hard Assignment)
            # 或者使用之前 fit 时的 labels，但为了简单直接实时算最近的原型
            # 两个视图应该属于同一个原型吗？理想情况下是的。
            # 我们用 z_expr 的聚类结果作为 z_spatial 的标签，反之亦然，或者用共同的聚类结果
            
            # 简单做法：每个点属于它在这个空间中最近的原型
            tgt_spa = torch.argmax(logits_spa, dim=1)
            tgt_expr = torch.argmax(logits_expr, dim=1)

        # 计算 Cross Entropy Loss
        # 强迫 z_spatial 预测出它属于哪个原型
        loss_spa = F.cross_entropy(logits_spa, tgt_spa)
        loss_expr = F.cross_entropy(logits_expr, tgt_expr)
        
        # 强迫两个视图的一致性 (可选，更强的约束)：
        # 要求 z_spatial 也能预测出 z_expr 所属的原型
        loss_cross_1 = F.cross_entropy(logits_spa, tgt_expr)
        loss_cross_2 = F.cross_entropy(logits_expr, tgt_spa)

        return (loss_spa + loss_expr + loss_cross_1 + loss_cross_2) / 4, True


# 🔥 新增：MSE重构损失
class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_rec, x_target):
        """
        Args:
            x_rec: 重构的表达 [n_spots × n_genes]
            x_target: 真实的表达 [n_spots × n_genes]
        """
        return torch.square(x_rec - x_target).mean()


# 🔥 新增：对称交叉熵损失（适合稀疏数据）
# 在loss.py末尾添加SCE Loss类

class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss
    适用于稀疏数据的重构损失
    """
    def __init__(self, alpha=0.1, beta=1.0, eps=1e-8):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, x_rec, x_target):
        """
        Args:
            x_rec: [n_samples × n_features] 重构的特征
            x_target: [n_samples × n_features] 真实特征
        
        Returns:
            loss: 标量
        """
        # 确保非负并归一化
        x_rec = torch.clamp(x_rec, min=self.eps)
        x_target = torch.clamp(x_target, min=self.eps)
        
        x_rec_norm = x_rec / (x_rec.sum(dim=1, keepdim=True) + self.eps)
        x_target_norm = x_target / (x_target.sum(dim=1, keepdim=True) + self.eps)
        
        # Cross Entropy: -Σ target·log(rec)
        ce = -torch.sum(x_target_norm * torch.log(x_rec_norm + self.eps), dim=1)
        
        # Reverse Cross Entropy: -Σ rec·log(target)
        rce = -torch.sum(x_rec_norm * torch.log(x_target_norm + self.eps), dim=1)
        
        # Symmetric CE
        loss = (self.alpha * ce + self.beta * rce).mean()
        
        return loss


# 🔥 新增：置信度加权的聚类对齐损失
# class ConfidenceWeightedClusterLoss(nn.Module):
#     """
#     基于置信度的聚类对齐损失
#     在高置信度区域强对齐，低置信度区域弱对齐
#     """
#     def __init__(
#         self,
#         n_classes,
#         weight_strategy='min',  # 'min', 'avg', 'max'
#         temperature=1.0,
#         eps=1e-8
#     ):
#         super(ConfidenceWeightedClusterLoss, self).__init__()
#         self.n_classes = n_classes
#         self.weight_strategy = weight_strategy
#         self.temperature = temperature
#         self.eps = eps
        
#         # 最大熵（用于归一化）
#         self.max_entropy = math.log(n_classes)
    
#     def compute_entropy(self, Q):
#         """
#         计算熵 H(Q) = -Σ Q·log(Q)
        
#         Args:
#             Q: [n_spots × n_clusters] 聚类概率分布
        
#         Returns:
#             H:  [n_spots] 每个spot的熵
#         """
#         # 避免log(0)
#         Q_safe = torch.clamp(Q, min=self.eps)
        
#         # H = -Σ P·log(P)
#         entropy = -(Q_safe * torch.log(Q_safe)).sum(dim=1)
        
#         return entropy
    
#     def compute_confidence(self, Q):
#         """
#         计算置信度 = 1 - H/H_max
        
#         Returns:
#             conf: [n_spots] 每个spot的置信度，范围[0, 1]
#         """
#         H = self.compute_entropy(Q)
#         conf = 1.0 - H / self.max_entropy
#         return conf
    
#     def compute_alignment_weight(self, conf1, conf2):
#         """
#         根据两个视图的置信度计算对齐权重
        
#         Args:
#             conf1, conf2: [n_spots] 两个视图的置信度
        
#         Returns:
#             weight: [n_spots] 对齐权重
#         """
#         if self.weight_strategy == 'min': 
#             # 保守策略：只在两个都确定时强对齐
#             weight = torch.min(conf1, conf2)
#         elif self.weight_strategy == 'avg':
#             # 平衡策略
#             weight = (conf1 + conf2) / 2.0
#         elif self.weight_strategy == 'max':
#             # 激进策略：只要有一个确定就强对齐
#             weight = torch.max(conf1, conf2)
#         else:
#             raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")
        
#         return weight*0.9 + 0.1 # 避免权重为0
    
#     def forward(self, Q1, Q2):
#         """
#         Args:
#             Q1: [n_spots × n_clusters] 视图1的聚类分配
#             Q2: [n_spots × n_clusters] 视图2的聚类分配
        
#         Returns:
#             loss: 标量
#         """
#         # 1.计算两个视图的置信度
#         conf1 = self.compute_confidence(Q1)
#         conf2 = self.compute_confidence(Q2)

#         entropy1 = self.compute_entropy(Q1)
#         entropy2 = self.compute_entropy(Q2)
        
#         # 2.计算对齐权重
#         weight = self.compute_alignment_weight(conf1, conf2)
        
#         # 3.计算加权的对齐损失（MSE）
#         alignment_loss = torch.square(Q1 - Q2).sum(dim=1)  # [n_spots]
#         weighted_loss = (weight * alignment_loss).mean()
        
#         # 4.负熵正则（鼓励均匀的聚类分布，防止所有点聚到一个类）
#         p1 = Q1.sum(dim=0) / (Q1.sum() + self.eps)
#         p2 = Q2.sum(dim=0) / (Q2.sum() + self.eps)
        
#         neg_entropy1 = self.max_entropy + (p1 * torch.log(p1 + self.eps)).sum()
#         neg_entropy2 = self.max_entropy + (p2 * torch.log(p2 + self.eps)).sum()
#         neg_entropy_loss = (neg_entropy1 + neg_entropy2) / 2.0

#         entropy_penalty = (entropy1.mean() + entropy2.mean()) / 2.0
        
#         # return weighted_loss + 0.5 * neg_entropy_loss + 0.05* entropy_penalty
#         return alignment_loss, weighted_loss,  neg_entropy_loss,  entropy_penalty
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpotLevelWeightedClusterLoss(nn.Module):
    def __init__(self, n_classes, temperature=1.0, eps=1e-8):
        super().__init__()
        self.n_classes = n_classes
        self.max_entropy = math.log(n_classes)
        self.eps = eps
        self.temperature = temperature # 这里的 temperature 主要用于缩放 logits，如果输入已经是 Q 则不用

    def compute_entropy(self, Q):
        """计算逐个 Spot 的熵"""
        Q_safe = torch.clamp(Q, min=self.eps)
        entropy = -(Q_safe * torch.log(Q_safe)).sum(dim=1)
        return entropy

    def compute_confidence(self, Q):
        """计算置信度 [0, 1]"""
        H = self.compute_entropy(Q)
        conf = 1.0 - H / self.max_entropy
        return torch.clamp(conf, 0.0, 1.0)

    def forward(self, Q1, Q2):
        """
        Args:
            Q1, Q2: [n_spots, n_classes] 经过 Softmax 的概率分布
        """
        # ---------------------------
        # 1.计算置信度和权重 (Spot Level)
        # ---------------------------
        conf1 = self.compute_confidence(Q1)
        conf2 = self.compute_confidence(Q2)
        
        entropy1 = self.compute_entropy(Q1)
        entropy2 = self.compute_entropy(Q2)

        # 策略：只有两个视图都自信时，才强迫对齐
        # 加上地板 0.1，避免梯度完全消失，但边界处给予极大宽容
        weight = torch.min(conf1, conf2)
        weight = weight * 0.7 + 0.3 
        
        # ---------------------------
        # 2.计算加权对齐损失 (KL Divergence)
        # ---------------------------
        # KL(P||Q) = sum(P * log(P/Q))
        # 我们使用对称 KL 或 MSE 均可，这里用 KL 更符合分布对齐的物理意义
        
        Q1_safe = torch.clamp(Q1, min=self.eps)
        Q2_safe = torch.clamp(Q2, min=self.eps)
        
        # KL(Q1 || Q2)
        kl_12 = (Q1_safe * (torch.log(Q1_safe) - torch.log(Q2_safe))).sum(dim=1)
        # KL(Q2 || Q1)
        kl_21 = (Q2_safe * (torch.log(Q2_safe) - torch.log(Q1_safe))).sum(dim=1)
        
        # 对称 KL
        alignment_raw = (kl_12 + kl_21) / 2.0
        
        # 🔥🔥🔥 关键：直接在 Spot 级别加权
        # 边界点 weight ≈ 0.1 -> Loss 很小 -> 梯度很小 -> 重构任务主导 -> 细节保留
        weighted_alignment_loss = (weight * alignment_raw).mean()

        # ---------------------------
        # 3.熵惩罚 (Entropy Penalty) - 锐化
        # ---------------------------
        # 强迫模型不要输出均匀分布，打破 Loss=0 的僵局
        # 这一项不加权！无论是否边界，我们都希望模型尽可能给出一个明确的判断
        entropy_penalty = (entropy1.mean() + entropy2.mean()) / 2.0

        # ---------------------------
        # 4.负熵正则 (Diversity) - 防坍塌
        # ---------------------------
        # 保证 Batch 内的 Class 分布是均匀的
        p1_mean = Q1.mean(dim=0)
        p2_mean = Q2.mean(dim=0)
        
        # neg_entropy_1 = - (p1_mean * torch.log(p1_mean + self.eps)).sum()
        # neg_entropy_2 = - (p2_mean * torch.log(p2_mean + self.eps)).sum()
        
        # # 我们希望 batch 级的熵越大越好 (均匀分布)，所以最小化 -H
        # diversity_loss = - (neg_entropy_1 + neg_entropy_2) / 2.0
        # 🔥 修正：改为正数形式（与原始Cluster Loss一致）
        # batch级别的熵
        entropy_batch_1 = -(p1_mean * torch.log(p1_mean + self.eps)).sum()
        entropy_batch_2 = -(p2_mean * torch.log(p2_mean + self.eps)).sum()
        avg_entropy_batch = (entropy_batch_1 + entropy_batch_2) / 2.0
        
        # diversity_loss = 最大熵 - 当前熵
        # 当分布均匀时（entropy ≈ log(K)）：diversity_loss ≈ 0
        # 当分布不均匀时（entropy << log(K)）：diversity_loss > 0，鼓励变均匀
        diversity_loss = self.max_entropy - avg_entropy_batch
        
        # ---------------------------
        # 5.总损失聚合
        # ---------------------------
        # 建议系数：
        # Alignment: 1.0 (主任务)
        # Entropy Penalty: 2.0 (必须够大，防止摆烂)
        # Diversity: 1.0 (辅助)
        
        total_loss = weighted_alignment_loss
        
        return total_loss, alignment_raw, weighted_alignment_loss, diversity_loss, entropy_penalty
# spCLUE/loss.py

# 🔥 修复：泊松损失
class PoissonLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood
    适合计数数据
    """
    def __init__(self):
        super(PoissonLoss, self).__init__()
        self.eps = 1e-10

    def forward(self, x_rec, x_target):
        """
        Args:  
            x_rec: 重构的λ (mean) [n_spots × n_genes]
            x_target: 真实的计数 [n_spots × n_genes]
        """
        # 🔥 确保在同一设备
        x_target = x_target.to(x_rec.device)
        
        # 确保λ > 0
        x_rec = torch.clamp(x_rec, min=self.eps)
        
        # Poisson NLL:   -log P(k|λ) = λ - k*log(λ) + log(k!)
        # 忽略常数项 log(k!)
        loss = x_rec - x_target * torch.log(x_rec + self.eps)
        
        return loss.mean()


# 🔥 修复：负二项分布损失
class NBLoss(nn.Module):
    """
    Negative Binomial Loss
    适合过度分散的计数数据
    """
    def __init__(self):
        super(NBLoss, self).__init__()
        self.eps = 1e-10
        # 可学习的分散参数
        self.theta = nn.Parameter(torch.ones(1))

    def forward(self, x_rec, x_target):
        """
        Args:
            x_rec: 重构的mean [n_spots × n_genes]
            x_target: 真实的计数 [n_spots × n_genes]
        """
        # 🔥 确保在同一设备
        x_target = x_target.to(x_rec.device)
        
        mu = torch.clamp(x_rec, min=self.eps)
        theta = torch.clamp(self.theta, min=self.eps)
        
        # NB log-likelihood
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(x_target + 1.0) - torch.lgamma(x_target + theta + self.eps)
        t2 = (theta + x_target) * torch.log(1.0 + (mu / (theta + self.eps))) + (x_target * (torch.log(theta + self.eps) - torch.log(mu + self.eps)))
        
        loss = t1 + t2
        return loss.mean()


# 🔥 修复：零膨胀负二项分布损失
class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss
    处理dropout和过度分散
    """
    def __init__(self):
        super(ZINBLoss, self).__init__()
        self.eps = 1e-10
        # 可学习的分散参数
        self.theta = nn.Parameter(torch.ones(1))

    def forward(self, x_rec, x_target, pi=None):
        """
        Args:
            x_rec: 重构的mean [n_spots × n_genes]
            x_target: 真实的计数 [n_spots × n_genes]
            pi: dropout概率 [n_spots × n_genes]，如果为None则不使用
        """
        # 🔥 确保在同一设备
        x_target = x_target.to(x_rec.device)
        
        mu = torch.clamp(x_rec, min=self.eps)
        theta = torch.clamp(self.theta, min=self.eps)
        
        if pi is None:
            # 🔥 修复：pi也要在正确的设备上
            pi = torch.zeros_like(mu, device=x_rec.device)
        else:
            pi = pi.to(x_rec.device)
            pi = torch.clamp(pi, min=self.eps, max=1-self.eps)
        
        # NB部分
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(x_target + 1.0) - torch.lgamma(x_target + theta + self.eps)
        t2 = (theta + x_target) * torch.log(1.0 + (mu / (theta + self.eps))) + (x_target * (torch.log(theta + self.eps) - torch.log(mu + self.eps)))
        nb_loss = t1 + t2
        
        # Zero-inflation部分
        nb_case = nb_loss - torch.log(1.0 - pi + self.eps)
        zero_nb = torch.pow(theta / (theta + mu + self.eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + self.eps)
        
        result = torch.where(torch.le(x_target, 1e-8), zero_case, nb_case)
        
        return result.mean()


# 🔥 新增：统一的重构损失接口
class ReconstructionLoss(nn.Module):
    """
    统一的重构损失接口
    支持多种损失类型
    """
    def __init__(self, loss_type='mse', **kwargs):
        """
        Args:
            loss_type: 'mse', 'sce', 'poisson', 'nb', 'zinb'
            **kwargs: 传递给具体损失的参数
        """
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = MSELoss()
        elif loss_type == 'sce':
            self.loss_fn = SCELoss(
                alpha=kwargs.get('alpha', 0.1),
                beta=kwargs.get('beta', 1.0)
            )
        elif loss_type == 'poisson':
            self.loss_fn = PoissonLoss()
        elif loss_type == 'nb':
            self.loss_fn = NBLoss()
        elif loss_type == 'zinb':
            self.loss_fn = ZINBLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        print(f"✅ 使用 {loss_type.upper()} 重构损失")

    def forward(self, x_rec, x_target, **kwargs):
        """
        Args:
            x_rec:  重构的表达
            x_target: 真实的表达
            **kwargs:  额外参数（如ZINB的pi）
        """
        if self.loss_type == 'zinb':
            return self.loss_fn(x_rec, x_target, pi=kwargs.get('pi', None))
        else:
            return self.loss_fn(x_rec, x_target)


# 🔥 新增：批次质心对齐损失
class CentroidAlignmentLoss(nn.Module):
    """
    批次质心对齐损失
    显存友好的跨批次对齐方法
    """
    def __init__(self, alignment_type='variance'):
        """
        Args:
            alignment_type: 'variance' 或 'pairwise'
                - variance: 最小化所有批次质心的方差
                - pairwise: 最小化批次对之间的距离
        """
        super(CentroidAlignmentLoss, self).__init__()
        self.alignment_type = alignment_type

    def forward(self, z, batch_list):
        """
        Args: 
            z: [n_spots × embed_dim] 嵌入
            batch_list: [n_spots] 批次标签
        """
        unique_batches = torch.unique(batch_list)
        n_batches = len(unique_batches)
        
        if n_batches <= 1:
            return torch.tensor(0.0, device=z.device)
        
        # 🔥 计算每个批次的质心
        centroids = []
        for batch_id in unique_batches:
            mask = (batch_list == batch_id)
            centroid = z[mask].mean(dim=0)
            centroids.append(centroid)
        
        centroids = torch.stack(centroids, dim=0)  # [n_batches × embed_dim]
        
        if self.alignment_type == 'variance':
            # 方法1: 最小化批次质心的方差
            # 让所有批次的质心尽可能接近
            loss = torch.var(centroids, dim=0).mean()
        else:
            # 方法2: 最小化批次对之间的距离
            loss = 0.0
            n_pairs = 0
            for i in range(n_batches):
                for j in range(i + 1, n_batches):
                    dist = torch.norm(centroids[i] - centroids[j], p=2)
                    loss += dist
                    n_pairs += 1
            loss = loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=z.device)
        
        return loss


# 🔥 新增：软聚类对齐损失（优化版）
class SoftClusterAlignmentLoss(nn.Module):
    """
    基于聚类分配概率的软对齐损失
    向量化 + 可选采样加速
    """
    def __init__(self, sample_size=None):
        """
        Args: 
            sample_size: 如果不为None，每个批次对最多采样sample_size个点
        """
        super(SoftClusterAlignmentLoss, self).__init__()
        self.sample_size = sample_size

    def forward(self, z, Q, batch_list):
        """
        Args: 
            z: [n_spots × embed_dim] 嵌入
            Q: [n_spots × n_clusters] 聚类分配概率
            batch_list: [n_spots] 批次标签
        """
        unique_batches = torch.unique(batch_list)
        n_batches = len(unique_batches)
        
        if n_batches <= 1:
            return torch.tensor(0.0, device=z.device)
        
        total_loss = 0.0
        n_pairs = 0
        
        for i in range(n_batches):
            for j in range(i + 1, n_batches):
                batch_i = unique_batches[i]
                batch_j = unique_batches[j]
                
                mask_i = (batch_list == batch_i)
                mask_j = (batch_list == batch_j)
                
                z_i = z[mask_i]
                z_j = z[mask_j]
                Q_i = Q[mask_i]
                Q_j = Q[mask_j]
                
                # 🔥 可选采样
                if self.sample_size is not None:
                    n_i, n_j = z_i.size(0), z_j.size(0)
                    if n_i > self.sample_size: 
                        idx_i = torch.randperm(n_i, device=z.device)[:self.sample_size]
                        z_i = z_i[idx_i]
                        Q_i = Q_i[idx_i]
                    if n_j > self.sample_size:
                        idx_j = torch.randperm(n_j, device=z.device)[:self.sample_size]
                        z_j = z_j[idx_j]
                        Q_j = Q_j[idx_j]
                
                # 计算相似度矩阵（基于聚类）
                similarity = torch.mm(Q_i, Q_j.T)  # [n_i × n_j]
                
                # 计算距离矩阵
                dist = torch.cdist(z_i, z_j, p=2)  # [n_i × n_j]
                
                # 归一化距离
                if dist.max() > 0:
                    dist = dist / dist.max()
                
                # 加权损失：相似度高的应该距离近
                loss = (similarity * dist).mean()
                
                total_loss += loss
                n_pairs += 1
        
        return total_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0, device=z.device)