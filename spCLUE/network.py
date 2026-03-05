import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.functional import normalize


class NoiseLayer(nn.Module):

    def __init__(self, alpha=0.01, dropout=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.drop = dropout

    def forward(self, x):
        gauss_x = x + self.alpha * torch.randn_like(x)
        return F.dropout(gauss_x, self.drop, training=self.training)


class AttentionBlock(nn.Module):

    def __init__(self, in_size, hidden_size=16):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        # shape of z: [N, n_vision, d]
        w = self.project(z)  # attention weight of each vision, shape: [N, n_vision, 1]
        beta = torch.softmax(w, dim=1)  # [N, n_vision, 1]
        return (beta * z).sum(1), beta

# STCF CAM 核心逻辑
# class STCFAttentionBlock(nn.Module):
#     def __init__(self, in_size):
#         super().__init__()
#         # W 直接投影到 3 个视图的权重空间 (公式 15)
#         self.W = nn.Linear(in_size * 3, 3) 
#         self.F_l = nn.Linear(in_size * 3, in_size) # 最终融合层 (公式 16)

#     def forward(self, z_spatial, z_feature, z_combined):
#         E_prime = torch.cat([z_spatial, z_feature, z_combined], dim=1) # [N, 3*d]
        
#         # 计算共享注意力信号 U 并进行 L2 归一化 (公式 15)
#         # 这里的 U 实际上就包含了 scalar attention weights (u1, u2, u3)
#         # 修改为 Softmax
#         # weights = F.softmax(self.W(E_prime), dim=1) 
#         weights = F.normalize(self.W(E_prime), p=2, dim=1) # [N, 3]
        
#         # 这里的权重通常需要通过 sigmoid 或 abs 确保为正，或直接使用原值
#         # 论文提到使用这些权重 re-calibrate 嵌入矩阵 (公式 16)
#         u1, u2, u3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
#         weighted_concat = torch.cat([u1 * z_spatial, u2 * z_feature, u3 * z_combined], dim=1)
#         return self.F_l(weighted_concat), weights
import torch
import torch.nn as nn
import torch.nn.functional as F

class STCFAttentionBlock(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        # 【修改点 1】W 不再是投影到 3 个标量，而是对每个视图的特征独立进行线性映射
        # 对应 MAFN 源码中的 MLP_L (Linear(64, 64))
        self.wl = nn.Linear(in_size, in_size) 
        
        # 最终融合层，对应 MAFN 源码中的 self.MLP (Linear(192, 64))
        self.F_l = nn.Linear(in_size * 3, in_size) 

    def forward(self, z_spatial, z_feature, z_combined):
        # 【修改点 2】不再使用拼接 (cat)，而是使用堆叠 (stack)
        # 对应源码：emb = torch.stack([emb1, com, emb2], dim=1)
        # 形状变为 [N, 3, in_size]
        stacked_emb = torch.stack([z_spatial, z_combined, z_feature], dim=1) 
        
        # 【修改点 3】计算特征级注意力权重
        # self.wl 作用于 [N, 3, in_size] 上，相当于对 3 个视图独立做 in_size -> in_size 的映射
        # 对应源码：a = self.MLP_L(emb)
        raw_weights = self.wl(stacked_emb) # 形状 [N, 3, in_size]
        
        # 【修改点 4】在视图维度 (dim=1) 上进行 L2 归一化
        # 这意味着对于每一个特征维度（共 in_size 个），都会在 3 个视图间分配 L2 归一化的权重
        # 对应源码：emb = F.normalize(a, p=2)
        weights = F.normalize(raw_weights, p=2, dim=1) # 形状 [N, 3, in_size]
        
        # 【修改点 5】提取特征级权重并进行逐元素相乘 (element-wise multiplication)
        # 此时的权重 weights[:, 0] 形状是 [N, in_size]，它不再是标量，而是一个特征向量
        u1 = weights[:, 0, :] # 空间图的特征级权重
        u2 = weights[:, 1, :] # 联合图的特征级权重
        u3 = weights[:, 2, :] # 特征图的特征级权重
        
        # 对应源码：emb = torch.cat((emb[:, 0].mul(emb1), emb[:, 1].mul(com), emb[:, 2].mul(emb2)), 1)
        weighted_concat = torch.cat([
            u1 * z_spatial,   # 特征级逐元素相乘
            u2 * z_combined,  # 注意 MAFN 源码中间放的是联合图 (com)
            u3 * z_feature
        ], dim=1) # 拼接后形状 [N, 3 * in_size]
        
        # 对应源码：emb = self.MLP(emb)
        out = self.F_l(weighted_concat)
        
        return out, weights
class ZINBDecoder(nn.Module):
    def __init__(self, z_dim, output_dim, hidden_dim=128):
        super().__init__()
        
        # 共享特征提取层
        self.decoder_base = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 分叉输出三个参数：pi (零膨胀), disp (离散度), mean (均值)
        self.pi_layer = nn.Linear(hidden_dim, output_dim)
        self.disp_layer = nn.Linear(hidden_dim, output_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6) 

    def forward(self, z):
        """
        不再接收 library_size 参数
        """
        h = self.decoder_base(z)
        
        pi = torch.sigmoid(self.pi_layer(h))
        disp = self.DispAct(self.disp_layer(h))
        mean = self.MeanAct(self.mean_layer(h))
        
        return mean, disp, pi
# class CCGCN(Module):

#     def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5) -> None:
#         super(CCGCN, self).__init__()
#         """
#         Args: 
#             dims_list (list): dimensions of GCNs.
#             n_clusters (int): number of clusters in the cluster-contrastive module.
#             graph_corr (float): corruption probability of the graph. (Edge Dropout).
#             dropout (float): probability of the dropout of networks.
#         """
#         self.input_dim = dims_list[0]
#         self.hidden_dim = dims_list[1]
#         self.z_dim = dims_list[2]
#         self.dropout = dropout
#         self.n_clusters = n_clusters
#         self.graph_corr = graph_corr

#         ### + encoders
#         self.noiseLayer = NoiseLayer(dropout=self.dropout)
#         self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
#         self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

#         self.act = nn.ELU()
#         self.relu = nn.ReLU()
#         self.attention = AttentionBlock(self.z_dim)

#         ## + instance projection head
#         self.projectInsHead = nn.Sequential(
#             nn.Linear(self.z_dim, self.z_dim),
#             nn.ReLU(),
#             nn.Linear(self.z_dim, self.z_dim),
#             nn.ReLU(),
#         )

#         ## + cluster projection head
#         self.projectClsHead = nn.Sequential(
#             nn.Linear(self.z_dim, self.z_dim),
#             nn.ReLU(),
#             nn.Linear(self.z_dim, self.n_clusters),
#             nn.Softmax(dim=1),
#         )

        
#     def encoder(self, data, adj):
#         feature = self.noiseLayer(data)
#         adj1 = torch.sparse_coo_tensor(
#             adj._indices(),
#             F.dropout(adj._values(), p=self.graph_corr, training=self.training),
#             size=adj.size(),
#         )
#         feature = self.act(torch.spmm(adj1, self.Transform1(feature)))
#         adj2 = torch.sparse_coo_tensor(
#             adj._indices(),
#             F.dropout(adj._values(), p=self.graph_corr, training=self.training),
#             size=adj.size(),
#         )
#         feature = self.act(torch.spmm(adj2, self.Transform2(feature)))
#         return feature

#     def getCluster(self, embed):
#         labels = self.projectClsHead(embed)
#         return torch.argmax(labels, dim=1)

#     def forward(self, data, adj1, adj2, batch_onehot=None):
#         """
#         Args:
#             data (torch.FloatTensor): pca input of gene expression data.
#             adj1 (torch.sparse_coo_tensor): normalized spatial graph.
#             adj2 (torch.sparse_coo_tensor): normalized expr graph.
#         """
#         feature1 = self.encoder(data, adj1)
#         feature2 = self.encoder(data, adj2)

#         # + L2 normalization
#         z1_norm = normalize(feature1, p=2, dim=1)
#         z2_norm = normalize(feature2, p=2, dim=1)

#         # + instance projection
#         h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
#         h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)

#         # + cluster projection
#         label1 = self.projectClsHead(z1_norm)
#         label2 = self.projectClsHead(z2_norm)

#         # + attention fuse
#         z = torch.stack([z1_norm, z2_norm], dim=1)
#         z, att_beta = self.attention(z)
       
#         # z = 0.5 * z1_norm + 0.5 * z2_norm
#         # z = normalize(z, p=2, dim=1)

#         x_Rec = self.relu(z @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

#         # rec_1 = z1_norm @ self.Transform2.W.t()
#         # rec_2 = z2_norm @ self.Transform2.W.t()

#         # # 合并隐藏层特征
#         # rec_hidden = (rec_1 + rec_2) / 2  # 或者直接相加

#         # # 最后一层解码
#         # x_Rec = self.relu(rec_hidden) @ self.Transform1.W.t()
#         return h1_norm, h2_norm, z1_norm, z2_norm, z, att_beta, label1, label2, x_Rec
class CCGCN(Module):

    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5, use_zinb=False) -> None:
        super(CCGCN, self).__init__()
        """
        Enhanced CCGCN with STCF-style three-view architecture
        
        Args: 
            dims_list (list): dimensions of GCNs [input_dim, hidden_dim, z_dim].
            n_clusters (int): number of clusters in the cluster-contrastive module.
            graph_corr (float): corruption probability (DEPRECATED in STCF, kept for compatibility).
            dropout (float): dropout rate for projection heads (not used in encoders).
            use_zinb (bool): whether to use ZINB decoder.
        """
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr  # Not used in STCF deterministic encoding
        self.use_zinb = use_zinb
        self.dropout = dropout

        ### === 三个独立的确定性GCN编码器 (STCF Style) ===
        
        # 空间视图编码器 (Spatial View Encoder)
        self.W_s1 = TransForm_W(self.input_dim, self.hidden_dim)  # 无dropout
        self.W_s2 = TransForm_W(self.hidden_dim, self.z_dim)
        
        # 特征视图编码器 (Feature View Encoder)
        self.W_f1 = TransForm_W(self.input_dim, self.hidden_dim)  # 无dropout
        self.W_f2 = TransForm_W(self.hidden_dim, self.z_dim)
        
        # 组合视图编码器 (Combined View Encoder)
        self.W_c1 = TransForm_W(self.input_dim, self.hidden_dim)  # 无dropout
        self.W_c2 = TransForm_W(self.hidden_dim, self.z_dim)

        # 激活函数
        self.act = nn.ELU()  # 或者 nn.ReLU()
        self.relu = nn.ReLU()
        
        # STCF风格注意力融合模块
        self.attention = STCFAttentionBlock(self.z_dim)

        ### === Projection Heads (保留原spCLUE设计用于对比学习) ===
        
        ## Instance-level projection head (用于ContrastiveLoss)
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),  # 这里使用dropout
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )

        ## Cluster-level projection head (用于ClusterLoss)
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )
        
        ### === 可选的ZINB解码器 ===
        if self.use_zinb:
            self.zinb_decoder = ZINBDecoder(self.z_dim, self.input_dim)
        else:
            self.zinb_decoder = None

    def deterministic_gcn_encoder(self, X, A_hat, W1, W2):
        """
        两层GCN（对齐MAFN）
        """
        # 第一层
        support1 = W1(X)
        h1 = torch.spmm(A_hat, support1)
        h1 = F.relu(h1)
        if self.dropout > 0:
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # 第二层
        support2 = W2(h1)
        z = torch.spmm(A_hat, support2)
        
        return z

    def getCluster(self, embed):
        """从嵌入向量预测聚类标签"""
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj_spatial, adj_feature, adj_combined, 
                batch_onehot=None, library_size=None):
        """
        Forward pass with three-view encoding and STCF-style fusion
        
        Args:
            data: 输入特征矩阵 [N, input_dim]
            adj_spatial: 归一化空间图邻接矩阵 (sparse tensor)
            adj_feature: 归一化特征图邻接矩阵 (sparse tensor)
            adj_combined: 归一化组合图邻接矩阵 (sparse tensor)
            library_size: ZINB解码所需的文库大小 [N, 1]
            
        Returns:
            Tuple containing:
                h_spatial, h_feature: instance-level projections for ContrastiveLoss
                z_spatial, z_feature: raw embeddings for CCR Loss
                z_combined: combined view embedding
                z_fused: attention-fused embedding
                label_spatial, label_feature: cluster assignments for ClusterLoss
                x_rec: reconstruction output (MSE or ZINB)
                attention_weights: CAM attention weights
        """
        ### === Step 1: 三个视图的独立确定性编码 ===
        
        # 空间视图编码: E_s = GCN_spatial(X, A_s)
        E_spatial = self.deterministic_gcn_encoder(
            data, adj_spatial, self.W_s1, self.W_s2
        )
        
        # 特征视图编码: E_f = GCN_feature(X, A_f)
        E_feature = self.deterministic_gcn_encoder(
            data, adj_feature, self.W_f1, self.W_f2
        )
        
        # 组合视图编码: E_c = GCN_combined(X, A_c)
        E_combined = self.deterministic_gcn_encoder(
            data, adj_combined, self.W_c1, self.W_c2
        )

        ### === Step 2: L2归一化 (用于后续相似度计算) ===
        z_spatial = normalize(E_spatial, p=2, dim=1)
        z_feature = normalize(E_feature, p=2, dim=1)
        z_combined = normalize(E_combined, p=2, dim=1)

        ### === Step 3: Instance-level Projection (用于ContrastiveLoss) ===
        h_spatial = normalize(self.projectInsHead(z_spatial), p=2, dim=1)
        h_feature = normalize(self.projectInsHead(z_feature), p=2, dim=1)

        ### === Step 4: Cluster-level Projection (用于ClusterLoss) ===
        label_spatial = self.projectClsHead(z_spatial)
        label_feature = self.projectClsHead(z_feature)

        ### === Step 5: STCF风格跨视图注意力融合 ===
        z_fused, attention_weights = self.attention(z_spatial, z_feature, z_combined)

        ### === Step 6: 重构解码 ===
        if self.use_zinb and self.zinb_decoder is not None:
            # ZINB解码器内部会处理library_size
            mean, disp, pi = self.zinb_decoder(z_fused)
            x_rec = (mean, disp, pi)
        else:
            # 原始MSE重构 (使用空间视图的权重矩阵)
            x_rec = self.relu(z_fused @ self.W_s2.W.data.T) @ self.W_s1.W.data.T

        ### === 返回所有需要的中间结果 ===
        return (
            h_spatial,          # [N, z_dim] - for ContrastiveLoss
            h_feature,          # [N, z_dim] - for ContrastiveLoss
            z_spatial,          # [N, z_dim] - for CCR Loss (raw spatial embedding)
            z_feature,          # [N, z_dim] - for CCR Loss (raw feature embedding)
            z_combined,         # [N, z_dim] - combined view embedding
            z_fused,            # [N, z_dim] - final fused embedding
            label_spatial,      # [N, n_clusters] - for ClusterLoss
            label_feature,      # [N, n_clusters] - for ClusterLoss
            x_rec,              # reconstruction (MSE or ZINB tuple)
            attention_weights   # [N, 3] - CAM attention weights
        )

class CCGCNs(Module):

    def __init__(
        self,
        dims_list,
        n_clusters,
        n_batches,
        graph_corr=0.4,
        dropout=0.5,
        device="cuda:0"
    ) -> None:
        super(CCGCNs, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.graph_corr = graph_corr

        ### + encoders
        self.noiseLayer = NoiseLayer()
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.batchPortion = torch.eye(self.n_batches).to(device)
        self.weightBatch = 0.01

        if self.weightBatch != 0:
            self.batchEmbed = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.z_dim))
            )
            self.batchPCA = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.input_dim))
            )
        else:
            self.batchEmbed = torch.ones(self.n_batches, self.z_dim).to(device)
            self.batchPCA = torch.ones(self.n_batches, self.input_dim).to(device)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
        self.attention = AttentionBlock(self.z_dim)

        ## + instance projection head
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )
        ## + cluster projection head
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )

    def encoder(self, data, adj):
        feature = self.noiseLayer(data)
        adj1 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj1, self.Transform1(feature)))
        adj2 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj2, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj1, adj2, batch_list=None):
        """
        Args:
            data (torch.FloatTensor): pca input of the gene expression data.
            adj1 (torch.sparse_coo_tensor): normalized spatial graph.
            adj2 (torch.sparse_coo_tensor): normalized expr graph.
            batch_list (): list of batch ID.
        """
        batch_noise = self.batchPortion[batch_list] @ self.batchPCA
        data = data - self.weightBatch * batch_noise 
        feature1 = self.encoder(data, adj1)
        feature2 = self.encoder(data, adj2)

        # + L2 normalization
        z1_norm = normalize(feature1, p=2, dim=1)
        z2_norm = normalize(feature2, p=2, dim=1)

        # + instance projection
        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)

        # + cluster projection
        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)

        # + attention fuse
        z = torch.stack([z1_norm, z2_norm], dim=1)
        z, _ = self.attention(z)

        # + add batch embedding
        norm_batch_embed = normalize(self.batchEmbed, p=2, dim=1)
        z_dec = z + self.weightBatch * self.batchPortion[batch_list] @ norm_batch_embed

        x_Rec = self.relu(z_dec @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, z1_norm, z2_norm, z, x_Rec, label1, label2


class InnerProductDec(nn.Module):

    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj_rec = z @ z.T
        return adj_rec


class IdentityMap(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


# class TransForm_W(nn.Module):

#     def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
#         super().__init__()
#         self.dropout = dropout
#         self.W = nn.Parameter(
#             nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
#         )  ## = initialize weight of transform layer

#     def forward(self, x):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         return x @ self.W
class TransForm_W(nn.Module):
    """线性变换层（对齐MAFN的GraphConvolution）"""
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, out_dim)
        
        # MAFN风格的uniform初始化
        stdv = 1. / math.sqrt(out_dim)
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return self.linear(x)