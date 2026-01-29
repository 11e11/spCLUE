from inspect import Parameter
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


# 🔥 新增：改进的解码器（重构HVG）
class Decoder(nn.Module):
    """
    解码器:  从嵌入重构HVG表达
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        """
        Args:
            in_dim: 输入维度（embedding维度）
            hidden_dim:  隐藏层维度
            out_dim: 输出维度（HVG数量，如2000）
            dropout:  Dropout率
        """
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            # nn.BatchNorm1d(hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, out_dim)
            # 注意：不加激活函数，让输出可以是任意实数
            # 如果需要非负（如泊松、NB），在forward中加ReLU/Softplus
        )
    
    def forward(self, z, activation='none'):
        """
        Args: 
            z:  Embedding [n_spots × in_dim]
            activation: 'none', 'relu', 'softplus'
        
        Returns:
            x_rec: 重构的HVG表达 [n_spots × out_dim]
        """
        x_rec = self.decoder(z)
        
        if activation == 'relu':
            x_rec = F.relu(x_rec)
        elif activation == 'softplus':
            x_rec = F.softplus(x_rec)
        
        return x_rec


# class Decoder(nn.Module):
#     """
#     解码器: 使用原生 PyTorch (spmm) 实现单层 GCN
#     """
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
#         """
#         Args:
#             in_dim: 输入维度（embedding维度）
#             hidden_dim: 兼容性保留参数（单层结构中未使用）
#             out_dim: 输出维度（HVG数量）
#             dropout: Dropout率
#         """
#         super(Decoder, self).__init__()
        
#         # --- 变化 1: 使用普通 Linear 层代替 GCNConv ---
#         # GCN 的本质就是: A * (X * W)
#         # 这里定义 W (权重矩阵)
#         self.transform = nn.Linear(in_dim, out_dim)
        
#         self.dropout = dropout

#     def forward(self, z, adj, activation='none'):
#         """
#         Args: 
#             z: Embedding [n_spots × in_dim]
#             adj: 归一化的邻接矩阵 (Sparse Tensor) [n_spots × n_spots]
#                  必须是 torch.sparse_coo_tensor 类型
#             activation: 'none', 'relu', 'softplus'
        
#         Returns:
#             x_rec: 重构的表达 [n_spots × out_dim]
#         """
        
#         # 1. Dropout (通常在进入层之前)
#         z = F.dropout(z, p=self.dropout, training=self.training)
        
#         # --- 变化 2: "手写" GCN 逻辑 ---
#         # 步骤 A: 线性变换 (Feature Transformation) -> 对应公式里的 XW
#         z_trans = self.transform(z)
        
#         # 步骤 B: 邻居聚合 (Message Passing) -> 对应公式里的 A(XW)
#         # 使用稀疏矩阵乘法，将邻居特征聚合到节点上
#         x_rec = torch.spmm(adj, z_trans)
        
#         # --- 变化 3: 激活函数 ---
#         if activation == 'relu':
#             x_rec = F.relu(x_rec)
#         elif activation == 'softplus':
#             x_rec = F.softplus(x_rec)
        
#         return x_rec
class AttentionBlock(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(dim=1), beta

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CCGCN(nn.Module):
    """
    CCGCN (SpaMask 风格改造版)
    逻辑：
    1. 训练时：双视图增强 (EdgeDrop vs FeatureMask) -> 对比学习 & 掩码重构
    2. 推理时：使用干净的原始图 -> 提取高质量嵌入
    """
    def __init__(self, dims_list, n_clusters, graph_corr=0.2, dropout=0.5, node_corr=0.2) -> None:
        super(CCGCN, self).__init__()
        
        # dims_list: [200, 64, 16]
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.enc_z_dim = dims_list[2]
        
        # 最终嵌入维度 (这里不再是 concat，而是单分支维度，除非你保留 concat 结构)
        # 注意：原代码 encoder 输出是 concat(z_mlp, z_gcn) = 32 维
        self.z_dim = self.enc_z_dim * 2 
        
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr
        self.node_corr = node_corr

        # === 1. 编码器层 (保持不变) ===
        self.MLP1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.MLP2 = nn.Linear(self.hidden_dim, self.enc_z_dim)
        
        self.GCN1 = nn.Linear(self.enc_z_dim, self.hidden_dim)
        self.GCN2 = nn.Linear(self.hidden_dim, self.enc_z_dim)
        
        # === 2. 解码器层 ===
        # 输入是 remask 后的 32 维向量，输出回 200 维
        self.decoder = nn.Linear(self.z_dim, self.input_dim) 

        # === 其他组件 ===
        self.noiseLayer = NoiseLayer(dropout=self.dropout)
        self.act = nn.ELU()
        self.attention = AttentionBlock(self.z_dim)
        
        # 输入层的 Mask Token (用于 Encoder 输入)
        self.mask_token = nn.Parameter(torch.zeros(1, self.input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 🔥 中间层的 Remask Token (用于 Decoder 输入)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.z_dim))
        nn.init.xavier_normal_(self.dec_mask_token)

        # 投影头 (用于对比学习 Loss)
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )
        
        # [注意] 移除了 AttentionBlock，因为 SpaMask 逻辑不需要融合视图来做输出
        # 如果你有专门的 Contrastive Head 可以加在这里
        ## + instance projection head
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )
    
    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)
        
    def mask_features(self, x, mask_rate):
        n_spots = x.size(0)
        if not self.training or mask_rate == 0:
            return x, torch.zeros(n_spots, dtype=torch.bool, device=x.device)
        
        perm = torch.randperm(n_spots, device=x.device)
        num_mask = int(mask_rate * n_spots)
        mask_nodes = torch.zeros(n_spots, dtype=torch.bool, device=x.device)
        mask_nodes[perm[:num_mask]] = True
        
        masked_x = x.clone()
        masked_x[mask_nodes] = self.mask_token # 替换为 token
        return masked_x, mask_nodes

    def dropout_edge(self, adj, drop_rate):
        if not self.training or drop_rate == 0:
            return adj
        remaining_adj = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=drop_rate, training=True),
            size=adj.size(),
        )
        return remaining_adj

    def encoder(self, data, adj):
        # 保持原有的 MLP+GCN 双流结构
        # MLP 流
        h_mlp = self.noiseLayer(data)
        h_mlp = self.act(self.MLP1(h_mlp))
        h_mlp = F.dropout(h_mlp, p=self.dropout, training=self.training)
        z_mlp = self.act(self.MLP2(h_mlp))
        
        # GCN 流
        h_gcn = self.GCN1(z_mlp)
        h_gcn = torch.spmm(adj, h_gcn)
        h_gcn = self.act(h_gcn)
        h_gcn = F.dropout(h_gcn, p=self.dropout, training=self.training)
        
        h_gcn = self.GCN2(h_gcn)
        z_gcn = torch.spmm(adj, h_gcn)
        
        feature = torch.cat([z_mlp, z_gcn], dim=1) # [N, 32]
        return feature

    # def forward(self, data, adj_spatial, adj_expr=None, rec_activation='none'):
    #     """
    #     🔥 训练专用 Forward 🔥
    #     产生两个损坏的视图，计算 Loss 所需的组件
    #     """
    #     # === 1. 视图构建 (View Construction) ===
        
    #     # 视图 1: 结构扰动 (Structure View)
    #     # 目的：学习结构不变性
    #     adj_view1 = self.dropout_edge(adj_spatial, self.graph_corr)
    #     x_view1 = data # 特征完整
        
    #     # 视图 2: 特征掩码 (Feature View)
    #     # 目的：学习通过邻居恢复特征 (Context Reconstruction)
    #     adj_view2 = adj_spatial # 结构完整
    #     x_view2, mask_nodes = self.mask_features(data, self.node_corr)
        
    #     # === 2. 编码 (Encoding) ===
    #     # 共享 Encoder
    #     z_view1 = self.encoder(x_view1, adj_view1) # [N, 32]
    #     z_view2 = self.encoder(x_view2, adj_view2) # [N, 32]

    #     # === 3. 聚类预测 (用于 KL Loss 或 Contrastive Loss) ===
    #     # 对两个视图都进行归一化
    #     z_view1_norm = F.normalize(z_view1, p=2, dim=1)
    #     z_view2_norm = F.normalize(z_view2, p=2, dim=1)
        
    #     pred_view1 = self.projectClsHead(z_view1_norm)
    #     pred_view2 = self.projectClsHead(z_view2_norm)

    #     # === 4. 重构 (Reconstruction with Remask) ===
    #     # 关键修改：只对“被掩码”的视图 (View 2) 进行重构
    #     # 我们的目标是： 输入(Masked) -> Encoder -> Remask -> Decoder -> 原始特征
        
    #     h_recon_input = z_view2.clone()
        
    #     # [Remask 操作]
    #     # 在进入解码器前，再次把 mask 位置的嵌入抹掉，替换为 dec_mask_token
    #     # 强迫 Decoder 只能利用 adj_spatial 聚合未 Mask 的邻居信息
    #     if self.training and mask_nodes.sum() > 0:
    #         h_recon_input[mask_nodes] = self.dec_mask_token
            
    #     # 解码: Linear
    #     rec_h = self.decoder(h_recon_input)
        
    #     # 解码: GCN Aggregation (利用完整的空间结构来修复)
    #     x_rec = torch.spmm(adj_spatial, rec_h)
        
    #     if rec_activation == 'softplus':
    #         x_rec = F.softplus(x_rec)
    #     elif rec_activation == 'relu':
    #         x_rec = F.relu(x_rec)

    #     # 返回训练 Loss 所需的所有变量
    #     # 此时不需要 Attention 融合的 z，对比学习直接对比 z_view1 和 z_view2
    #     return (
    #         z_view1_norm,   # View 1 Embedding
    #         z_view2_norm,   # View 2 Embedding
    #         pred_view1,     # View 1 Cluster Prob
    #         pred_view2,     # View 2 Cluster Prob
    #         adj_view1,      # View 1 Adjacency
    #         adj_view2,      # View 2 Adjacency
    #         x_rec,          # 重构结果
    #         mask_nodes      # Mask 索引
    #     )
    def forward(self, data, adj_spatial, adj_expr, rec_activation='none'):
        # 1. 数据增强
        # adj_spatial_dropped = self.dropout_edge(adj_spatial, self.graph_corr)
        # # 获取 mask_nodes (True 表示该节点被 Mask 了)
        # data_masked, mask_nodes = self.mask_features(data, self.node_corr)
        adj_view1 = self.dropout_edge(adj_spatial, self.graph_corr)
        x_view1 = data

        adj_view2 = adj_spatial
        x_view2, mask_nodes = self.mask_features(data, self.node_corr)
        
        # 2. 编码
        feature_spatial = self.encoder(x_view1, adj_view1) 
        feature_expr = self.encoder(x_view2, adj_view2)       
        # 3. 归一化 & 聚类
        z_spatial_norm = F.normalize(feature_spatial, p=2, dim=1)
        z_expr_norm = F.normalize(feature_expr, p=2, dim=1)

        predlabel_spatial = self.projectClsHead(z_spatial_norm)
        predlabel_expr = self.projectClsHead(z_expr_norm)

        # 4. 融合
        z_stack = torch.stack([z_spatial_norm, z_expr_norm], dim=1)
        z_fuse, attention_weights = self.attention(z_stack) # [N, 32]

        # 5. 用于 Loss 的输出
        output1 = feature_spatial
        output2 = feature_expr

        # === 🔥 6. 重构 (加入 Remask 逻辑) ===
        
        # [关键步骤]: Remask
        # 即使 z_fuse 此时可能包含了通过 feature_spatial 泄露的原始信息，
        # 我们在这里强制把 mask_nodes 对应的嵌入替换成 mask_token。
        h_decoder_input = z_fuse.clone()
        
        # 只有在训练时才进行 Remask
        if self.training and mask_nodes.sum() > 0:
            h_decoder_input[mask_nodes] = self.dec_mask_token
        
        # Step A: 线性变换 (Embedding 32 -> Feature 200)
        # 此时，被 Mask 的位置是纯粹的 dec_mask_token
        rec_h = self.decoder(h_decoder_input) 
        
        # Step B: 图聚合 (利用空间邻居信息)
        # 解码器被迫使用 adj_spatial 中的邻居信息来填补 dec_mask_token 造成的空缺
        x_rec = torch.spmm(adj_spatial, rec_h)
        
        # 激活函数
        if rec_activation == 'softplus':
            x_rec = F.softplus(x_rec)
        elif rec_activation == 'relu':
            x_rec = F.relu(x_rec)
            
        return (
            output1,
            output2,
            z_spatial_norm,
            z_expr_norm,
            adj_view1,
            adj_view2,
            z_fuse,
            predlabel_spatial,
            predlabel_expr,
            x_rec,
            mask_nodes # 返回 mask 索引供 Loss 使用
        )

    @torch.no_grad()
    def get_embedding(self, data, adj_spatial):
        """
        🔥 推理专用 (Inference) 🔥
        用于最终生成聚类结果。输入最干净的数据，不带任何 Mask。
        """
        self.eval() 
        # 直接通过编码器，没有任何 Dropout 或 Mask
        z_final = self.encoder(data, adj_spatial)
        z_final_norm = F.normalize(z_final, p=2, dim=1)
        
        return z_final_norm


class CCGCNs(Module):
    """多切片模型 - 改进版（重构HVG + 批次级校正）"""
    
    def __init__(
        self,
        dims_list,
        n_clusters,
        n_batches,
        n_hvgs,
        graph_corr=0.4,
        dropout=0.5,
        device="cuda:0",
        use_conditional_prompt=True,
        prompt_hidden_dim=128,
        batch_embed_dim=32,
    ) -> None:
        super(CCGCNs, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.n_hvgs = n_hvgs
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.graph_corr = graph_corr
        self.use_conditional_prompt = use_conditional_prompt

        ### + encoders
        self.noiseLayer = NoiseLayer()
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.batchPortion = torch.eye(self.n_batches).to(device)
        self.weightBatch = 0.01

        # 🔥 改进的条件批次Prompt（增加batch-level校正）
        if self.use_conditional_prompt:
            # 批次编码器
            self.batch_encoder = nn.Embedding(self.n_batches, batch_embed_dim)
            
            # 🔥 batch-level全局校正强度预测器
            self.batch_level_correction_predictor = nn.Sequential(
                nn.Linear(batch_embed_dim, 64),
                nn.ELU(),
                nn.Dropout(dropout * 0.3),
                nn.Linear(64, 32),
                nn.ELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
            # 条件噪声生成器（编码阶段）
            self.conditional_noise_generator = nn.Sequential(
                nn.Linear(self.input_dim + batch_embed_dim, prompt_hidden_dim),
                nn.LayerNorm(prompt_hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(prompt_hidden_dim, prompt_hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(prompt_hidden_dim // 2, self.input_dim),
            )
            
            # spot-level局部微调预测器
            self.spot_level_correction_predictor = nn.Sequential(
                nn.Linear(self.input_dim + batch_embed_dim, 64),
                nn.ELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            # 条件嵌入生成器（解码阶段）
            self.conditional_embed_generator = nn.Sequential(
                nn.Linear(self.z_dim + batch_embed_dim, prompt_hidden_dim // 2),
                nn.LayerNorm(prompt_hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(prompt_hidden_dim // 2, self.z_dim),
            )
            
            print(f"✅ 使用改进的条件批次Prompt")
            print(f"   - batch-level全局校正 + spot-level局部微调")
        else:
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
        
        # 🔥 新增：解码器（重构HVG）
        self.decoder = Decoder(
            in_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.n_hvgs,
            dropout=dropout
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

    def forward(self, data, adj1, adj2, batch_list=None, rec_activation='none'):
        """
        Args: 
            rec_activation: 重构时的激活函数
        """
        # ========== 编码阶段：改进的条件化批次噪声移除 ==========
        if self.use_conditional_prompt:
            batch_embed = self.batch_encoder(batch_list)
            
            # # 🔥 batch-level全局校正强度（向量化）
            # unique_batches = torch.unique(batch_list)
            # batch_embed_means = []
            # for batch_id in unique_batches:
            #     mask = (batch_list == batch_id)
            #     batch_embed_mean = batch_embed[mask].mean(dim=0, keepdim=True)
            #     batch_embed_means.append(batch_embed_mean)
            
            # batch_embed_means = torch.cat(batch_embed_means, dim=0)
            # global_weights_per_batch = self.batch_level_correction_predictor(batch_embed_means)
            
            # # 映射到每个spot
            # batch_to_idx = {batch_id.item(): idx for idx, batch_id in enumerate(unique_batches)}
            # batch_indices = torch.tensor(
            #     [batch_to_idx[b.item()] for b in batch_list], 
            #     device=data.device
            # )
            # global_correction_weight = global_weights_per_batch[batch_indices]
            
            # 条件输入
            condition_input = torch.cat([data, batch_embed], dim=1)
            batch_noise = self.conditional_noise_generator(condition_input)
            
            # spot-level局部微调
            local_correction_weight = self.spot_level_correction_predictor(condition_input)
            
            # 最终校正权重
            # final_correction_weight = global_correction_weight * local_correction_weight
            final_correction_weight = local_correction_weight
            
            data_cleaned = data - self.weightBatch * final_correction_weight * batch_noise
        else:
            batch_noise = self.batchPortion[batch_list] @ self.batchPCA
            data_cleaned = data - self.weightBatch * batch_noise
            final_correction_weight = torch.ones(len(batch_list), 1, device=data.device)
        
        # ========== 图编码 ==========
        feature1 = self.encoder(data_cleaned, adj1)
        feature2 = self.encoder(data_cleaned, adj2)

        z1_norm = normalize(feature1, p=2, dim=1)
        z2_norm = normalize(feature2, p=2, dim=1)

        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)

        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)

        z = torch.stack([z1_norm, z2_norm], dim=1)
        z, _ = self.attention(z)

        # ========== 解码阶段 ==========
        if self.use_conditional_prompt:
            z_with_batch = torch.cat([z, batch_embed], dim=1)
            batch_embed_adaptive = self.conditional_embed_generator(z_with_batch)
            z_dec = z + self.weightBatch * final_correction_weight * batch_embed_adaptive
        else:
            norm_batch_embed = normalize(self.batchEmbed, p=2, dim=1)
            z_dec = z + self.weightBatch * self.batchPortion[batch_list] @ norm_batch_embed

        # 🔥 使用解码器重构HVG
        x_Rec = self.decoder(z_dec, activation=rec_activation)

        return h1_norm, h2_norm, z1_norm, z2_norm, z, x_Rec, label1, label2


class TransForm_W(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
        )

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x @ self.W