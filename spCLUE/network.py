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

class Decoder(nn.Module):
    """
    改进的Decoder:  重构HVGs
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        """
        Parameters:
        -----------
        in_dim : int
            输入维度（embedding维度）
        hidden_dim :  int
            隐藏层维度
        out_dim :  int
            输出维度（HVG数量，如2000）
        dropout : float
            Dropout率
        """
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, out_dim)
            # 注意：不加激活函数，让loss决定输出范围
            # 如果用SCE，会在loss中加softplus/sigmoid
        )
    
    def forward(self, z):
        """
        Parameters: 
        -----------
        z : torch.Tensor
            Embedding [n_spots × in_dim]
        
        Returns:
        --------
        x_rec : torch.Tensor
            重构的HVG表达 [n_spots × out_dim]
        """
        return self.decoder(z)


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


class CCGCN(Module):

    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5) -> None:
        super(CCGCN, self).__init__()
        """
        Args: 
            dims_list (list): dimensions of GCNs.
            n_clusters (int): number of clusters in the cluster-contrastive module.
            graph_corr (float): corruption probability of the graph.(Edge Dropout).
            dropout (float): probability of the dropout of networks.
        """
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr

        ### + encoders
        self.noiseLayer = NoiseLayer(dropout=self.dropout)
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

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

    def forward(self, data, adj1, adj2, batch_onehot=None):
        """
        Args:
            data (torch.FloatTensor): pca input of gene expression data.
            adj1 (torch.sparse_coo_tensor): normalized spatial graph.
            adj2 (torch.sparse_coo_tensor): normalized expr graph.
        """
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

        x_Rec = self.relu(z @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, z1_norm, z2_norm, z, label1, label2, x_Rec


class CCGCNs(Module):
    """多切片模型 - 使用条件批次Prompt"""
    
    def __init__(
        self,
        dims_list,
        n_clusters,
        n_batches,
        graph_corr=0.4,
        dropout=0.5,
        device="cuda: 0",
        use_conditional_prompt=True,  # 🔥 新增：是否使用条件prompt
        prompt_hidden_dim=128,        # 🔥 新增：prompt生成器隐藏层维度
        batch_embed_dim=32,           # 🔥 新增：批次嵌入维度
    ) -> None:
        super(CCGCNs, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.graph_corr = graph_corr
        self.use_conditional_prompt = use_conditional_prompt  # 🔥 新增

        ### + encoders
        self.noiseLayer = NoiseLayer()
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.batchPortion = torch.eye(self.n_batches).to(device)
        self.weightBatch = 0.01

        # 🔥🔥🔥 条件批次Prompt核心组件
        if self.use_conditional_prompt:
            # 批次编码器：将批次ID映射到嵌入空间
            self.batch_encoder = nn.Embedding(self.n_batches, batch_embed_dim)
            
            # 条件噪声生成器（编码阶段）
            self.conditional_noise_generator = nn.Sequential(
                nn.Linear(self.input_dim + batch_embed_dim, prompt_hidden_dim),
                nn.LayerNorm(prompt_hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),  # 较小的dropout防止过拟合
                nn.Linear(prompt_hidden_dim, prompt_hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(prompt_hidden_dim // 2, self.input_dim),
            )
            
            # 条件嵌入生成器（解码阶段）
            self.conditional_embed_generator = nn.Sequential(
                nn.Linear(self.z_dim + batch_embed_dim, prompt_hidden_dim // 2),
                nn.LayerNorm(prompt_hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(prompt_hidden_dim // 2, self.z_dim),
            )

            # 🔥 新增：校正强度预测器
            self.correction_weight_predictor = nn.Sequential(
                nn.Linear(self.input_dim + batch_embed_dim, 64),
                nn.ELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出0-1之间
            )
            
            print(f"✅ 使用条件批次Prompt (batch_embed_dim={batch_embed_dim}, hidden_dim={prompt_hidden_dim})")
        else:
            # 保留原始的线性批次校正
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
            print("ℹ️ 使用原始线性批次校正")

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

    # 🔥🔥🔥 核心修改：forward函数
    def forward(self, data, adj1, adj2, batch_list=None):
        """
        Args:
            data (torch.FloatTensor): pca input of the gene expression data.[n_spots × input_dim]
            adj1 (torch.sparse_coo_tensor): normalized spatial graph.
            adj2 (torch.sparse_coo_tensor): normalized expr graph.
            batch_list (torch.LongTensor): list of batch ID.[n_spots]
        """
        
        # ========== 编码阶段：条件化批次噪声移除 ==========
        if self.use_conditional_prompt:
            # 🔥 步骤1：获取批次嵌入
            batch_embed = self.batch_encoder(batch_list)  # [n_spots × batch_embed_dim]
            
            # 🔥 步骤2：拼接数据和批次信息作为条件
            condition_input = torch.cat([data, batch_embed], dim=1)  # [n_spots × (input_dim + batch_embed_dim)]
            
            # 🔥 步骤3：通过条件生成器生成数据依赖的批次噪声
            batch_noise = self.conditional_noise_generator(condition_input)  # [n_spots × input_dim]
            
            # 🔥 步骤4：移除批次噪声
            # 🔥 预测校正强度（数据依赖）
            correction_weight = self.correction_weight_predictor(condition_input)
            # # correction_weight ∈ [0, 1]
            # # 如果数据"看起来需要强校正" → 接近1
            # # 如果数据"看起来正常" → 接近0
            
            # # 应用自适应权重
            data_cleaned = data - self.weightBatch * correction_weight * batch_noise
            # data_cleaned = data - self.weightBatch * batch_noise
        else:
            # 原始线性方法
            batch_noise = self.batchPortion[batch_list] @ self.batchPCA
            data_cleaned = data - self.weightBatch * batch_noise
        
        # ========== 图编码 ==========
        feature1 = self.encoder(data_cleaned, adj1)
        feature2 = self.encoder(data_cleaned, adj2)

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

        # ========== 解码阶段：条件化批次嵌入添加 ==========
        if self.use_conditional_prompt:
            # 🔥 步骤1：拼接融合特征和批次信息
            z_with_batch = torch.cat([z, batch_embed], dim=1)  # [n_spots × (z_dim + batch_embed_dim)]
            
            # 🔥 步骤2：生成条件化的批次嵌入
            batch_embed_adaptive = self.conditional_embed_generator(z_with_batch)  # [n_spots × z_dim]
            
            # 🔥 步骤3：添加批次嵌入
            z_dec = z + self.weightBatch * correction_weight * batch_embed_adaptive
        else:
            # 原始线性方法
            norm_batch_embed = normalize(self.batchEmbed, p=2, dim=1)
            z_dec = z + self.weightBatch * self.batchPortion[batch_list] @ norm_batch_embed

        # + reconstruction
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


class TransForm_W(nn.Module):

    def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
        )  ## = initialize weight of transform layer

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x @ self.W