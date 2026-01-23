# import torch
# import numpy as np

# from .network import CCGCN, CCGCNs
# from tqdm import tqdm
# from .loss import ContrastiveLoss, ClusterLoss, MSELoss
# from .utils import sparse_mx_to_torch_sparse_tensor, adjust_learning_rate, fix_seed

# from sklearn.metrics import adjusted_rand_score


# class spCLUE: 

#     def __init__(
#         self,
#         input_data,
#         graph_dict,
#         n_clusters=12,
#         batch_list=None,
#         epochs=500,
#         random_seed=0,
#         device=torch.device("cuda:0"),
#         learning_rate=0.001,
#         weight_decay=0.001,
#         dim_input=200,
#         dim_hidden=64,
#         dim_embed=24,
#         graph_corr=0.4,
#         dropout=0.5,
#         gamma=1,
#         beta=1,
#         kappa=0.1,
#         batch_train=False,
#         # 🔥🔥🔥 新增：条件批次Prompt参数
#         use_conditional_prompt=True,   # 是否使用条件prompt
#         prompt_hidden_dim=64,         # prompt生成器隐藏层维度
#         batch_embed_dim=24,            # 批次嵌入维度
#     ):
#         self.device = device
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.epochs = epochs

#         self.n_clusters = n_clusters

#         self.random_seed = random_seed
#         self.graph_corr = graph_corr
#         self.gamma = gamma
#         self.beta = beta
#         self.kappa = kappa
#         self.dims_list = [dim_input, dim_hidden, dim_embed]
#         self.n_spot = input_data.shape[0]
        
#         # 🔥 新增：保存条件prompt参数
#         self.use_conditional_prompt = use_conditional_prompt
#         self.prompt_hidden_dim = prompt_hidden_dim
#         self.batch_embed_dim = batch_embed_dim

#         fix_seed(self.random_seed)
#         self.input_data = torch.FloatTensor(input_data).to(self.device)
#         self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(
#             self.device
#         )
#         self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(
#             self.device
#         )

#         if batch_list is None:
#             # 单切片：使用CCGCN
#             self.model = CCGCN(
#                 self.dims_list, self.n_clusters, self.graph_corr, dropout
#             ).to(self.device)
#             print("ℹ️ 初始化单切片模型 (CCGCN)")
#         else:
#             # 多切片：使用CCGCNs
#             self.n_batches = len(set(batch_list))
#             self.epochs = 500
#             self.batchList = torch.LongTensor(batch_list).to(self.device)
#             self.batch_train = batch_train
            
#             # 🔥🔥🔥 使用条件批次Prompt初始化CCGCNs
#             self.model = CCGCNs(
#                 self.dims_list, 
#                 self.n_clusters, 
#                 self.n_batches, 
#                 self.graph_corr,
#                 dropout=dropout,
#                 device=self.device,
#                 use_conditional_prompt=self.use_conditional_prompt,      # 🔥 新增
#                 prompt_hidden_dim=self.prompt_hidden_dim,                # 🔥 新增
#                 batch_embed_dim=self.batch_embed_dim,                    # 🔥 新增
#             ).to(self.device)
            
#             if self.use_conditional_prompt:
#                 print(f"✅ 初始化多切片模型 (CCGCNs) with 条件批次Prompt")
#                 print(f"   - 批次数量: {self.n_batches}")
#                 print(f"   - 批次嵌入维度: {self.batch_embed_dim}")
#                 print(f"   - Prompt隐藏层维度: {self.prompt_hidden_dim}")
#             else:
#                 print(f"ℹ️ 初始化多切片模型 (CCGCNs) with 线性批次校正")

#     def loss_idx(self):
#         """防止OOM的批次采样"""
#         tmp = np.arange(self.n_spot)
#         if self.batch_train:
#             np.random.shuffle(tmp)
#             return tmp[: 20000]
#         return tmp

#     def updateResult(self, batch_case=False):
#         with torch.no_grad():
#             self.model.eval()
#             if batch_case:
#                 _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
#                     self.input_data, self.g_spatial, self.g_expr, self.batchList
#                 )
#                 features_fuse = features_fuse.detach().cpu().numpy()
#                 return features_fuse

#             _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
#                 self.input_data, self.g_spatial, self.g_expr
#             )
#             predLabel = self.model.getCluster(features_fuse)
#             features_fuse = features_fuse.detach().cpu().numpy()
#             predLabel = predLabel.detach().cpu().numpy()
#             return predLabel, features_fuse

#     # 单切片训练 - 保持不变
#     def train(self):
#         self.instance_crit = ContrastiveLoss()
#         self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
#         self.rec_crit = MSELoss()

#         self.optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )
#         max_ari = 0.3 if self.n_spot <= 10000 else 1.1
#         print("Training Start =========================>")
#         for epoch in tqdm(range(self.epochs)):
#             self.model.train()
#             adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
#             self.optimizer.zero_grad()

#             (
#                 output1,
#                 output2,
#                 output_spa,
#                 output_expr,
#                 output_fuse,
#                 predlabel1,
#                 predlabel2,
#                 x_rec,
#             ) = self.model(self.input_data, self.g_spatial, self.g_expr)

#             cur_contrastive_loss = (
#                 self.instance_crit(output1, output2)
#                 + self.instance_crit(output2, output1)
#             ) / 2
#             cur_cluster_loss = self.cluster_crit(predlabel1, predlabel2)
#             cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

#             cur_batch_loss = (
#                 self.kappa * cur_contrastive_loss
#                 + self.beta * cur_cluster_loss
#                 + self.gamma * cur_rec_expr_loss
#             )
#             cur_batch_loss.backward()
#             self.optimizer.step()
            
#             if (epoch + 1) % 100 == 0:
#                 predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
#                 predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
#                 cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
#                 print(f"epoch {epoch + 1}: {cur_ari}")
#                 if cur_ari >= max_ari:
#                     predLabel, features_fuse = self.updateResult()
#                     return predLabel, features_fuse

#         print("Training Finished =================<")
#         with torch.no_grad():
#             self.model.eval()
#             _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
#                 self.input_data, self.g_spatial, self.g_expr
#             )
#             predLabel = self.model.getCluster(features_fuse)
#             features_fuse = features_fuse.detach().cpu().numpy()
#             predLabel = predLabel.detach().cpu().numpy()

#         return predLabel, features_fuse

#     # 🔥 多切片训练 - 添加调试信息
#     def trainBatch(self):
#         self.instance_crit = ContrastiveLoss()
#         self.rec_crit = MSELoss()
#         self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        
#         self.optimizer = torch.optim.Adam(
#             filter(lambda p: p.requires_grad, self.model.parameters()),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )
        
#         max_ari = 0.5
#         print("Training Start =========================>")
        
#         for epoch in tqdm(range(self.epochs)):
#             self.model.train()
#             adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
#             self.optimizer.zero_grad()

#             (
#                 output1,
#                 output2,
#                 _,
#                 _,
#                 output_fuse,
#                 x_rec,
#                 predlabel1,
#                 predlabel2,
#             ) = self.model(self.input_data, self.g_spatial, self.g_expr, self.batchList)

#             cur_loss_id = self.loss_idx()
#             cur_contrastive_loss = (
#                 self.instance_crit(output1[cur_loss_id], output2[cur_loss_id])
#                 + self.instance_crit(output2[cur_loss_id], output1[cur_loss_id])
#             ) / 2
#             cur_cluster_loss = self.cluster_crit(
#                 predlabel1[cur_loss_id], predlabel2[cur_loss_id]
#             )
#             cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

#             cur_batch_loss = (
#                 self.kappa * cur_contrastive_loss
#                 + self.gamma * cur_rec_expr_loss
#                 + self.beta * cur_cluster_loss
#             )

#             cur_batch_loss.backward()
#             self.optimizer.step()

#             if (epoch + 1) % 100 == 0:
#                 predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
#                 predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
#                 cur_ari = round(adjusted_rand_score(predLabel1_np, predLabel2_np), 2)
                
#                 # 🔥 新增：打印更详细的训练信息
#                 print(f"epoch {epoch + 1}: ARI={cur_ari:.3f}, "
#                       f"contrast={cur_contrastive_loss.item():.4f}, "
#                       f"cluster={cur_cluster_loss.item():.4f}, "
#                       f"recon={cur_rec_expr_loss.item():.4f}")
                
#                 if epoch + 1 == 100:
#                     self.kappa, self.beta = 0.0, 1.0

#                 # if cur_ari >= max_ari:
#                 #     features_fuse = self.updateResult(batch_case=True)
#                 #     return "hello", features_fuse
                    
#         print("Training Finished =================<")

#         with torch.no_grad():
#             self.model.eval()
#             _, _, _, _, features_fuse, _, *_ = self.model(
#                 self.input_data, self.g_spatial, self.g_expr, self.batchList
#             )
#             features_fuse = features_fuse.detach().cpu().numpy()

#         return "hello", features_fuse

# spCLUE/spCLUE.py (Part 1: 初始化)
import torch
import numpy as np

from .network import CCGCN, CCGCNs
from tqdm import tqdm
from .loss import (
    ContrastiveLoss, 
    ClusterLoss, 
    # ConfidenceWeightedClusterLoss,  # 🔥 新增：置信度加权聚类损失
    SpotLevelWeightedClusterLoss,
    LocalAggregationContrastiveLoss,
    PrototypeContrastiveLoss,
    ReconstructionLoss,  # 🔥 统一的重构损失接口
    SCELoss,
    CentroidAlignmentLoss,  # 🔥 质心对齐
    SoftClusterAlignmentLoss  # 🔥 软聚类对齐
)
from .utils import sparse_mx_to_torch_sparse_tensor, adjust_learning_rate, fix_seed

from sklearn.metrics import adjusted_rand_score


class spCLUE: 
    def __init__(
        self,
        input_data,
        graph_dict,
        n_clusters=12,
        n_hvgs=2000,  # 🔥 新增：HVG数量
        hvg_input=None,  # 🔥 新增：HVG表达数据（用于重构）
        batch_list=None,
        epochs=500,
        random_seed=0,
        device=torch.device("cuda:0"),
        learning_rate=0.001,
        weight_decay=0.001,
        dim_input=200,
        dim_hidden=64,
        dim_embed=24,
        graph_corr=0.4,
        node_corr=0.2,  # 节点遮蔽概率
        dropout=0.5,
        gamma=1,
        gamma_mask=0.5,
        beta=1,
        kappa=0.1,
        batch_train=False,
        # 🔥 新增：重构损失类型
        reconstruction_loss='mse',  # 'mse', 'sce', 'poisson', 'nb', 'zinb'
        # 🔥 新增：跨切片一致性损失权重
        lambda_centroid=0.3,
        lambda_structure=0.2,
        # 🔥 修改：早停参数
        patience=50,
        min_delta=1e-4,
        # 条件批次Prompt参数
        use_conditional_prompt=True,
        prompt_hidden_dim=128,
        batch_embed_dim=32,
        # 🔥 新增：软聚类对齐的采样大小
        structure_sample_size=None,
        cluster_weight_strategy='min',
        cluster_warmup_epochs=50,  
        prototype_temperature=0.2,  # 🔥 新增：原型对比温度参数
        prototype_update_interval=10,  # 🔥 新增：原型更新间隔
        # lambda_local_agg=0.5,  # 🔥 新增：局部聚合损失权重
        # local_agg_negative_samples=256,  # 🔥 新增：负样本数量
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.n_clusters = n_clusters
        self.n_hvgs = n_hvgs
        self.hvg_input = hvg_input

        self.random_seed = random_seed
        self.graph_corr = graph_corr
        self.node_corr = node_corr
        self.gamma = gamma
        self.gamma_mask = gamma_mask
        self.beta = beta
        self.kappa = kappa
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]
        
        # 🔥 新增参数
        self.prototype_update_interval = prototype_update_interval
        self.prototype_temperature = prototype_temperature

        # 🔥 新增参数
        self.reconstruction_loss = reconstruction_loss
        self.lambda_centroid = lambda_centroid
        self.lambda_structure = lambda_structure
        self.patience = patience
        self.min_delta = min_delta
        self.structure_sample_size = structure_sample_size
        
        self.use_conditional_prompt = use_conditional_prompt
        self.prompt_hidden_dim = prompt_hidden_dim
        self.batch_embed_dim = batch_embed_dim

        self.cluster_weight_strategy = cluster_weight_strategy
        self.cluster_warmup_epochs = cluster_warmup_epochs

        fix_seed(self.random_seed)
        
        # PCA输入（用于图编码）
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        
        # 🔥 HVG输入（用于重构）
        if hvg_input is None:
            print("重构PCA")
            # raise ValueError("必须提供hvg_input用于重构！")
        else:
            self.hvg_data = torch.FloatTensor(hvg_input).to(self.device)
        
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(
            self.device
        )
        self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(
            self.device
        )

        if batch_list is None:
            # 单切片：使用CCGCN
            self.model = CCGCN(
                self.dims_list, 
                self.n_clusters, 
                self.n_hvgs,  # 🔥 传入HVG数量
                self.graph_corr, 
                self.node_corr,
                dropout
            ).to(self.device)
            print("ℹ️ 初始化单切片模型 (CCGCN)")
            print(f"   - 输入:  {dim_input}维PCA")
            print(f"   - 重构: {n_hvgs}个HVG")
        else:
            # 多切片：使用CCGCNs
            self.n_batches = len(set(batch_list))
            self.batchList = torch.LongTensor(batch_list).to(self.device)
            self.batch_train = batch_train
            
            self.model = CCGCNs(
                self.dims_list, 
                self.n_clusters, 
                self.n_batches,
                self.n_hvgs,  # 🔥 传入HVG数量
                self.graph_corr,
                dropout=dropout,
                device=self.device,
                use_conditional_prompt=self.use_conditional_prompt,
                prompt_hidden_dim=self.prompt_hidden_dim,
                batch_embed_dim=self.batch_embed_dim,
            ).to(self.device)
            
            if self.use_conditional_prompt:
                print(f"✅ 初始化多切片模型 (CCGCNs)")
                print(f"   - 批次数量: {self.n_batches}")
                print(f"   - 输入: {dim_input}维PCA")
                print(f"   - 重构: {n_hvgs}个HVG")
                print(f"   - 批次校正: batch-level + spot-level")
            else:
                print(f"ℹ️ 初始化多切片模型 (CCGCNs) with 线性批次校正")

    def loss_idx(self):
        """随机采样用于计算对比损失的索引"""
        if self.batch_train:
            batch = int(self.n_spot * 0.1)
            rid = np.random.choice(self.n_spot, size=batch, replace=False)
            return rid
        else: 
            return np.arange(self.n_spot)

    def updateResult(self, batch_case=False):
        with torch.no_grad():
            self.model.eval()
            if batch_case:
                _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
                    self.input_data, self.g_spatial, self.g_expr, self.batchList
                )
                features_fuse = features_fuse.detach().cpu().numpy()
                return features_fuse

            _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
                self.input_data, self.g_spatial, self.g_expr
            )
            predLabel = self.model.getCluster(features_fuse)
            features_fuse = features_fuse.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            return predLabel, features_fuse

    def train(self):
        self.instance_crit = ContrastiveLoss()
        
        # 🔥 使用新的置信度加权聚类损失
        # self.cluster_crit = ConfidenceWeightedClusterLoss(
        #     n_classes=self.n_clusters,
        #     weight_strategy=self.cluster_weight_strategy,
        #     temperature=0.2
        # ).to(self.device)
        self.cluster_crit = SpotLevelWeightedClusterLoss(
            n_classes=self.n_clusters,
            temperature=0.2 # 这个参数目前主要作为保留，如果输入是logits才用，现在输入是prob影响不大
        ).to(self.device)

        self.cluster_crit2 = ClusterLoss(self.n_clusters, self.device)
        
        # 🔥 新增：局部聚合级对比学习损失
        self.local_agg_crit = LocalAggregationContrastiveLoss(
            temperature=0.2,
            n_negative_samples=1  # 负样本数量
        ).to(self.device)

        self.rec_crit = ReconstructionLoss(loss_type=self.reconstruction_loss)
        self.rec_crit = self.rec_crit.to(self.device)
        self.sce_crit = SCELoss(alpha=0.1, beta=1.0).to(self.device)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        best_loss = float('inf')
        patience_counter = 0
        best_embeddings = None
        best_labels = None
        
        print("Training Start =========================>")
        print(f"重构损失类型: {self.reconstruction_loss.upper()}")
        print(f"聚类对齐策略: {self.cluster_weight_strategy}")
        print(f"Warmup epochs: {self.cluster_warmup_epochs}")
        print(f"早停策略: patience={self.patience}, min_delta={self.min_delta}")
        print(f"损失权重: kappa={self.kappa}, beta={self.beta}, gamma={self.gamma}")
        
        if self.reconstruction_loss in ['poisson', 'nb', 'zinb']:
            rec_activation = 'softplus'
        else:
            rec_activation = 'none'
        
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            (
                output1,
                output2,
                output_spa,
                output_expr,
                output_fuse,
                predlabel1,
                predlabel2,
                x_rec,
                mask_nodes
            ) = self.model(self.input_data, self.g_spatial, self.g_expr, 
                          rec_activation=rec_activation)

            cur_contrastive_loss = (
                self.instance_crit(output1, output2)
                + self.instance_crit(output2, output1)
            ) / 2
            cur_contrastive_loss = 0
            if mask_nodes.sum() > 0:
                cur_mask_rec_loss = self.sce_crit(
                    x_rec[mask_nodes], self.hvg_data[mask_nodes])
            else:
                cur_mask_rec_loss = torch.tensor(0.0,device=self.device)
            
            # 🔥 置信度加权的聚类损失
            # (
            #     soft_total_loss,       # 包含加权对齐的总Loss
            #     alignment_raw,    # 🔥 未加权的硬对齐 (Warmup用这个)
            #     weighted_alignment_loss, # 加权的软对齐
            #     diversity_loss,        # 负熵
            #     entropy_penalty        # 熵惩罚
            # ) = self.cluster_crit(predlabel1, predlabel2)
            # alignment_raw_mean = alignment_raw.mean()
            # cur_cluster_loss = total_loss            
            cur_rec_expr_loss = self.rec_crit(x_rec, self.hvg_data)
            
            # 🔥 Local aggregation contrastive loss
            # cur_local_agg_loss = self.local_agg_crit(
            #     output_spa,  # 空间视图嵌入
            #     output_expr,  # 表达视图嵌入
            #     self.g_spatial,  # 空间图
            #     self.g_expr  # 表达图
            # )

            cur_loss, neg_loss = self.cluster_crit2(predlabel1, predlabel2)
            cur_cluster_loss = cur_loss + 1. *neg_loss

            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                # + self.beta * cluster_weight * cur_cluster_loss  # 🔥 应用warmup权重
                + self.beta * cur_cluster_loss  
                + self.gamma * cur_rec_expr_loss
                # + self.gamma_mask * cur_mask_rec_loss
                # + 5.0 * cur_local_agg_loss  
            )
            
            cur_batch_loss.backward()
            self.optimizer.step()
            
            current_loss = cur_batch_loss.item()
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
                predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
                # print(f"\nEpoch {epoch + 1}:   Loss={current_loss:.4f}, ARI={cur_ari:.3f}, ClusterWeight={cluster_weight:.2f}")
                print(f"\nEpoch {epoch + 1}:   Loss={current_loss:.4f}, ARI={cur_ari:.3f}")
                print(f" Cluster={cur_cluster_loss.item():.4f}, Recon={cur_rec_expr_loss.item():.4f}, MaskRecon={cur_mask_rec_loss.item():.4f}, LocalAgg={cur_local_agg_loss.item():.4f}")
                # print(f"  AlignmentRaw={alignment_raw_mean.item():.4f}, WeightedLoss={weighted_alignment_loss.item():.4f}, Diversity={diversity_loss.item():.4f}, EntropyPenalty={entropy_penalty.item():.4f}")
                print(f"  ClusterLoss={cur_loss.item():.4f}, NegLoss={neg_loss.item():.4f}")
            
            # 🔥 相对早停
            if current_loss < best_loss * (1 - self.min_delta):
                best_loss = current_loss
                patience_counter = 0
                best_labels, best_embeddings = self.updateResult()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best loss: {best_loss:.4f}")
                break

        print("Training Finished =================<")
        
        if best_embeddings is not None:
            return best_labels, best_embeddings
        else:
            return self.updateResult()
        
    # spCLUE/spCLUE.py (Part 2: 多切片训练)

    # 🔥 修改：多切片训练（加入质心对齐和软聚类对齐）
    def trainBatch(self):
        self.instance_crit = ContrastiveLoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        
        # 🔥 使用统一的重构损失接口
        self.rec_crit = ReconstructionLoss(loss_type=self.reconstruction_loss)
        
        # # 🔥 新增：跨切片一致性损失
        # self.centroid_crit = CentroidAlignmentLoss(alignment_type='variance')
        # self.structure_crit = SoftClusterAlignmentLoss(sample_size=self.structure_sample_size)
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # 🔥 早停相关变量
        best_loss = float('inf')
        patience_counter = 0
        best_embeddings = None
        
        print("Training Start =========================>")
        print(f"重构损失类型: {self.reconstruction_loss.upper()}")
        print(f"早停策略: patience={self.patience}, min_delta={self.min_delta}")
        # print(f"跨切片损失权重: lambda_centroid={self.lambda_centroid}, lambda_structure={self.lambda_structure}")
        # if self.structure_sample_size: 
        #     print(f"软聚类对齐采样大小: {self.structure_sample_size}")
        
        # 根据损失类型确定激活函数
        if self.reconstruction_loss in ['poisson', 'nb', 'zinb']:
            rec_activation = 'softplus'
        else:
            rec_activation = 'none'
        
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            (
                output1,
                output2,
                z1_norm,
                z2_norm,
                output_fuse,
                x_rec,
                predlabel1,
                predlabel2,
            ) = self.model(self.input_data, self.g_spatial, self.g_expr, 
                          self.batchList, rec_activation=rec_activation)

            cur_loss_id = self.loss_idx()
            
            # 原有损失
            cur_contrastive_loss = (
                self.instance_crit(output1[cur_loss_id], output2[cur_loss_id])
                + self.instance_crit(output2[cur_loss_id], output1[cur_loss_id])
            ) / 2
            cur_cluster_loss = self.cluster_crit(
                predlabel1[cur_loss_id], predlabel2[cur_loss_id]
            )
            
            # 🔥 重构HVG（而非PCA）
            cur_rec_expr_loss = self.rec_crit(x_rec, self.hvg_data)
            
            # 🔥 新增：批次质心对齐损失
            # cur_centroid_loss = self.centroid_crit(output_fuse, self.batchList)
            
            # 🔥 新增：软聚类引导的跨切片对齐
            # Q_avg = (predlabel1 + predlabel2) / 2.0
            # cur_structure_loss = self.structure_crit(output_fuse, Q_avg, self.batchList)

            # 🔥 总损失（加入跨切片损失）
            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                + self.gamma * cur_rec_expr_loss
                + self.beta * cur_cluster_loss
                # + self.lambda_centroid * cur_centroid_loss  # 🔥 新增
                # + self.lambda_structure * cur_structure_loss  # 🔥 新增
            )

            cur_batch_loss.backward()
            self.optimizer.step()

            # 记录当前损失
            current_loss = cur_batch_loss.item()
            
            if (epoch + 1) % 50 == 0:
                predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
                predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
                
                print(f"Epoch {epoch + 1}:")
                print(f"  Total Loss={current_loss:.4f}, ARI={cur_ari:.3f}")
                print(f"  Instance={cur_contrastive_loss.item():.4f}, Cluster={cur_cluster_loss.item():.4f}")
                print(f"  Recon={cur_rec_expr_loss.item():.4f}")
                # print(f"  🔥 Centroid={cur_centroid_loss.item():.4f}, Structure={cur_structure_loss.item():.4f}")
                
                # # 动态调整权重（可选）
                # if epoch + 1 == 100: 
                #     self.kappa, self.beta = 0.0, 1.0
                #     print("  ⚙️ 调整权重:  kappa=0.0, beta=1.0")
            
            # 🔥 早停逻辑
            if current_loss < best_loss - self.min_delta:
                best_loss = current_loss
                patience_counter = 0
                # 保存最佳嵌入
                best_embeddings = self.updateResult(batch_case=True)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Best loss: {best_loss:.4f}")
                break
                    
        print("Training Finished =================<")

        # 返回最佳结果
        if best_embeddings is not None:
            return "hello", best_embeddings
        else:
            with torch.no_grad():
                self.model.eval()
                _, _, _, _, features_fuse, _, *_ = self.model(
                    self.input_data, self.g_spatial, self.g_expr, self.batchList
                )
                features_fuse = features_fuse.detach().cpu().numpy()
            return "hello", features_fuse