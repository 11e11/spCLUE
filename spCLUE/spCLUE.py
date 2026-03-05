import torch
import numpy as np

from .network import CCGCN, CCGCNs
from tqdm import tqdm
from .loss import ContrastiveLoss, ClusterLoss, MSELoss,CCRLoss
from .utils import sparse_mx_to_torch_sparse_tensor, adjust_learning_rate, fix_seed

from sklearn.metrics import adjusted_rand_score


class spCLUE:
    def __init__(
    self,
    input_data,
    graph_dict,
    n_clusters=12,
    batch_list=None,
    epochs=250,
    random_seed=0,
    device=torch.device("cuda:0"),
    learning_rate=0.001,
    weight_decay=5e-4,
    dim_input=3000,
    dim_hidden=128,
    dim_embed=64,
    graph_corr=0.4,
    dropout=0.0,
    gamma=1,
    beta=1,
    kappa=0.1,
    lambda_ccr=0.5,  # 新增：CCR损失权重
    use_zinb=False,  # 新增：是否使用ZINB解码器
    batch_train=False,
    lambda_reg=0.1,  # 正则化损失权重
    reg_mode='spatial',  # 'spatial', 'combined', 'both' - 使用哪个图
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.n_clusters = n_clusters

        self.random_seed = random_seed
        self.graph_corr = graph_corr
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.lambda_ccr = lambda_ccr  # 新增
        self.use_zinb = use_zinb  # 新增
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]
        self.lambda_reg = lambda_reg  # 新增
        self.reg_mode = reg_mode  # 新增

        fix_seed(self.random_seed)
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        
        # 三个图
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(self.device)
        self.g_feature = sparse_mx_to_torch_sparse_tensor(graph_dict["feature"]).to(self.device)
        self.g_combined = sparse_mx_to_torch_sparse_tensor(graph_dict["combined"]).to(self.device)
        self.adj_s = torch.FloatTensor(graph_dict["adj_s"]).to(self.device)
        
        # 保存raw count用于ZINB
        if use_zinb and "raw_count" in graph_dict:
            self.raw_count = torch.FloatTensor(graph_dict["raw_count"]).to(self.device)
            self.library_size = self.raw_count.sum(dim=1, keepdim=True)
        else:
            self.raw_count = None
            self.library_size = None

        if batch_list is None:
            self.model = CCGCN(
                self.dims_list, self.n_clusters, self.graph_corr, dropout, use_zinb
            ).to(self.device)
        else:
            # 批次处理版本暂不修改，可类似扩展
            self.n_batches = len(set(batch_list))
            self.epochs = 500
            self.batchList = torch.LongTensor(batch_list).to(self.device)
            self.batch_train = batch_train
            self.model = CCGCNs(
                self.dims_list, self.n_clusters, self.n_batches, self.graph_corr
            ).to(self.device)

    def loss_idx(self):
        # train with batch, prevent OOM (out of memory)
        tmp = np.arange(self.n_spot)
        if self.batch_train:
            np.random.shuffle(tmp)
            return tmp[:20000]
        return tmp

    def updateResult(self, batch_case=False):
        with torch.no_grad():
            self.model.eval()
            if batch_case:
                # 批次版本保持原逻辑
                _, _, feature_spa, feature_expr, feature_comb, features_fuse, _, *_ = self.model(
                    self.input_data, self.g_spatial, self.g_expr, self.batchList
                )
                features_fuse = features_fuse.detach().cpu().numpy()
                return features_fuse

            (_, _, _, _, _, z_fused, *_) = self.model(
                self.input_data, self.g_spatial, self.g_feature, self.g_combined
            )
            predLabel = self.model.getCluster(z_fused)
            features_fuse = z_fused.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            return predLabel, features_fuse

    def train(self):
        from .loss import CCRLoss, ZINBLoss, RegularizationLoss  # 新增导入
        
        self.instance_crit = ContrastiveLoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        self.ccr_crit = CCRLoss()  
        # 🔥 新增: 初始化正则化损失
        self.reg_crit = RegularizationLoss()
        
        if self.use_zinb:
            self.rec_crit = ZINBLoss()
        else:
            self.rec_crit = MSELoss()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_ari = 0.3 if self.n_spot <= 10000 else 1.1
        print("Training Start (Enhanced 3-View with CCR) =========================>")
        
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            (
                h_spatial,
                h_feature,
                z_spatial,
                z_feature,
                z_combined,
                z_fused,
                label_spatial,
                label_feature,
                x_rec,
                attention_weights
            ) = self.model(self.input_data, self.g_spatial, self.g_feature, 
                        self.g_combined, library_size=self.library_size)

            # 实例级对比损失
            # cur_contrastive_loss = (
            #     self.instance_crit(h_spatial, h_feature)
            #     + self.instance_crit(h_feature, h_spatial)
            # ) / 2
            
            # 聚类级对比损失
            cur_cluster_loss = self.cluster_crit(label_spatial, label_feature)
            
            # CCR一致性损失（新增）
            cur_ccr_loss = self.ccr_crit(z_spatial, z_feature)
            
            # 重构损失
            if self.use_zinb:
                mean, disp, pi = x_rec
                
                cur_rec_expr_loss = self.rec_crit(self.input_data, mean, disp, pi, ridge_lambda=0)
            else:
                cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            # 🔥 新增: 正则化损失
            if self.reg_mode == 'spatial':
                # 仅使用空间图
                cur_reg_loss = self.reg_crit(z_fused, self.adj_s, mode='spatial')
            elif self.reg_mode == 'combined':
                # 仅使用联合图
                cur_reg_loss = self.reg_crit(z_fused, self.g_combined, mode='combined')
            elif self.reg_mode == 'both':
                # 同时使用两个图,取平均
                reg_loss_spatial = self.reg_crit(z_fused, self.g_spatial, mode='spatial')
                reg_loss_combined = self.reg_crit(z_fused, self.g_combined, mode='combined')
                cur_reg_loss = (reg_loss_spatial + reg_loss_combined) / 2
            else:
                cur_reg_loss = torch.tensor(0.0, device=self.device)

            # 总损失
            cur_batch_loss = (
                # self.kappa * cur_contrastive_loss
                + self.beta * cur_cluster_loss
                + self.lambda_ccr * cur_ccr_loss  # 新增
                + self.gamma * cur_rec_expr_loss
                + self.lambda_reg * cur_reg_loss  
            )
            
            cur_batch_loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                predLabel1_np = label_spatial.detach().cpu().numpy().argmax(axis=1)
                predLabel2_np = label_feature.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
                print(f"epoch {epoch + 1}: ARI={cur_ari:.4f}, CCR={cur_ccr_loss.item():.4f}, CLU={cur_cluster_loss.item():.4f}, REC={cur_rec_expr_loss.item():.4f}, REG={cur_reg_loss.item():.4f}")
                # print(x_rec[0])
            # if (epoch + 1) % 100 == 0:
            #     if cur_ari >= max_ari:
            #         predLabel, features_fuse = self.updateResult()
            #         return predLabel, features_fuse, attention_weights.detach().cpu().numpy()

        print("Training Finished =================<")
        with torch.no_grad():
            self.model.eval()
            (_, _, _, _, _, z_fused, *_) = self.model(
                self.input_data, self.g_spatial, self.g_feature, self.g_combined
            )
            predLabel = self.model.getCluster(z_fused)
            features_fuse = z_fused.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()

        return predLabel, features_fuse, attention_weights.detach().cpu().numpy()


    def trainBatch(self):
        self.instance_crit = ContrastiveLoss()
        self.rec_crit = MSELoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_ari = 0.5
        print("Training Start =========================>")
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            (
                output1,
                output2,
                _,
                _,
                output_fuse,
                x_rec,
                predlabel1,
                predlabel2,
            ) = self.model(self.input_data, self.g_spatial, self.g_expr, self.batchList)

            cur_loss_id = self.loss_idx()
            cur_contrastive_loss = (
                self.instance_crit(output1[cur_loss_id], output2[cur_loss_id])
                + self.instance_crit(output2[cur_loss_id], output1[cur_loss_id])
            ) / 2
            cur_cluster_loss = self.cluster_crit(
                predlabel1[cur_loss_id], predlabel2[cur_loss_id]
            )
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                + self.gamma * cur_rec_expr_loss
                + self.beta * cur_cluster_loss
            )

            cur_batch_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
                predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
                cur_ari = round(adjusted_rand_score(predLabel1_np, predLabel2_np), 2)
                print(f"epoch {epoch + 1}: {cur_ari}")

                if epoch + 1 == 100:
                    self.kappa, self.beta = 0.0, 1.0

                if cur_ari >= max_ari:
                    features_fuse = self.updateResult(batch_case=True)
                    return "hello", features_fuse
        print("Training Finished =================<")

        with torch.no_grad():
            self.model.eval()
            _, _, _, _, features_fuse, _, *_ = self.model(
                self.input_data, self.g_spatial, self.g_expr, self.batchList
            )
            features_fuse = features_fuse.detach().cpu().numpy()

        return "hello", features_fuse
