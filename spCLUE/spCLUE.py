import torch
import numpy as np

from .network import CCGCN, CCGCNs
from tqdm import tqdm
from .loss import ContrastiveLoss, ClusterLoss, MSELoss, GraphGuidedContrastiveLoss
from .utils import sparse_mx_to_torch_sparse_tensor, adjust_learning_rate, fix_seed, build_consensus_graph

from sklearn.metrics import adjusted_rand_score


class spCLUE:

    def __init__(
        self,
        input_data,
        graph_dict,
        n_clusters=12,
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
        dropout=0.5,
        gamma=1,
        beta=1,
        kappa=0.1,
        batch_train=False,

        delta=0.5,          # Weight for graph-guided contrastive loss
        consensus_alpha=0.85, # Weight for spatial graph in consensus
        consensus_k=20,     # Number of consensus neighbors
        warmup_epochs=50,  # Epochs before enabling graph-guided loss
        loss_freq=5,         # 新增：每3个epoch计算一次图损失
        k_pos=3,             # 新增：每个anchor的正样本数
        k_neg=256,           # 新增：每个anchor的负样本数
        n_anchors=1024,      # 新增：每次采样的anchor数
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.n_clusters = n_clusters # default to be 15

        self.random_seed = random_seed
        self.graph_corr = graph_corr
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]

        self.delta = delta
        self.consensus_alpha = consensus_alpha
        self.consensus_k = consensus_k
        self.warmup_epochs = warmup_epochs
        self.loss_freq = loss_freq
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.n_anchors = n_anchors

        fix_seed(self.random_seed)
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(
            self.device
        )
        self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(
            self.device
        )
        # Build consensus graph
        print("Building gated consensus graph...")
        from .utils import build_consensus_graph
        
        consensus_neighbors, consensus_weights, weight_threshold = build_consensus_graph(
            graph_dict["spatial"], 
            graph_dict["expr"], 
            alpha=getattr(self, 'consensus_alpha', self.consensus_alpha),      # 提高到0.9
            k_c=getattr(self, 'consensus_k', self.consensus_k),             # 降低到15
            weight_percentile=getattr(self, 'weight_percentile', 80),  # 只用top 20%
            mutual_nn=getattr(self, 'mutual_nn', False)      # 可选：互为近邻
        )
        
        # Convert to torch and move to device
        self.consensus_neighbors = torch.LongTensor(consensus_neighbors).to(self.device)
        self.consensus_weights = torch.FloatTensor(consensus_weights).to(self.device)
        self.weight_threshold = weight_threshold  # 存储阈值用于loss中的门控
        
        print(f"✅ Gated consensus graph built (alpha={getattr(self, 'consensus_alpha', self.consensus_alpha)}, "
              f"k={getattr(self, 'consensus_k', self.consensus_k)}, weight_threshold={weight_threshold:.4f})")
        

        if batch_list is None:
            self.model = CCGCN(
                self.dims_list, self.n_clusters, self.graph_corr, dropout
            ).to(self.device)
        else:
            self.n_batches = len(set(batch_list))
            self.epochs = 500
            self.batchList = torch.LongTensor(batch_list).to(self.device)
            # self.batch_train = True if self.n_spot > 26000 else False
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
            features_spa = feature_spa.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            return predLabel, features_fuse, features_spa

    def train(self):
        self.instance_crit = ContrastiveLoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        self.rec_crit = MSELoss()
        self.graph_guided_crit = GraphGuidedContrastiveLoss(
            temperature=0.2,
            k_pos=self.k_pos,
            k_neg=self.k_neg,
            n_anchors=self.n_anchors
        )


        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_ari = 0.3 if self.n_spot <= 10000 else 1.1
        print("Training Start =========================>")
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
                output_z_grp,
                att_beta,
                predlabel1,
                predlabel2,
                x_rec,
            ) = self.model(self.input_data, self.g_spatial, self.g_expr)

            cur_contrastive_loss = (
                self.instance_crit(output1, output2)
                + self.instance_crit(output2, output1)
            ) / 2
            cur_cluster_loss = self.cluster_crit(predlabel1, predlabel2)
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            # === 修改：每隔loss_freq个epoch才计算，并加warm-up ===
            # === 修改：传递门控参数（预测标签和边权阈值） ===
            cur_delta = 0.0
            if epoch + 1 >= self.warmup_epochs and ((epoch + 1) % self.loss_freq == 0):
                warmup_progress = min(1.0, (epoch + 1 - self.warmup_epochs) / 50.0)
                cur_delta = self.delta * warmup_progress
                
                # 传递predlabel1, predlabel2用于边界检测
                cur_graph_guided_loss = self.graph_guided_crit(
                    output_fuse, 
                    self.consensus_neighbors,
                    self.consensus_weights,      # 新增
                    predlabel1,                   # 新增：用于一致性检测
                    predlabel2,                   # 新增
                    weight_threshold=0.06  # 新增：边权阈值
                )
            else:
                cur_graph_guided_loss = torch.tensor(0.0).to(self.device)

            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                + self.beta * cur_cluster_loss
                + self.gamma * cur_rec_expr_loss
                + cur_delta * cur_graph_guided_loss
            )
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            cur_batch_loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                predLabel1_np = predlabel1.detach().cpu().numpy().argmax(axis=1)
                predLabel2_np = predlabel2.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel1_np, predLabel2_np)
                print(f"epoch {epoch + 1}: {cur_ari}")
                print(f"  Batch Loss: {cur_batch_loss.item():.4f}, Cluster Loss: {cur_cluster_loss.item():.4f}, Rec Loss: {cur_rec_expr_loss.item():.4f}, Contrastive Loss: {cur_contrastive_loss.item():.4f},GraphGuided Loss: {cur_graph_guided_loss.item():.4f},Delta: {cur_delta:.4f}, Beta: {self.beta}, Kappa: {self.kappa}")
                
            if (epoch + 1) % 100 == 0:
                if cur_ari >= max_ari:
                    predLabel, features_fuse, features_spa = self.updateResult()
                    return predLabel, features_fuse, features_spa, att_beta

        print("Training Finished =================<")
        with torch.no_grad():
            self.model.eval()
            _, _, feature_spa, feature_expr, features_fuse, _, *_ = self.model(
                self.input_data, self.g_spatial, self.g_expr
            )
            predLabel = self.model.getCluster(features_fuse)
            features_fuse = features_fuse.detach().cpu().numpy()
            features_spa = feature_spa.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()

        return predLabel, features_fuse,features_spa, att_beta

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
