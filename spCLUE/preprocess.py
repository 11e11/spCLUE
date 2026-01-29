# import scanpy as sc
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cdist
# import numpy as np
# import scipy.sparse as sp



# def preprocess(adata, hvgNumber=None):
#     print("normalized data ---------------->")
#     sc.pp.filter_genes(adata, min_counts=1)
#     sc.pp.filter_cells(adata, min_counts=1)
#     if not hvgNumber is None:
#         print(f"========== selecting HVG ============")
#         adata.layers["count"] = adata.X.copy()
#         sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count",n_top_genes=hvgNumber, subset=False)
#         adata = adata[:, adata.var["highly_variable"] == True]
#         sc.pp.scale(adata)
#         return adata
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.scale(adata)
#     return adata

# # utils.py 或 preprocessing.py

# # preprocess.py 或 utils.py

# import scanpy as sc
# import numpy as np
# from scipy.sparse import issparse


# def prepare_hvg_input(adata, n_top_genes=2000, return_indices=False):
#     """
#     准备HVG输入数据（单切片或已合并的数据）
    
#     Parameters:  
#     -----------
#     adata :  AnnData
#         输入数据
#     n_top_genes : int
#         选择的HVG数量
#     return_indices : bool
#         是否返回HVG的索引
    
#     Returns: 
#     --------
#     X_hvg : np.ndarray
#         HVG表达矩阵 [n_spots × n_hvgs]
#     hvg_indices : np.ndarray (optional)
#         HVG的索引
#     """
#     print(f"选择top {n_top_genes}个高变基因...")
    
#     # 1.选择HVGs
#     if 'highly_variable' not in adata.var.columns:
#         sc.pp.highly_variable_genes(
#             adata, 
#             n_top_genes=n_top_genes,
#             flavor='seurat_v3'  # 对稀疏数据更稳定
#         )
    
#     hvg_mask = adata.var['highly_variable'].values
#     hvg_indices = np.where(hvg_mask)[0]
    
#     print(f"  ✓ 选择了 {hvg_mask.sum()} 个HVGs")
    
#     # 2.提取HVG表达
#     X_hvg = adata[: , hvg_mask].X
    
#     # 3.转换为dense（如果是sparse）
#     if issparse(X_hvg):
#         X_hvg = X_hvg.toarray()
    
#     # 4.数据已经log归一化，直接使用
#     # （归一化应该在调用此函数之前完成）
#     X_hvg_normalized = X_hvg
    
#     print(f"  ✓ HVG数据提取完成")
#     print(f"  ✓ 形状: {X_hvg_normalized.shape}")
#     print(f"  ✓ 范围: [{X_hvg_normalized.min():.3f}, {X_hvg_normalized.max():.3f}]")
    
#     if return_indices:
#         return X_hvg_normalized, hvg_indices
#     else:
#         return X_hvg_normalized


# def prepare_single_slice(adata, n_hvgs=2000):
#     """
#     🔥 为spCLUE准备单切片数据
    
#     Parameters:
#     -----------
#     adata : AnnData
#         单个切片的数据
#     n_hvgs : int
#         HVG数量
    
#     Returns:
#     --------
#     adata :  AnnData
#         准备好的数据，包含X_hvg和X_pca
#     """
#     print("\n" + "="*70)
#     print("准备单切片数据")
#     print("="*70)
    
#     print(f"\n原始数据:")
#     print(f"  - Spots: {adata.n_obs}")
#     print(f"  - Genes: {adata.n_vars}")
#     print(f"  - 稀疏矩阵:  {issparse(adata.X)}")
    
#     # 1.归一化（如果还没做）
#     if 'log1p' not in adata.uns:
#         print(f"\n归一化表达数据...")
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         adata.uns['log1p'] = True
#     else:
#         print(f"\n✓ 数据已归一化")
    
#     # 2.选择HVGs并提取
#     print(f"\n提取HVG表达...")
#     X_hvg = prepare_hvg_input(adata, n_top_genes=n_hvgs)
#     adata.obsm['X_hvg'] = X_hvg
    
#     # 3.计算PCA（用于构建表达图）
#     print(f"\n计算PCA（用于表达图）...")
#     if 'X_pca' not in adata.obsm:
#         sc.tl.pca(adata, n_comps=200, random_state=0)
#         print(f"  ✓ PCA完成:  {adata.obsm['X_pca'].shape}")
#     else:
#         print(f"  ✓ 已存在PCA结果")
    
#     print(f"\n✅ 单切片数据准备完成！")
#     print(f"  - adata.obsm['X_hvg']: {adata.obsm['X_hvg'].shape} (用于重构)")
#     print(f"  - adata.obsm['X_pca']: {adata.obsm['X_pca'].shape} (用于表达图)")
    
#     return adata


# def prepare_multi_slices(adata_list, n_hvgs=2000):
#     """
#     🔥 为spCLUE准备多切片数据
    
#     Parameters:
#     -----------
#     adata_list : list of AnnData
#         多个切片的数据列表
#     n_hvgs :  int
#         HVG数量
    
#     Returns:
#     --------
#     adata_concat : AnnData
#         合并后的数据，包含X_hvg和X_pca
#     """
#     print("\n" + "="*70)
#     print("准备多切片数据")
#     print("="*70)
    
#     # 1.合并所有切片
#     print(f"\n合并 {len(adata_list)} 个切片...")
#     adata_concat = sc.concat(adata_list, label='batch')
    
#     print(f"\n合并后数据:")
#     print(f"  - Spots: {adata_concat.n_obs}")
#     print(f"  - Genes:  {adata_concat.n_vars}")
#     print(f"  - Batches:  {adata_concat.obs['batch'].nunique()}")
#     print(f"  - 稀疏矩阵: {issparse(adata_concat.X)}")
    
#     # 2.归一化（如果还没做）
#     if 'log1p' not in adata_concat.uns:
#         print(f"\n归一化表达数据...")
#         sc.pp.normalize_total(adata_concat, target_sum=1e4)
#         sc.pp.log1p(adata_concat)
#         adata_concat.uns['log1p'] = True
#     else: 
#         print(f"\n✓ 数据已归一化")
    
#     # 3.选择HVGs并提取
#     print(f"\n提取HVG表达...")
#     X_hvg = prepare_hvg_input(adata_concat, n_top_genes=n_hvgs)
#     adata_concat.obsm['X_hvg'] = X_hvg
    
#     # 4.计算PCA（用于构建表达图）
#     print(f"\n计算PCA（用于表达图）...")
#     if 'X_pca' not in adata_concat.obsm:
#         sc.tl.pca(adata_concat, n_comps=200, random_state=0)
#         print(f"  ✓ PCA完成: {adata_concat.obsm['X_pca'].shape}")
#     else:
#         print(f"  ✓ 已存在PCA结果")
    
#     print(f"\n✅ 多切片数据准备完成！")
#     print(f"  - adata.obsm['X_hvg']: {adata_concat.obsm['X_hvg'].shape} (用于重构)")
#     print(f"  - adata.obsm['X_pca']: {adata_concat.obsm['X_pca'].shape} (用于表达图)")
    
#     return adata_concat


# def prepare_data_for_spclue(data, n_hvgs=2000):
#     """
#     🔥 通用的数据准备函数（自动判断单切片或多切片）
    
#     Parameters:
#     -----------
#     data : AnnData or list of AnnData
#         单个切片或多个切片的数据
#     n_hvgs : int
#         HVG数量
    
#     Returns: 
#     --------
#     adata :  AnnData
#         准备好的数据
#     """
#     if isinstance(data, list):
#         # 多切片
#         return prepare_multi_slices(data, n_hvgs=n_hvgs)
#     else:
#         # 单切片
#         return prepare_single_slice(data, n_hvgs=n_hvgs)
# def calcGAEParams(graph, n_samples):
#     '''graph is a bipartite graph, return pos_weight and norm_val
#     '''
#     non_zero_cnt = graph.sum()
#     norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
#     pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
#     return norm_val, pos_weight


# def calcGraphWeight(coor, eps=1e-6):
#     dist = cdist(coor, coor, "euclidean")
#     dist = dist / (np.max(dist) + eps)
#     return dist


# def correlation_graph(A, B):
#     '''calculate correlation between A and B.
#     Args:
#         A (np.ndarray): sample matrix, shape: [samples, features].
#         B (np.ndarray): sample matrix, shape: [samples, features].
#     Returns: 
#         corr (np.ndarray): correlation matrix of features, shape: [features, features].
#     '''
#     am = A - np.mean(A, axis=0, keepdims=True)
#     bm = B - np.mean(B, axis=0, keepdims=True)
#     return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


# def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, eps=1e-8, svd_solver="randomized", self_weight=0.3):
#     n_spots = adata.shape[0]
#     assert key in ["spatial", "expr"], "case should be [spatial] or [expr]"
#     if key == "spatial":
#         print("create adjacent matrix from spatial idx --------------->")
#         expr = adata.obsm[key]
#         weights = 1./ (cdist(expr, expr, "euclidean") + eps)
#     else:
#         print("create adjacent matrix from pca expr --------------->")
#         expr = PCA(n_components=n_comps, random_state=0, svd_solver=svd_solver).fit_transform(adata.X)
#         weights = correlation_graph(expr.T, expr.T)

#     print("create knn graph ---->")
#     threshold = np.sort(weights)[:, -n_neighbors - 1:-n_neighbors]
#     weights[weights < threshold] = 0
#     weights = (weights + weights.T) / 2
#     weights = weights * (1 - np.eye(n_spots))  # drop the diag

#     adjFilter = 0.if key == "spatial" else 0.1
#     # convert to bipartite case
#     adjBip = np.where(weights > adjFilter, 1, 0)
#     print(f"{key} knn graph created ----<")

#     return sp.coo_matrix(symm_norm(adjBip, weightDiag=self_weight))

# def symm_norm(adj, weightDiag=.3, eps=1e-8):
#     '''
#     args: adjacent matrix with diag = 0
#     return: D^{-1/2} (A + I) D^{-1 / 2}
#     '''
#     n_spot = adj.shape[0]
#     adj_self = (1 - weightDiag) * adj + np.eye(n_spot) * weightDiag  
#     degrees = 1./ np.sqrt((np.sum(adj_self, axis=1) + eps))
#     adj_self *= degrees
#     adj_self *= degrees[:, None]
#     return adj_self.astype(np.float32)

# spCLUE/preprocess.py
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


# ========== 原有接口函数 ========== 

# def preprocess(adata, n_top_genes=2000):
#     """
#     🔥 原有的预处理接口（保持不变）
#     对单个切片进行预处理，包括归一化、HVG选择
    
#     Parameters:
#     -----------
#     adata : AnnData
#         单个切片的数据
#     n_top_genes : int
#         高变基因数量
    
#     Returns:
#     --------
#     adata : AnnData
#         预处理后的数据，包含: 
#         - adata.obsm['X_hvg']: HVG表达矩阵（用于重构）
#         - 归一化的 adata.X（用于后续PCA）
#     """
#     print("\n" + "="*70)
#     print("预处理单切片数据")
#     print("="*70)
    
#     print(f"\n原始数据:")
#     print(f"  - Spots: {adata.n_obs}")
#     print(f"  - Genes: {adata.n_vars}")
#     print(f"  - 稀疏矩阵:  {issparse(adata.X)}")
    
#     # 1.归一化
#     if 'log1p' not in adata.uns: 
#         print(f"\n归一化表达数据...")
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         adata.uns['log1p'] = True
#         print(f"  ✓ 归一化完成")
#     else:
#         print(f"\n✓ 数据已归一化")
    
#     # 2.选择HVG
#     print(f"\n选择top {n_top_genes}个高变基因...")
#     sc.pp.highly_variable_genes(
#         adata, 
#         n_top_genes=n_top_genes, 
#         subset=False,
#         flavor='seurat_v3'
#     )
    
#     # 3.🔥 提取HVG表达矩阵（用于重构）
#     hvg_genes = adata.var['highly_variable']
#     if issparse(adata.X):
#         X_hvg = adata[: , hvg_genes].X.toarray()
#     else:
#         X_hvg = adata[:, hvg_genes].X
    
#     adata.obsm['X_hvg'] = X_hvg
#     print(f"  ✓ HVG表达矩阵:  {X_hvg.shape}")
    
#     # 4.🔥 转换为dense matrix（方便后续PCA）
#     if issparse(adata.X):
#         adata.X = adata.X.toarray()
    
#     print(f"\n✅ 预处理完成！")
#     print(f"  - adata.X:  {adata.X.shape} (归一化表达，用于PCA)")
#     print(f"  - adata.obsm['X_hvg']: {adata.obsm['X_hvg'].shape} (HVG表达，用于重构)")
    
#     return adata
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA

def preprocess(adata, n_top_genes=2000):
    """
    实现方案：
    1. 使用 Raw Counts 正确挑选 HVG
    2. 全量数据进行 Log-normalization
    3. 基于全量归一化数据计算 PCA
    4. 提取归一化后的 HVG 用于重构
    """
    print("\n" + "="*70)
    print("预处理单切片数据 (全基因 PCA + 归一化 HVG)")
    print("="*70)
    
    # # 1. 基础过滤
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    
    # 2. 备份原始计数 (为了 seurat_v3 能够正确运行)
    if issparse(adata.X):
        adata.layers['count'] = adata.X.copy() # 保持稀疏以节省内存
    else:
        adata.layers['count'] = adata.X.copy()
    
    print(f"\n根据原始计数挑选 top {n_top_genes} 个高变基因...")
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_top_genes, 
        layer='count',
        subset=False,
        flavor='seurat_v3',
    )

    # 4. 全量归一化 (作用于 adata.X)
    print(f"全量数据归一化 (target_sum=1e4) & Log1p...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns['log1p'] = True

    # 3. 挑选 HVG (虽然基于 count 选，但之后我们会提取归一化后的值)
    

    # 5. 提前提取【归一化后】的高变基因矩阵 (用于重构目标)
    hvg_mask = adata.var['highly_variable']
    X_hvg = adata[:, hvg_mask].X
    if issparse(X_hvg):
        X_hvg = X_hvg.toarray()
    adata.obsm['X_hvg'] = X_hvg
    print(f"  ✓ 已保存归一化后的 HVG 重构目标: {X_hvg.shape}")

    # 6. 🔥 关键步骤：在计算 PCA 前将全量数据转换为 Dense
    # 这样 PCA(adata.X) 才能处理全基因
    if issparse(adata.X):
        print("将全量稀疏矩阵转换为稠密矩阵以计算 PCA...")
        adata.X = adata.X.toarray()

    # sc.pp.scale(adata, zero_center=False,max_value=10) 
    # 7. 基于【全基因归一化数据】计算 PCA
    print(f"计算 PCA (基于全基因归一化数据, n_components=200)...")
    pca = PCA(n_components=200, random_state=0)
    X_pca = pca.fit_transform(adata.X) 
    adata.obsm['X_pca'] = X_pca
    
    print(f"\n✅ 预处理完成！")
    print(f"  - adata.X (全量归一化): {adata.X.shape}")
    print(f"  - adata.obsm['X_hvg'] (归一化HVG): {adata.obsm['X_hvg'].shape}")
    print(f"  - adata.obsm['X_pca'] (全基因PCA): {adata.obsm['X_pca'].shape}")
    
    return adata


def prepare_graph(adata, graph_type="spatial", n_neighbors=10):
    """
    🔥 原有的图构建接口（保持不变）
    构建空间图或表达图
    
    Parameters:
    -----------
    adata : AnnData
        预处理后的数据
    graph_type : str
        图类型:  'spatial' 或 'expr'
    n_neighbors : int
        邻居数量
    
    Returns:
    --------
    adj_norm : scipy.sparse.coo_matrix
        归一化的邻接矩阵
    """
    print(f"\n构建 {graph_type} 图...")
    
    if graph_type == "spatial": 
        # 空间图：基于空间坐标
        if 'spatial' not in adata.obsm:
            raise ValueError("未找到空间坐标 (adata.obsm['spatial'])！")
        
        coords = adata.obsm['spatial']
        print(f"  - 使用空间坐标: {coords.shape}")
        
        # 构建KNN图
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # 构建邻接矩阵
        n_spots = coords.shape[0]
        adj = sp.lil_matrix((n_spots, n_spots))
        
        for i in range(n_spots):
            for j in range(1, n_neighbors + 1):  # 跳过自己（索引0）
                neighbor_idx = indices[i, j]
                dist = distances[i, j]
                # 使用高斯核权重
                weight = np.exp(-dist**2 / (2 * distances[: , 1:].std()**2))
                adj[i, neighbor_idx] = weight
                adj[neighbor_idx, i] = weight  # 对称
        
        adj = adj.tocoo()
        print(f"  ✓ 空间图构建完成:  {adj.shape}, 边数={adj.nnz}")
    
    elif graph_type == "expr":
        # 表达图：基于基因表达（PCA）
        if 'X_pca' not in adata.obsm:
            raise ValueError("未找到PCA结果 (adata.obsm['X_pca'])！请先运行PCA。")
        
        pca_data = adata.obsm['X_pca']
        print(f"  - 使用PCA:  {pca_data.shape}")
        
        # 构建KNN图
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(pca_data)
        distances, indices = nbrs.kneighbors(pca_data)
        
        # 构建邻接矩阵
        n_spots = pca_data.shape[0]
        adj = sp.lil_matrix((n_spots, n_spots))
        
        for i in range(n_spots):
            for j in range(1, n_neighbors + 1):
                neighbor_idx = indices[i, j]
                dist = distances[i, j]
                weight = np.exp(-dist**2 / (2 * distances[: , 1:].std()**2))
                adj[i, neighbor_idx] = weight
                adj[neighbor_idx, i] = weight
        
        adj = adj.tocoo()
        print(f"  ✓ 表达图构建完成: {adj.shape}, 边数={adj.nnz}")
    
    else:
        raise ValueError(f"未知的图类型:  {graph_type}，应为 'spatial' 或 'expr'")
    
    # 归一化邻接矩阵 (D^{-1/2} A D^{-1/2})
    adj_norm = normalize_adj(adj)
    
    return adj_norm


def normalize_adj(adj):
    """
    归一化邻接矩阵:  D^{-1/2} A D^{-1/2}
    
    Parameters:
    -----------
    adj : scipy.sparse matrix
        邻接矩阵
    
    Returns: 
    --------
    adj_norm : scipy.sparse.coo_matrix
        归一化的邻接矩阵
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    return adj_norm


# ========== 新增的辅助函数 ========== 

def prepare_hvg_input(adata, n_top_genes=2000):
    """
    提取高变基因的表达矩阵（用于重构）
    
    Parameters:
    -----------
    adata : AnnData
        已归一化的数据
    n_top_genes : int
        高变基因数量
    
    Returns:  
    --------
    X_hvg : np.ndarray
        HVG表达矩阵 [n_obs × n_top_genes]
    """
    # 如果还没有选择HVG，先选择
    if 'highly_variable' not in adata.var.columns:
        print(f"  选择top {n_top_genes}个高变基因...")
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_top_genes, 
            subset=False,
            flavor='seurat_v3'
        )
    
    # 提取HVG的表达
    hvg_genes = adata.var['highly_variable']
    
    if issparse(adata.X):
        X_hvg = adata[:, hvg_genes].X.toarray()
    else:
        X_hvg = adata[:, hvg_genes].X
    
    print(f"  ✓ HVG表达矩阵: {X_hvg.shape}")
    
    return X_hvg


def prepare_single_slice(adata, n_hvgs=2000):
    """
    🔥 为spCLUE准备单切片数据（新接口）
    
    Parameters: 
    -----------
    adata :  AnnData
        单个切片的数据
    n_hvgs : int
        HVG数量
    
    Returns:
    --------
    adata : AnnData
        准备好的数据，包含:  
        - adata.obsm['X_hvg']:  HVG表达矩阵（用于重构）
        - adata.obsm['X_pca']: PCA降维结果（用于图编码）
    """
    print("\n" + "="*70)
    print("准备单切片数据")
    print("="*70)
    
    print(f"\n原始数据:")
    print(f"  - Spots: {adata.n_obs}")
    print(f"  - Genes: {adata.n_vars}")
    print(f"  - 稀疏矩阵: {issparse(adata.X)}")
    
    # 1.归一化（如果还没做）
    if 'log1p' not in adata.uns:
        print(f"\n归一化表达数据...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns['log1p'] = True
        print(f"  ✓ 归一化完成")
    else:
        print(f"\n✓ 数据已归一化")
    
    # 2.🔥 提取HVG表达矩阵（用于重构）
    print(f"\n提取HVG表达矩阵（用于重构）...")
    X_hvg = prepare_hvg_input(adata, n_top_genes=n_hvgs)
    adata.obsm['X_hvg'] = X_hvg
    
    # 3.计算PCA（用于构建表达图和模型输入）
    print(f"\n计算PCA（用于图编码）...")
    if 'X_pca' not in adata.obsm:
        # 🔥 在HVG上计算PCA
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        sc.tl.pca(adata_hvg, n_comps=200, random_state=0)
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        print(f"  ✓ PCA完成:  {adata.obsm['X_pca'].shape}")
    else:
        print(f"  ✓ 已存在PCA结果")
    
    print(f"\n✅ 单切片数据准备完成！")
    print(f"  - adata.obsm['X_hvg']: {adata.obsm['X_hvg'].shape} (用于重构)")
    print(f"  - adata.obsm['X_pca']:  {adata.obsm['X_pca'].shape} (用于图编码)")
    
    return adata


def prepare_multi_slices(adata_list, n_hvgs=2000):
    """
    🔥 为spCLUE准备多切片数据（修复版）
    
    Parameters:
    -----------
    adata_list : list of AnnData
        多个切片的数据列表
    n_hvgs :  int
        HVG数量
    
    Returns:
    --------
    adata_list : list of AnnData
        处理后的切片列表，每个包含: 
        - adata.obsm['X_hvg']:  HVG表达矩阵
        - adata.obsm['X_pca']: PCA降维结果
        - adata.obs['batch_id']: 批次ID（整数）
    batch_list : np.ndarray
        所有切片的批次标签 [total_spots]
    """
    print("\n" + "="*70)
    print(f"准备多切片数据 ({len(adata_list)} 个切片)")
    print("="*70)
    
    # 🔥 1.先合并用于选择共同的HVG
    print(f"\n临时合并切片用于HVG选择...")
    adata_concat_temp = sc.concat(adata_list, label='batch')
    
    print(f"  - 总Spots: {adata_concat_temp.n_obs}")
    print(f"  - 总Genes: {adata_concat_temp.n_vars}")
    print(f"  - 批次数: {adata_concat_temp.obs['batch'].nunique()}")
    
    # 归一化
    if 'log1p' not in adata_concat_temp.uns:
        print(f"\n归一化表达数据...")
        sc.pp.normalize_total(adata_concat_temp, target_sum=1e4)
        sc.pp.log1p(adata_concat_temp)
    
    # 选择共同的HVG
    print(f"\n选择top {n_hvgs}个高变基因（基于所有切片）...")
    sc.pp.highly_variable_genes(
        adata_concat_temp, 
        n_top_genes=n_hvgs, 
        subset=False,
        flavor='seurat_v3',
        batch_key='batch'  # 🔥 批次感知的HVG选择
    )
    
    hvg_genes = adata_concat_temp.var['highly_variable']
    hvg_names = adata_concat_temp.var_names[hvg_genes].tolist()
    print(f"  ✓ 选择了 {len(hvg_names)} 个HVG")
    
    # 🔥 2.分别处理每个切片
    print(f"\n分别处理每个切片...")
    processed_adata_list = []
    batch_list = []
    
    for i, adata in enumerate(adata_list):
        print(f"\n  切片 {i}:")
        print(f"    - Spots: {adata.n_obs}")
        
        # 归一化
        if 'log1p' not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.uns['log1p'] = True
        
        # 转dense
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        
        # 提取HVG表达（用于重构）
        hvg_mask = adata.var_names.isin(hvg_names)
        if issparse(adata.X):
            X_hvg = adata[:, hvg_mask].X.toarray()
        else:
            X_hvg = adata[:, hvg_mask].X
        
        adata.obsm['X_hvg'] = X_hvg
        print(f"    - X_hvg: {X_hvg.shape}")
        
        # 计算PCA（在HVG上，用于构图）
        adata_hvg = adata[:, hvg_mask].copy()
        sc.tl.pca(adata_hvg, n_comps=200, random_state=0)
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        print(f"    - X_pca: {adata.obsm['X_pca'].shape}")
        
        # 标记批次ID
        adata.obs['batch_id'] = i
        
        processed_adata_list.append(adata)
        batch_list.extend([i] * adata.n_obs)
    
    batch_list = np.array(batch_list)
    
    print(f"\n✅ 多切片数据准备完成！")
    print(f"  - 返回:  {len(processed_adata_list)} 个处理后的adata")
    print(f"  - batch_list: {batch_list.shape}")
    print(f"\n批次分布:")
    unique, counts = np.unique(batch_list, return_counts=True)
    for batch_id, count in zip(unique, counts):
        print(f"  - 批次 {batch_id}: {count} spots")
    
    return processed_adata_list, batch_list

def construct_graph_multi_slices(adata_list, n_neighbors_spa=10, n_neighbors_expr=10):
    """
    🔥 为多切片构建块对角图
    
    Parameters:
    -----------
    adata_list :  list of AnnData
        处理后的切片列表（需包含X_pca和spatial）
    n_neighbors_spa :  int
        空间图的邻居数
    n_neighbors_expr : int
        表达图的邻居数
    
    Returns:
    --------
    graph_dict : dict
        {'spatial': block_diag_spatial, 'expr': block_diag_expr}
    """
    from scipy.sparse import block_diag
    
    print("\n" + "="*70)
    print("构建多切片块对角图")
    print("="*70)
    
    # 空间图
    print(f"\n构建空间图...")
    g_spatial_list = []
    for i, adata in enumerate(adata_list):
        print(f"  切片 {i}:")
        g_spatial = prepare_graph(adata, "spatial", n_neighbors=n_neighbors_spa)
        g_spatial_list.append(g_spatial)
    
    g_spatial_block = block_diag(g_spatial_list)
    print(f"\n✓ 块对角空间图:  {g_spatial_block.shape}, 总边数={g_spatial_block.nnz}")
    
    # 表达图
    print(f"\n构建表达图...")
    g_expr_list = []
    for i, adata in enumerate(adata_list):
        print(f"  切片 {i}:")
        g_expr = prepare_graph(adata, "expr", n_neighbors=n_neighbors_expr)
        g_expr_list.append(g_expr)
    
    g_expr_block = block_diag(g_expr_list)
    print(f"\n✓ 块对角表达图: {g_expr_block.shape}, 总边数={g_expr_block.nnz}")
    
    return {"spatial": g_spatial_block, "expr": g_expr_block}

def prepare_data_for_spclue(data, n_hvgs=2000):
    """
    🔥 通用的数据准备函数（自动判断单切片或多切片）
    
    Parameters:
    -----------
    data : AnnData or list of AnnData
        单个切片或多个切片的数据
    n_hvgs : int
        HVG数量
    
    Returns:  
    --------
    adata : AnnData
        准备好的数据
    """
    if isinstance(data, list):
        # 多切片
        return prepare_multi_slices(data, n_hvgs=n_hvgs)
    else:
        # 单切片
        return prepare_single_slice(data, n_hvgs=n_hvgs)


def prepare_raw_counts_for_reconstruction(adata, n_hvgs=2000):
    """
    🔥 准备原始计数数据用于重构（适用于Poisson/NB/ZINB损失）
    
    Parameters: 
    -----------
    adata :  AnnData
        原始计数数据（未归一化）
    n_hvgs : int
        HVG数量
    
    Returns: 
    --------
    adata : AnnData
        准备好的数据，包含:
        - adata.obsm['X_hvg_raw']: HVG原始计数（用于重构）
        - adata.obsm['X_hvg']: HVG归一化表达（用于PCA）
        - adata.obsm['X_pca']: PCA降维结果（用于图编码）
    """
    print("\n" + "="*70)
    print("准备原始计数数据（用于Poisson/NB/ZINB重构）")
    print("="*70)
    
    print(f"\n原始数据:")
    print(f"  - Spots: {adata.n_obs}")
    print(f"  - Genes: {adata.n_vars}")
    print(f"  - 稀疏矩阵: {issparse(adata.X)}")
    
    # 1.🔥 先保存原始计数（用于选择HVG和重构）
    if issparse(adata.X):
        X_raw = adata.X.toarray()
    else:
        X_raw = adata.X.copy()
    
    # 2.归一化（仅用于选择HVG和计算PCA）
    print(f"\n归一化表达数据（仅用于HVG选择和PCA）...")
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    # 3.选择HVG
    print(f"\n选择top {n_hvgs}个高变基因...")
    sc.pp.highly_variable_genes(
        adata_norm, 
        n_top_genes=n_hvgs, 
        subset=False,
        flavor='seurat_v3'
    )
    hvg_genes = adata_norm.var['highly_variable']
    
    # 4.🔥 提取HVG的原始计数（用于重构）
    X_hvg_raw = X_raw[:, hvg_genes.values]
    adata.obsm['X_hvg_raw'] = X_hvg_raw
    print(f"  ✓ HVG原始计数:  {X_hvg_raw.shape}")
    
    # 5.提取HVG的归一化表达（用于PCA）
    if issparse(adata_norm.X):
        X_hvg_norm = adata_norm[: , hvg_genes].X.toarray()
    else:
        X_hvg_norm = adata_norm[:, hvg_genes].X
    adata.obsm['X_hvg'] = X_hvg_norm
    
    # 6.计算PCA（在归一化的HVG上）
    print(f"\n计算PCA（用于图编码）...")
    adata_hvg_norm = adata_norm[:, hvg_genes].copy()
    sc.tl.pca(adata_hvg_norm, n_comps=200, random_state=0)
    adata.obsm['X_pca'] = adata_hvg_norm.obsm['X_pca']
    print(f"  ✓ PCA完成: {adata.obsm['X_pca'].shape}")
    
    # 7.保存HVG的基因名
    adata.uns['hvg_names'] = adata_norm.var_names[hvg_genes].tolist()
    adata.var['highly_variable'] = hvg_genes
    
    print(f"\n✅ 原始计数数据准备完成！")
    print(f"  - adata.obsm['X_hvg_raw']: {adata.obsm['X_hvg_raw'].shape} (原始计数，用于重构)")
    print(f"  - adata.obsm['X_hvg']: {adata.obsm['X_hvg'].shape} (归一化表达，用于PCA)")
    print(f"  - adata.obsm['X_pca']: {adata.obsm['X_pca'].shape} (用于图编码)")
    
    return adata


def prepare_multi_slices_raw_counts(adata_list, n_hvgs=2000):
    """
    🔥 准备多切片原始计数数据（适用于Poisson/NB/ZINB损失）
    
    Parameters:
    -----------
    adata_list : list of AnnData
        多个切片的原始计数数据
    n_hvgs : int
        HVG数量
    
    Returns:
    --------
    adata_concat : AnnData
        合并后的数据，包含原始计数
    """
    print("\n" + "="*70)
    print("准备多切片原始计数数据（用于Poisson/NB/ZINB重构）")
    print("="*70)
    
    # 1.合并所有切片
    print(f"\n合并 {len(adata_list)} 个切片...")
    adata_concat = sc.concat(adata_list, label='batch')
    
    print(f"\n合并后数据:")
    print(f"  - Spots: {adata_concat.n_obs}")
    print(f"  - Genes:  {adata_concat.n_vars}")
    print(f"  - Batches: {adata_concat.obs['batch'].nunique()}")
    
    # 2.使用原始计数准备函数
    adata_concat = prepare_raw_counts_for_reconstruction(adata_concat, n_hvgs=n_hvgs)
    
    # 打印每个批次的spot数量
    print(f"\n批次分布:")
    for batch_id in adata_concat.obs['batch'].cat.categories:
        n_spots = (adata_concat.obs['batch'] == batch_id).sum()
        print(f"  - {batch_id}: {n_spots} spots")
    
    return adata_concat


def get_hvg_input_for_reconstruction(adata, reconstruction_loss='mse'):
    """
    根据重构损失类型返回合适的HVG输入
    
    Parameters:
    -----------
    adata : AnnData
        准备好的数据
    reconstruction_loss : str
        重构损失类型
    
    Returns:
    --------
    X_hvg : np.ndarray
        用于重构的HVG数据
    """
    if reconstruction_loss in ['poisson', 'nb', 'zinb']:
        # 需要原始计数
        if 'X_hvg_raw' in adata.obsm:
            print(f"✅ 使用原始计数数据进行{reconstruction_loss.upper()}重构")
            return adata.obsm['X_hvg_raw']
        else:
            print(f"⚠️ 未找到原始计数数据，使用归一化数据（可能不适合{reconstruction_loss.upper()}）")
            return adata.obsm['X_hvg']
    else:
        # MSE/SCE使用归一化数据
        print(f"✅ 使用归一化数据进行{reconstruction_loss.upper()}重构")
        return adata.obsm['X_hvg']


def construct_graph(adata, graph_type="both", n_neighbors_spa=10, n_neighbors_expr=10):
    """
    🔥 便捷的图构建函数
    
    Parameters:
    -----------
    adata : AnnData
        预处理后的数据（需包含 X_pca 和 spatial）
    graph_type : str
        'spatial', 'expr', 或 'both'
    n_neighbors : int
        邻居数量
    
    Returns:
    --------
    graph_dict : dict
        如果 graph_type='both'，返回 {'spatial': adj_spatial, 'expr': adj_expr}
        否则返回单个邻接矩阵
    """
    if graph_type == "both":
        print("\n" + "="*70)
        print("构建双视图图")
        print("="*70)
        g_spatial = prepare_graph(adata, "spatial", n_neighbors=n_neighbors_spa)
        g_expr = prepare_graph(adata, "expr", n_neighbors=n_neighbors_expr)
        return {"spatial": g_spatial, "expr": g_expr}
    else:
        return prepare_graph(adata, graph_type, n_neighbors=n_neighbors_spa)