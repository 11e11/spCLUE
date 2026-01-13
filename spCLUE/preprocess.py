import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np
import scipy.sparse as sp



def preprocess(adata, hvgNumber=None):
    print("normalized data ---------------->")
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if not hvgNumber is None:
        print(f"========== selecting HVG ============")
        adata.layers["count"] = adata.X.copy()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count",n_top_genes=hvgNumber, subset=False)
        adata = adata[:, adata.var["highly_variable"] == True]
        sc.pp.scale(adata)
        return adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    return adata

# utils.py 或 preprocessing.py

# preprocess.py 或 utils.py

import scanpy as sc
import numpy as np
from scipy.sparse import issparse


def prepare_hvg_input(adata, n_top_genes=2000, return_indices=False):
    """
    准备HVG输入数据（单切片或已合并的数据）
    
    Parameters:  
    -----------
    adata :  AnnData
        输入数据
    n_top_genes : int
        选择的HVG数量
    return_indices : bool
        是否返回HVG的索引
    
    Returns: 
    --------
    X_hvg : np.ndarray
        HVG表达矩阵 [n_spots × n_hvgs]
    hvg_indices : np.ndarray (optional)
        HVG的索引
    """
    print(f"选择top {n_top_genes}个高变基因...")
    
    # 1.选择HVGs
    if 'highly_variable' not in adata.var.columns:
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_top_genes,
            flavor='seurat_v3'  # 对稀疏数据更稳定
        )
    
    hvg_mask = adata.var['highly_variable'].values
    hvg_indices = np.where(hvg_mask)[0]
    
    print(f"  ✓ 选择了 {hvg_mask.sum()} 个HVGs")
    
    # 2.提取HVG表达
    X_hvg = adata[: , hvg_mask].X
    
    # 3.转换为dense（如果是sparse）
    if issparse(X_hvg):
        X_hvg = X_hvg.toarray()
    
    # 4.数据已经log归一化，直接使用
    # （归一化应该在调用此函数之前完成）
    X_hvg_normalized = X_hvg
    
    print(f"  ✓ HVG数据提取完成")
    print(f"  ✓ 形状: {X_hvg_normalized.shape}")
    print(f"  ✓ 范围: [{X_hvg_normalized.min():.3f}, {X_hvg_normalized.max():.3f}]")
    
    if return_indices:
        return X_hvg_normalized, hvg_indices
    else:
        return X_hvg_normalized


def prepare_single_slice(adata, n_hvgs=2000):
    """
    🔥 为spCLUE准备单切片数据
    
    Parameters:
    -----------
    adata : AnnData
        单个切片的数据
    n_hvgs : int
        HVG数量
    
    Returns:
    --------
    adata :  AnnData
        准备好的数据，包含X_hvg和X_pca
    """
    print("\n" + "="*70)
    print("准备单切片数据")
    print("="*70)
    
    print(f"\n原始数据:")
    print(f"  - Spots: {adata.n_obs}")
    print(f"  - Genes: {adata.n_vars}")
    print(f"  - 稀疏矩阵:  {issparse(adata.X)}")
    
    # 1.归一化（如果还没做）
    if 'log1p' not in adata.uns:
        print(f"\n归一化表达数据...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns['log1p'] = True
    else:
        print(f"\n✓ 数据已归一化")
    
    # 2.选择HVGs并提取
    print(f"\n提取HVG表达...")
    X_hvg = prepare_hvg_input(adata, n_top_genes=n_hvgs)
    adata.obsm['X_hvg'] = X_hvg
    
    # 3.计算PCA（用于构建表达图）
    print(f"\n计算PCA（用于表达图）...")
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=200, random_state=0)
        print(f"  ✓ PCA完成:  {adata.obsm['X_pca'].shape}")
    else:
        print(f"  ✓ 已存在PCA结果")
    
    print(f"\n✅ 单切片数据准备完成！")
    print(f"  - adata.obsm['X_hvg']: {adata.obsm['X_hvg'].shape} (用于重构)")
    print(f"  - adata.obsm['X_pca']: {adata.obsm['X_pca'].shape} (用于表达图)")
    
    return adata


def prepare_multi_slices(adata_list, n_hvgs=2000):
    """
    🔥 为spCLUE准备多切片数据
    
    Parameters:
    -----------
    adata_list : list of AnnData
        多个切片的数据列表
    n_hvgs :  int
        HVG数量
    
    Returns:
    --------
    adata_concat : AnnData
        合并后的数据，包含X_hvg和X_pca
    """
    print("\n" + "="*70)
    print("准备多切片数据")
    print("="*70)
    
    # 1.合并所有切片
    print(f"\n合并 {len(adata_list)} 个切片...")
    adata_concat = sc.concat(adata_list, label='batch')
    
    print(f"\n合并后数据:")
    print(f"  - Spots: {adata_concat.n_obs}")
    print(f"  - Genes:  {adata_concat.n_vars}")
    print(f"  - Batches:  {adata_concat.obs['batch'].nunique()}")
    print(f"  - 稀疏矩阵: {issparse(adata_concat.X)}")
    
    # 2.归一化（如果还没做）
    if 'log1p' not in adata_concat.uns:
        print(f"\n归一化表达数据...")
        sc.pp.normalize_total(adata_concat, target_sum=1e4)
        sc.pp.log1p(adata_concat)
        adata_concat.uns['log1p'] = True
    else: 
        print(f"\n✓ 数据已归一化")
    
    # 3.选择HVGs并提取
    print(f"\n提取HVG表达...")
    X_hvg = prepare_hvg_input(adata_concat, n_top_genes=n_hvgs)
    adata_concat.obsm['X_hvg'] = X_hvg
    
    # 4.计算PCA（用于构建表达图）
    print(f"\n计算PCA（用于表达图）...")
    if 'X_pca' not in adata_concat.obsm:
        sc.tl.pca(adata_concat, n_comps=200, random_state=0)
        print(f"  ✓ PCA完成: {adata_concat.obsm['X_pca'].shape}")
    else:
        print(f"  ✓ 已存在PCA结果")
    
    print(f"\n✅ 多切片数据准备完成！")
    print(f"  - adata.obsm['X_hvg']: {adata_concat.obsm['X_hvg'].shape} (用于重构)")
    print(f"  - adata.obsm['X_pca']: {adata_concat.obsm['X_pca'].shape} (用于表达图)")
    
    return adata_concat


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
    adata :  AnnData
        准备好的数据
    """
    if isinstance(data, list):
        # 多切片
        return prepare_multi_slices(data, n_hvgs=n_hvgs)
    else:
        # 单切片
        return prepare_single_slice(data, n_hvgs=n_hvgs)
def calcGAEParams(graph, n_samples):
    '''graph is a bipartite graph, return pos_weight and norm_val
    '''
    non_zero_cnt = graph.sum()
    norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
    pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
    return norm_val, pos_weight


def calcGraphWeight(coor, eps=1e-6):
    dist = cdist(coor, coor, "euclidean")
    dist = dist / (np.max(dist) + eps)
    return dist


def correlation_graph(A, B):
    '''calculate correlation between A and B.
    Args:
        A (np.ndarray): sample matrix, shape: [samples, features].
        B (np.ndarray): sample matrix, shape: [samples, features].
    Returns: 
        corr (np.ndarray): correlation matrix of features, shape: [features, features].
    '''
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, eps=1e-8, svd_solver="randomized", self_weight=0.3):
    n_spots = adata.shape[0]
    assert key in ["spatial", "expr"], "case should be [spatial] or [expr]"
    if key == "spatial":
        print("create adjacent matrix from spatial idx --------------->")
        expr = adata.obsm[key]
        weights = 1./ (cdist(expr, expr, "euclidean") + eps)
    else:
        print("create adjacent matrix from pca expr --------------->")
        expr = PCA(n_components=n_comps, random_state=0, svd_solver=svd_solver).fit_transform(adata.X)
        weights = correlation_graph(expr.T, expr.T)

    print("create knn graph ---->")
    threshold = np.sort(weights)[:, -n_neighbors - 1:-n_neighbors]
    weights[weights < threshold] = 0
    weights = (weights + weights.T) / 2
    weights = weights * (1 - np.eye(n_spots))  # drop the diag

    adjFilter = 0.if key == "spatial" else 0.1
    # convert to bipartite case
    adjBip = np.where(weights > adjFilter, 1, 0)
    print(f"{key} knn graph created ----<")

    return sp.coo_matrix(symm_norm(adjBip, weightDiag=self_weight))

def symm_norm(adj, weightDiag=.3, eps=1e-8):
    '''
    args: adjacent matrix with diag = 0
    return: D^{-1/2} (A + I) D^{-1 / 2}
    '''
    n_spot = adj.shape[0]
    adj_self = (1 - weightDiag) * adj + np.eye(n_spot) * weightDiag  
    degrees = 1./ np.sqrt((np.sum(adj_self, axis=1) + eps))
    adj_self *= degrees
    adj_self *= degrees[:, None]
    return adj_self.astype(np.float32)
