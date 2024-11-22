import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from torch.utils.data import Dataset, Subset
from collections import defaultdict

class IIDSplitter:
    """
    A simple class to split a dataset into IID subsets for federated learning.
    
    Args:
        client_num: The number of clients to split the dataset into.
        seed: Random seed for reproducibility (default: None).
    """
    def __init__(self, client_num, seed=None):
        self.client_num = client_num
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)  

    def __call__(self, dataset):
        """
        Split the dataset into `client_num` IID subsets.
        
        Args:
            dataset: A PyTorch Dataset object or a list of samples.
            
        Returns:
            A list of datasets (or subsets) for each client.
        """
        # 样本总数
        total_samples = len(dataset)
        
        # 创建一个索引数组并随机打乱
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # 将索引拆分为‘ client_num ’相等或几乎相等的部分
        idx_slices = np.array_split(indices, self.client_num)
        
        # 创建子数据集
        if isinstance(dataset, Dataset):
            # 如果属于torch的Dataset类，则创建Subset对象
            subsets = [Subset(dataset, idxs) for idxs in idx_slices]
        else:
            # 否则创建原始分片
            subsets = [[dataset[idx] for idx in idxs] for idxs in idx_slices]
        
        return subsets


class LDASplitter:
    """
    Split dataset using LDA-inspired distribution to simulate non-IID data.
    
    Args:
        client_num: Number of clients.
        alpha: Dirichlet distribution parameter controlling the data heterogeneity.
        seed: Random seed for reproducibility (default: None).
    """
    def __init__(self, client_num, alpha=0.5, seed=None):
        self.client_num = client_num
        self.alpha = alpha
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, dataset, labels):
        """
        Split dataset into non-IID subsets using LDA-like method.
        
        Args:
            dataset: A PyTorch Dataset object.
            labels: A list or array of dataset labels.
            
        Returns:
            A list of subsets for each client.
        """
        # 获取类别分布
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # 根据 Dirichlet 分布生成客户端的类别占比
        class_dist = np.random.dirichlet([self.alpha] * num_classes, self.client_num)
        
        # 构造每个类别的样本索引
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)
        
        # 按照分布分配样本
        client_indices = [[] for _ in range(self.client_num)]
        for label, indices in label_indices.items():
            np.random.shuffle(indices)  # 打乱每个类别的索引
            splits = np.array_split(indices, self.client_num)  # 将类别样本分为客户端数量
            for client_id, split in enumerate(splits):
                num_samples = int(len(indices) * class_dist[client_id, label])
                client_indices[client_id].extend(split[:num_samples])
        
        # 构造客户端数据子集
        subsets = [Subset(dataset, indices) for indices in client_indices]
        return subsets
    



