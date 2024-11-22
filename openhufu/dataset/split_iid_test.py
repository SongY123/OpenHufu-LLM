import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from load_dataset import CodeAlpacaDataset
from splitter import IIDSplitter
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # 加载数据集
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "code_alpaca_20k.json")
    dataset = CodeAlpacaDataset(dataset_path)
    
    # 定义客户端数量
    client_num = 5
    
    # 创建分割器实例
    splitter = IIDSplitter(client_num, seed=42)  
    
    # 执行数据划分
    client_datasets = splitter(dataset)
    
    # 打印每个客户端的数据量
    print("=== IID Splitting Results ===")
    for i, client_data in enumerate(client_datasets):
        print(f"Client {i + 1}: {len(client_data)} samples")

    
    # 测试第一个客户端的数据加载
    client_loader = DataLoader(client_datasets[0], batch_size=4, shuffle=True)
    print("=== First Client DataLoader Test ===")
    for batch in client_loader:
        print("Batch:")
        print(batch)
        break
