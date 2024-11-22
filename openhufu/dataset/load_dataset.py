import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
from torch.utils.data import Dataset, DataLoader

class CodeAlpacaDataset(Dataset):
    def __init__(self, CodeAlpaca_path):
        """
        初始化数据集，加载 JSON 文件
        :param CodeAlpaca_path: 数据集文件路径
        """
        # 检查文件是否存在
        if not os.path.exists(CodeAlpaca_path):
            raise FileNotFoundError(f"Dataset file not found at {CodeAlpaca_path}")
        
        # 加载 JSON 数据
        with open(CodeAlpaca_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
    def __len__(self):
        """
        返回数据集样本数量
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        根据索引返回一个数据样本
        :param idx: 索引
        :return: 一个样本，包括 instruction, input, 和 output
        """
        sample = self.data[idx]
        instruction = sample["instruction"]
        input_text = sample["input"]
        output_text = sample["output"]
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
        }
    
# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接数据集路径
CodeAlpaca_path = os.path.join(current_dir, "code_alpaca_20k.json")

if __name__ == "__main__":
    # 加载自定义数据集
    dataset = CodeAlpacaDataset(CodeAlpaca_path)

    # 使用 DataLoader 进行批量加载
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 测试 DataLoader 输出
    print("=== DataLoader Test ===")
    for batch in dataloader:
        print("Batch:")
        print(batch)
        break
