import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from load_dataset import CodeAlpacaDataset
from splitter  import LDASplitter  

# 定义instruction中涉及的编程语言及其关键词
LANGUAGE_KEYWORDS = {
    "Python": ["Python", "py"],
    "Java": ["Java"],
    "C++": ["C++", "cpp"],
    "JavaScript": ["JavaScript", "JS"],
    "C#": ["C#"],
    "Ruby": ["Ruby"],
    "Go": ["Go"],
    "PHP": ["PHP"],
    "Others": []  # 未匹配的默认归为 "其他"
}

def classify_by_language(instruction):
    """
    根据 instruction 中的关键词分类到特定编程语言。
    """
    for language, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in instruction.lower():
                return language
    return "Others"

def generate_labels_by_language(dataset):
    """
    为数据集生成编程语言分类标签。
    """
    labels = []
    language_to_label = {language: idx for idx, language in enumerate(LANGUAGE_KEYWORDS)}
    for sample in dataset:
        instruction = sample["instruction"]
        language = classify_by_language(instruction)
        labels.append(language_to_label[language])
    return labels, language_to_label

if __name__ == "__main__":
    # 加载数据集
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CodeAlpaca_path = os.path.join(current_dir, "code_alpaca_20k.json")
    dataset = CodeAlpacaDataset(CodeAlpaca_path)

    # 根据编程语言分类生成标签
    labels, language_to_label = generate_labels_by_language(dataset)

    # 使用 LDA Splitter 进行划分
    client_num = 5
    alpha = 0.5
    lda_splitter = LDASplitter(client_num, alpha, seed=42)
    subsets = lda_splitter(dataset, labels)

    # 打印测试结果
    print("=== LDA Split Test ===")
    for client_id, subset in enumerate(subsets):
        subset_labels = [labels[idx] for idx in subset.indices]
        print(f"Client {client_id}:")
        print(f"  Number of samples: {len(subset)}")
        label_distribution = np.bincount(subset_labels, minlength=len(language_to_label))
        for language, idx in language_to_label.items():
            print(f"    {language}: {label_distribution[idx]}")

