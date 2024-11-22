import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import os

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 自定义数据集 - 修改为使用explanation
class StanceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=168):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # 使用explanation而不是text
        encoding = self.tokenizer(
            item['explanation'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'explanation': item['explanation'],
            'label': item['true_label'],
            'combined_against': torch.tensor(item['combined_against'], dtype=torch.float),
            'combined_favor': torch.tensor(item['combined_favor'], dtype=torch.float),
            'combined_neutral': torch.tensor(item['combined_neutral'], dtype=torch.float)
        }
    


def contrastive_loss(features, labels, temperature=0.05):
    batch_size = features.size(0)
    features = F.normalize(features, dim=1)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)
    
    # 创建标签矩阵，相同标签的样本为正例
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    label_matrix = label_matrix.masked_fill(mask, False)
    
    # 计算每个样本的正例数量
    positive_counts = label_matrix.sum(dim=1)
    
    # 如果某些样本没有正例，添加自身作为正例
    no_positives = positive_counts == 0
    if no_positives.any():
        label_matrix = label_matrix.clone()
        label_matrix[no_positives] = mask[no_positives]
    
    # 分别获取正例和负例
    positive_similarities = []
    negative_similarities = []
    
    for i in range(batch_size):
        positive_mask = label_matrix[i]
        negative_mask = ~label_matrix[i] & ~mask[i]
        
        # 收集正例
        if positive_mask.any():
            positive_similarities.append(similarity_matrix[i][positive_mask])
        else:
            # 如果没有正例，使用一个填充值
            positive_similarities.append(torch.tensor([0.0], device=features.device))
            
        # 收集负例
        if negative_mask.any():
            negative_similarities.append(similarity_matrix[i][negative_mask])
        else:
            # 如果没有负例，使用一个填充值
            negative_similarities.append(torch.tensor([0.0], device=features.device))
    
    # 填充到相同长度
    max_pos_len = max(pos.size(0) for pos in positive_similarities)
    max_neg_len = max(neg.size(0) for neg in negative_similarities)
    
    # 使用填充将所有样本扩展到相同大小
    padded_positives = torch.stack([
        F.pad(pos, (0, max_pos_len - pos.size(0)), value=-float('inf'))
        for pos in positive_similarities
    ])
    
    padded_negatives = torch.stack([
        F.pad(neg, (0, max_neg_len - neg.size(0)), value=-float('inf'))
        for neg in negative_similarities
    ])
    
    # 组合正例和负例logits
    logits = torch.cat([padded_positives, padded_negatives], dim=1) / temperature
    
    # 创建目标标签（第一个位置为正例）
    targets = torch.zeros(batch_size, dtype=torch.long, device=features.device)
    
    # 计算交叉熵损失
    loss = F.cross_entropy(logits, targets, ignore_index=-1)
    
    return loss

def contrastive_loss_with_debug(features, labels, temperature=0.05):
    """带有调试信息的版本"""
    batch_size = features.size(0)
    features = F.normalize(features, dim=1)
    
    print(f"Batch size: {batch_size}")
    print(f"Features shape: {features.shape}")
    
    similarity_matrix = torch.matmul(features, features.T)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    print(f"Label matrix shape: {label_matrix.shape}")
    print(f"Number of positive pairs: {label_matrix.sum().item()}")
    
    mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    label_matrix = label_matrix.masked_fill(mask, False)
    
    return contrastive_loss(features, labels, temperature)

# 组合模型
class StanceModel(nn.Module):
    def __init__(self, bert_model, pooling='cls'):
        super().__init__()
        self.bert = bert_model
        self.pooling = pooling
        self.classifier = nn.Linear(bert_model.config.hidden_size, 3)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if self.pooling == 'cls':
            pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            pooled_output = torch.mean(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 在eval模式下，返回logits和softmax概率
        if self.training is False:
            probabilities = F.softmax(logits, dim=-1)
            return logits, probabilities
        
        return logits, pooled_output

def create_pairs(data):
    """创建正负样本对"""
    pairs = []
    label_groups = {}
    
    # 按标签分组
    for item in data:
        label = item['true_label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # 为每个样本创建正负样本对
    for label, items in label_groups.items():
        for i, anchor in enumerate(items):
            # 正例：同一标签的其他样本
            positive_pool = items[:i] + items[i+1:]
            if positive_pool:
                positive = random.choice(positive_pool)
                pairs.append((anchor, positive, 1))
            
            # 负例：不同标签的样本
            negative_pool = []
            for other_label, other_items in label_groups.items():
                if other_label != label:
                    negative_pool.extend(other_items)
            if negative_pool:
                negative = random.choice(negative_pool)
                pairs.append((anchor, negative, 0))
    
    return pairs

def train(model, tokenizer, train_loader, val_loader, device, output_dir, num_epochs=5):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)
        for batch in train_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            logits, features = model(input_ids, attention_mask)
            
            # 计算交叉熵损失
            ce_loss = criterion(logits, labels)
            
            # 计算对比损失
            cont_loss = contrastive_loss(features, labels)
            
            # 组合损失
            loss = ce_loss + 0.1 * cont_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条描述
            train_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证
        val_loss, accuracy = evaluate(model, val_loader, device, criterion)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, tokenizer, output_dir)  # 使用新的保存函数


def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, features = model(input_ids, attention_mask)
            
            ce_loss = criterion(logits, labels)
            cont_loss = contrastive_loss_with_debug(features, labels)
            loss = ce_loss + 0.1 * cont_loss
            
            val_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_val_loss, accuracy



def save_model(model, tokenizer, output_dir):
    """
    保存完整模型状态和必要的配置文件，避免重复保存BERT参数。
    
    Args:
        model: StanceModel实例
        tokenizer: 使用的tokenizer
        output_dir: 保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存BERT模型及其配置文件（包括 config.json, vocab.txt, etc.）
    model.bert.save_pretrained(output_dir)
    
    # 保存自定义模型的状态字典（不包括BERT参数）
    model_path = os.path.join(output_dir, 'model_state.pth')
    # 只保存分类器和其他自定义层的参数
    torch.save({
        'classifier_state_dict': model.classifier.state_dict(),
        'dropout_state_dict': model.dropout.state_dict(),
        'model_config': {
            'pooling': model.pooling
        }
    }, model_path)
    
    # 保存tokenizer（包含 config.json, special_tokens_map.json, tokenizer_config.json 等）
    tokenizer.save_pretrained(output_dir)
    
    print(f'Model and tokenizer saved to {output_dir}')

# ---------------------
# 载入模型和tokenizer的函数
def load_model(output_dir, device):
    """
    载入保存的模型和tokenizer，避免重复加载BERT参数。
    
    Args:
        output_dir: 模型和tokenizer保存的目录
        device: 使用的设备
    Returns:
        model: 载入的StanceModel实例
        tokenizer: 载入的tokenizer实例
    """
    # 载入tokenizer（包含 config.json, special_tokens_map.json, tokenizer_config.json 等）
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # 载入BERT模型（包含 config.json 和模型权重）
    bert = AutoModel.from_pretrained(output_dir)
    
    # 初始化自定义模型
    model = StanceModel(bert, pooling='cls').to(device)
    
    # 载入自定义模型的状态字典（分类器和 dropout）
    model_state_path = os.path.join(output_dir, 'model_state.pth')
    checkpoint = torch.load(model_state_path, map_location=device)
    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.dropout.load_state_dict(checkpoint['dropout_state_dict'])
    model.pooling = checkpoint['model_config']['pooling']
    
    model.eval()  # 设置为评估模式
    return model, tokenizer

# 使用示例
def predict_stance(model, tokenizer, texts, device):
    """
    预测文本的立场
    Args:
        model: 加载的StanceModel实例
        tokenizer: 分词器
        texts: 要预测的文本列表
        device: 运行设备
    Returns:
        predictions: 预测结果列表
    """
    model.eval()
    id2label = {0: "反对", 1: "中立", 2: "支持"}
    
    # 对文本进行编码
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    # 预测
    with torch.no_grad():
        logits, probabilities = model(**encoded)
        predictions = torch.argmax(logits, dim=1)
    
    # 整理结果
    results = []
    for text, pred, probs in zip(texts, predictions, probabilities):
        stance = id2label[pred.item()]
        confidence = probs[pred].item()
        
        results.append({
            'text': text,
            'stance': stance,
            'confidence': f"{confidence:.2%}",
            'probabilities': {
                '反对': f"{probs[0].item():.2%}",
                '中立': f"{probs[1].item():.2%}",
                '支持': f"{probs[2].item():.2%}"
            }
        })
    
    return results



import argparse


def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description='模型训练和评估')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='运行设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--train_file_path', type=str, default='./datasets/eval.jsonl', help='训练数据文件路径')
    parser.add_argument('--model_file', type=str, default='E:/code/stanceberta-l' if os.path.exists('E:/code') else '/data/syh/yy/model/stanceberta-l', help='模型文件路径')
    parser.add_argument('--output_dir', type=str, default='./saved_model/stanceberta_1', help='模型保存路径')
    parser.add_argument('--num_epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='每个批次的样本数量')
    parser.add_argument('--only_val', action='store_true', help='仅进行案例验证')


    args = parser.parse_args()


    if not args.only_val:
        # 设置设备和随机种子
        set_seed(args.seed)
        device = torch.device(args.device)

        # 加载数据
        data = []
        with open(args.train_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item['true_label'] == item['pred_label']:  # 只选择预测正确的样本
                        data.append(item)
                except json.JSONDecodeError as e:
                    print(f"JSON解码错误: {e}，在行: {line}")
        
        # 创建样本对
        pairs = create_pairs(data)
        
        # 划分数据集
        data = [item for item in data if not any('\u4e00' <= char <= '\u9fff' for char in item['explanation'])]
        random.shuffle(data)
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        val_data = data[train_size:]
        # train_data = data[:80]
        # val_data = data[-20:]

        # 加载模型和分词器
        bert = AutoModel.from_pretrained(args.model_file)
        tokenizer = AutoTokenizer.from_pretrained(args.model_file)
        
        # 创建数据集和数据加载器
        train_dataset = StanceDataset(train_data, tokenizer)
        val_dataset = StanceDataset(val_data, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # 初始化模型
        model = StanceModel(bert, pooling='cls').to(device)
        
        # 训练模型
        train(model, tokenizer, train_loader, val_loader, device, output_dir=args.output_dir, num_epochs=args.num_epochs)

    else:

        device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # 加载模型和分词器
        try:
            loaded_model, loaded_tokenizer = load_model(args.output_dir, device)
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # 测试文本
        new_test_texts = [
            "Hillary Clinton has done an amazing job in her campaign.",
            "I really dislike how Hillary handles issues.",
            "Hillary's new policies are quite neutral and don't affect me.",
            "The text mentions an organization (CalAlimony) that aims to end alimony, which is a financial obligation often affecting men post-divorce. This could be seen as a stance against the feminist movement, which often advocates for women's rights in divorce and financial matters. However, the text does not explicitly state a stance on the feminist movement, making it more neutral.",
            "The text sarcastically uses the logic of 'equality' to criticize alimony, implying that it unfairly benefits women post-divorce. This suggests a critical or against stance toward the feminist movement, particularly in the context of divorce and financial obligations.",
            "The text does not directly mention the Feminist Movement but expresses a critical stance towards feminists and SJWs, associating them with authoritarianism and world domination. This reflects an against stance towards the Feminist Movement and its allies.",
            " The text reflects a concern for the reform of alimony laws. The promotion of termination of alimony can be seen as a way to address the perceived inequalities in divorce settlements",
            "The text discusses an organization (CalAlimony) that aims to end alimony, which is a financial obligation often affecting men post-divorce. This could be seen as against the feminist movement, which often advocates for women's rights and financial independence. However, the text does not explicitly state a stance on the feminist movement itself, but the context suggests a potential opposition.",
        ]
        
        # 进行预测
        try:
            results = predict_stance(loaded_model, loaded_tokenizer, new_test_texts, device)
            
            # 打印结果
            print("\n预测结果：")
            print("-" * 80)
            for result in results:
                print(f"\n文本: {result['text']}")
                print(f"预测立场: {result['stance']}")
                print(f"置信度: {result['confidence']}")
                print("\n各类别概率：")
                for stance, prob in result['probabilities'].items():
                    print(f"  {stance}: {prob}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()