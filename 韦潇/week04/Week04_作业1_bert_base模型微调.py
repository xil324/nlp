import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset

# 本地BERT模型路径
MODEL_LOCAL_PATH = r"E:\AI\miniconda3\envs\py312\models\google-bert\bert-base-chinese"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 定义标签ID与中文名称的映射关系（核心修复点）
label_map = {
    0: "新闻",
    1: "情感",
    2: "科技",
    3: "生活"
}
# 反向映射，用于后续解码预测结果
id2label = {v: k for k, v in label_map.items()}
num_labels = len(label_map)

# 构建标注数据集，直接使用标签ID，无需额外编码
data = [
    # 类别0：新闻
    ("央行下调存款准备金率，释放长期流动性约5000亿元", 0),
    ("全国铁路暑运收官，累计发送旅客超8亿人次", 0),
    ("多地出台新政，优化房地产市场调控措施", 0),
    ("上半年国内生产总值同比增长5.2%，经济运行平稳", 0),
    ("外交部回应国际经贸合作新进展，坚持多边主义", 0),
    # 类别1：情感
    ("今天吃到了超好吃的蛋糕，心情特别棒！", 1),
    ("加班到深夜，真的又累又烦躁", 1),
    ("和老朋友见面聊天，幸福感直接拉满", 1),
    ("新买的耳机坏了，太让人失望了", 1),
    ("看到晚霞铺满天空，所有烦恼都消失了", 1),
    # 类别2：科技
    ("国产大模型迭代升级，推理速度提升30%", 2),
    ("新款智能手机发布，搭载自研AI图像处理芯片", 2),
    ("量子计算机实现重大突破，成功模拟复杂分子结构", 2),
    ("5G-A技术商用加速，万物互联场景全面落地", 2),
    ("开源深度学习框架新增轻量化模型部署功能", 2),
    # 类别3：生活
    ("自制番茄牛腩教程，软烂入味超下饭", 3),
    ("周末去郊野公园露营，感受自然风景", 3),
    ("家用扫地机器人选购攻略，避坑指南", 3),
    ("冬季保湿护肤技巧，告别干燥起皮", 3),
    ("城市骑行路线推荐，沿途打卡网红景点", 3),
]

# 扩充数据集
data_expand = data * 10
# 拆分文本和标签ID
texts = [item[0] for item in data_expand]
labels = [item[1] for item in data_expand]

# 分层划分训练集/测试集
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)
print(f"数据集分类总数：{num_labels}，分类名称：{list(label_map.values())}")

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)
# 加载分类模型，传入标签映射，模型内部也会识别中文标签
model = BertForSequenceClassification.from_pretrained(
    MODEL_LOCAL_PATH,
    num_labels=num_labels,
    local_files_only=True,
    id2label=label_map,  # 传入标签映射
    label2id=id2label
)

# 文本编码预处理
max_seq_len = 64
train_encodings = tokenizer(
    x_train, truncation=True, padding=True, max_length=max_seq_len
)
test_encodings = tokenizer(
    x_test, truncation=True, padding=True, max_length=max_seq_len
)

# 构建Dataset并转换为PyTorch张量
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": y_train
}).with_format("torch")

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": y_test
}).with_format("torch")


# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}


# 配置训练参数
training_args = TrainingArguments(
    output_dir="./bert_text_classify_results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=SEED,
    overwrite_output_dir=True
)

# 初始化训练器并启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("—————— 开始BERT模型微调 ——————")
trainer.train()

# 模型评估
print("——————— 模型评估结果 ———————")
eval_result = trainer.evaluate()
print(eval_result)


def predict_single_text(text: str):
    """单条文本预测，输出中文分类名称"""
    # 文本编码
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_seq_len,
        return_tensors="pt"
    )
    # 推理模式
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测标签ID
    pred_label_id = torch.argmax(outputs.logits, dim=1).item()
    # 通过映射字典获取中文名称
    pred_label_name = label_map[pred_label_id]

    # 格式化输出
    print(f"\n——————— 预测结果 ————————")
    print(f"输入文本：{text}")
    print(f"预测分类ID：{pred_label_id}")
    print(f"预测分类名称：{pred_label_name}")
    print(f"———————————————————————————\n")
    return pred_label_name


# 测试新样本
# 新闻类样本
predict_single_text("国家统计局发布11月消费物价指数，市场运行总体平稳")
# 科技类样本
predict_single_text("全新AI助手上线，支持多轮对话与代码生成功能")
# 情感类样本
predict_single_text("终于完成了项目答辩，整个人都轻松了！")
# 生活类样本
predict_single_text("自制寿喜烧锅底，搭配肥牛和蔬菜，冬日暖心美食")
# 自定义测试样本
predict_single_text("新款电动汽车发布，续航里程突破1000公里")
