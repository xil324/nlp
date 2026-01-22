import os

from torch.utils.data import TensorDataset, DataLoader

# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# 设置中文字体（如果系统中有中文字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("使用默认字体")


# 生成模拟数据
np.random.seed(42)  # 固定随机种子，确保可重现
X_numpy = np.random.rand(200, 1) * 10
y_numpy = 2 * np.sin(X_numpy) + 3 + np.random.randn(200, 1)*0.2
# 将数据转换为张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
print(f"数据生成完成。X形状: {X.shape}, y范围: [{y.min():.2f}, {y.max():.2f}]")
print("---" * 10)

class SimpleClassifier(nn.Module):

    # 调整隐藏层为多层
    def __init__(self, hidden_dim, hidden_dim2, hidden_dim3): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


# 调整不同隐藏层的节点数
hidden_dim = 64
hidden_dim2 = 128
hidden_dim3 = 64

# 创建模型
model = SimpleClassifier(hidden_dim, hidden_dim2, hidden_dim3)

# 定义损失函数
loss_fn = torch.nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 准备数据加载器（数据封装）
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 循环训练
num_epochs = 1000
losses = []  # 记录损失
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for idx, (x, y) in enumerate(dataloader):
        # 清除梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(x)
        # 计算损失
        loss = loss_fn(outputs, y)
        # 反向传播和优化
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        running_loss += loss.item()
        # if idx % 50 == 0:
            # print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 计算平均损失
    avg_loss = running_loss / len(dataloader)
    losses.append(avg_loss)

    # 每50轮打印一次
    if (epoch + 1) % 50 == 0:
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')

print("训练完成！")
print(f"最终Loss: {loss.item():.4f}")
print("---" * 10)

# 绘制结果

# 生成测试数据（0-10间生成200个等间距的点）
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
# 将数据转换为张量（因为模型只能使用张量计算）
X_test_tensor = torch.from_numpy(X_test).float()

# 切换模型预测
model.eval()
# 上下文管理器，关闭梯度计算
with torch.no_grad():
    # 用训练好的参数计算预测值，并转换为NumPy数组
    y_test_pred = model(X_test_tensor).numpy()

# 创建新的图形窗口，figsize=(12, 5)：设置图形大小为12英寸宽，5英寸高
plt.figure(figsize=(12, 5))

# 绘制散点图函数：x轴、y轴、标签、颜色、透明度
plt.scatter(X_numpy, y_numpy, label='原始数据', color='blue', alpha=0.6)

# 绘制预测值线（r:red;-:实线）
plt.plot(X_test, y_test_pred, 'r-', linewidth=3, label='神经网络拟合')

# 设置x轴标签
plt.xlabel('X')

# 设置y轴标签
plt.ylabel('y')

plt.title('神经网络拟合非线性函数')

# 显示图例
plt.legend()

# 添加网格
plt.grid(True)

# 显示图形窗口
plt.show()
print("绘制完成！")


