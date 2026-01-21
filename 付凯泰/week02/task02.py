import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成sin函数模拟数据
X_numpy = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
# 形状为 (1000, 1) 的二维数组，在 [-π, π] 范围内生成 1000 个等间距的点

y_numpy = np.sin(X_numpy) + np.random.randn(1000,1)*0.1
X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X shape：{X.shape}，y shape：{y.shape}")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
class SinFittingNet(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim):
        super(SinFittingNet,self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim,hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim,output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self,x):
        return self.network(x)

model = SinFittingNet(input_dim = 1,hidden_dims = [64,32,16],output_dim = 1)

print("===========神经网络模型创建完成===========")
print(model)
print("---" * 10)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)

# 4. 训练模型
num_epochs = 2000
loss_history = []
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 记录损失，每10个epoch记录一次，避免太多点
    if epoch % 10 == 0:
        loss_history.append(loss.item())

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print(f"最终损失：{loss.item():.6f}")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
model.eval()
with torch.no_grad():
    y_predicted = model(X)

# 准备绘图数据
X_plot = X_numpy.flatten()
y_true_plot = np.sin(X_numpy).flatten()  # 真实的sin函数值
y_pred_plot = y_predicted.detach().numpy().flatten()  # 预测值

plt.figure(figsize=(12, 8))

# 绘制真实sin函数和神经网络拟合结果
plt.subplot(2, 1, 1)
plt.scatter(X_plot, y_numpy.flatten(), label='Noisy Data (sin(x) + noise)', color='lightblue', alpha=0.5, s=10)
plt.plot(X_plot, y_true_plot, label='True sin(x)', color='green', linewidth=2)
plt.plot(X_plot, y_pred_plot, label='Neural Network Fit', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Fitting to sin(x) Function')
plt.legend()
plt.grid(True, alpha=0.3)


# 第二个子图：显示训练损失曲线
plt.subplot(2, 1, 2)
epochs_recorded = list(range(0, num_epochs, 10))  # 对应的epoch数
plt.plot(epochs_recorded, loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
