import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的数据
X_numpy = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)  # 更密集的数据点
y_numpy = np.sin(X_numpy)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin函数数据生成完成。")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)

# 2. 定义多层神经网络
class MultiLayerNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MultiLayerNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())  # 使用ReLU激活函数增加非线性
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 3. 定义不同的网络结构进行对比
network_configs = [
    {
        'hidden_dims': [10],
        'name': '单隐藏层_10节点'
    },
    {
        'hidden_dims': [20],
        'name': '单隐藏层_20节点'
    },
    {
        'hidden_dims': [10, 10],
        'name': '双隐藏层_10_10节点'
    },
    {
        'hidden_dims': [20, 20],
        'name': '双隐藏层_20_20节点'
    },
    {
        'hidden_dims': [30, 20, 10],
        'name': '三隐藏层_30_20_10节点'
    }
]

# 4. 训练不同配置的网络并比较
results = []

for config in network_configs:
    print(f"开始训练: {config['name']}")
    
    # 创建网络
    model = MultiLayerNet(input_dim=1, hidden_dims=config['hidden_dims'], output_dim=1)
    
    # 损失函数和优化器
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器效果更好
    
    # 训练参数
    num_epochs = 2000
    
    # 记录训练过程中的损失
    losses = []
    
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(X)
        
        # 计算损失
        loss = loss_fn(y_pred, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 每200个epoch打印一次损失
        if (epoch + 1) % 200 == 0:
            print(f'  Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    # 保存结果
    results.append({
        'model': model,
        'losses': losses,
        'name': config['name'],
        'final_loss': losses[-1]
    })
    
    print(f"  {config['name']} 训练完成，最终损失: {losses[-1]:.6f}")
    print("---" * 10)

# 5. 绘制损失曲线对比图
plt.figure(figsize=(15, 10))

# 绘制所有模型的损失曲线
for result in results:
    # 只绘制每10个点，使图形更清晰
    plot_losses = result['losses'][::10]
    epochs = range(0, len(result['losses']), 10)
    plt.plot(epochs, plot_losses, label=result['name'], linewidth=1.5)

plt.title('不同网络结构的训练损失对比')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数刻度以便更好地显示损失变化
plt.show()

# 6. 绘制各个模型对sin函数的拟合效果对比
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2行3列
axes = axes.ravel()

x_test = torch.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1).float()

for i, result in enumerate(results):
    model = result['model']
    
    # 使用训练好的模型预测
    with torch.no_grad():
        y_pred = model(x_test)
    
    # 绘制原始sin函数和模型预测结果
    axes[i].plot(x_test.numpy(), np.sin(x_test.numpy()), label='真实sin(x)', color='blue', linewidth=2)
    axes[i].plot(x_test.numpy(), y_pred.numpy(), label='模型预测', color='red', linestyle='--', linewidth=2)
    axes[i].set_title(f'{result["name"]}\n最终Loss: {result["final_loss"]:.6f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# 隐藏最后一个子图（因为我们只有5个模型，但预留了6个位置）
axes[5].axis('off')

plt.tight_layout()
plt.show()

# 7. 打印各模型的最终损失对比
print("\n各模型最终损失对比:")
for result in results:
    print(f"{result['name']}: 最终Loss = {result['final_loss']:.6f}")

# 8. 选择表现最好的模型进行详细分析
best_result = min(results, key=lambda x: x['final_loss'])
print(f"\n最佳模型: {best_result['name']}, 最终Loss = {best_result['final_loss']:.6f}")

# 使用最佳模型进行预测并可视化
with torch.no_grad():
    best_predictions = best_result['model'](x_test)

plt.figure(figsize=(12, 6))
plt.plot(x_test.numpy(), np.sin(x_test.numpy()), label='真实sin(x)', color='blue', linewidth=2)
plt.plot(x_test.numpy(), best_predictions.numpy(), label=f'{best_result["name"]} 预测', color='red', linestyle='--', linewidth=2)
plt.title(f'最佳模型拟合效果: {best_result["name"]}\n最终Loss: {best_result["final_loss"]:.6f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 9. 显示网络结构信息
print(f"\n最佳网络结构详情:")
best_model = best_result['model']
print(best_model)
