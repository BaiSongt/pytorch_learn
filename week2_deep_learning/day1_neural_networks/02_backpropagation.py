import torch

# --- 前言 ---
# 反向传播 (Backpropagation) 是驱动神经网络学习的核心算法。
# 它本质上是“链式法则”在神经网络中的应用，用于高效地计算损失函数关于网络中每一个参数（权重、偏置）的梯度。
# PyTorch的autograd引擎为我们自动完成了这个过程，但理解其工作原理至关重要。
# 本脚本将通过一个极简的例子，手动模拟反向传播，并与PyTorch的自动计算结果进行对比。

# --- 1. 定义一个简单的前向传播 ---
# 假设一个非常简单的网络：
# 输入 x -> 线性层1 (w1, b1) -> ReLU激活 -> 线性层2 (w2, b2) -> 输出 y_pred
# 损失函数：MSE = (y_pred - y_true)^2

# 初始化参数和数据
x = torch.tensor(2.0) # 输入
y_true = torch.tensor(1.0) # 真实标签

# 权重和偏置，设置 requires_grad=True 来让PyTorch追踪它们的梯度
w1 = torch.tensor(0.5, requires_grad=True)
b1 = torch.tensor(0.1, requires_grad=True)
w2 = torch.tensor(-0.8, requires_grad=True)
b2 = torch.tensor(0.2, requires_grad=True)

# 手动执行前向传播
print("---" + " 1. Forward Pass " + "---")
# Layer 1
lin1_out = w1 * x + b1
relu_out = torch.relu(lin1_out)
# Layer 2
y_pred = w2 * relu_out + b2

# 计算损失
loss = (y_pred - y_true)**2

print(f"Input x: {x}")
print(f"Linear 1 Output: {lin1_out:.4f}")
print(f"ReLU Output: {relu_out:.4f}")
print(f"Predicted Output y_pred: {y_pred:.4f}")
print(f"True Label y_true: {y_true}")
print(f"Loss: {loss:.4f}")
print("-"*30)

# --- 2. 手动反向传播 (使用链式法则) ---
# 我们的目标是计算 dLoss/dw1, dLoss/db1, dLoss/dw2, dLoss/db2

print("---" + " 2. Manual Backpropagation " + "---")

# a. 计算最末端的梯度: dLoss/dy_pred
# Loss = (y_pred - y_true)^2  =>  dLoss/dy_pred = 2 * (y_pred - y_true)
grad_loss_ypred = 2 * (y_pred - y_true)
print(f"dLoss/dy_pred = {grad_loss_ypred:.4f}")

# b. 计算关于w2和b2的梯度
# y_pred = w2 * relu_out + b2
# dLoss/dw2 = (dLoss/dy_pred) * (dy_pred/dw2) = grad_loss_ypred * relu_out
grad_loss_w2 = grad_loss_ypred * relu_out
# dLoss/db2 = (dLoss/dy_pred) * (dy_pred/db2) = grad_loss_ypred * 1
grad_loss_b2 = grad_loss_ypred
print(f"dLoss/dw2 = {grad_loss_w2:.4f}")
print(f"dLoss/db2 = {grad_loss_b2:.4f}")

# c. 将梯度传播到ReLU层之前: dLoss/drelu_out
# dLoss/drelu_out = (dLoss/dy_pred) * (dy_pred/drelu_out) = grad_loss_ypred * w2
grad_loss_relu_out = grad_loss_ypred * w2
print(f"dLoss/drelu_out = {grad_loss_relu_out:.4f}")

# d. 计算通过ReLU层的梯度: dLoss/dlin1_out
# relu_out = relu(lin1_out)
# drelu/dx = 1 if x > 0, else 0
grad_relu_lin1_out = 1.0 if lin1_out > 0 else 0.0
grad_loss_lin1_out = grad_loss_relu_out * grad_relu_lin1_out
print(f"dLoss/dlin1_out = {grad_loss_lin1_out:.4f}")

# e. 计算关于w1和b1的梯度
# lin1_out = w1 * x + b1
# dLoss/dw1 = (dLoss/dlin1_out) * (dlin1_out/dw1) = grad_loss_lin1_out * x
grad_loss_w1 = grad_loss_lin1_out * x
# dLoss/db1 = (dLoss/dlin1_out) * (dlin1_out/db1) = grad_loss_lin1_out * 1
grad_loss_b1 = grad_loss_lin1_out
print(f"dLoss/dw1 = {grad_loss_w1:.4f}")
print(f"dLoss/db1 = {grad_loss_b1:.4f}")
print("-"*30)

# --- 3. 使用PyTorch自动反向传播 ---
print("---" + " 3. Automatic Backpropagation with PyTorch " + "---")

# 调用 .backward() 会自动计算loss关于所有requires_grad=True的张量的梯度
loss.backward()

print(f"PyTorch gradient for w1: {w1.grad.item():.4f}")
print(f"PyTorch gradient for b1: {b1.grad.item():.4f}")
print(f"PyTorch gradient for w2: {w2.grad.item():.4f}")
print(f"PyTorch gradient for b2: {b2.grad.item():.4f}")
print("\n观察：自动计算的结果与我们手动推导的结果完全一致！")
print("-"*30)

# --- 4. 手动更新权重 ---
# 在PyTorch的优化器（如SGD）中，`optimizer.step()` 会执行这一步。
# 更新规则: new_weight = old_weight - learning_rate * gradient

print("---" + " 4. Manual Weight Update " + "---")
learning_rate = 0.01

# 使用 .no_grad() 上下文，因为我们不希望更新权重的操作本身被梯度追踪
with torch.no_grad():
    w1_new = w1 - learning_rate * w1.grad
    b1_new = b1 - learning_rate * b1.grad
    w2_new = w2 - learning_rate * w2.grad
    b2_new = b2 - learning_rate * b2.grad

print(f"Original w1: {w1.item():.4f} -> New w1: {w1_new.item():.4f}")
print(f"Original w2: {w2.item():.4f} -> New w2: {w2_new.item():.4f}")

# 总结:
# 1. 反向传播是应用链式法则从后向前逐层计算梯度的过程。
# 2. PyTorch的 `loss.backward()` 自动完成了所有这些复杂的计算。
# 3. 计算出的梯度指明了让损失函数下降最快的方向。
# 4. 优化器 (Optimizer) 根据这些梯度和学习率来更新模型的权重，从而使模型在下一次预测时表现得更好。
