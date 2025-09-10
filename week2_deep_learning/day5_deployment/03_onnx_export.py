
import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np

# --- 前言 ---
# ONNX (Open Neural Network Exchange) 是一个开放的模型表示标准。
# 它的目标是让AI模型具有互操作性。也就是说，你可以在一个框架（如PyTorch）中训练模型，
# 然后将其转换为ONNX格式，最后在另一个框架（如TensorFlow）或专门的推理引擎（如ONNX Runtime, TensorRT）中运行它。
# 这对于需要将模型部署到不同硬件或平台的场景非常有用。

# --- 1. 准备一个简单的PyTorch模型 ---
# 我们使用一个简单的CNN模型作为示例。
# 注意：要导出到ONNX，模型最好是直接的、没有复杂控制流的。

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10) # 假设输入是 3x32x32

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

model = SimpleCNN()
model.eval() # 必须设置为评估模式

# 创建一个虚拟输入
# ONNX导出过程需要一个示例输入来“追踪”模型的计算图
batch_size = 1
dummy_input = torch.randn(batch_size, 3, 32, 32)

print("--- 1. A standard PyTorch model ---")
print(model)
print("-"*30)

# --- 2. 导出到ONNX格式 ---
# 使用 `torch.onnx.export()` 函数

print("--- 2. Exporting to ONNX format ---")

onnx_model_path = "simple_cnn.onnx"

# 导出模型
torch.onnx.export(
    model,               # 要导出的模型
    dummy_input,         # 一个示例输入
    onnx_model_path,     # 输出文件的路径
    export_params=True,  # 将训练好的模型权重也一并导出
    opset_version=11,    # ONNX算子集版本，11或更高通常是好的选择
    do_constant_folding=True, # 是否执行常量折叠优化
    input_names=['input'],   # 为输入张量指定一个易于理解的名字
    output_names=['output'], # 为输出张量指定一个易于理解的名字
    dynamic_axes={'input' : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}}
)

print(f"Model has been converted to ONNX format and saved as {onnx_model_path}")
print("-"*30)

# --- 3. 验证和使用ONNX模型 ---
# 我们可以使用 `onnx` 库来检查模型的有效性，并使用 `onnxruntime` 来运行它。
# 你需要先安装这两个库: `pip install onnx onnxruntime`

print("--- 3. Verifying and Running the ONNX model ---")

# 1. 检查模型是否符合ONNX规范
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model check passed.")

# 2. 使用 ONNX Runtime 创建一个推理会话
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 3. 准备输入数据
# ONNX Runtime 需要numpy数组作为输入
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

# 4. 执行推理
ort_outs = ort_session.run(None, ort_inputs)

# 5. 比较PyTorch和ONNX Runtime的输出
# 获取原始PyTorch模型的输出
with torch.no_grad():
    pytorch_output = model(dummy_input)

# 比较两者输出是否接近
np.testing.assert_allclose(pytorch_output.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
print("\nPyTorch and ONNX Runtime outputs match!")
print(f"PyTorch output shape: {pytorch_output.shape}")
print(f"ONNX Runtime output shape: {ort_outs[0].shape}")

# 总结:
# 1. ONNX是实现模型跨平台、跨框架部署的行业标准。
# 2. `torch.onnx.export()` 是将PyTorch模型转换为ONNX格式的核心函数。
# 3. 在导出时，使用 `dynamic_axes` 参数来指定动态维度（如batch_size）至关重要，这大大增加了部署的灵活性。
# 4. `onnxruntime` 是一个高性能的推理引擎，可以加载 `.onnx` 文件并在多种硬件上（CPU, GPU等）高效运行。
# 5. 导出后，务必用相同的输入数据来验证ONNX模型的输出是否与原始PyTorch模型一致。
