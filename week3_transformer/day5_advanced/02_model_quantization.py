
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import os
import copy

# --- 前言 ---
# 模型量化是通过降低模型参数的数值精度来压缩和加速模型的技术。
# 最常见的做法是将模型从32位浮点数（FP32）转换为8位整数（INT8）。
# PyTorch提供了多种量化方法，本脚本将重点介绍最容易使用的“动态量化”。

# --- 1. 准备一个已训练好的模型 ---
# 量化是在模型训练**之后**进行的一个步骤。
# 我们首先创建一个简单的模型，并假装它已经被训练好了。

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM的输出是 (outputs, (h_n, c_n))
        lstm_out, _ = self.lstm(x)
        # 我们只取序列最后一个时间步的输出来进行分类
        final_hidden_state = lstm_out[:, -1, :]
        return self.fc(final_hidden_state)

# 实例化一个FP32模型
fp32_model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=5)
# 在真实场景中，这里会加载已训练好的权重
# fp32_model.load_state_dict(torch.load("trained_model.pth"))
fp32_model.eval() # 必须设置为评估模式

print("--- 1. A pre-trained FP32 model ---")
print(fp32_model)
print("---" + "-"*30)

# --- 2. 动态量化 (Dynamic Quantization) ---
# - **原理**: 
#   - **权重 (Weights)**: 被预先转换（量化）为INT8格式。
#   - **激活值 (Activations)**: 在推理时，当它们在层与层之间传递时，被“动态地”量化为INT8，计算完成后再转换回FP32。
# - **优点**: 使用非常简单，只需一行代码。不需要校准数据。
# - **适用场景**: 对模型大小和内存带宽敏感的场景，尤其是在CPU上能带来显著的加速。对于LSTM和Transformer这类模型效果很好，因为它们的计算瓶颈通常在于权重的加载。

print("--- 2. Applying Dynamic Quantization ---")

# 使用 `torch.quantization.quantize_dynamic()` 进行量化
# - `model`: 要量化的模型
# - `{nn.LSTM, nn.Linear}`: 一个要量化的层类型的集合
# - `dtype=torch.qint8`: 指定量化为8位整数
quantized_model = quantize_dynamic(
    model=fp32_model, 
    qconfig_spec={nn.LSTM, nn.Linear}, 
    dtype=torch.qint8
)

print("Model successfully quantized.")
print("\n--- Quantized Model Architecture ---")
# 打印量化后的模型，可以看到Linear层被替换为了DynamicQuantizedLinear
print(quantized_model)
print("---" + "-"*30)

# --- 3. 比较模型大小 ---
print("--- 3. Comparing Model Size ---")

def print_model_size(model, label):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6 # MB
    print(f"Size of {label}: {size:.2f} MB")
    os.remove("temp.p")

print_model_size(fp32_model, "FP32 Model")
print_model_size(quantized_model, "Quantized INT8 Model")
print("\nObservation: The quantized model is roughly 4x smaller!")
print("---" + "-"*30)

# --- 4. 比较性能和精度 ---
# **性能 (速度)**: 
# - 在支持INT8计算的硬件（特别是CPU）上，量化模型的推理速度通常会比FP32模型快2到4倍。
# - 这需要在一个真实的推理环境中进行基准测试才能准确衡量。

# **精度 (Accuracy)**:
# - 量化通常会导致轻微的精度下降。在应用到生产之前，必须在验证集上仔细评估量化模型的性能，
#   确保其精度仍然在可接受的范围内。

# 我们可以用一个虚拟输入来验证两个模型的输出是否大致相似
input_fp32 = torch.randn(1, 10, 10) # (batch, seq_len, input_size)

with torch.no_grad():
    output_fp32 = fp32_model(input_fp32)
    output_quantized = quantized_model(input_fp32)

# 比较输出
print("--- 4. Comparing Output and Accuracy ---")
print("FP32 model output:", output_fp32.squeeze())
print("Quantized model output:", output_quantized.squeeze())
print("\nOutputs are similar, but not identical due to precision loss.")
print("It is crucial to evaluate the quantized model on a validation set.")

# --- 其他量化方法 ---
# - **静态量化 (Static Quantization)**: 除了量化权重，还预先通过一个“校准”过程（在一些代表性数据上运行模型）来确定激活值的量化参数。通常能获得比动态量化更好的性能，但过程更复杂。
# - **量化感知训练 (Quantization Aware Training, QAT)**: 在训练过程中就模拟量化的效应。它能达到最佳的精度，但需要重新训练模型。

# 总结:
# 1. **量化**是降低模型大小、加速CPU推理的有效手段。
# 2. **动态量化**是最简单的方法，只需一行代码即可应用，尤其适用于LSTM和Transformer。
# 3. 量化会带来**精度损失**，部署前必须在验证集上进行严格的性能评估。
# 4. 对于需要更高精度的场景，可以考虑使用**静态量化**或**量化感知训练**。
