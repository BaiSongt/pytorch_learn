
import torch
import torch.nn as nn

# --- 前言 ---
# 当我们想将PyTorch模型部署到生产环境时，Python的性能和依赖问题可能会成为瓶颈。
# 例如，在C++后端服务、移动端(iOS/Android)等环境中，我们无法或不希望运行一个完整的Python解释器。
# 
# TorchScript 就是为了解决这个问题而生的。它提供了一种将PyTorch模型序列化和优化的方法，
# 使其可以脱离Python环境，在高性能的C++环境中被加载和执行。

# --- 1. 准备一个简单的PyTorch模型 ---
# 我们将使用一个包含一些逻辑控制流的模型来演示TorchScript的两种模式。

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        # 演示一个简单的控制流
        if x.mean() > 0:
            x = torch.relu(x)
        else:
            x = torch.sigmoid(x)
        x = self.linear2(x)
        return x

model = MyModel()
model.eval() # 设置为评估模式

# 创建一个虚拟输入
dummy_input = torch.randn(8, 10)

print("--- 1. A standard PyTorch model ---")
print(model)
print("---" * 30)

# --- 2. 方法一: 追踪 (Tracing) ---
# - **原理**: `torch.jit.trace` 通过传入一个示例输入来“执行”一次模型，并记录下所有遇到的操作，
#   然后将这些操作序列转换成一个静态的计算图（一个`ScriptModule`对象）。
# - **优点**: 使用简单，对于没有复杂控制流的直接网络（如标准ResNet）非常有效。
# - **缺点**: 它**无法**捕捉到任何与数据相关的控制流。在上面的例子中，`if x.mean() > 0:`
#   这个判断的结果取决于输入数据。追踪只会记录下示例输入执行的那条路径。

print("--- 2. Tracing the model with torch.jit.trace ---")

# 追踪模型
traced_model = torch.jit.trace(model, dummy_input)

print("Traced model graph:")
print(traced_model.graph)
print("\nNotice that the if/else logic is NOT captured in the traced graph.")

# 保存和加载追踪后的模型
traced_model.save("traced_model.pt")
loaded_traced_model = torch.jit.load("traced_model.pt")
print("Traced model saved and loaded successfully.")
print("---" * 30)

# --- 3. 方法二: 脚本化 (Scripting) ---
# - **原理**: `torch.jit.script` 直接分析模型的Python源代码（`forward`方法以及任何被调用的函数），
#   并将其编译成TorchScript的中间表示。它是一个真正的代码编译器。
# - **优点**: 功能更强大，能够正确地处理各种复杂的控制流（if语句、for循环等）。
# - **缺点**: 对Python语法有一定限制，不是所有的Python特性都能被正确编译。

print("--- 3. Scripting the model with torch.jit.script ---")

# 脚本化模型
scripted_model = torch.jit.script(model)

print("Scripted model graph:")
print(scripted_model.graph)
print("\nNotice that the if/else logic IS correctly captured now!")

# 保存和加载脚本化后的模型
scripted_model.save("scripted_model.pt")
loaded_scripted_model = torch.jit.load("scripted_model.pt")
print("Scripted model saved and loaded successfully.")
print("---" * 30)

# --- 4. 如何选择？---
# - **首选 `torch.jit.script`**: 因为它更健壮，能正确处理各种模型结构。你可以先尝试脚本化，如果遇到不支持的语法，再考虑其他方法。
# - **使用 `torch.jit.trace`**: 当你的模型是一个简单的、没有数据依赖控制流的直接网络时，追踪是一个非常好的选择。
# - **混合使用**: 你甚至可以在一个脚本化的模块中调用一个追踪过的模块，反之亦然，以获得最大的灵活性。

# --- 5. 在C++中加载模型 (概念) ---
# 保存的 `.pt` 文件可以在C++中通过LibTorch库加载和执行，完全脱离Python。
# 
#   ```cpp
#   #include <torch/script.h>
#   #include <memory>
#
#   int main() {
#     // 加载模型
#     torch::jit::script::Module module;
#     try {
#       module = torch::jit::load("scripted_model.pt");
#     } catch (const c10::Error& e) {
#       return -1;
#     }
#
#     // 创建一个输入张量
#     std::vector<torch::jit::IValue> inputs;
#     inputs.push_back(torch::randn({8, 10}));
#
#     // 执行模型
#     at::Tensor output = module.forward(inputs).toTensor();
#
#     return 0;
#   }
#   ```

# 总结:
# 1. TorchScript是PyTorch模型部署和优化的关键工具，它将模型转换为可独立于Python运行的格式。
# 2. **追踪 (Trace)** 简单快捷，但无法处理数据依赖的控制流。
# 3. **脚本化 (Script)** 功能更强大，能正确编译复杂的模型代码，是更推荐的选择。
# 4. 保存的 `.pt` 文件可以被LibTorch (C++) 库加载，从而实现高性能的后端或移动端部署。
