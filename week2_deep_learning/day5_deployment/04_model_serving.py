
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import json

# --- 0. 准备工作：训练并保存一个简单的模型 ---
# 在真实场景中，我们会加载一个已经训练好的复杂模型。
# 这里，我们先创建一个简单的模型并保存它的state_dict，以供我们的API服务加载。

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2) # 接收10个特征，输出2个类别的logits

    def forward(self, x):
        return self.linear(x)

# 保存模型的state_dict
model_to_serve = SimpleClassifier()
# 假设它已经被训练过了，我们这里只保存其随机初始化的权重
MODEL_PATH = "simple_classifier.pth"
torch.save(model_to_serve.state_dict(), MODEL_PATH)

# --- 1. 初始化服务和加载模型 ---

print("--- Initializing Flask App and Loading Model ---")

# 创建Flask应用实例
app = Flask(__name__)

# 在全局作用域加载模型
# 这样可以确保模型只在服务启动时加载一次，而不是每次收到请求都重新加载，从而提高效率。
model = SimpleClassifier()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # 必须设置为评估模式！

print("Model loaded and ready to serve.")

# --- 2. 创建API端点 (Endpoint) ---
# 我们定义一个路径（例如 /predict），并指定它只接受HTTP POST请求。

@app.route("/predict", methods=['POST'])
def predict():
    """接收JSON数据，执行推理，并返回JSON结果。"""
    if request.method == 'POST':
        try:
            # 1. 从POST请求中获取JSON数据
            data = request.get_json()
            # 假设JSON数据格式为: {"features": [0.1, 0.2, ..., 0.9]}
            input_features = data['features']
            
            # 2. 数据预处理
            # 将输入的列表转换为模型可以接受的PyTorch张量
            # 需要增加一个批次维度 (batch_size=1)
            input_tensor = torch.tensor(input_features).float().unsqueeze(0)
            
            # 检查输入维度是否正确
            if input_tensor.shape[1] != 10:
                return jsonify({"error": f"Expected 10 features, got {input_tensor.shape[1]}"}), 400

            # 3. 执行推理
            # 使用 torch.no_grad() 来关闭梯度计算，以加速推理
            with torch.no_grad():
                output_logits = model(input_tensor)
                # 应用Softmax来获取概率
                output_probs = torch.softmax(output_logits, dim=1)
                # 获取预测的类别
                predicted_class = torch.argmax(output_probs, dim=1).item()

            # 4. 格式化输出结果
            response = {
                'predicted_class': predicted_class,
                'probabilities': output_probs.squeeze().tolist() # [prob_class_0, prob_class_1]
            }
            
            return jsonify(response)

        except Exception as e:
            # 统一的错误处理
            return jsonify({"error": str(e)}), 500

# --- 3. 运行Flask服务 ---
# `if __name__ == '__main__':` 确保只有在直接运行此脚本时，
# `app.run()` 才会被调用。如果此脚本被其他模块导入，则不会自动运行。

if __name__ == '__main__':
    # `host='0.0.0.0'` 使你的服务可以从网络中的任何计算机访问，而不仅仅是本机。
    # `port=5000` 是Flask默认的端口。
    print("--- Starting Flask Server ---")
    print("To test, run the following command in a new terminal:")
    # 使用json.dumps来确保JSON字符串中的双引号被正确转义
    test_data = json.dumps({"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    print(f'curl -X POST -H "Content-Type: application/json" -d \'{test_data}\' http://127.0.0.1:5000/predict')
    app.run(host='0.0.0.0', port=5000, debug=False)

# 总结:
# 1. **Flask** 是一个用于快速创建Web API的优秀工具。
# 2. **模型加载**: 在服务启动时一次性加载模型，而不是在每次请求时加载。
# 3. **API端点**: 创建一个如 /predict 的路由来接收 POST 请求。
# 4. **数据流**: JSON -> Python Dict -> PyTorch Tensor -> Model -> PyTorch Tensor -> Python Dict -> JSON。
# 5. **推理模式**: 务必使用 `model.eval()` 和 `with torch.no_grad()` 来确保推理的正确性和效率。
# 6. 在生产环境中，通常会使用更专业的服务器（如 Gunicorn）来运行Flask应用，以获得更好的性能和稳定性。
