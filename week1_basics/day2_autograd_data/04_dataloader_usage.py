
import torch
from torch.utils.data import Dataset, DataLoader
import time

# --- 1. 准备一个Dataset ---
# DataLoader需要一个Dataset对象作为输入。
# 我们在这里定义一个简单的自定义数据集，它会模拟加载一些有延迟的数据。
# 这有助于我们后续观察 `num_workers` 的效果。

class SlowCustomDataset(Dataset):
    """
    一个模拟加载慢速数据的自定义数据集。
    每个样本的获取都会有轻微的延迟。
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 模拟IO操作或复杂的预处理，导致0.01秒的延迟
        time.sleep(0.01)
        # 返回样本索引和它的平方
        return torch.tensor(idx, dtype=torch.float32), torch.tensor(idx**2, dtype=torch.float32)

# 实例化数据集
dataset = SlowCustomDataset(num_samples=100)

# --- 2. DataLoader 的基本使用 ---
# DataLoader 是一个迭代器，它将Dataset包装起来，并提供批量加载、打乱等功能。

print("--- Basic DataLoader Usage ---")
# - dataset: 要加载的数据集。
# - batch_size: 每个批次加载的样本数。这是训练神经网络时调优的一个重要超参数。
# - shuffle: 是否在每个epoch开始时打乱数据顺序。在训练时设为True可以增加模型的泛化能力。
data_loader_basic = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

# 遍历DataLoader
print("Iterating through the basic DataLoader (one batch):")
start_time = time.time()
for i, (data_batch, label_batch) in enumerate(data_loader_basic):
    if i == 0:
        print(f"  Batch {i+1}:")
        print(f"    Data batch shape: {data_batch.shape}")   # (batch_size,)
        print(f"    Label batch shape: {label_batch.shape}") # (batch_size,)
        break # 只演示一个批次
end_time = time.time()
print(f"Time to load one batch: {end_time - start_time:.4f} seconds")
print("-"*30)

# --- 3. 使用 num_workers 加速数据加载 ---
# `num_workers` 参数可以指定使用多少个子进程来预加载数据。
# 当 `num_workers > 0` 时，数据加载和模型训练可以并行进行，从而大大提高GPU的利用率。
# 在数据预处理或IO操作成为瓶颈时，这个参数尤其有用。

# 注意：在Windows或macOS上，多进程代码必须放在 `if __name__ == '__main__':` 块中。
# 这是因为子进程会重新导入主脚本，需要这个入口点来防止无限递归。

def run_dataloader_test(num_workers):
    print(f"--- Testing DataLoader with num_workers = {num_workers} ---")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers
    )

    start_time = time.time()
    # 遍历整个数据集
    for _ in data_loader:
        pass # 我们只关心加载数据的时间，所以循环体是空的
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time to iterate through dataset: {total_time:.4f} seconds")
    return total_time

if __name__ == '__main__':
    # 在主进程中加载 (num_workers=0)
    # 数据加载是串行的，一个批次加载完才能加载下一个。
    time_0_workers = run_dataloader_test(num_workers=0)

    # 使用多个子进程加载 (例如 num_workers=4)
    # PyTorch会在后台启动4个worker。当主进程在处理一个批次时，worker已经在准备接下来的批次了。
    # 这就像餐厅里，主厨在炒菜，而帮厨已经在准备下一道菜的食材了。
    # 注意：num_workers的最佳值取决于你的CPU核心数、内存和数据处理的复杂度。
    # 过多的worker可能会因为进程间通信的开销而降低效率。
    time_multi_workers = run_dataloader_test(num_workers=4)

    print("\n--- Comparison ---")
    print(f"With num_workers=0, time taken: {time_0_workers:.4f}s")
    print(f"With num_workers=4, time taken: {time_multi_workers:.4f}s")
    if time_multi_workers < time_0_workers:
        print(f"Speed-up factor: {time_0_workers / time_multi_workers:.2f}x")
    print("\nNote: The first run with num_workers > 0 might be slower due to process initialization overhead.")

# 总结:
# 1. DataLoader 是PyTorch数据加载流程的核心，它负责将Dataset中的数据整理成批次(batch)。
# 2. `batch_size` 和 `shuffle` 是两个最常用的参数，分别控制批量大小和是否打乱数据。
# 3. `num_workers` 是一个关键的性能参数。通过设置 `num_workers > 0`，可以利用多进程并行加载数据，
#    避免CPU数据处理成为GPU训练的瓶颈，从而提升整体训练效率。
# 4. 在使用多进程加载时，尤其是在Windows/macOS上，需要将执行代码放入 `if __name__ == '__main__':` 块中。
