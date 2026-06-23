# Qwen CUDA Fused RMSNorm

面向大语言模型推理的 CUDA Fused RMSNorm 算子优化与 Qwen2.5 集成实验。

本项目针对 PyTorch eager 模式下 RMSNorm 由多个小算子组成、存在多次 CUDA Kernel 启动和中间张量读写的问题，实现了自定义 CUDA Fused RMSNorm，并将其接入 Qwen2.5-0.5B-Instruct。项目包含 PyTorch 基线、自定义 CUDA Kernel、正确性验证、Block Size 调优、Warp-level Reduction V2 对比、真实 Qwen 替换、端到端 Benchmark 和 PyTorch Profiler 分析。

---

## 1. 项目简介

RMSNorm 是 Qwen、LLaMA 等大语言模型中频繁使用的归一化算子。原始 PyTorch 实现通常会被拆分为多个操作：

```text
dtype conversion
→ square
→ mean reduction
→ rsqrt
→ normalization
→ weight scaling
→ dtype conversion
```

这些步骤在 PyTorch eager 模式下可能对应多次 CUDA Kernel 启动，并产生多个中间张量。

本项目将上述计算融合到一个自定义 CUDA Kernel 中：

```text
读取输入
→ FP32 平方和累加
→ Block 内归约
→ 计算逆均方根
→ 归一化并乘权重
→ 写出最终结果
```

核心优化方法包括：

- CUDA Kernel Fusion
- Shared Memory Reduction
- FP16/BF16 输入与 FP32 累加
- Block Size 调优
- Warp-level Reduction V2 对比
- PyTorch C++/CUDA Extension
- Qwen2RMSNorm 模块替换
- 端到端稳定性 Benchmark
- PyTorch Profiler 分析

---

## 2. 主要成果

### 单算子结果

在 NVIDIA GeForce RTX 4090D 上，自定义 CUDA Fused RMSNorm 相比 PyTorch baseline 获得：

| 数据类型 | 单算子加速比 |
| -------- | -----------: |
| FP32     |        约 8× |
| FP16     |       约 12× |
| BF16     |       约 12× |

自定义 CUDA Kernel 的典型延迟约为：

```text
0.008 ms
```

### Qwen 端到端结果

实验将 Qwen2.5-0.5B-Instruct 中的 49 个 `Qwen2RMSNorm` 模块替换为自定义 `CudaQwenRMSNorm`。

Alternating Benchmark 结果：

| 指标              |       结果 |
| ----------------- | ---------: |
| 替换 RMSNorm 层数 |         49 |
| FP16 平均加速比   | 约 1.3057× |
| BF16 平均加速比   | 约 1.3016× |
| 整体平均加速比    | 约 1.3037× |
| 平均延迟下降      |   约 23.3% |

说明：以上结果仅对应本项目实验环境，即 RTX 4090D、Qwen2.5-0.5B-Instruct、PyTorch eager、batch size 为 1、prefill 场景。不同模型规模、推理框架和硬件环境下的结果可能不同。

---

## 3. 实验环境

| 项目                | 配置                     |
| ------------------- | ------------------------ |
| 操作系统            | Ubuntu Linux             |
| GPU                 | NVIDIA GeForce RTX 4090D |
| GPU 显存            | 24 GB                    |
| Compute Capability  | 8.9                      |
| NVIDIA Driver       | 570.195.03               |
| Driver 支持 CUDA    | 最高 12.8                |
| Python              | 3.11.15                  |
| PyTorch             | 2.7.1+cu126              |
| CUDA Runtime        | 12.6                     |
| CUDA Toolkit / nvcc | 12.6.85                  |
| 模型                | Qwen2.5-0.5B-Instruct    |

验证环境：

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA Runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute Capability:", torch.cuda.get_device_capability(0))
PY
```

检查 CUDA 编译器：

```bash
nvcc --version
```

---

## 4. 项目结构

```text
cuda-rmsnorm-llm/
├── src/
│   ├── __init__.py
│   ├── rmsnorm.py
│   ├── rmsnorm_cuda.py
│   ├── rmsnorm_kernel.cu
│   ├── rmsnorm_kernel_v1_shared.cu
│   └── binding.cpp
│
├── tests/
│   ├── test_cuda_rmsnorm.py
│   └── test_block_size_correctness.py
│
├── benchmarks/
│   ├── benchmark_rmsnorm.py
│   ├── benchmark_cuda_rmsnorm.py
│   ├── benchmark_cuda_rmsnorm_stable.py
│   ├── benchmark_block_size.py
│   └── benchmark_block_size_warp.py
│
├── llm/
│   ├── __init__.py
│   ├── test_qwen_load.py
│   ├── qwen_rmsnorm_patch.py
│   ├── test_qwen_patch.py
│   ├── benchmark_qwen_patch.py
│   └── benchmark_qwen_patch_alternating.py
│
├── profiling/
│   ├── __init__.py
│   ├── profile_rmsnorm_operator.py
│   └── profile_qwen_original_vs_patched.py
│
├── results/
│   ├── profiler/
│   └── *.csv
│
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md
```

不同版本项目中的文件名可能略有差异，请以实际目录为准。

---

## 5. 安装依赖

### 5.1 创建 Conda 环境

```bash
conda create -n cuda-rmsnorm python=3.11 -y
conda activate cuda-rmsnorm
```

### 5.2 安装 PyTorch

请根据本机驱动和 CUDA 环境选择兼容版本。本项目使用：

```text
PyTorch 2.7.1+cu126
```

安装完成后确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 5.3 安装 Python 依赖

```bash
pip install transformers accelerate safetensors tokenizers sentencepiece huggingface_hub pandas
```

也可以使用：

```bash
pip install -r requirements.txt
```

一个参考 `requirements.txt`：

```text
torch==2.7.1
transformers
accelerate
safetensors
tokenizers
sentencepiece
huggingface_hub
pandas
```

注意：PyTorch 的安装方式与 CUDA 版本有关，建议根据实际环境单独安装，不要直接照搬版本。

---

## 6. 环境变量设置

选择实验 GPU：

```bash
export CUDA_VISIBLE_DEVICES=3
```

设置 CUDA Toolkit：

```bash
export CUDA_HOME=/path/to/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

设置 RTX 4090D 对应架构：

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

---

## 7. 编译 CUDA Extension

在项目根目录执行：

```bash
python setup.py build_ext --inplace
```

编译成功后，项目根目录会生成类似文件：

```text
rmsnorm_cuda_ext.cpython-311-x86_64-linux-gnu.so
```

如果修改了 `.cu` 或 `.cpp` 文件，建议清理后重新编译：

```bash
rm -rf build
rm -f rmsnorm_cuda_ext*.so
python setup.py build_ext --inplace
```

---

## 8. RMSNorm 原理

RMSNorm 的计算公式为：

\[
y_i =
\frac{x_i}
{\sqrt{\frac{1}{n}\sum_{j=1}^{n}x_j^2+\epsilon}}
w_i
\]

其中：

- \(x_i\)：输入隐藏向量的第 \(i\) 个元素
- \(n\)：hidden size
- \(\epsilon\)：避免除零的小常数
- \(w_i\)：可学习权重
- \(y_i\)：归一化输出

计算步骤：

1. 对输入元素平方
2. 对 hidden dimension 求平均
3. 加上 epsilon
4. 计算逆平方根
5. 对输入进行归一化
6. 乘可学习权重

---

## 9. CUDA V1：Shared Memory Reduction

V1 Kernel 采用以下映射方式：

```text
一个 CUDA Block 处理一个 Token
一个 Block 内的多个 Thread 共同处理 hidden dimension
```

每个 Thread 负责间隔读取若干元素：

```text
thread 0: 0, 128, 256, ...
thread 1: 1, 129, 257, ...
...
```

每个 Thread 计算自己的局部平方和：

```cpp
float local_sum = 0.0f;

for (int64_t i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
    float value = static_cast<float>(x[base + i]);
    local_sum += value * value;
}
```

随后将所有 Thread 的 `local_sum` 写入 Shared Memory，并进行 Block-level Reduction。

归约结束后计算：

```cpp
float inv_rms = rsqrtf(
    block_sum / static_cast<float>(hidden_size) + eps
);
```

最后在同一个 Kernel 内完成归一化和权重缩放：

```cpp
out[base + i] = x[base + i] * inv_rms * weight[i];
```

### V1 的主要优势

- 将多个 PyTorch 操作融合为一个 CUDA Kernel
- 减少 Kernel Launch 次数
- 减少中间张量
- 减少全局显存读写
- FP16/BF16 输入使用 FP32 进行平方和累加
- 实现简单、稳定，在当前测试规模下性能最好

---

## 10. Block Size 调优

本项目测试了以下配置：

```text
128 threads/block
256 threads/block
512 threads/block
```

测试覆盖：

- hidden size：1024、2048、4096
- dtype：FP32、FP16、BF16
- 多种 batch size 和 sequence length

实验结果表明：

- 128 threads/block 在多数配置中表现最好
- 256 和 512 没有稳定优势
- 不同 Block Size 之间差异较小
- 最终默认选择 128 threads/block

最终配置：

```text
Shared Memory Reduction
+
128 threads/block
```

---

## 11. CUDA V2：Warp-level Reduction

V2 尝试使用 Warp Shuffle 优化 Block 内归约。

核心思想：

```text
先在每个 Warp 内归约
再对多个 Warp 的结果进行归约
```

每个 Warp 包含 32 个 Thread。Warp 内使用：

```cpp
__shfl_down_sync()
```

进行寄存器级数据交换，无需先写入 Shared Memory。

V2 主要流程：

1. 每个 Thread 计算 `local_sum`
2. 每个 Warp 内部使用 `__shfl_down_sync` 求和
3. 每个 Warp 的 lane 0 将结果写入 `warp_sums`
4. 第一个 Warp 对 `warp_sums` 进行第二次归约
5. 得到整个 Block 的平方和

### V2 实验结论

理论上 Warp-level Reduction 可以减少 Shared Memory 访问和同步，但本项目实测中：

```text
V1 平均延迟约 0.0083 ms
V2 平均延迟约 0.0087 ms
```

V2 平均慢约 4%～5%。

可能原因：

- 当前 Kernel 已经处于微秒级
- Kernel Launch 和基本显存访问占比较高
- Reduction 不是唯一瓶颈
- V2 增加了 Shuffle、lane 判断和跨 Warp 汇总逻辑
- 当前 hidden size 和 Block Size 下，V1 已经足够高效

因此最终采用：

```text
V1 Shared Memory Reduction
+
128 threads/block
```

该结果说明：CUDA 优化必须以真实硬件测试为依据，更复杂的实现不一定更快。

---

## 12. 正确性测试

运行：

```bash
python tests/test_cuda_rmsnorm.py
python tests/test_block_size_correctness.py
```

测试内容：

- PyTorch reference 与 CUDA output 对比
- FP32、FP16、BF16
- 多种输入形状
- 不同 Block Size
- 最大绝对误差
- 平均绝对误差

典型误差范围：

| 数据类型 |        最大绝对误差 |
| -------- | ------------------: |
| FP32     |      约 \(10^{-6}\) |
| FP16     |      约 \(10^{-3}\) |
| BF16     | 约 \(10^{-2}\) 以内 |

---

## 13. 单算子 Benchmark

### 13.1 PyTorch Baseline

```bash
python -m benchmarks.benchmark_rmsnorm
```

### 13.2 稳定版 CUDA Benchmark

```bash
python -m benchmarks.benchmark_cuda_rmsnorm_stable
```

稳定版 Benchmark 使用：

- CUDA Event 计时
- 正式计时前 Warmup
- 多轮重复执行
- `torch.cuda.synchronize()`
- Median latency
- 相同输入、shape 和 dtype

加速比计算：

```text
Speedup = PyTorch median latency / CUDA median latency
```

注意：单算子加速比只代表 RMSNorm 本身，不代表整个 Qwen 模型。

---

## 14. 下载 Qwen2.5-0.5B-Instruct

项目不包含模型权重。

可使用 ModelScope：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir="/your/model/path",
)

print(model_dir)
```

也可以使用 Hugging Face：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /your/model/path/Qwen2.5-0.5B-Instruct
```

下载后修改以下脚本中的 `MODEL_PATH`：

```text
llm/test_qwen_load.py
llm/test_qwen_patch.py
llm/benchmark_qwen_patch.py
llm/benchmark_qwen_patch_alternating.py
profiling/profile_qwen_original_vs_patched.py
```

例如：

```python
MODEL_PATH = "/mnt/new_4tdisk/wbc/models/Qwen2.5-0.5B-Instruct"
```

---

## 15. 测试 Qwen 加载

```bash
python llm/test_qwen_load.py
```

测试成功后，模型应能正常生成中文文本。

---

## 16. 替换 Qwen RMSNorm

`qwen_rmsnorm_patch.py` 会递归遍历模型模块，将：

```text
Qwen2RMSNorm
```

替换为：

```text
CudaQwenRMSNorm
```

替换时保留：

- 原始 weight
- epsilon
- dtype
- device

本项目中共替换：

```text
49 个 Qwen2RMSNorm
```

运行正确性测试：

```bash
python llm/test_qwen_patch.py
```

典型结果：

```text
Replaced RMSNorm layers: 49
logits max_abs_err ≈ 6.88e-02
logits mean_abs_err ≈ 7.38e-03
```

---

## 17. Qwen 端到端 Benchmark

普通 Benchmark：

```bash
python llm/benchmark_qwen_patch.py
```

交替顺序稳定性 Benchmark：

```bash
python llm/benchmark_qwen_patch_alternating.py
```

Alternating Benchmark 会交替执行：

```text
original → patched
patched → original
```

目的是减少以下因素对结果的影响：

- CUDA Warmup
- Cache 状态
- 测试顺序
- GPU 状态变化

结果文件通常保存在：

```text
results/qwen_rmsnorm_patch_benchmark.csv
results/qwen_rmsnorm_patch_alternating_summary.csv
results/qwen_rmsnorm_patch_alternating_raw.csv
```

---

## 18. Profiler 分析

### 18.1 单算子 Profiler

```bash
python -m profiling.profile_rmsnorm_operator
```

PyTorch RMSNorm 通常包含：

```text
aten::to
aten::_to_copy
aten::copy_
aten::pow
aten::mean
aten::rsqrt
aten::mul
```

自定义 CUDA 版本将主要计算融合到一个 CUDA Kernel 中。

### 18.2 Qwen Profiler

```bash
python -m profiling.profile_qwen_original_vs_patched
```

结果通常保存为：

```text
results/profiler/pytorch_rmsnorm_trace.json
results/profiler/cuda_fused_rmsnorm_trace.json
results/profiler/qwen_original_fp16_seq512_trace.json
results/profiler/qwen_patched_fp16_seq512_trace.json
```

大型 Trace 文件默认不建议提交到 GitHub。

---

## 19. 复现实验

一个推荐的完整执行顺序：

```bash
# 1. 激活环境
conda activate cuda-rmsnorm

# 2. 设置 GPU 与 CUDA 环境
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/path/to/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"

# 3. 编译扩展
python setup.py build_ext --inplace

# 4. 正确性测试
python tests/test_cuda_rmsnorm.py
python tests/test_block_size_correctness.py

# 5. 单算子 Benchmark
python -m benchmarks.benchmark_cuda_rmsnorm_stable

# 6. Block Size Benchmark
python -m benchmarks.benchmark_block_size

# 7. Qwen 模型加载测试
python llm/test_qwen_load.py

# 8. Qwen RMSNorm 替换测试
python llm/test_qwen_patch.py

# 9. Qwen 端到端稳定性 Benchmark
python llm/benchmark_qwen_patch_alternating.py

# 10. Profiler
python -m profiling.profile_rmsnorm_operator
python -m profiling.profile_qwen_original_vs_patched
```

---

## 20. Benchmark 说明

本项目中的 Benchmark 为针对该实验自定义编写的 Microbenchmark 和端到端测试脚本，并非 NVIDIA、PyTorch 或 Qwen 官方 Benchmark。

测试代码由项目作者根据常见 CUDA 性能测试方法编写，实验数据由 RTX 4090D 服务器实际运行得到。

主要测试规范：

- 使用 CUDA Event 计时
- 正式计时前执行 Warmup
- 调用 `torch.cuda.synchronize()`
- 多次重复测试
- 主要报告 Median latency
- PyTorch 与 CUDA 使用相同输入和 dtype
- 使用 Alternating Benchmark 降低顺序影响
- 使用 Profiler 分析性能变化原因

---

## 21. 项目限制

当前项目仍存在以下限制：

- 仅重点优化 RMSNorm
- 主要测试 Qwen2.5-0.5B-Instruct
- 主要测试 batch size 为 1
- 主要面向 PyTorch eager 推理
- 未与 Triton、Apex、vLLM、TensorRT-LLM 的融合算子直接对比
- 未实现 FP16 `half2` 向量化
- 未使用 Nsight Compute 分析带宽和 Occupancy
- 未测试更多 GPU 架构
- 未对完整 Decode 阶段进行深入优化

---

## 22. 后续工作

后续可继续完成：

- FP16 `half2` 向量化
- BF16 向量化
- CUDA Graph
- 减少输出张量分配
- Nsight Systems / Nsight Compute 分析
- Qwen 1.5B、3B、7B 测试
- vLLM 集成
- TensorRT-LLM 集成
- Attention / RoPE / KV Cache 优化
- 自动选择最优 Block Size

---



## 25. License

本项目主要用于异构计算课程实验与学习研究。

如需公开发布，建议添加合适的开源许可证，例如：

```text
MIT License
```
