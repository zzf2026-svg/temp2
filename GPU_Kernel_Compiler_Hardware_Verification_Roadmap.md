# GPU Kernel → Compiler Lowering → Hardware Mapping → Verification 学习路线与论文资料

## 目标

目标不是成为只会写 CUDA kernel 的工程师，而是掌握：

Algorithm → Kernel → Compiler IR → ISA → Hardware → Verification

完整 GPU 软件硬件协同设计链路。

------------------------------------------------------------------------

# 一、GPU基础与架构

## 1. Programming Massively Parallel Processors (PMPP)

推荐理由：

GPU计算领域经典教材，系统介绍 CUDA、SIMT、memory
hierarchy、optimization。

学习重点：

-   GPU execution model
-   Warp
-   Shared memory
-   Reduction
-   Memory coalescing

网址：

https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861

------------------------------------------------------------------------

## 2. NVIDIA CUDA Programming Guide

学习重点：

-   Thread hierarchy
-   Memory hierarchy
-   CUDA execution model
-   Cooperative groups

网址：

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

------------------------------------------------------------------------

# 二、GPU Kernel Optimization

## 3. CUTLASS

论文：

CUTLASS: Fast Linear Algebra in CUDA C++

学习内容：

-   GEMM decomposition
-   Threadblock tile
-   Warp tile
-   Tensor Core mapping

网址：

https://github.com/NVIDIA/cutlass

------------------------------------------------------------------------

## 4. Efficient GEMM Optimization

重点理解：

矩阵乘：

C = A × B

如何优化：

-   global memory
-   shared memory
-   register blocking
-   tensor core

参考：

https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

------------------------------------------------------------------------

# 三、深度学习 Compiler

## 5. TVM

论文：

TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

作者：

Tianqi Chen 等

核心思想：

通过 compiler 自动搜索：

-   tiling
-   fusion
-   memory optimization
-   hardware mapping

网址：

论文： https://arxiv.org/abs/1802.04799

代码： https://github.com/apache/tvm

------------------------------------------------------------------------

## 6. MLIR

论文：

MLIR: A Compiler Infrastructure for the End of Moore's Law

学习：

-   IR设计
-   dialect
-   lowering pipeline

网址：

https://mlir.llvm.org/

论文：

https://arxiv.org/abs/2002.11054

------------------------------------------------------------------------

# 四、Triton GPU Kernel

## 7. Triton

论文：

Triton: An Intermediate Language and Compiler for Tiled Neural Network
Computations

学习：

-   GPU DSL
-   tile programming
-   compiler lowering

网址：

论文：

https://arxiv.org/abs/2103.06989

代码：

https://github.com/triton-lang/triton

------------------------------------------------------------------------

# 五、Attention与LLM Kernel

## 8. FlashAttention

论文：

FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness

核心：

重新设计 Attention 数据流：

HBM → SRAM → Register

减少 memory traffic。

网址：

论文：

https://arxiv.org/abs/2205.14135

代码：

https://github.com/Dao-AILab/flash-attention

------------------------------------------------------------------------

## 9. FlashAttention-2

学习：

-   warp specialization
-   better parallelism
-   GPU mapping

网址：

https://arxiv.org/abs/2307.08691

------------------------------------------------------------------------

## 10. FlashAttention-3 / Hopper Optimization

学习：

-   Hopper Tensor Core
-   WGMMA
-   TMA
-   asynchronous pipeline

网址：

https://arxiv.org/abs/2407.08608

------------------------------------------------------------------------

# 六、Compiler Lowering 到 Hardware

需要理解：

CUDA:

CUDA C

↓

NVVM IR

↓

LLVM

↓

PTX

↓

SASS

↓

GPU SM

学习资料：

## NVIDIA PTX ISA

网址：

https://docs.nvidia.com/cuda/parallel-thread-execution/

------------------------------------------------------------------------

## NVIDIA Nsight Compute

用于分析：

-   occupancy
-   memory bandwidth
-   instruction
-   warp stall

网址：

https://developer.nvidia.com/nsight-compute

------------------------------------------------------------------------

# 七、Verification（工业级Kernel开发）

一个高质量kernel流程：

数学公式

↓

Reference implementation

↓

GPU kernel

↓

Numerical comparison

↓

Race checking

↓

Benchmark

------------------------------------------------------------------------

## CUDA Compute Sanitizer

包括：

-   memcheck
-   racecheck
-   initcheck

网址：

https://docs.nvidia.com/cuda/compute-sanitizer/

------------------------------------------------------------------------

# 八、TIRx方向推荐

## 11. Apache TVM TensorIR

学习：

-   TensorIR
-   scheduling
-   structural mapping

网址：

https://tvm.apache.org/

------------------------------------------------------------------------

## 12. MLC LLM

学习：

端到端：

Model → Compiler → GPU

网址：

https://github.com/mlc-ai/mlc-llm

------------------------------------------------------------------------

# 九、推荐阅读顺序（12个月）

## Month 1

CUDA:

-   PMPP
-   CUDA Programming Guide

目标：

手写 reduction。

------------------------------------------------------------------------

## Month 2-3

Kernel:

-   GEMM
-   CUTLASS

目标：

理解 tensor core mapping。

------------------------------------------------------------------------

## Month 4-5

Compiler:

-   TVM
-   MLIR
-   Triton

目标：

理解 lowering。

------------------------------------------------------------------------

## Month 6-8

LLM Kernel:

-   FlashAttention
-   vLLM
-   MoE kernel

------------------------------------------------------------------------

## Month 9-12

贡献开源：

目标：

-   Triton
-   TVM
-   TIRx

方向：

-   kernel optimization
-   compiler pass
-   verification

------------------------------------------------------------------------

# 十、研究者思维

普通工程师：

如何写 kernel？

优秀工程师：

为什么这个 kernel 在 GPU 上这样映射？

研究者：

算法、compiler、hardware 是否应该共同重新设计？

目标：

成为 GPU software-hardware co-design researcher。
