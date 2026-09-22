# 陈天奇导师版 GPU Kernel 研究训练计划

## 目标

训练目标：

成为能够完成 GPU 软件硬件协同设计的人。

能力链：

    数学模型
      ↓
    算子设计
      ↓
    GPU Kernel
      ↓
    Compiler Lowering
      ↓
    ISA Mapping
      ↓
    Hardware Execution
      ↓
    Verification

核心能力：

-   理解模型数学结构
-   设计高性能 GPU kernel
-   理解 TensorIR / Triton / LLVM lowering
-   分析 PTX/SASS
-   建立 reference + numerical verification 流程
-   能够向 TVM/Triton/TIRx 等项目贡献代码

------------------------------------------------------------------------

# 第一阶段（第1个月）：建立 GPU 心智模型

## 目标

理解：

GPU 为什么快？

GPU 为什么慢？

性能瓶颈在哪里？

## 学习资料

### 1. Programming Massively Parallel Processors

重点：

-   SIMT
-   Warp
-   Memory hierarchy
-   Reduction
-   Parallel algorithm design

网址：

https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861

### 2. CUDA Programming Guide

重点：

-   Thread hierarchy
-   CUDA memory model
-   Shared memory
-   Synchronization

网址：

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## 实践任务

实现：

1.  vector add

2.  reduction

3.  softmax

要求：

解释：

-   thread mapping
-   warp behavior
-   memory access pattern

------------------------------------------------------------------------

# 第二阶段（第2-3个月）：GPU Kernel Optimization

目标：

从数学公式到高性能 kernel。

------------------------------------------------------------------------

# 重点1：GEMM

数学：

C=A×B

学习：

-   tiling
-   shared memory
-   register blocking
-   tensor core

## 推荐资料

### CUTLASS

NVIDIA 高性能矩阵计算模板库。

网址：

https://github.com/NVIDIA/cutlass

学习：

-   threadblock tile
-   warp tile
-   MMA instruction

------------------------------------------------------------------------

# 重点2：性能分析

学习：

NVIDIA Nsight Compute

网址：

https://developer.nvidia.com/nsight-compute

掌握：

-   occupancy
-   memory bandwidth
-   warp stall
-   instruction throughput

------------------------------------------------------------------------

# 第三阶段（第4-6个月）：Compiler Lowering

目标：

理解：

Python模型如何变成GPU指令。

完整路径：

    PyTorch
     ↓
    IR
     ↓
    TensorIR / Triton IR
     ↓
    LLVM
     ↓
    PTX
     ↓
    SASS
     ↓
    GPU

------------------------------------------------------------------------

# TVM

论文：

TVM: An Automated End-to-End Optimizing Compiler for Deep Learning

核心：

-   graph optimization
-   operator optimization
-   schedule search
-   hardware mapping

论文：

https://arxiv.org/abs/1802.04799

官网：

https://tvm.apache.org/

代码：

https://github.com/apache/tvm

------------------------------------------------------------------------

# MLIR

论文：

MLIR: A Compiler Infrastructure for the End of Moore's Law

学习：

-   IR设计
-   dialect
-   lowering

论文：

https://arxiv.org/abs/2002.11054

官网：

https://mlir.llvm.org/

------------------------------------------------------------------------

# Triton

论文：

Triton: An Intermediate Language and Compiler for Tiled Neural Network
Computations

学习：

-   GPU DSL
-   tile programming
-   compiler backend

论文：

https://arxiv.org/abs/2103.06989

代码：

https://github.com/triton-lang/triton

------------------------------------------------------------------------

# 第四阶段（第6-9个月）：LLM Kernel 深入

目标：

进入当前 GPU kernel 最前沿。

------------------------------------------------------------------------

# FlashAttention

论文：

FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness

重点：

理解：

为什么 Attention 是 memory bound。

网址：

https://arxiv.org/abs/2205.14135

代码：

https://github.com/Dao-AILab/flash-attention

------------------------------------------------------------------------

# FlashAttention-2

重点：

-   warp parallelism
-   better GPU utilization

网址：

https://arxiv.org/abs/2307.08691

------------------------------------------------------------------------

# FlashAttention-3

重点：

Hopper GPU：

-   Tensor Core
-   WGMMA
-   TMA
-   asynchronous pipeline

网址：

https://arxiv.org/abs/2407.08608

------------------------------------------------------------------------

# 第五阶段（第9-12个月）：Verification 与开源贡献

工业级 kernel 流程：

    数学公式

    ↓

    Reference implementation

    ↓

    GPU kernel

    ↓

    Numerical check

    ↓

    Race check

    ↓

    Benchmark

------------------------------------------------------------------------

# Reference设计

原则：

reference必须：

-   简单
-   数学直接
-   易验证

例如：

FlashAttention：

reference：

PyTorch Attention

kernel：

FlashAttention

比较：

误差范围：

    allclose(reference, kernel)

------------------------------------------------------------------------

# CUDA Compute Sanitizer

用途：

检测：

-   memory error
-   race condition

网址：

https://docs.nvidia.com/cuda/compute-sanitizer/

------------------------------------------------------------------------

# TIRx / TVM贡献训练

目标：

从读PR到提交PR。

推荐流程：

## Step 1

阅读kernel：

回答：

-   数学是什么？
-   tensor如何切分？
-   CTA负责什么？
-   register保存什么？

## Step 2

建立reference

## Step 3

benchmark

## Step 4

修改kernel

## Step 5

加入：

-   correctness test
-   performance test

------------------------------------------------------------------------

# 推荐论文阅读顺序

## 基础

1.  Programming Massively Parallel Processors

2.  CUDA Programming Guide

## Compiler

3.  TVM

https://arxiv.org/abs/1802.04799

4.  Learning to Optimize Tensor Programs

https://arxiv.org/abs/1805.08166

5.  MLIR

https://arxiv.org/abs/2002.11054

## Kernel

6.  CUTLASS

https://github.com/NVIDIA/cutlass

7.  Triton

https://arxiv.org/abs/2103.06989

## LLM System

8.  FlashAttention

https://arxiv.org/abs/2205.14135

9.  FlashAttention-2

https://arxiv.org/abs/2307.08691

------------------------------------------------------------------------

# 陈天奇式研究思维

普通工程师：

"如何写一个kernel？"

高级工程师：

"如何让kernel适配GPU？"

研究者：

"算法、compiler、hardware是否应该一起重新设计？"

最终目标：

成为 GPU Software-Hardware Co-design researcher。

------------------------------------------------------------------------

# 推荐课程

## Machine Learning Compilation

课程：

https://mlc.ai/

内容：

-   TVM
-   TensorIR
-   GPU compilation
-   deployment

------------------------------------------------------------------------

# 每周训练模板

周一：

阅读论文

周二：

分析kernel

周三：

写实验

周四：

benchmark

周五：

总结优化原因

周末：

阅读开源PR
