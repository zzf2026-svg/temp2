# 从 LLM 数学基础薄弱到参与高性能 LLM 系统开发：6个月学习路线

## 学习目标

目标不是成为普通 LLM 使用者，而是建立：

-   Transformer 数学理解能力
-   LLM 训练目标推导能力
-   Attention/FlashAttention 算法理解能力
-   GPU Kernel 与软硬件协同优化能力
-   参与 TIRx、TVM、Triton 等项目贡献的能力

适合背景：

-   已有信号处理、矩阵、向量基础
-   希望通过数学公式推导学习 LLM

------------------------------------------------------------------------

# 总体路线

    数学基础
        ↓
    深度学习基本原理
        ↓
    Transformer
        ↓
    GPT训练目标
        ↓
    FlashAttention
        ↓
    FlashAttention-2
        ↓
    MoE
        ↓
    GPU Kernel / Triton / TVM
        ↓
    TIRx贡献

------------------------------------------------------------------------

# 第一阶段（第1-2周）：补齐深度学习数学基础

## 学习目标

掌握：

-   Forward propagation
-   Loss function
-   Gradient descent
-   Back propagation

核心推导：

假设：

y = Wx+b

损失：

L = 1/2(y-y_hat)\^2

需要能够推导：

∂L/∂W

## 推荐资料

Neural Networks and Deep Learning

网址：

http://neuralnetworksanddeeplearning.com/

------------------------------------------------------------------------

# 第二阶段（第3-6周）：Transformer数学核心

## 必读论文

Attention Is All You Need

论文：

https://arxiv.org/abs/1706.03762

目标：

完整推导：

Attention(Q,K,V)

公式：

Attention(Q,K,V)=softmax(QK\^T/sqrt(d_k))V

重点理解：

## 1. Query / Key / Value

输入：

X

生成：

Q=XW_Q

K=XW_K

V=XW_V

理解：

为什么需要三个投影矩阵。

## 2. 注意力矩阵

计算：

S=QK\^T

其中：

S_ij=q_i\^T k_j

理解 token 之间如何建立关系。

## 3. Scaling

为什么：

除以 sqrt(d_k)

理解：

高维向量点积方差增长导致 softmax 饱和。

------------------------------------------------------------------------

# 第三阶段（第7-8周）：GPT语言模型数学

## 推荐论文

GPT-1:

Improving Language Understanding by Generative Pre-Training

网址：

https://openai.com/research/language-unsupervised

------------------------------------------------------------------------

## 必须理解

语言模型概率：

P(x)=Π_i P(x_i\|x\_\<i)

取log：

logP(x)=Σ_i logP(x_i\|x\_\<i)

训练目标：

Cross Entropy Loss

目标：

理解为什么 GPT 本质是：

预测下一个 token。

------------------------------------------------------------------------

# 第四阶段（第9-10周）：Transformer工程结构

## 推荐论文

GPT-2

Language Models are Unsupervised Multitask Learners

网址：

https://openai.com/research/better-language-models

重点：

Transformer Block:

    Input
     |
    LayerNorm
     |
    Attention
     |
    Residual
     |
    MLP
     |
    Residual
     |
    Output

重点公式：

LayerNorm:

(x-μ)/sqrt(σ²+ε)

Residual:

x_next=x+F(x)

------------------------------------------------------------------------

# 第五阶段（第11-14周）：FlashAttention

## 必读论文

FlashAttention

网址：

https://arxiv.org/abs/2205.14135

目标：

理解：

为什么 Attention 慢。

普通 Attention：

QK\^T

需要保存：

N×N attention matrix

问题：

显存访问成为瓶颈。

------------------------------------------------------------------------

## 必须推导

Online Softmax：

m_i=max(m\_(i-1),x_i)

l_i=

exp(m\_(i-1)-m_i)l\_(i-1) + exp(x_i-m_i)

理解：

如何避免保存完整 attention matrix。

------------------------------------------------------------------------

# 第六阶段（第15-18周）：FlashAttention-2

## 论文

FlashAttention-2

网址：

https://arxiv.org/abs/2307.08691

重点：

理解 GPU 并行：

-   thread block
-   warp
-   shared memory
-   occupancy

目标：

看到 kernel：

知道对应数学步骤。

------------------------------------------------------------------------

# 第七阶段（第19-20周）：MoE模型

## 推荐论文

Switch Transformers

网址：

https://arxiv.org/abs/2101.03961

重点：

理解：

普通 FFN：

FFN(x)

MoE:

y=Σg_i(x)E_i(x)

学习：

-   router
-   expert
-   load balancing
-   communication cost

------------------------------------------------------------------------

# 第八阶段（第21-24周）：GPU Kernel 与系统优化

## Triton

官方教程：

https://triton-lang.org/main/getting-started/tutorials/

重点：

学习：

-   GPU memory hierarchy
-   tiling
-   kernel fusion
-   vectorization

## FlashAttention Triton实现

https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

------------------------------------------------------------------------

# TIRx贡献路线

目标：

从读代码到提交 PR。

步骤：

## Step 1

复现数学公式：

例如：

Attention:

QK\^T

softmax

×V

## Step 2

写 reference 实现：

使用 PyTorch 验证数学正确性。

## Step 3

写 kernel：

优化：

-   memory access
-   register usage
-   synchronization

## Step 4

验证：

包括：

-   numerical correctness
-   racecheck
-   benchmark

------------------------------------------------------------------------

# 推荐阅读顺序

  顺序   资料
  ------ -----------------------------------
  1      Neural Networks and Deep Learning
  2      Attention Is All You Need
  3      GPT
  4      GPT-2
  5      FlashAttention
  6      FlashAttention-2
  7      Switch Transformer
  8      Triton
  9      TIRx/TVM kernel

------------------------------------------------------------------------

# 学习方法

不要：

-   只看论文
-   只看博客
-   只跑代码

正确方式：

每篇论文：

1.  找核心公式
2.  手推公式
3.  写小实验验证
4.  对应源码实现
5.  思考 GPU 优化点

最终目标：

从数学公式：

Attention(Q,K,V)

走到：

GPU kernel实现。
