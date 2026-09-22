# QComp 与 ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization 对比分析

## 1. 背景

ASER（Activation Smoothing and Error Reconstruction for Large Language
Model Quantization）是一种面向 LLM PTQ
的低秩误差补偿方法。论文包含两个核心部分：

1.  Error Reconstruction：利用低秩补偿矩阵重构量化误差。
2.  Activation Smoothing：通过处理 activation outlier 改善量化误差分布。

ASER 的核心观点是：量化误差存在低秩结构，可以使用 LoRA-style
矩阵进行补偿。

参考： ASER: Activation Smoothing and Error Reconstruction for Large
Language Model Quantization, AAAI 2025.

------------------------------------------------------------------------

# 2. ASER 方法

ASER关注量化前后输出误差：

$$
Y = WX
$$

$$
Y_q=W_qX
$$

误差：

$$
\Delta Y=(W-W_q)X
$$

ASER通过低秩结构恢复误差：

$$
W-W_q\approx AB
$$

推理时：

$$
Y_q+ABX
$$

------------------------------------------------------------------------

# 3. QComp 方法

当前 QComp目标：

$$
Y_{fp16}-Y_{awq}\approx(XB^T)A^T
$$

定义：

$$
T=Y_{fp16}-Y_{awq}
$$

使用 AWQ runtime 输入：

$$
X=X_{awq}
$$

拟合：

$$
T\approx XB^TA^T
$$

推理：

$$
Y=Y_{awq}+\alpha(XB^T)A^T
$$

------------------------------------------------------------------------

# 4. 相似点

## 4.1 低秩误差补偿

ASER 和 QComp 都认为量化误差可以通过低秩结构恢复。

## 4.2 LoRA-style 参数化

都采用：

$$
A,B
$$

避免完整矩阵存储。

## 4.3 使用 activation 信息

ASER使用 activation smoothing 和 whitening SVD。

QComp直接在 activation/output space 建模误差。

------------------------------------------------------------------------

# 5. 核心区别

## 5.1 补偿空间不同

ASER：

主要进行 weight-space error reconstruction：

$$
W_q+AB
$$

QComp：

进行 output-space error correction：

$$
Y_q+\alpha(XB^T)A^T
$$

这是两者最大的区别。

------------------------------------------------------------------------

## 5.2 为什么 QComp 不采用 W-Q

AWQ不是简单：

$$
Q=round(W)
$$

而包含 scaling 和 reparameterization。

因此：

$$
W-Q
$$

可能混合：

-   AWQ变换误差
-   量化误差

无法准确表示 runtime 输出误差。

QComp直接优化：

$$
Y_{fp16}-Y_{awq}
$$

匹配真实推理行为。

------------------------------------------------------------------------

# 6. QComp 相对 ASER 的区别和潜在贡献

## 6.1 Runtime-consistent compensation

QComp训练和部署使用同一分布：

训练：

$$
X_{awq}
$$

部署：

$$
X_{awq}
$$

避免 FP16 activation 与 AWQ activation mismatch。

------------------------------------------------------------------------

## 6.2 AWQ-specific

QComp针对：

-   AutoAWQ W4A16
-   WQLinear_GEMM
-   A10部署

考虑：

-   AWQ scaling
-   runtime activation shift
-   推理额外开销

------------------------------------------------------------------------

## 6.3 Hardware-aware inference

QComp不展开：

$$
A@B
$$

保持：

$$
X\rightarrow B\rightarrow A
$$

额外计算：

$$
O(dr)
$$

满足吞吐约束。

------------------------------------------------------------------------

# 7. 当前实验结果

QComp MVP：

单层：

model.layers.0.mlp.down_proj

rank=8。

Layer output error：

64.6392 → 61.4744

下降：

4.89%。

PPL：

AWQ:

21.5365

AWQ+QComp:

21.5117

改善：

0.11%。

说明 activation-space compensation 方向有效。

------------------------------------------------------------------------

# 8. 当前问题

多层 QComp主要问题：

calibration overfitting。

后续：

1.  增加 calibration 数据。
2.  使用 ridge regression。
3.  validation alpha search。
4.  layer selection。
5.  benchmark throughput。

------------------------------------------------------------------------

# 总结

ASER：

$$
\Delta W
$$

主要关注 weight/error reconstruction。

QComp：

$$
\Delta Y
$$

主要关注 AWQ runtime output correction。

QComp不是重新提出低秩补偿，而是：

针对 AWQ runtime 的 activation-consistent、hardware-aware
低秩输出误差补偿方法。
