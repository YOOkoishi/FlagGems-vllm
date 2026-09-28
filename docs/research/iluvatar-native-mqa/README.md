**已找到：天数 IxFormer 的分页 MQA logits 入口**

2026-09-24，从天数公开发布的原生镜像中提取调用代码后确认：

```python
import ixformer.inference.functions as ixfops

# 天数 V3.2 indexer 调用
ixfops.dsa_indexer_mqa_logits_with_blocks

# 该镜像中的天数 V4 indexer 调用
ixfops.dsa_indexer_mqa_logits_with_blocks_bf16
```

这两个入口承担 `QK → ReLU → 乘 head weights → 沿 head 求和` 的分页索引打分，对应当前正在研究的 paged MQA logits 计算。V4 调用后再单独执行 top-k，所以不是只提供完整 attention 或融合后无法拆出的接口。

但找到的 V4 路径使用 **BF16 Q/K/weights**，并明确关闭 FP4。它是天数的同类计算实现，**不是同输入格式的 FP8/FP4 直接替换接口**。同名 FP8/FP4 原生版本目前仍未确认。

两个保留原始行号的源码快照：

- [V4 调用代码](/home/yoo/Documents/AIC/FlagGems-vllm/docs/research/iluvatar-native-mqa/ilu_deepseek_v4_attention.py.txt:1379)：第 41 行 import IxFormer；1349/1379 行分别是 prefill/decode 调用；1181 行关闭 FP4；1229 行指定 BF16 cache；1304 行 weights 使用 Q 的 dtype。
- [V3.2 调用代码](/home/yoo/Documents/AIC/FlagGems-vllm/docs/research/iluvatar-native-mqa/ilu_sparse_attn_indexer.py.txt:158)：第 5 行 import IxFormer；130/158 行分别是 prefill/decode 调用。

V4 函数的调用参数依次是：Q、query 累计长度、KV 累计长度、分页 K cache、页表、weights；另传 `max_q_len`、`max_kv_len`、`max_context_len`。Q 是 `[num_tokens,H,D]`，K cache 是 `[pages,page_size,D]`，没有目标 FP8 cache 的 scales 尾部。

V4 调用侧说明老的无 `_bf16` 入口输出 FP32 logits、新入口输出 BF16 logits。此说明尚未通过 IxFormer 函数本体与真机输出核实。源码内有关旧入口在 page=256 回退 PyTorch 的注释，不能直接套到新的 `_bf16` 入口；正式测速时仍需确认实际调用路径。初次对齐应使用 next_n=1，厂商调用侧对 padding decode 显式报不支持。

具体来源：

```text
registry.iluvatar.com.cn:10443/k8s/vllm:corex450-rc12-dsv4-flash-w4a8-bnxt239
IxFormer: 0.7.0+corex.4.5.0.rc.12.20260722（镜像构建记录）
vLLM: 0.25.1+corex.4.5.0.rc.12.20260722（镜像构建记录）
vllm-iluvatar: 0.1.0（安装包 METADATA）
```

发现过程：从 [llm-d 官方合并的天数支持 PR](https://github.com/llm-d/llm-d/pull/2381) 追到天数公开镜像仓库，再通过匿名 Registry API 查询标签，找到这个 CoreX 4.5 / DSV4 镜像。只下载约 5 MiB 的厂商源码与安装层，没有安装软件或运行镜像。源码层和安装层的 SHA256 已与 manifest 一致。

这次找到的是厂商发布包里的具体 API 调用和完整调用方源码；IxFormer 函数本体在预装基础层中，尚未提取，底层 C++/设备 kernel 源码也未取得。不能把本次只读核查写成在 BI-V150 上运行通过。

镜像及源码文件的 digest、版本和路径记录在 [provenance.json](/home/yoo/Documents/AIC/FlagGems-vllm/docs/research/iluvatar-native-mqa/provenance.json)。快照仅用于本地调研，保留了原始文件内容，不参与本仓库算子实现或测试。
