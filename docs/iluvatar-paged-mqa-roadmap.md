**FP8/FP4 paged MQA logits：天数平台调研与优化路线**

调研日期：2026-09-24。代码基准：FlagGems-vllm `868379a`。
后续专项查找已定位到天数原生入口 `ixformer.inference.functions.dsa_indexer_mqa_logits_with_blocks` / `_bf16`，来自公开 CoreX 4.5 DSV4 镜像。该路径使用 BF16，与目标 FP8/FP4 格式不同。具体源码与证据见[专项查找结果](/home/yoo/Documents/AIC/FlagGems-vllm/docs/research/iluvatar-native-mqa/README.md)。下文首次调研的“未找到”结论应按此更新理解。
实际文件名是 `fp8_fp4_paged_mqa_logits.py`。本轮为源码与公开资料调研，没有修改算子，也没有由本助手在天数设备执行测试或测量性能。

用户补充的服务器状态：目标卡为 **BI-V150**，CoreX **可能为 4.5，准确版本待核实**；用户已按服务器文档跑通该算子。FP8/FP4 分别覆盖哪些路径、是否完成 reference 对拍、测试 shapes 与运行命令尚未记录。此状态取代此前“尚未跑通”的环境假设；不需要重新安装或从零搭建环境。

建议主线：**复用已跑通环境，保存运行命令与覆盖范围；原生基线查找并行推进 → 冻结当前性能与测量口径 → 直接读取分页 KV → 去除 host 同步和多余准备工作 → 天数矩阵指令验证与调优 → 逐 shape 验收 95%。**

**1. 算子语义与需要先读的代码**

这是索引器的打分计算。对 query 行 `r=(batch,next_n)` 和历史 token `t`：

```text
p = block_tables[batch, t // page_size]
o = t % page_size
logits[r,t] = sum_h(weights[r,h] * relu(sum_d(Q_real[r,h,d] * K_fp8[p,o,d]) * K_scale[p,o]))
```

没有 softmax，也没有乘 V。ReLU 先于乘 weights；不能把权重移入 ReLU，因为 weights 可以为负。

| 对象 | 当前支持路径/布局 |
| --- | --- |
| FP8 Q | `[B,next_n,H,D]`，E4M3，`q_scale=None` |
| FP4 Q | `[B,next_n,H,D/2]`，两枚 E2M1 打包为一字节；每 32 通道一个 UE8M0 scale，当前 D=128 时四个 scale 打包成 int32 |
| K | 两条路径中均为 FP8，附每 token 的 FP32 scale |
| 输出 | `[B*next_n,max_model_len]`，FP32 |
| 初期冻结范围 | H=64、D=128、page=64；按真实模型再扩展 page=256 等路径 |

FP4 路径当前在 Triton 内把 Q 解码为 FP16、把 K 转 FP16，再做 dot；它不要求已经存在原生 FP4 矩阵乘。

每页实际物理布局是 `[page_size*D 字节的 K][page_size*4 字节的 scales]`。虽然外部 view 是 `[pages,page_size,1,D+4]`，**并非每个 token 的 K 后面紧跟 scale**。直接读取时，以 uint8 为地址单位：

```text
page_base = p * page_size * (D+4)
K_byte    = page_base + o*D + d
scale_byte= page_base + page_size*D + o*4
```

推荐阅读顺序：

1. [host 入口与参数合同](/home/yoo/Documents/AIC/FlagGems-vllm/src/flaggems_vllm/ops/fp8_fp4_paged_mqa_logits.py:511)。
2. [预处理与 block 选择](/home/yoo/Documents/AIC/FlagGems-vllm/src/flaggems_vllm/ops/fp8_fp4_paged_mqa_logits.py:255)。
3. [general kernel](/home/yoo/Documents/AIC/FlagGems-vllm/src/flaggems_vllm/ops/fp8_fp4_paged_mqa_logits.py:126)。
4. [测试中的真实 KV 打包方式](/home/yoo/Documents/AIC/FlagGems-vllm/tests/test_fp8_fp4_paged_mqa_logits.py:101)。
5. [已有 benchmark shapes](/home/yoo/Documents/AIC/FlagGems-vllm/benchmark/test_fp8_fp4_paged_mqa_logits.py:60)。

**2. 原生实现调研：已确认的内容与尚未补齐的证据**

截至本次调研，尚未定位到一个能确认在目标天数环境中直接调用、且满足完整 FP8/FP4 paged logits 合同的原生实现。公开搜索未发现不等于不存在；它可能随厂商镜像、wheel 或扩展库分发。

| 入口 | 本次可确认的事实 | 是否可直接作为天数原生基线 |
| --- | --- | --- |
| 上游 DeepGEMM | 有 indexer/MQA scoring 算子族；README 的运行要求是 NVIDIA SM90/SM100。新旧接口应同时查 `fp8_fp4_paged_mqa_logits`、`fp8_paged_mqa_logits` | 否，必须确认天数移植版本 |
| 本仓库 NVIDIA reference | 使用 `vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits` | 否，这是当前 NVIDIA 对比入口 |
| DeepSpark/vLLM 分发 | DeepSpark 有公开 vLLM fork；还需核验实际天数发布分支与容器内容，默认分支不能代表厂商内部实现 | 查找入口，尚未确认具体 kernel |
| llm-d 天数部署资料 | 明确采用 CoreX/vLLM fork，并有 DeepSeek-V4-Flash 部署路径 | 证明有部署线索，不能证明本接口或 FP4 路径可用 |
| 本仓库天数 `qwen4_qsa_mqa_paged_dot` | BF16、4 heads，合同与本算子不同 | 否，可参考分页寻址方式 |

来源：[DeepGEMM 官方导出](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/__init__.py)、[DeepGEMM 硬件要求](https://github.com/deepseek-ai/DeepGEMM#requirements)、[DeepSpark 官方组织](https://github.com/Deep-Spark)、[DeepSpark/vLLM](https://github.com/Deep-Spark/vllm)、[llm-d 天数部署说明](https://github.com/llm-d/llm-d/blob/main/docs/getting-started/accelerators.md)。
本地相近实现见 [qsa_mqa.py](/home/yoo/Documents/AIC/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_iluvatar/ops/qsa_mqa.py:190)。

厂商软件获取入口也已找到：[DeepSpark V3.1 部署文档](https://github.com/Deep-Spark/DeepSparkInference/blob/master/models/nlp/llm/deepseek-v3.1/vllm/README.md) 指向[天数支持中心](https://support.iluvatar.com/)获取 SDK，并要求模型仓库 release 分支与 SDK 匹配。这是获取实际软件栈的依据；V3.1 部署本身不能证明后续 DSA/indexer 的这个算子存在。

查找原生实现时按以下顺序推进：

1. 以团队当前天数推理容器为准，记录镜像 tag/digest、卡型号、驱动/CoreX、torch、FlagTree、vLLM 版本。
2. 找到容器里真实安装的 vLLM、DeepGEMM、厂商扩展路径，搜索下面的新旧 symbol 及 `indexer`、`DSA`。
3. 从模型 decode/indexer 调用处沿 dispatch 向下追，确认最终进入哪个 Python/Triton/C++/扩展库实现。留意同名 wrapper 实际走其他 fallback 的情况。
4. 使用相同的量化输入和分页布局，先调用一次、与独立 reference 对拍，再记录 profiler 中的实际 kernel。
5. 固定最小调用样例和版本，封装成 benchmark native adapter。FP8 与 FP4 的支持状态分别登记。

在目标容器运行的只读定位命令（本轮未执行）：

```bash
ixsmi
python -m pip show torch triton flagtree vllm deep-gemm
python - <<'PY'
import importlib.util
import site
for name in ("torch", "triton", "vllm", "deep_gemm"):
    spec = importlib.util.find_spec(name)
    print(name, None if spec is None else spec.origin)
    if spec and spec.submodule_search_locations:
        print(list(spec.submodule_search_locations))
print("site-packages:", site.getsitepackages())
PY
```

对输出的实际包目录和镜像中的源码目录执行，替换占位路径：

```bash
rg -n 'fp8_fp4_paged_mqa_logits|fp8_paged_mqa_logits|get_paged_mqa_logits_metadata|paged_mqa|indexer' /实际/vllm目录 /实际/deep_gemm目录
```

若只找到二进制扩展，可检查它的 Python 导出和绑定文档；不要求拿到全部源码才能测性能，但需要一个能独立调用且语义明确的接口。若原生把 logits 与 top-k 融合，只输出 indices，不能直接拿它的延迟与单独 logits 对比；应索取独立入口，或另设完整 indexer 对比任务。

给带教/平台同事的简短询问稿（仅供复制，本轮未发送）：

> 我在接 `fp8_fp4_paged_mqa_logits` 的天数对比，确认了上游 DeepGEMM 接口，但还没定位到天数原生入口。咱们验收用的是哪个 CoreX/vLLM 镜像或算子包？能否给一个 native 调用样例、FP8/FP4 支持范围和验收 shapes？95% 是要求每个关键 shape 都达到，还是按约定聚合指标？

**3. 优化前必须修通的测试与测量**

以下障碍来自本地 `868379a` checkout 的静态检查，不否定用户已经在服务器上跑通的结果；服务器可能使用不同提交、补丁或独立脚本。先比对运行命令与代码版本，再补缺失项：

- [量化 helper](/home/yoo/Documents/AIC/FlagGems-vllm/tests/fp8_fp4_quant.py:108) 对 iluvatar 在 import 阶段报错。只删除测试 skip 不够。
- [paged tests](/home/yoo/Documents/AIC/FlagGems-vllm/tests/test_fp8_fp4_paged_mqa_logits.py:70) 与 [benchmark](/home/yoo/Documents/AIC/FlagGems-vllm/benchmark/test_fp8_fp4_paged_mqa_logits.py:43) 也未接入天数 reference。
- benchmark 顶层从 tests 导入平台函数；需要把可选 native adapter 与独立 correctness 解耦。
- 当前 [set_shapes](/home/yoo/Documents/AIC/FlagGems-vllm/benchmark/test_fp8_fp4_paged_mqa_logits.py:198) 固定使用八组 BENCH_SHAPES，不能假定 `--level core` 或 `--shape_file` 已控制这个算子的 workload。
- `--mode kernel` 是对公共 Python 函数调用 `do_bench`，仍包含它发射的拷贝、填充和主 kernel，并非只测一个 Triton kernel。见 [计时实现](/home/yoo/Documents/AIC/FlagGems-vllm/benchmark/base.py:287)。

先做不依赖原生包的小规模数学 reference：解码已量化的 Q/K，按 block table 取页，在 FP32 中计算公式。可在测试内使用 torch；生产路径不得用 torch compute fallback。

优先复用用户已经跑通的命令。如果缺少独立对拍或某个 dtype 路径，可补 smoke：`B=4,next_n=1,H=64,D=128,page=64,context=512`，先 FP8 后 FP4。当前 Gems 不使用 schedule_metadata，独立测试可传 `None`。对本 checkout 新增独立测试时，先关闭当前 NVIDIA 风格 TLE 路径，并记录与已跑通环境的配置差异：

```bash
export GEMS_VENDOR=iluvatar
export FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE=0
```

随后补 `context=0/1/63/64/65`、next_n=2 且长度不同、请求间长度不同、乱序/共享物理页、clean_logits 两种情况。FP4 应验证解码与 scales，而不是只比较原始未量化的 BF16 输入。除总体误差外，检查逐行/最大误差和输出有效区/尾部。

源码还存在需要处理的静态风险：[TLE guard](/home/yoo/Documents/AIC/FlagGems-vllm/src/flaggems_vllm/ops/fp8_fp4_paged_mqa_logits.py:441) 未排除 FP4，但 TLE kernel 按 FP8 Q 和 H*D stride 读取、没有 Q_scale 参数。在满足 TLE 条件的 FP4 输入上存在错读风险，尚未设备复现。当前 page=64 的测试覆盖不到 page=256 的 TLE 路径。应先完善 dtype/vendor/shape guard，再测该路径。

**4. 按收益依据排序的优化实验**

| 顺序 | 具体改动/实验 | 为什么值得先做 | 验证与交付 |
| --- | --- | --- | --- |
| 0 | 冻结环境、合同、native adapter；建立当前版本 baseline | 把编译问题、错误计算和性能问题分开 | 独立正确性通过；native 有或无均记录证据 |
| 1 | kernel 直接从原分页 cache 读 K/scales | 当前每次把整个物理 cache 拆分成连续数组，即使只访问少量页 | 保持其他策略不变，对拍后比较 operator 延迟和复制流量 |
| 2 | 原地读取 block table/context；移除每次 `max().item()` | 当前展开/转换有额外 kernel，取 GPU 标量会同步 | 测 host 同步、kernel 数；使用可信 host 上界或设备调度，避免极大空 grid |
| 3 | 重新处理输出初始化 | 当前 `torch.full` 填满 `[rows,max_model_len]` | 先确认尾部合同；需要清零/-inf 时用 Triton 填充或融合写回 |
| 4 | 核对天数 FP8 dot lowering，分别测试 FP8 与 kernel 内升 FP16 路径 | FP8 dtype 可用不等于原生 FP8 矩阵指令可用 | 记录编译目标/IR/汇编与误差；按目标硬件选路径 |
| 5 | 接入天数配置，调每 CTA 页数、num_warps、num_stages | 当前只有按最大长度的四档 BLOCK_KV，未覆盖 batch/设备差异 | 配置 key 纳入 rows、长度桶、page、dtype；冻结配置后计时 |
| 6 | 有 profiler 证据后再做流水、next_n 的 K 复用、变长任务调度 | 此时主 kernel 才可能成为主要瓶颈 | 每轮只改变一个主要假设，记录 keep/revert 与关键 shape 回归 |

可量化的源码线索：benchmark 的物理 KV 池约 42.93 MiB，每次拆分都产生额外数据移动；`rows=256,max_model_len=111*1024` 时完整 FP32 输出为 111 MiB。context=1024 时仅 1/111 列有效。**这些是按尺寸算出的字节量，不是已测得的耗时占比或加速比。**

`clean_logits=False` 时当前 Gems 仍承诺/实现尾部为零，不能为提速擅自换成未初始化输出。必须核对目标 native 和调用方合同，再决定是否调整；`clean_logits=True` 的无效区 -inf 语义应保留。

本文件没有 `libtuner` 接入，也没有 paged 算子的 tune config。NVIDIA YAML 中的 `fp8_fp4_mqa_logits` 属于另一个 dense 算子。后续实现需要增加 paged 配置/路径，并检查 exports 与后端替换机制。

当前相邻 FlagTree 源码的天数 TLE whitelist 不包含 `gpu.wgmma`；编译器还有架构相关的 FP8 支持分支。这些只说明源码支持边界，不能替代目标容器验证。先测基础 dot 和通用路径，不照搬 H800 TMA/WGMMA 参数。
见 [TLE whitelist](/home/yoo/Documents/AIC/FlagTree/third_party/iluvatar/backend/tle_supported.py:1) 和 [FP8 配置](/home/yoo/Documents/AIC/FlagTree/third_party/iluvatar/backend/compiler.py:208)。

**5. 验收 95% 的正确口径**

保持同一工作量时：

```text
性能比例 = native_latency / gems_latency >= 0.95
等价：gems_latency <= native_latency / 0.95 ≈ native_latency * 1.05263
```

例如原生 100 μs，Gems 不超过约 105.26 μs 即达到其 95% 性能。不是要求 Gems 耗时不超过原生的 95%。本任务应按用户要求采用 0.95，不能沿用仓库通用 0.9 门槛。

需要保留三种不同基线：

| 基线 | 用途 |
| --- | --- |
| 测试专用 FP32 reference | 判定数值和分页语义正确 |
| 同卡当前 Gems 提交 | 判定优化是否有效及是否回归 |
| 同卡天数 native 实现 | 判定是否达到原生 95% |

native 未找到时第三列必须为 N/A；前两者不能替代原生验收。

统一输入 bytes、page size、max_model_len、clean_logits、context 分布、页共享方式、设备与软件版本。输入量化/随机数据构造在计时外。每次调用必需的布局转换、metadata 生成和输出准备，按同一 API 边界计入；可另报双方都预先准备 metadata 的核心计算延迟。

JIT/autotune 完成后再测稳定耗时，冷启动另记。交错重复 native/Gems，报告中位数和波动；eager/graph 模式一致。95% 附近不能用单次测量宣布通过。

先用三组性能种子定位瓶颈：`(B,next_n,L)=(4,1,1024)、(256,1,8192)、(32,1,65536)`；再加入 next_n=2 和严重长短不均请求，最后用模型实际 shapes 验收。当前仓库八组 BENCH_SHAPES 可作为起点，但其等长请求与重复物理页不一定代表生产负载。

报告按 shape 展示：格式、page、B/next_n/L、native μs、before μs、after μs、native/after、波动、正确性结果。聚合值另列，不能掩盖关键 shape 未达标。

**6. 建议拆成四个可交付阶段**

1. **固化已跑通结果**：保存 BI-V150 环境版本、服务器代码提交、运行命令、FP8/FP4 覆盖范围和对拍结果；只补缺失的独立 reference 与边界测试。
2. **对比可用**：native 最小调用样例与固定版本，或明确“尚无 native 基线”；冻结 active set 和计时边界。
3. **首个优化**：只做直接分页读取，保留 before/after 和正确性证据；之后逐项处理同步与初始化。
4. **平台调优与验收**：dot lowering、autotune、必要的调度优化，逐 shape 检查 95%，记录回归和未支持范围。

目标卡已由用户确认为 BI-V150；CoreX 暂记约 4.5。准确软件/镜像版本、已跑通的测试范围和正式验收 shapes 尚待记录，因此本路线不预估加速比，也不声称原生 95% 已可达。编码前按 [workflow.md](/home/yoo/Documents/AIC/FlagGems-vllm/workflow.md) 补齐项目真相、接口合同、实现路径三张表；本 checkout 未找到其引用的 `optimization.md` 和 `deep_opt.md`。当前任务是这个 fused logits 算子，工作流末尾旧的 conv1d 示例不适用于本次。
