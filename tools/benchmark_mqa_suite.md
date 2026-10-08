# MQA 五档尺寸、两个算子、三方对比

统一入口是 `tools/benchmark_mqa_suite.py`。默认测试：

- 普通版 `fp8_fp4_mqa_logits` 和分页版 `fp8_fp4_paged_mqa_logits`。
- 每个算子分别运行 FP8 Q 和 MXFP4 Q；K 始终为 FP8，输出为 FP32。
- 三方：Torch.compile 分块 FP32、冻结的原始 Gems、当前 TLE 候选 Gems。
- 16 个 Torch 可见设备协作处理同一个任务，按 K 的长度维分片。

| 档位 | M（普通版）/ B（分页版） | 全局 N / L | 16 rank 时每 rank 的 N / L |
|---|---:|---:|---:|
| 小 | 4 | 4,096 | 256 |
| 中小 | 16 | 16,384 | 1,024 |
| 中 | 64 | 65,536 | 4,096 |
| 大 | 128 | 262,144 | 16,384 |
| 超大 | 256 | 1,048,576 | 65,536 |

所有档位固定 H=64、D=128、page=256；paged 固定 next_n=1、所有请求等长。
一共 20 个「算子 × 尺寸 × Q 格式」组合，每个组合对比三份实现。

普通版所有 M 个 Q 共享一个 `[N,128]` 的 K 张量；分页版每个请求有各自的
逻辑 KV 序列。两者都输出 `[M/B,N/L]`，但 KV 容量和复用不同，不能把两个
算子的耗时直接算成“分页带来的加速比”。同一算子、尺寸、格式的三方使用
相同的确定性输入、分片、容差和计时设置。

## 修改算子之前：冻结原始实现

在天数容器里，使用此前已经跑通算子的 Python 环境：

```bash
cd /root/FlagGems-vllm

python tools/benchmark_mqa_suite.py \
  --freeze-baseline results/mqa_original
```

这一步不运行 GPU。它保存整个 `src/flaggems_vllm` 包，包括两个算子、相互
import 的 helper、runtime、autotune 配置，并记录每个文件的 SHA256。
已有快照不会被覆盖；快照被修改后会拒绝使用。需要在修改算子**之前**执行，
不能修改完以后再把新代码冻结为“原始版本”。

快照冻结 Python 包源码和配置；Torch、FlagTree、CoreX 等外部依赖仍使用当前
环境。一次三方比较共用这套环境，不是给三个实现各装一份编译器。

## 一条命令运行五档、两个算子、三方

```bash
python tools/benchmark_mqa_suite.py \
  --baseline-dir results/mqa_original \
  --nproc-per-node 16 \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --warmup 3 --iterations 10 --repeats 5 \
  --output-dir results/mqa_threeway_after
```

用 **python** 启动，入口会自行运行 torchrun；不要再在外面套 torchrun。
目前统一调度入口面向单机多设备，沿用 static rendezvous、127.0.0.1:29501，
可用 `--master-port` 改端口。编号遵循 `CUDA_VISIBLE_DEVICES` 中的逻辑设备，
`--nproc-per-node 16` 不等于确认存在 16 块物理板卡。

默认候选来自当前仓库的 `src/`；也可以用 `--candidate-src /另一个checkout/src`
指定候选。每次运行会把原始版、候选版再次复制到结果目录的 `sources/`，保证
运行过程中改动工作目录不会改变正在比较的算子。测试入口和 helper 在启动
后发生变化会报错，避免一轮测试混入两个测试协议。

三方分别在独立进程组中运行，避免同名 Python 模块、Triton launcher 或 import
缓存串用。进程组串行调度，不让三份实现同时争抢同一批设备。默认共 30 个
进程组，每个组运行 FP8 和 FP4 两种格式。数据生成、参考计算、首次编译可能
耗时较长；控制台给出当前进度和对应日志路径。

如果 TLE 优化还没开始，可以先运行同样的两方基线：

```bash
python tools/benchmark_mqa_suite.py \
  --baseline-dir results/mqa_original \
  --backends torch baseline \
  --nproc-per-node 16 \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --output-dir results/mqa_before
```

优化后保留同一份 `mqa_original`，去掉 `--backends torch baseline`，换一个新的
输出目录，就会在同一轮里重新测量三方。不要直接拿不同软件环境或不同计时
协议的旧数字作为本轮分母。

## TLE 路径如何确认

原始版强制关闭两个算子的 `FLAGGEMS_*_TLE` 开关；候选版开启这两个开关。
候选仍调用真实公开 API，不跳过其 host dispatch。

只设置环境变量不代表实际进入 TLE。脚本在一次不计时的正确性调用中，观察
每个非空 rank 是否调用候选模块的 `_launch_tle_kernel`，然后恢复原函数再计时。
这是 host dispatch 证据，不能代替 compiler IR 或 profiler 的指令级证据。

- `PASS`：数值检查通过；TLE 列还要求所有非空 rank 都调用了 TLE launcher。
- `FALLBACK`：候选数值正确，但至少一个非空 rank 没有调用 TLE launcher。
  仍可记录实际回退耗时，但不生成 TLE 加速比。
- `UNVERIFIED_TLE`：候选没有可观察的 launcher，不能确认 TLE 路径，不生成耗时比较。
- `BASELINE_USED_TLE`：原始版关闭开关后仍调用了 TLE launcher，视为错误。

通用算子内的 TLE 路径针对 NVIDIA 的 WGMMA/TMA，而且有尺寸限制。天数候选
位于 `runtime/backend/_iluvatar/fused/`，使用 `gpu.alloc/local_ptr` 做同步
FP16 shared staging，尚未通过天数真机验收，默认不替换通用实现。
在**导入包之前**设置以下开关，才会启用两个天数候选的顶层 API：

```bash
export FLAGGEMS_ILUVATAR_MQA_EXPERIMENTAL=1
```

脚本现在调用 `flaggems_vllm.<op>`，并从公开函数的 `__module__` 找到实际
launcher；报告中的 `implementation.module/file` 会记录真正执行的厂商模块。
直接导入 `flaggems_vllm.ops.<op>` 会绕过厂商替换，不应用它验证此候选。
修改前冻结的 baseline 不包含此候选，在同一开关下仍执行原始实现。

候选先覆盖 D=128、H=1..64、FP8/MXFP4 Q，paged 支持 page=16/32/64/128/256
及 `[B]` / `[B,next_n]` context lengths。先在已配置好的天数环境运行：

```bash
python tools/check_iluvatar_mqa_tle.py --bench --output /tmp/mqa-tle-check.json
```

此工具对独立 FP32 reference 检查边界，比较候选的普通 Triton 与 TLE 分支，
记录各自 autotune 的 best config。两者可能选择不同 block size，因此其耗时
之比不是固定配置的原语消融。原始 Gems 与候选的比较仍使用本文三方 suite。
TLE staging 不保证异步重叠或提速；需在设备上检查 IR 与实测结果。
`--cpu-only` 只检查输入和参考计算，不代表 kernel 正确性。

若优化后的 host launcher 改名，传入真实入口名称：

```text
--dense-tle-launcher <普通版模块中的host函数名>
--paged-tle-launcher <分页版模块中的host函数名>
```

建议保留 `_launch_tle_kernel` 作为实际优化路径的主机入口。不要将整个公共
API 指定为 launcher；那只能证明 API 被调用，不能证明选中了优化路径。

## 结果和计时口径

- `summary.md`：三个计时范围各一张表，包含三方耗时、状态和加速比。
- `summary.csv`：同一组数值，方便导入表格工具。
- `summary.json`：结构化汇总与错误原因。
- `reports/*.json`：每个 rank 的原始样本、误差、TLE launcher 记录、环境和源码。
- `logs/*.log`：完整标准输出和报错；rank 崩溃还会留下 `.rankN.error.json`。
- `run.json`、`jobs/`、`sources/`：冻结的测试参数、子进程任务和两份 Gems 源码。

| 指标 | 包含范围 |
|---|---|
| compute_only | 本地公开算子完整调用，包括其解码、分配、初始化、拷贝和 host 开销 |
| communication_and_assembly_only | Q/scales/weights 广播、FP32 AllGather、输出拼接 |
| end_to_end | 广播 → 本地算子 → AllGather → 拼接 |

每个 rank 保留完整全局输出；KV 预先驻留，初始生成/传输、JIT、参考计算和
预热不计时。每轮重复 `iterations` 次取每次调用平均耗时，取所有 rank 中最慢
者，最后取 `repeats` 轮的中位数。三个范围独立实测，不能相加替代端到端值。

正确性：每个 rank 全部有效结果对独立逐请求 FP32 reference，再对全局输出的
分片边界、页边界、首尾和随机列抽查；默认 `rtol=0.02, atol=0.05`。
只对通过检查、耗时有效、输入与环境签名相同的结果生成加速比：

```text
baseline_vs_torch = Torch_ms / 原始Gems_ms
tle_vs_torch      = Torch_ms / TLE候选_ms
tle_vs_baseline   = 原始Gems_ms / TLE候选_ms
```

显存或主机内存估计超过预算记录 `SKIPPED_RESOURCE`，不算 PASS；估计包含
主要中间张量但不保证杜绝厂商 workspace 或碎片导致的 OOM。失败组不会阻止
其余组继续测试；中断或超时会终止本脚本启动的进程组并保留部分结果。
退出码 0 表示所有选中项通过；2 表示有 FALLBACK/未验证/资源跳过；1 表示错误；
130 表示用户中断。

## 查看计划、缩小调试、重建汇总

```bash
python tools/benchmark_mqa_suite.py --plan

# 先测最小档：两个算子、两种格式、三方。
python tools/benchmark_mqa_suite.py \
  --baseline-dir results/mqa_original --shapes 4x4096 \
  --nproc-per-node 16 \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --warmup 1 --iterations 2 --repeats 2 \
  --output-dir results/mqa_threeway_smoke

python tools/benchmark_mqa_suite.py --summarize-only results/mqa_threeway_after
```

开发机没有可访问 GPU 时，可验证参考计算和通信：

```bash
python -m unittest discover -s tools/tests -v
python tools/benchmark_mqa_suite.py \
  --device cpu --backends torch --torch-mode eager --nproc-per-node 2 \
  --shapes 3x70 --heads 16 --page-size 16 --chunk-tokens 16 \
  --warmup 1 --iterations 2 --repeats 2 \
  --output-dir /tmp/mqa_cpu_check
```

CPU 模式不运行真实 Gems，也不代表天数 GPU 或 TLE 的正确性和性能验证。
