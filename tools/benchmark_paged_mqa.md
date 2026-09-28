**Paged MQA：torch.compile 与 Gems 对比**

如需“同一个大任务由多卡协作”，使用新增的 [分布式测试说明](/home/yoo/Documents/AIC/FlagGems-vllm/tools/benchmark_paged_mqa_distributed.md)：按KV长度分片、广播Q、汇总logits。下面suite的多卡模式仍是各卡独立测不同case。

当前默认只运行 `torch_compile_chunked_fp32 → gems`。不导入或调用 IxFormer 算子；native 保留为显式 opt-in 的历史诊断入口。`torch_fp32` 是 eager 名称，不能把旧 summary 的这一行当作 torch.compile。

**先修好编译路径，再跑一个小 case**

在已安装 CoreX/FlagTree 的服务器容器里，从仓库根目录运行。下面的 `.bc` 路径须换成当前容器内实际存在、此前让 Gems 跑通的路径：

```bash
python3 tools/benchmark_paged_mqa.py \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --quant both --warmup 2 --iterations 5 --repeats 3 \
  --output compile_gems_smoke.json
```

`--libdevice-path` 只设置 `TRITON_LIBDEVICE_PATH`，不改 CoreX root、PATH、LD_LIBRARY_PATH 或 SDK 软链接。环境检查会读取当前 Triton backend 的有效 libdevice 路径；已知缺失/不可读则在造数据前失败。内部探测接口不兼容时明确报告 UNVERIFIED，真实小 case 编译仍是判断依据。

之前出现所有 Gems ERROR 的那轮，实际错误是找不到 `corex-4.5.0.20260509/.../libdevice.compute_bi.10.bc`，且环境里的 TRITON_LIBDEVICE_PATH 为 null。共享库运行成功不代表 Triton 编译依赖齐全。用命令行传文件路径，可避免换终端后丢失一次性的 export。

默认 shape：FP8、B4、L512、H64、D128、page64、next_n1。`--quant both` 分别测试 FP8、FP4。`--torch-mode eager` 可显式改成 eager，`--torch-mode both` 则额外保留 eager；默认是 compile。

**批量测试**

小 case 的 compile 和 Gems 都通过后：

```bash
python3 tools/run_paged_mqa_suite.py \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --preset core --devices 0 --jobs 1 \
  --output-dir results/compile_gems_before
```

suite 会在启动批量任务前检查编译环境，显式路径传给每个子进程。每个 case 内部仍串行运行两个实现，每个逻辑设备最多一个 active case。

| 集合 | 内容 |
| --- | --- |
| smoke：3个shape | 跨页边界、混合长度、B4/L512 |
| core：17个shape，含smoke | B1～256、2K～128K上下文，page64/256、长短混合、较大输出宽度及物理cache |
| stress：6个shape | B1024/L8K、B512/L32K、B256/L64K、B64/L128K、大cache池、B256/L128K |
| all：23个shape | core + stress；默认FP8/FP4分别运行，共46个独立进程 |

查看清单和内存估计，无需GPU：

```bash
python3 tools/run_paged_mqa_suite.py --preset all --list-cases
```

利用多卡做覆盖检查，先从4个并发任务开始：

```bash
python3 tools/run_paged_mqa_suite.py \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --preset all --devices all --jobs 4 \
  --output-dir results/compile_gems_sweep
```

`--devices` 使用当前 CUDA_VISIBLE_DEVICES 下的 Torch 逻辑编号，脚本保留原mask。只调度本机可见设备；多节点须各自运行。每个case必须放进单个设备的显存，512GB集群总量不能直接当作单case预算。脚本检查实际free memory、主机可用RAM和常见cgroup-v2限制；估计超过预算记为 SKIPPED_MEMORY，不算通过。估计不是绝对的OOM保证。

多卡并发会竞争主机内存、编译资源，可能影响耗时。正式比较关键shape用 `--jobs 1` 和空闲设备复测。数据生成、native历史适配器均已分块，PyTorch计算也分块，避免一次构造完整 `[B,H,L]` 中间分数。

suite 默认普通case预热5次、每组20次；长上下文/大工作量预热2次、每组5次；均重复3组。可用 `--warmup`、`--iterations`、`--repeats` 覆盖。每个case默认超时1200秒，包含首次编译和正确性；按Ctrl-C终止本次启动的子进程组并保存部分结果。

**结果怎么看**

每个实现与同一份量化输入的独立FP32参考对拍，通过后才计时。默认比较有效token，容差 rtol=0.02、atol=0.05，可显式收紧。首次编译和预热不计入稳态延迟。

```text
Gems speedup vs torch.compile = compile wall time / Gems wall time
大于1：Gems更快；小于1：torch.compile更快。
```

只有两者都PASS且耗时有效，才生成加速比。没有比较结果的case不能算性能通过。native的95%标准已不适用于当前默认测试。

- `summary.md`：case总状态、各backend状态、耗时、Gems加速比、错误正文和日志链接。
- `summary.json`：完整汇总。
- `cases.json`：冻结的shape清单。
- 每个case的 `.json` / `.log`：实际算子路径与SHA256、版本、环境、原始耗时、失败阶段与完整traceback。

已有旧summary只有ERROR、不显示原因时，可以从原JSON重新生成可读摘要，不重跑GPU：

```bash
python3 tools/run_paged_mqa_suite.py --summarize-only results/old_run
```

也可以传 `results/old_run/summary.json`。此操作保留原JSON内容，只更新summary.md。

SKIPPED_MEMORY返回退出码2；失败/超时返回1；中断返回130。超时后的未完成实现标NOT_COMPLETED，不会拿前面的PASS掩盖失败。已有非空结果目录默认不覆盖。

**计时范围与 torch.compile 范围**

两个实现都接收相同的 packed FP8/FP4 Q、FP8 K cache、FP32 scale/weights，并输出FP32。当前 compile 版本仅编译固定大小的分页gather、dot、ReLU和head归约块；解码、cache拆分、外层循环、输出分配/复制也计入其完整调用时间。它不应被描述为整个函数融合成一个kernel。

Gems计时包含当前公开函数的KV拷贝、host同步和输出初始化。`wall_us_median` 是主要指标；GPU event给的是设备时间线区间，不等于主kernel独立耗时。没有使用CUDA Graph。JSON中显存峰值是PyTorch allocator统计，可能不含厂商库内部显存。

**修改算子后重跑**

单shape默认读取当前checkout的src；可以用 `--op-file /实际路径/候选算子.py` 指定替代文件。记录中包含实际source路径与hash。

```bash
python3 tools/run_paged_mqa_suite.py \
  --cases-file results/compile_gems_before/cases.json \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --devices 0 --jobs 1 --output-dir results/compile_gems_after
```

其他参数也应与before保持相同，尤其quant、torch-mode、chunk、seed、计时次数和并发方式。

若要单独检查算子语义，可运行 `check_paged_mqa_equivalence.py --libdevice-path ...`：默认18组精确小数值，将 torch.compile、Gems、手算答案直接对拍，检查分页、负权重、ReLU顺序和FP4 scale。`--cpu-only`仅检查fixture，不代表GPU正确性或性能。

本地已做CPU数据与编译测试、参数/调度/错误报告测试。用户此前已在BI-V150上验证B4/L512的FP8 Gems；新的compile+Gems组合和大矩阵仍需在目标容器实跑。
