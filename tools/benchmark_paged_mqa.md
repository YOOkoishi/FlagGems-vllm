**分页 MQA baseline 脚本**

现在有三份脚本，各做一件事：

- `check_paged_mqa_equivalence.py`：小数值、可手算的三方对拍，查 native/Gems 是否在计算同一个分页 logits。
- `benchmark_paged_mqa.py`：测一个 shape，适合调优时反复跑。
- `run_paged_mqa_suite.py`：覆盖不同 batch、上下文长度、页大小和 cache 容量，支持多卡分配 case。

**先验证计算，再扩大规模**

在目标服务器仓库根目录运行：

```bash
python3 tools/check_paged_mqa_equivalence.py --output equivalence.json
```

默认18组：FP8/FP4、page64/256、跨页长度、乱序物理页、负 Q/权重、零 Q，以及 FP4 非单位 scale。四条请求覆盖不同 head 位置及四个32通道分组。输入都是各格式能精确表示的小整数/二进制小数，同时检查 native、Gems、手算答案，要求有效 logits 完全相等。它用于识别公式、寻址或 scale 错误，不用于测性能，也不能证明任意输入都等价。

**多规模测量**

原来的 B4/L512 是 smoke。当前矩阵固定 H64/D128/next_n1，主要规模如下：

| 集合 | 内容 |
| --- | --- |
| smoke（3个shape） | 单请求跨页、混合长度、B4/L512 |
| core（17个shape，含smoke） | B1～256、上下文2K～128K；page64/256；长短混合；较大输出宽度；物理cache大但有效请求小 |
| stress（6个shape） | B1024/L8K、B512/L32K、B256/L64K、B64/L128K、大cache池、B256/L128K |
| all（23个shape） | core + stress；默认每个shape分别跑FP8和FP4，共46个独立进程 |

查看完整矩阵和内存估计，不需要GPU：

```bash
python3 tools/run_paged_mqa_suite.py --preset all --list-cases
```

先做单卡串行 baseline：

```bash
python3 tools/run_paged_mqa_suite.py --preset core --devices 0 --jobs 1 --output-dir results/before
```

扫全矩阵，可在本机可见设备之间分配 case：

```bash
python3 tools/run_paged_mqa_suite.py --preset all --devices all --jobs 4 --output-dir results/parallel-sweep
```

`--jobs 4` 表示最多4个case同时运行，每个逻辑device上最多1个进程；同一case内部的 native→torch→Gems 始终串行。确认主机RAM与编译资源够用后，可以改为 `--jobs 16`。如需 eager 和 compile 同时测，加 `--torch-mode both`。编译开销大，推荐先只在几个代表shape上测，例如：

```bash
python3 tools/run_paged_mqa_suite.py --preset core --case b4_l512 b32_l16384 b32_l65536 --devices 0 --jobs 1 --torch-mode both --output-dir results/compile-check
```

**16卡的512GB显存不会自动合成一个算子的显存池。** 每个case仍需装进它分配到的那个Torch可见设备。脚本读取实际free memory，在默认70%的可用显存预算内安排case；主机RAM和常见cgroup-v2内存限制也会检查。B256/L64K的当前保守估计约16.9GiB，B256/L128K约31.5GiB；这是估计，不是实测峰值或“保证不OOM”。后者在32GiB设备和默认预算下通常会跳过。native/编译器的内部workspace仍可能超出估计。

多卡同时跑可能竞争CPU、内存带宽和板卡功耗，作为覆盖检查和优化初筛；最终关键shape用 `--jobs 1`、空闲机器再测。`--devices` 是当前 `CUDA_VISIBLE_DEVICES` 下的逻辑编号，脚本保留原mask。只调度当前主机可见设备；若16卡分布在多台主机，分别运行并使用不同结果目录，记录里的hostname用于区分。

输入生成和native BF16准备已改为每次处理最多约65536个物理token，减少大case的临时内存。PyTorch计算仍分块，不创建整个 `[B,H,L]` 中间分数。每个case使用独立Python进程，隔离解释器和CUDA上下文，并保留各自日志。按Ctrl-C会终止本次启动的子进程组，并保存已完成和取消的结果。

suite 默认普通case预热5次、每组20次调用；长上下文或大工作量预热2次、每组5次；均重复3组。可以用 `--warmup`、`--iterations`、`--repeats` 覆盖。小规模先确认能跑，大规模先收集一轮结果，再对关键shape增加次数。

结果目录包含：

- `summary.md`：可直接阅读的各shape/各实现耗时，显示case总状态和每条实现状态。
- `summary.json`：完整汇总，包括设备、运行参数、内存估计与错误。
- `cases.json`：这次实际使用的shape清单。
- 每个case的 `.json` 和 `.log`：原始结果、错误栈与首次编译日志。

`SKIPPED_MEMORY` 不算通过，suite退出码为2；失败/超时退出码为1。超时或半途失败时，尚未完成的实现显示 `NOT_COMPLETED`，不会用前面某一条PASS掩盖。已有非空结果目录不会覆盖。

调优后复用冻结的矩阵：

```bash
python3 tools/run_paged_mqa_suite.py --cases-file results/before/cases.json --devices 0 --jobs 1 --output-dir results/after
```

其他参数（量化类型、torch模式、chunk、seed、计时次数等）也要保持相同。suite记录了它们；如before用了非默认值，after需同样传入。不要用不同并发方式或不同设备的数字直接判断微小收益。

**单个 shape 的原用法**

在已经跑通算子的天数容器里，用原来的 Python 环境，从服务器上的仓库根目录运行：

```bash
python tools/benchmark_paged_mqa.py --output before.json
```

默认串行执行：IxFormer 原生接口 → PyTorch eager → 当前仓库的 `fp8_fp4_paged_mqa_logits.py`。默认输入是 FP8、B=4、H=64、D=128、page=64、context=512、next_n=1。

同时跑 FP8、FP4 和 torch.compile：

```bash
python tools/benchmark_paged_mqa.py --quant both --torch-mode both --output before.json
```

修改算子后，用完全相同的参数再次运行，输出改为 `after.json`。脚本默认读取当前 checkout 的 `src`，并记录实际算子文件路径、SHA256、Git commit、包版本和 benchmark 脚本 SHA256。也可以用 `--op-file /实际路径/候选算子.py` 指定其他文件；该文件须保持原入口签名，且它依赖的 FlagGems 包仍需可导入。

更换输入例子：

```bash
python tools/benchmark_paged_mqa.py --batch 32 --context 8192 --output b32-l8192.json
python tools/benchmark_paged_mqa.py --contexts 63,64,65,512 --max-model-len 1024 --quant both --output variable-length.json
```

`--contexts` 覆盖 `--batch`、`--context`。当前只支持 next_n=1、D=128；heads 可选16/32/64，page可选16/32/64/128/256。输入页表会随机打乱，每条请求分配独立物理页；`--cache-pages` 可以增加物理 cache 容量，用于观察当前实现拆分整个 cache 的开销。FP4 输入直接生成合法 E2M1/UE8M0 编码，不声称复现某个模型量化器的舍入过程。

每条实现先与独立 FP32 参考计算对拍，再预热、测速。默认 warmup=10次，每组100次调用，重复5组，报告每次调用的中位耗时（微秒），各组原始耗时也保存在 JSON。首次 JIT/compile 放在计时外，单独记录 `first_call_ms`。默认容差 `rtol=0.02, atol=0.05`，可以通过同名参数收紧；比较有效 token，padding 值不作为三个接口共同合同。

终端主要看 `wall=... us`，这是包含 host 调度、同步和算子准备工作的实际调用时间。JSON 另有 GPU event 时间，字段名为 `device_timeline_us_median`，它是设备时间线区间，不是某个主 kernel 的独立耗时。脚本不使用 CUDA Graph，不同时运行多个候选。

各行含义：

| 名称 | 实际测量 |
| --- | --- |
| native | 已准备好的 BF16 Q/K/weights → IxFormer 无后缀接口 → FP32 logits；每次输出分配计时，量化输入转 BF16 不计时 |
| torch_fp32 | 原始量化输入的解码、KV 拆分、分页 gather、FP32 dot/ReLU/加权归约、输出分配与复制 |
| torch_compile_chunked_fp32 | 与 eager 同算法，仅用 torch.compile 编译固定大小的 gather/dot/ReLU/归约块；解码和外层循环仍计时 |
| gems | 当前源码的完整公开函数，包括原有 KV 拷贝、host 同步和输出初始化 |

torch 按 `--chunk-tokens`（默认512）计算，只算到当前最长上下文，避免长序列的巨大中间张量或整段循环展开成巨型编译图。token 位置索引在计时前准备，eager/compile 使用相同口径。编译块使用 `fullgraph=True`，无法捕获时会报错，详见 [PyTorch torch.compile 文档](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)。脚本不会把编译失败伪装成 eager 成功。

**native 与 Gems 输入精度不同，因此这里的 native/Gems 比值只是初始性能参考，不能当作“同规格达到95%”验收。** native 从同一份实际量化数据反量化并舍入成 BF16，按它实际收到的 BF16 值另建参考；JSON 同时记录它和量化输入参考的数值差，终端也会打印。显式传 FP32 输出并不会消除输入舍入差异。

某条实现不正确或不可用时，打印 `INCORRECT` / `ERROR` 并返回非零退出码；可恢复错误后继续其余实现，已完成结果逐条写入 JSON。发生 GPU 上下文错误后，后续同步也可能报错，需要新进程重跑。只想跑部分实现，可用：

```bash
python tools/benchmark_paged_mqa.py --backends torch gems --torch-mode both
```

当前默认关闭 TLE。`--enable-tle` 只允许 FP8；现有算子的 FP4 TLE 路由尚不安全。

本地已验证CPU编解码、分页/负权重、全部5种页大小的45个精确输入fixture，以及PyTorch eager/Inductor和native adapter参数合同。用户已在BI-V150上跑通B4/L512的FP8 native和Gems，Gems约288.80微秒；新增加的矩阵、FP4和三方精确输入检查尚需在目标设备验证。JSON中的显存峰值是PyTorch allocator统计，不包含所有厂商库内部显存。
