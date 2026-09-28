**一个大任务，多卡共同计算：Paged MQA**

需要一次覆盖五档尺寸、普通版和分页版，并比较 Torch / 原始 Gems / TLE 候选，
使用新的[统一测试入口](benchmark_mqa_suite.md)。本文件对应原来的单 shape、
仅 paged、两方比较入口。

原来的 `run_paged_mqa_suite.py --jobs 16` 是16张卡独立跑不同case，不包含卡间协作。这个新脚本把同一批请求的KV长度维分给多个rank，共用Q/weights，再通过collective通信拼回完整logits。

生产kernel没有修改；测试的是现有单卡kernel加分片和通信编排后的性能。默认仍只比较 torch.compile 和 Gems，不运行native。

依赖同目录的四份文件：

- `benchmark_paged_mqa_distributed.py`：torchrun入口、通信、计时、检查。
- `paged_mqa_distributed_inputs.py`：按全局坐标生成相同逻辑输入。
- `benchmark_paged_mqa.py`：已有两种本地实现及参考计算。
- `run_paged_mqa_suite.py`：复用主机内存查询函数。

**默认大任务**

```text
global B = 256
global L = 1,048,576
H = 64, D = 128, next_n = 1, page = 256
quant = FP8（可改成fp4或both）
```

| 项目 | 数量 |
| --- | ---: |
| 总packed KV，含每token的scale | 33 GiB |
| 相比旧B128/L16384的KV容量 | 128倍 |
| 16 rank时，每rank上下文 | 65,536 |
| 16 rank时，每rank packed KV | 2.0625 GiB |
| 每rank最终完整FP32 logits | 1 GiB |
| 每rank all-gather缓冲 | 1 GiB |
| 当前保守GPU内存估计 | 约19.16 GiB/rank |
| 当前保守host内存估计，16 rank合计 | 约54 GiB |

估计包括Torch FP32解码等中间张量，不只是KV容量；厂商库workspace和碎片仍可能额外占内存。启动时查询实际free memory与主机/cgroup容量，超过预算写 `SKIPPED_RESOURCE` 并退出。不是要求一次占满集群512GB。

离线查看计划：

```bash
python3 tools/benchmark_paged_mqa_distributed.py --plan --plan-world-size 16
```

**先做16卡小通信检查，再上大任务**

在已经跑通Gems的容器中，确认 `torch.cuda.device_count()` 至少等于本机 `--nproc-per-node`。这里的rank对应Torch可见逻辑设备，不直接等于物理板卡。下面假设16个设备都在同一主机；libdevice路径替换为你的实际文件。

单机启动使用static rendezvous与显式loopback地址，避免容器主机名（如p-iluvatar-01）不能解析导致TCPStore一直重试。不要同时添加 `--standalone`，它会强制切回c10d/localhost:0；`--master-addr`无法覆盖该路径。若旧任务正在重试，先Ctrl-C停止，再用新的输出文件名重跑。OMP_NUM_THREADS=1的提示不是此次阻塞原因。[PyTorch启动逻辑](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/run.py)、[static rendezvous实现](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py)。

```bash
torchrun --nnodes=1 --node-rank=0 --nproc-per-node=16 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tools/benchmark_paged_mqa_distributed.py \
  --batch 4 --context 4096 --page-size 64 --quant both \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --warmup 2 --iterations 5 --repeats 3 \
  --output results/cooperative_smoke16.json
```

确认通信和两种实现正确后运行默认大任务：

```bash
torchrun --nnodes=1 --node-rank=0 --nproc-per-node=16 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tools/benchmark_paged_mqa_distributed.py \
  --batch 256 --context 1048576 --page-size 256 --quant both \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --warmup 3 --iterations 10 --repeats 5 \
  --output results/cooperative_large16.json
```

CoreX的PyTorch分布式接口默认用 `--dist-backend nccl`，由平台通信栈提供实现。天数官方示例也是先设置LOCAL_RANK对应设备，再初始化nccl后端；不能仅凭名字判断使用NVIDIA硬件。[天数官方初始化示例](https://github.com/Deep-Spark/DeepSparkHub/blob/master/models/cv/classification/resnet50/pytorch/common_utils/dist.py)、[Iluvatar配置说明](https://github.com/verl-project/verl-hardware-plugin/blob/main/docs/user_guide_iluvatar/README.md)。

脚本会先实测uint8/int32广播与FP32 all-gather。若失败，应先处理通信栈/设备可见性；CPU Gloo通过不能代替IXCCL真机验证。

**每一步具体测什么**

端到端计时包含：

```text
rank0广播Q / FP4 scales / weights
→ 各rank调用本地torch.compile或Gems，处理自己的KV片段
→ all-gather各rank的FP32 logits
→ 恢复成连续的[B,global L]结果，每个rank都拿到完整输出
```

KV已预先分片驻留；初始数据生成、H2D传输、进程组初始化、编译和reference不计入稳态延迟。FP8 Q按uint8 view传输原始位，不要求通信库原生支持FP8 datatype。

输出三组独立测量：

- `compute_only`：每卡本地算子完整API，不含通信；取所有rank耗时的最大值。
- `communication_and_assembly_only`：用已计算的局部logits，测Q广播、all-gather与布局恢复。
- `end_to_end`：完整广播→计算→汇总，用于评估多卡协同性能。

每个repeat共同同步起跑，结束后同步设备，再在计时窗口外汇总各rank耗时，取最慢rank；不能把各rank平均耗时当整个任务耗时。前两项来自不同实验，不能简单相加预测第三项。

16rank默认大任务中，每rank每次all-gather逻辑接收约960MiB，所有rank合计约15GiB。另有每rank1GiB输出布局写回。因此可能出现“本地计算变快，但通信抵消收益”；这正是该测试要量出来的。逻辑payload不是实测链路流量或物理带宽。

**如何测1/2/4/8/16卡加速比**

强扩展要保持全局B/L、数据、dtype、代码和软件版本相同。默认33GiB KV大任务不适合拿单卡硬跑；要评估相对单卡的加速比，先选择单卡也能容纳的规模，例如B32/L131072：

```bash
torchrun --nnodes=1 --node-rank=0 --nproc-per-node=1 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tools/benchmark_paged_mqa_distributed.py \
  --batch 32 --context 131072 --quant both \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --output results/scaling_1.json

torchrun --nnodes=1 --node-rank=0 --nproc-per-node=16 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tools/benchmark_paged_mqa_distributed.py \
  --batch 32 --context 131072 --quant both \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --single-rank-baseline results/scaling_1.json \
  --output results/scaling_16.json
```

将16改成2/4/8、输出文件名对应修改，即可补齐曲线。

```text
strong_scaling_speedup = 单rank端到端耗时 / 多rank端到端耗时
strong_scaling_efficiency = strong_scaling_speedup / rank数量
```

没有匹配的单rankPASS结果时，不会凭空生成这个加速比。还会检查源码hash、参数、设备型号和包版本；不要拿单卡脚本的时间直接作为此分布式脚本的单rank分母。

**多节点**

如果是两台机器各8个可见设备，在每台机器启动一个torchrun，使用相同master地址和端口；节点0用 `--node-rank=0`，节点1用 `--node-rank=1`：

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --rdzv-backend=static \
  --master-addr=<节点0可达IP> --master-port=29501 \
  tools/benchmark_paged_mqa_distributed.py \
  --batch 256 --context 1048576 --quant both \
  --libdevice-path /usr/local/corex-4.5.0/nvvm/libdevice/libdevice.compute_bi.10.bc \
  --output results/cooperative_large16.json
```

两节点的计算与循环参数必须一致，脚本会交换参数和源码hash检查。网卡/RDMA选择沿用管理员已配置好的CoreX通信环境，不自动修改NCCL/IXCCL网络参数。主报告由global rank0写入，因此不要求结果目录跨节点共享。[torchrun说明](https://docs.pytorch.org/docs/2.10/elastic/run.html)

**正确性、限制与本地验证**

- 每rank所有有效local logits与独立FP32 reference对拍。
- 全局输出额外抽查分片边界、页边界、首尾和随机位置，reference直接按全局坐标重新生成，不依赖拼接函数。
- stateless KV生成器保证改变rank数量不改变逻辑K值；物理页顺序可以不同。它是合成uniform数据，不是模型trace。
- 支持最后片不满和空rank；空rank返回零输出但仍参加全部collective。
- 仅uniform global context、next_n1、D128；不包含top-k，不代表完整vLLM推理吞吐。
- 目前每rank局部FP8 K字节偏移超过保守int32范围时拒绝运行，避免仅因显存够就跨过原kernel的地址范围。
- 每rank都会保留完整全局logits，超长上下文时该复制策略可能成为瓶颈；这是本次明确选择的协同方式。
- 使用新 `--output` 文件，防止早期失败后误读旧PASS。失败rank写独立error JSON，由torchrun终止同组worker；没有完成的主报告不会被视为PASS。

本地已用两rank CPU Gloo验证广播、部分页拼接、全局样本和分阶段计时；helper也做了不同world-size逐byte输入一致性检查。尚未在BI-V150/IXCCL上实测多卡性能。
