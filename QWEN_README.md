# Qwen3.5-0.8B 中文风险文本分类（LoRA SFT + 分类头）

在 Colab 免费 T4 上，对本地 `Qwen3.5-0.8B`（752.4 M 参数）做 **LoRA SFT + 分类头**，
完成 6 类中文风险短文本识别（0 辱骂 / 1 色情 / 2 涉政 / 3 违禁 / 4 广告 / 5 正常）。

- **数据**：`data/test.txt` 150,019 条已标注（`文本<TAB>标签`，0 条脏数据，6 类基本均衡）+ `data/label.txt`
- **切分**：固定种子 42 分层 8:1:1 → train 120,015 / dev 15,002 / test 15,002
- **训练环境**：Colab 免费层 Tesla T4（15.6 GB，可用 14.56 GiB），Python 3.13，transformers 5.16.1，peft 0.20.0
- **方案对比与实测数据**：见 [`docs/方案对比.md`](docs/方案对比.md)；本机 CPU 能力边界见 [`reports/bench.md`](reports/bench.md)

---

## 1. 目录结构

```
src/
  lora_sft.py        训练主脚本：LoRA 注入、长度档位分批、AMP、断点续训、评测、导出
  common.py          共用：线程/切分读取/分桶/pooling/路径常量
  metrics.py         accuracy / macro-F1 / 每类 P-R-F1 / 混淆矩阵
  infer.py           单条推理（直接复用训练用的 LoraClassifier，加载路径不会漂移）
scripts/
  prepare_data.py    可复现切分 + 数据报告
  make_colab_bundle.py  打上传包（代码+数据，不含模型）
data/
  test.txt label.txt        原始数据
  splits/{train,dev,test}.tsv + split_manifest.sha256 + manifest.json
reports/             所有实测证据（bench.md / data.md / eval_curve*.jsonl / *_result.json）
artifacts/           checkpoint、LoRA adapter、分类头、推理配置
docs/方案对比.md      方案对比与推荐
```

**路径约定**：`common.py` 用 `ROOT = Path(__file__).resolve().parents[1]`。所以把仓库放在
`/content/`（即代码在 `/content/src/`）时，`ROOT` 自动等于 `/content`，模型、数据、产物全部落在根目录。
**无需改任何代码**即可切换工作根目录。

---

## 2. 完整训练命令

```bash
cd /content && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python src/lora_sft.py \
  --device cuda --dtype auto --train-limit full --epochs 1 \
  --batch-size 16 --grad-accum 1 --max-len 64 \
  --lora-r 16 --lora-alpha 32 --pooling mean+last \
  --head ffn --ffn-hidden 512 --lr 1e-4 --head-lr 1e-3 --patience 2 \
  --eval-every 1 --eval-dev-limit 96 \
  --save-every 200 --log-every 500 \
  --save-dir /content/artifacts/lora_full \
  --report /content/reports/lora_full_result.json \
  --curve-out /content/reports/eval_curve_full.jsonl \
  > /content/train_full.log 2>&1 &
```

最小冒烟（任何设备，约 1 分钟）：

```bash
python src/lora_sft.py --smoke
```

只测吞吐不训练：

```bash
python src/lora_sft.py --bench --device cuda
```

单条推理：

```bash
python src/infer.py --export-dir artifacts/lora_full --text "办证刻章 快速出证 加微信"
```

---

## 3. 参数逐项说明

### 3.1 进程与内存环境（shell 层）

| 片段 | 作用 | 为什么 |
|---|---|---|
| `cd /content &&` | 工作目录 = 根目录 | 代码在 `/content/src/` → `ROOT` 自动解析为 `/content` |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | CUDA 分配器用可扩展段 | **【实测】** 曾出现显存碎片 OOM（报错含 "reserved but unallocated"），此开关显著降低碎片 OOM |
| `nohup` | 忽略挂断信号 | 让训练**不依赖内核会话**存活；会话断了进程仍可能继续 |
| `> log 2>&1` | stdout+stderr 都入日志 | `2>&1` 必须在后，否则 traceback 丢失 |
| `&` | 放后台 | 否则单次工具调用会阻塞到训练结束（而调用超时只有 15 分钟） |

### 3.2 设备与精度

| 参数 | 值 | 说明 |
|---|---|---|
| `--device` | `cuda` | `auto`/`cuda`/`cpu`。显式写 `cuda` 避免静默跑成 CPU |
| `--dtype` | `auto` | **【实测】** 按算力自动选：sm_80+ → bf16；sm_70/75（T4/V100）→ **fp16 + GradScaler**；sm_60（P100）→ fp32。T4 是 `(7,5)`，所以实际跑的是 **fp16** |

补充：权重以 **fp32** 加载（`from_pretrained(dtype=torch.float32)`），只有计算走 fp16 autocast，
属于**混合精度（fp32 主权重 + fp16 计算）**，不是整个模型 fp16。

### 3.3 数据与规模

| 参数 | 值 | 说明 |
|---|---|---|
| `--train-limit` | `full` | `full` = 全部 120,015 条；也可传整数（如 `20000`）做子集实验 |
| `--epochs` | `1` | 1 个 epoch = 完整过一遍 = 7,501 步。日志里 epoch **从 0 计**，完成时打印 `[epoch 0]` |
| `--batch-size` | `16` | **【实测】** bs16/seq64 峰值显存约 9.5 GiB / 可用 14.56 GiB |
| `--grad-accum` | `1` | 不累积，每步更新；有效批 = 16 × 1 |
| `--max-len` | `64` | **【实测】** token 长度 p99 = 84，`max_len=64` 覆盖 **97.95%** |
| `--eval-limit` | 0（默认，未传） | 0 = epoch 末用**全量** dev 评测 |
| `--test-limit` | 0（默认，未传） | 0 = 最后用**全量** test 评测 |

### 3.4 LoRA 结构（最关键的一组）

| 参数 | 值 | 说明 |
|---|---|---|
| `--lora-r` | `16` | 秩 16 → 可训练参数约 **4.92 M** |
| `--lora-alpha` | `32` | 缩放 α=32 → 实际缩放 = α/r = **2.0** |
| `--lora-dropout` | 0.05（默认，未传） | 适配器 dropout |
| `--targets` | 默认 **7 个名字**（未传） | `self_attn.{q,k,v,o}_proj` + `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` |

**最后一行是本脚本最重要的防坑设计。** Qwen3.5 的 24 层里 **18 层是 `linear_attention`**，
其投影名是 `linear_attn.in_proj_qkv / in_proj_z / out_proj`，**没有 `q_proj`**。
若照抄 Qwen2/LLaMA 的 `["q_proj","k_proj","v_proj","o_proj"]`，只会命中 6 个 `full_attention` 层
（0.54 M 可训练参数），**漏掉 75% 的网络且不报错** —— 属于静默失效。
脚本因此把 7 个目标模块显式写死，并在注入后断言可训练参数 > 1 M。

### 3.5 分类头

| 参数 | 值 | 说明 |
|---|---|---|
| `--pooling` | `mean+last` | 均值池化 + 最后一个非 pad token，各 1024 维 → 拼成 **2048 维**。【实测】数据 token 中位数仅 4，单用 mean 会稀释信号 |
| `--head` | `ffn` | 两层 MLP：`2048 → 512 → GELU → Dropout(0.2) → 6`，参数 1.05 M。也可选 `linear` |
| `--ffn-hidden` | `512` | 隐层宽度 |
| `--dropout` | 0.2（默认，未传） | 头与池化后的 dropout |

**不使用 `lm_head`**：分类头接在池化 hidden state 上，主干调用走内层 `get_base_model().model`，
完全不算 248,320 词表的 logits。【实测】那一步在 bs16/seq64 下每次需分配 **486 MB**，是早期 OOM 的直接原因；
去掉后吞吐 **+23%**（CPU 实测；`lm_head` = 254.3 M 参数 = 总参数 33.8%）。

### 3.6 优化器

| 参数 | 值 | 说明 |
|---|---|---|
| `--lr` | `1e-4` | LoRA 学习率（低秩增量，常用 1e-4 量级） |
| `--head-lr` | `1e-3` | 分类头学习率，比 LoRA **高 10 倍** —— 头是随机初始化，LoRA 从零增量开始 |
| `--weight-decay` | 0.01（默认） | AdamW 权重衰减 |
| `--warmup-frac` | 0.05（默认） | 前 5% 步数线性 warmup，之后余弦衰减 |
| `--patience` | `2` | epoch 末 dev macro-F1 连续 2 次不提升即早停。**1 个 epoch 时永不触发** |
| `--seed` | 42（默认） | 【实测】同种子**逐步可复现**：两次运行在 step 49 的 loss / dev_acc 完全相同 |

### 3.7 评测探针（每步看效果）

| 参数 | 值 | 说明 |
|---|---|---|
| `--eval-every` | `1` | **每训练 1 步**就跑一次 dev 评测。`0` = 关闭 |
| `--eval-dev-limit` | `96` | 频繁评测只用 96 条 dev。【实测】全量 dev 15,002 条一次要 **4.4 分钟**（≈313 个训练步），96 条只需约 2 秒 |
| `--log-every` | `500` | 每 500 步打一行汇总（每步已有 `[eval]` 行，汇总不必太密） |
| `--save-every` | `200` | 每 200 micro-batch 存一次 checkpoint = 每 3,200 条样本 ≈ **每 6–7 分钟** |

**只有 96 条 dev → 1 条样本 = 1.04 pt，单点噪声大**（实测相邻步 |Δacc| 均值 2.9 pt、最大 11.5 pt）。
所以每步曲线**只用于看趋势**；权威数字来自 **epoch 末的全量 dev** 与最后的 **全量 test** 评测。

### 3.8 输出路径

| 参数 | 路径 | 内容 |
|---|---|---|
| `--save-dir` | `/content/artifacts/lora_full` | `checkpoint.pt` + `adapter/` + `head.pt` + `label_map.json` + `inference_config.json` |
| `--report` | `/content/reports/lora_full_result.json` | 环境、配置、逐 epoch 历史、test 的 accuracy / macro-F1 / 每类指标 / 混淆矩阵 |
| `--curve-out` | `/content/reports/eval_curve_full.jsonl` | 每步一行：`{epoch, step, global_step, train_loss, dev_accuracy, dev_macro_f1, dev_n, elapsed_s}` |

checkpoint 的三个设计细节：

1. **只存 LoRA + 头**（不存 752 M 参数的完整 state_dict）—— 否则单文件约 3 GB，写盘会拖慢每一步
2. **原子写**：先写 `.tmp` 再 `os.replace`，写到一半被中断不会毁掉上一个好 checkpoint
3. **支持 `--resume`**：确定性地跳过已训 micro-batch 继续（同种子下数据顺序可重放）

---

## 4. 未在命令中显式传入的默认值

| 参数 | 默认 | 含义 |
|---|---|---|
| `--model-dir` | `<ROOT>/Qwen3.5-0.8B` | 基座路径 |
| `--eval-limit` / `--test-limit` | `0` | 0 = 用全量 dev / test |
| `--lora-dropout` | `0.05` | LoRA adapter dropout |
| `--targets` | 7 个显式模块名 | 见 3.4 |
| `--dropout` | `0.2` | 头内 dropout |
| `--weight-decay` | `0.01` | AdamW |
| `--warmup-frac` | `0.05` | warmup 比例 |
| `--seed` | `42` | 影响切分抽样、批次顺序、初始化 |
| `--grad-checkpoint` | 关 | 梯度检查点，省显存换时间；当前不需要 |
| `--num-threads` | `0` | 仅 CPU 用；0 = 物理核数 |

---

## 5. 输出产物

训练结束后 `artifacts/lora_full/` 下：

| 文件 | 内容 |
|---|---|
| `checkpoint.pt` | 续训用：LoRA+头权重、优化器、调度器、scaler、进度、历史 |
| `adapter/adapter_config.json` + `adapter_model.safetensors` | peft LoRA 适配器 |
| `head.pt` | 分类头 state_dict + `pooling`/`head`/`ffn_hidden`/`in_dim` |
| `label_map.json` | `{"0": "辱骂", ...}` |
| `inference_config.json` | `base_model_dir` / `max_len` / `pooling` / `head` / LoRA 参数 / 环境信息 |

`infer.py` 直接读取上述配置，并**复用训练时的 `LoraClassifier` 类**重建模型，
因此推理与评测的加载路径不可能漂移。

---

## 6. 实测约束（踩过的坑，均已固化进代码）

| # | 现象 | 原因 | 已固化 |
|---|---|---|---|
| 1 | `get_peft_model` 直接抛 `ImportError: incompatible version of torchao` | Colab 镜像自带 torchao 0.10.0，peft ≥0.20 的 `is_torchao_available()` **抛异常而非返回 False** | 用前 `pip uninstall -y torchao`；**每次运行时被回收后都需重做** |
| 2 | 训练无任何进展，`GPU util 0%`，CPU 满载 | flash-linear-attention 的 Triton 内核按 shape 编译：新长度首次 **119 s**，同长度后续 0.94 s。动态 padding 几乎每批都是新 shape → 无限重编译 | `LEN_LADDER = [16,32,48,64]` 固定 padding 档位 |
| 3 | bs16/seq64 时 CUDA OOM（要分配 486 MB） | 白算了 `lm_head` 的 248,320 词表 logits | 改调内层 `get_base_model().model` |
| 4 | 训练报 OOM，但显卡看起来是空的 | 内核里残留的实验张量占着显存；`%reset -f` 不释放（IPython 输出历史仍引用 traceback 帧） | 用 `os.kill(os.getpid(), SIGKILL)` 重启内核；pip 包与 HF 缓存都在磁盘上，重启很廉价 |
| 5 | loss 正常降到 0.085，但 dev_acc 只有 **17.97%**（低于随机 16.67%） | 长度分桶会重排样本，而 `evaluate()` 拿「重排后的预测」对比「原始顺序的标签」→ 全部错位 | `evaluate()` 从 batch 取标签（`golds`），与 `preds` 同序 |
| 6 | 每步显存/时间额外开销 | `use_cache=True` 时模型会创建 KV / recurrent state 缓存 | 训练前向传 `use_cache=False` |

---

## 7. 断点续训

```bash
# 中断后在原命令基础上加 --resume（其余参数保持一致）
python src/lora_sft.py ... --resume --save-dir /content/artifacts/lora_full
```

恢复时会打印：

```
[resume] checkpoint.pt: epoch=0 skip_micro=1500 global_step=1500 best_dev_f1=... 
[resume] epoch 0: deterministic skip of first 1500 micro-batches
```

因为 `make_batches` 用 `seed + epoch` 做确定性 shuffle，重放前 N 个 micro-batch 可精确复现被中断时的数据顺序。

**注意**：若换了 `--epochs` 或 `--train-limit`，恢复出来的 LR 调度会与 checkpoint 里的进度混合，
此时建议清空 `--save-dir` 从零重跑（删除 `checkpoint.pt` 即可）。

---

## 8. 环境与依赖

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 仅本地 CPU 需要
pip install "transformers>=5.0" peft scikit-learn psutil pyyaml joblib sentencepiece
pip uninstall -y torchao        # 必须：否则 peft 无法注入 LoRA（见 6.1）
```

| 项 | 要求 |
|---|---|
| `transformers` | **≥ 5.0**，硬要求。Qwen3.5 是混合 linear_attention + 视觉塔架构，4.x 无法加载 |
| `peft` | ≥ 0.20（实测 0.20.0 可用） |
| `torchao` | **必须卸载** |
| GPU | 任意；T4（sm_75）会自动走 fp16 + GradScaler。纯 CPU 也能跑，但慢约 10 倍（见 `reports/bench.md`） |
| 显存 | bs16/seq64 约 9.5 GiB；14.56 GiB 可用时余量充足 |

**免费层注意事项**：Colab 会因闲置/关闭标签页回收运行时，
被回收后 `/content` 会清空（代码、数据、checkpoint 全丢）。建议：

- 保持 Colab 标签页打开
- 把 `artifacts/` 定期复制到 Google Drive
- 用 `reports/eval_curve_full.jsonl` 实时确认进度；若长时间不增长即为断线
