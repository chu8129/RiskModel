# ELECTRA 风控 6 类文本分类（Colab T4）— 交接 / 重启文档

> 本文档记录 **当前有效代码、已踩的坑与修正、以及明天上班如何一键重启**。
> 与仓库既有的 Qwen3.5 LoRA SFT 方案并列，是**第二套方案**，**复用同一份数据与切分**。

---

## 0. 结论 / 最终指标（**训练已完成**）

`hfl/chinese-electra-small-discriminator`（12.35M 参数）在 Colab 免费 **T4** 上微调完成，
6 类中文风控短文本分类。**最终 test 集（15,002 条）结果：**

| 指标 | 值 |
|---|---|
| **accuracy** | **93.22%** |
| **macro-F1** | **92.89%** |
| weighted-F1 | 93.21% |
| test loss | 0.2418 |
| n | 15,002 |

### 每类 precision / recall / F1（test，用户要求重点看）

| 类别 | **precision** | **recall** | F1 | support |
|---|---:|---:|---:|---:|
| 辱骂 | 94.70% | 95.58% | 95.14% | 2,805 |
| 色情 | 93.12% | 92.10% | 92.61% | 2,469 |
| 涉政 | 93.47% | **90.78%**（最低） | 92.10% | 2,114 |
| 违禁 | **98.19%**（最高） | **89.18%**（最低） | 93.47% | 2,367 |
| 广告 | 93.33% | **99.07%**（最高） | 96.12% | 3,135 |
| 正常 | **86.17%**（最低） | 89.68% | 87.89% | 2,112 |

> 误判规律（见下方混淆矩阵）：**正常 → 涉政**（159 条）、**违禁 → 广告**（199 条）、
> **正常 → 色情**（67 条）是三个最大的错误源。
> 违禁 precision 极高（98.19%）但 recall 最低（89.18%）：宁可漏判也不误判，符合风控「少误伤」的取向。

### 混淆矩阵（行=真实，列=预测）

| 真实\预测 | 辱骂 | 色情 | 涉政 | 违禁 | 广告 | 正常 |
|---|---:|---:|---:|---:|---:|---:|
| **辱骂** | 2,681 | 67 | 6 | 5 | 0 | 46 |
| **色情** | 99 | 2,274 | 29 | 7 | 4 | 56 |
| **涉政** | 9 | 20 | 1,919 | 2 | 5 | 159 |
| **违禁** | 10 | 10 | 5 | 2,111 | 199 | 32 |
| **广告** | 1 | 4 | 7 | 6 | 3,106 | 11 |
| **正常** | 31 | 67 | 87 | 19 | 14 | 1,894 |

### dev（验证集）训练过程

| 进度 | dev accuracy | dev macro-F1 | dev weighted-F1 | dev loss |
|---|---|---|---|---|
| epoch 0.533 | 0.9001 | 0.8942 | 0.8997 | 0.3732 |
| epoch 1.066 | 0.9233 | 0.9197 | 0.9234 | 0.2735 |
| epoch 1.599 | 0.9317 | 0.9286 | 0.9317 | 0.2493 |
| epoch 1.866 | 0.9328 | 0.9297 | 0.9328 | 0.2418 |
| **test（终）** | — | — | — | **acc 93.22% / macro-F1 92.89%** |

实测耗时：**train_runtime 501 秒**（≈8.4 分钟，2 epoch，479 samples/s）；eval 约 1500 samples/s。
最佳 checkpoint：`artifacts/electra/checkpoints/best_model`（= `checkpoint-3752`，末步）。

**对照基线**：仓库 `docs/方案对比.md` 里 char n-gram TF-IDF + 逻辑回归 **test accuracy 91.91% / macro-F1 91.48%**。
**ELECTRA-small 提升 +1.31pt accuracy / +1.41pt macro-F1**，且仅 12.35M 参数、8 分钟训完。

---

## 1. 交付文件清单

| 文件 | 作用 |
|---|---|
| `src/electra_finetune.py` | **主脚本**：读切分 → 分词 → 微调 → 评估 → 保存。含 transformers 4.x/5.x 兼容层 |
| `scripts/make_electra_bundle.py` | 打 Colab 上传包 `electra_bundle.zip`（代码+数据，**不含模型**） |
| `scripts/build_colab_electra.py` | 用 nbformat 生成 `colab_electra.ipynb`（可复现的 notebook 源码） |
| `colab_electra.ipynb` | Colab notebook（14 cells，`nbformat.validate` 通过） |
| `electra_bundle.zip` | 上传包，**sha256 = `84538fcdb772314e59340dff0c19d5b66b5eb8b248f371ba6c65d63e6e056582`** |

数据契约（**复用现有，不新建切分**）：

- `data/test.txt` 150,019 条 `文本<TAB>标签`
- `data/label.txt` → `0辱骂 1色情 2涉政 3违禁 4广告 5正常`
- `data/splits/{train,dev,test}.tsv` → 120015 / 15002 / 15002，seed 42，8:1:1 分层
- manifest `split_manifest_sha256 = 6fcc57e9f363eb741a87538b46c8171110eb86b891ba8479948fe5b235d13ce7`

共享代码：`src/common.py`（`read_split` / `load_label_map` / `set_seed`）、`src/metrics.py`（`compute_metrics` / `save_metrics`）。

---

## 2. 关键参数（当前有效）

```python
MODEL_NAME  = "hfl/chinese-electra-small-discriminator"  # 12.35M 参数, 256 hidden, 12 层
MAX_LENGTH  = 128
EPOCHS      = 2
BATCH_SIZE  = 64
LR          = 3e-5
WEIGHT_DECAY= 0.01
WARMUP      = 0.06        # 脚本内部换算成 warmup_steps
EVAL_STEPS  = 500
SEED        = 42
```

实测资源：显存约 **1.7 GB / 15.6 GB**，GPU 利用率 65–75%，eval 约 **1500 samples/s**（15002 条 dev ≈ 10 秒）。

**为什么不是 `google/electra-small-discriminator`**：数据是**中文**，google 英文版词表不含中文字符，会被切成 UNK。
中文对等模型是 `hfl/chinese-electra-small-discriminator`（同为 small、更小、原生中文，已验证分词正确：
`不好意思，我是dx主播` → `['[CLS]','不','好','意','思','，','我','是','dx','主','播','[SEP]']`）。
在 notebook / 脚本里改 `MODEL_NAME` 即可换其它中文 Electra（如 `hfl/chinese-electra-base-discriminator`，约 102M，更准更慢）。

---

## 3. Colab 有效代码（可直接复制粘贴）

> 前提：`~/.pi/agent/mcp.json` 里 colab MCP 的 `toolPrefix: "none"`；`open_colab_browser_connection` 返回 `true`
> **只代表连上了浏览器，不代表有 GPU**。必须靠下面的 `cell 1` 证明。

### cell 1 — GPU 探针（**必做闸门**）

```python
import os, subprocess, torch
print("NOTEBOOK_ID:", os.environ.get("COLAB_NOTEBOOK_ID"))
print("IMAGE_TYPE :", os.environ.get("COLAB_IMAGE_TYPE"))
print("COLAB_GPU  :", repr(os.environ.get("COLAB_GPU")))
print("torch      :", torch.__version__)
print("cuda       :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU        :", p.name, "| VRAM GB:", round(p.total_memory/1e9,1))
    print("capability :", torch.cuda.get_device_capability(0))
print("uptime     :", subprocess.run("uptime", shell=True, text=True, capture_output=True).stdout.strip())
```

**判据**：`IMAGE_TYPE: gpu` + `COLAB_GPU: '1'` + `cuda: True`。
若为 `cpu`：在**同一个标签页**里 Runtime → Change runtime type → **T4 GPU** → Save，
然后 **Runtime → Disconnect and delete runtime**（强制用带 GPU 的镜像重建运行时，这一步不能省）。
`uptime` 不变 = 没重启 = 设置没生效。

实测本机结果：`IMAGE_TYPE: gpu`，`COLAB_GPU: '1'`，`Tesla T4`，`15.6 GB`，`capability (7,5)`。

### cell 2 — 安装依赖 + 解压上传包

```python
import subprocess, sys, zipfile, glob, os
# Colab 镜像自带 torchao，会与新版 transformers 冲突，先卸载
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "transformers", "datasets", "accelerate", "scikit-learn",
                "matplotlib", "seaborn"], check=True)

# 解压从本地上传的 electra_bundle.zip（Colab 左侧「文件」面板拖入）
z = glob.glob("*.zip")[0]
with zipfile.ZipFile(z) as zf:
    zf.extractall(".")
print("extracted", z)
print("train lines:", sum(1 for _ in open("data/splits/train.tsv", encoding="utf-8")))
```

### cell 3 — **清除模块缓存**（关键，见 §4 坑 5）

```python
import sys, importlib
for m in ["electra_finetune", "common", "metrics"]:
    sys.modules.pop(m, None)
sys.path.insert(0, "src")
import electra_finetune as ef
print("classes:", ef.NUM_LABELS, ef.LABEL_MAP)
print("has compat shims:", hasattr(ef, "make_training_args"), hasattr(ef, "make_trainer"))
```

### cell 4 — 启动训练（**后台执行**，别塞在一个 cell 里）

```python
import subprocess
cmd = ("cd /content && "
       "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
       "nohup python src/electra_finetune.py "
       "--device cuda --model-name hfl/chinese-electra-small-discriminator "
       "--epochs 2 --batch-size 64 --max-length 128 --eval-steps 500 --seed 42 "
       "--save-dir artifacts/electra --report reports/electra_result.json "
       "> /content/electra_train.log 2>&1 & echo $!")
print("pid:", subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout.strip())
```

### cell 5 — 轮询（短 cell，输出要小）

```python
import os, subprocess
txt = open("/content/electra_train.log", encoding="utf-8", errors="replace").read()
lines = [l for l in txt.splitlines()
         if not any(k in l for k in ["it/s","s/it","Loading weights","LOAD REPORT",
                                     "UNEXPECTED","MISSING","Notes:","---"])]
print("\n".join(lines[-30:]))
print("--- GPU ---")
print(os.popen("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").read().strip())
print("running:", subprocess.run("pgrep -f electra_finetune.py", shell=True, text=True,
                                 capture_output=True).stdout.strip() or "NONE")
```

### 训练结束后（脚本自动完成，无需额外 cell）

- 最佳模型 → `artifacts/electra/checkpoints/best_model/`
- test 报告 → `reports/electra_result.json` + `reports/electra_result.md`
- loss 曲线原始数据 → `artifacts/electra/training_curve.jsonl`

---

## 4. 已踩的坑与修正（**明天重启前务必先读**）

### 坑 1 — `build_trainer() got an unexpected keyword argument 'test_ds'`

`main()` 调 `build_trainer` 时多传了 `test_ds`（该函数只在 `evaluate_and_report` 里用 test）。
**修正**：`src/electra_finetune.py` 的 `main()` 里删掉 `test_ds=test_ds`（已修）。

### 坑 2 — `TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`

**transformers 5.x 移除了 `warmup_ratio`**（本地确认 5.17 只有 `warmup_steps`）。
**修正**：`build_trainer` 里按 `epochs × len(train) / (batch × grad_accum)` 先算出 `total_steps`，
再 `warmup_steps = max(1, int(warmup_ratio * total_steps))` 传给 `TrainingArguments`（已修）。

### 坑 3 — `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`

**transformers 5.x 把 `Trainer(tokenizer=...)` 改名为 `processing_class=...`**。
**修正**：统一走 `make_trainer()`，内部按签名自动在 `processing_class` / `tokenizer` 之间映射（已修）。

> 另外 transformers 5.x 把 `evaluation_strategy` 改名为 `eval_strategy`；`make_training_args()` 已一并适配。
> 脚本现在通过 `make_training_args(**kw)` / `make_trainer(**kw)` 两个入口构造，**4.x / 5.x 通用**：
> 按 `inspect.signature` 过滤掉当前版本不支持的 key，避免再炸。

### 坑 4 — 自检 cell 自己写错（不是脚本问题）

手写自检 cell 时又给 `build_trainer` 传了 `test_ds`，报同样的错。
**教训**：`build_trainer` 签名是
`(model, tokenizer, train_ds, dev_ds, max_length, seed, epochs, batch_size, learning_rate, weight_decay, warmup_ratio, eval_steps, save_dir, report_path, device, gradient_accumulation_steps=1)`
—— **没有 `test_ds`**。

### 坑 5 — ⚠️ **Python 模块缓存陷阱（最容易误判成"文件没改"）**

现象：本地已修好、zip 已重传解压，但 Colab 仍报**旧版本**的错
（例如签名里 `warmup_ratio` 还在、`make_training_args` 明明在文件里却没用上）。

原因：前面的 cell 已经 `import electra_finetune`，模块进了 `sys.modules`；
**重新解压覆盖 `.py` 文件不会让已导入的模块重载**，内存里仍是旧对象。
（曾据此误判为"兼容层没生效"，实际文件是对的。）

**修正**（每次覆盖 `.py` 后必做）：

```python
import sys
for m in ["electra_finetune", "common", "metrics"]:
    sys.modules.pop(m, None)
import importlib; importlib.invalidate_caches()
sys.path.insert(0, "src")
import electra_finetune as ef
```

> 后台 `nohup python src/electra_finetune.py` 是**新起进程**，读磁盘新文件，**不受此坑影响**。

### 坑 6 — 数据不要用 cell 传

6 MB 数据转 base64 会膨胀到 ~8 MB 并吃掉上下文。**用 zip 拖进文件面板**（本次 3.28 MB，一次拖拽）。

### 坑 7 — `torchao` 冲突

Colab 镜像自带 torchao，与新版 transformers/peft 冲突，报 `ImportError: incompatible version of torchao`。
**修正**：用前 `pip uninstall -y torchao`（**每次运行时被回收后都要重做**）。

### 坑 8 — 长任务别塞一个 cell

单 cell 有 15 分钟上限，训练会超。**必须** `nohup ... &` + 日志文件 + 轮询 cell。

---

## 5. 明天上班如何重启（5 步）

1. **本地**确认产物在：
   - `src/electra_finetune.py`、`electra_bundle.zip`（sha256 见 §1）
   - 若改过代码，先 `python scripts/make_electra_bundle.py` 重打包
2. 连 Colab：`open_colab_browser_connection` → **先告诉用户把 T4 设好并"断开并删除运行时"重建**（见 §3 cell 1 判据）
3. **把 `electra_bundle.zip` 拖进文件面板** → 跑 cell 2
4. 跑 cell 3（**清模块缓存**）→ cell 4（后台启动）→ cell 5（轮询）
5. 训练完，从 Colab 取回：
   - `artifacts/electra/checkpoints/best_model/`（模型+tokenizer）
   - `reports/electra_result.json` + `.md`
   - `artifacts/electra/training_curve.jsonl`
   然后**独立复算 test 指标再引用**（见 §6）

---

## 6. 待办 / 下一步

- [x] 完整 2 epoch 的 **test 指标**（`reports/electra_result.json` + `.md`）—— **acc 93.22% / macro-F1 92.89%**
- [x] 每类 precision / recall / F1 + 混淆矩阵（见 §0）
- [ ] 画 loss 曲线 + 混淆矩阵热图（notebook cell 10/11 已就绪；或读 `training_curve.jsonl` + 报告 json）
- [ ] 取回 best_model，**独立复算 test 指标**（不要直接引用训练日志里的数）
- [ ] 与 Qwen3.5 LoRA SFT（`src/lora_sft.py`）及 TF-IDF 基线（test acc 91.91%）做同表对比
- [ ] 把结论补进 `docs/方案对比.md`

---

## 7. 兼容性备忘（transformers 版本差异速查）

| 旧写法（4.x） | 新写法（5.x） | 说明 |
|---|---|---|
| `Trainer(tokenizer=...)` | `Trainer(processing_class=...)` | 5.x 改名 |
| `TrainingArguments(warmup_ratio=...)` | `TrainingArguments(warmup_steps=...)` | 5.x 移除 ratio，需自算 |
| `TrainingArguments(evaluation_strategy=...)` | `TrainingArguments(eval_strategy=...)` | 改名 |

当前 Colab 环境：**Python 3.x + torch 2.11.0+cu128 + transformers 5.17.0**，GPU = Tesla T4 (sm_75)。
脚本用 `make_training_args` / `make_trainer` 兼容层自动适配，**不建议再直接在源码里写裸 `TrainingArguments(...)` / `Trainer(...)`**。
