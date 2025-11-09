# HuChenFeng — Qwen1.5 风格微调 & RAG 问答示例 

> 仓库说明：基于 Qwen1.5-4B-Chat 的 LoRA 微调（SFT） + FAISS 向量索引（RAG）演示，用于把“户晨风”风格的直播对话素材做成可检索、可回答的风格化聊天机器人。

---

## ✨ 项目亮点

* 🔥 **LoRA 微调（低成本，高效率）**：只训练适配器参数（无需微调整个模型），节省显存、时间与存储。
* 🧠 **RAG（检索增强生成）**：把 Markdown 语料构建为 FAISS 向量库，结合检索结果生成更准确、更有上下文的回答。
* ⚡ **推理缓存与量化支持**：示例演示 4-bit/半精度加载以节省内存，且全局 `MODEL_CACHE` 避免重复加载，提高交互速度。
* 🛡️ **工程友好**：数据持久化（缓存处理后的 Dataset 与 FAISS 索引）、断点检测（已有 LoRA 权重/索引则跳过相应步骤）。

---

## 🧩 目录结构（示例）

```
./
├─ hu_chen_feng_train_and_rag.py   # 主脚本（您提供的核心代码）
├─ README.md                       # （本文件）
├─ processed_sft_dataset_ratio_*    # 缓存的处理后数据集（由脚本生成）
├─ hu_chen_feng_style_adapter_*     # LoRA 适配器权重（训练后保存）
├─ hu_chen_feng_faiss_index/        # FAISS 索引目录
└─ data/                            # 原始 Markdown 语料（请替换为实际路径）
```

当然可以 👍
下面是我帮你重新排好格式的版本，段落清晰、Markdown 结构更规范，也方便读者在 GitHub 上阅读 👇

---

## 🧩 环境安装说明

本项目的依赖环境经过多次调试与冲突排查，**尤其是 LangChain 与 Hugging Face 版本兼容问题较多**。
请直接使用我们提供的 `requirements.txt` 来安装环境，确保可复现性与稳定性。

```bash
pip install -r requirements.txt
```

## 🔐 关于 Hugging Face Token（重要）

**请务必不要将任何明文 Token（硬编码）提交到 GitHub。**

脚本中有 `hf_login(secret_name)` 函数，会尝试以下方式登录 Hugging Face：

1. **环境变量**：脚本会读取
   `FUCHENFENG_TOKEN`（由 `secret_name = "Fuchenfeng"` → `FUCHENFENG_TOKEN`）
2. **交互式登录**：如果未找到环境变量，会调用 `huggingface_hub.login()` 进行交互输入。

---

### 🧱 使用示例（Linux / macOS）

```bash
export FUCHENFENG_TOKEN="YOURTOKEN"
python hcf_train.py
```

---

### ⚙️ 在 CI / GitHub Actions 中使用 Secrets

请在项目设置中添加 Secret，命名为：

```
FUCHENFENG_TOKEN
```

并通过环境变量自动读取。


---

## 🚀 快速开始（一键流程）

1. 准备好原始 Markdown 语料，放到 `BASE_DATA_DIR` 指定目录（默认为 `/root/IRLbiddingPT/HuChenFeng`，请根据实际修改）。
2. 设置环境变量 `FUCHENFENG_TOKEN`（Hugging Face 访问 token）。
3. 运行脚本：

```bash
python hcf_train.py
```

脚本默认会按顺序执行：

1. `train_sft_model()` — 若检测到 LoRA 权重则跳过训练。
2. `build_rag_index()` — 构建并持久化 FAISS 索引（若已存在则跳过）。
3. `run_inference()` — 加载 LoRA + 基座模型并启动交互式问答。

---

## 🛠️ 参数与配置说明（脚本关键常量）

* `BASE_MODEL_ID`：基础对话模型（默认为 `Qwen/Qwen1.5-4B-Chat`）。
* `TRAIN_SCALE_RATIO`：训练数据采样比例（示例 `1/10`），用于快速原型和节省训练资源。
* `RATIO_STR`：将 `TRAIN_SCALE_RATIO` 序列化到路径，便于保存不同配置的适配器。
* `LORA_ADAPTER_PATH`：LoRA 权重保存目录。
* `FAISS_INDEX_PATH`：FAISS 索引持久化目录。
* `PROCESSED_DATASET_PATH`：处理后 dataset 的持久化路径。
* `EMBEDDING_MODEL_NAME`：向量化模型（示例 `BAAI/bge-small-zh-v1.5`）。

您可以直接在脚本顶部修改这些常量来控制训练 / 索引 / 推理行为。

---

## 📚 数据处理说明

* 脚本会遍历 `BASE_DATA_DIR` 下的 `**/*.md` 文件（排除 README/SUMMARY 等），用自定义正则解析“某网友：... / 户晨风：...”样式的问答对，生成训练样本。
* 当没有可解析的样例时，脚本会使用示例占位句对以保证流程可运行。
* 处理后数据会调用 `.save_to_disk(PROCESSED_DATASET_PATH)`，后续运行会优先加载该缓存以加速迭代。

---

## 🧪 训练流程要点（LoRA + SFT）

* 使用 `peft.LoraConfig` 配置 LoRA 相关超参（`r=16`, `lora_alpha=32` 等为示例值，可按需调节）。
* 使用 `trl.SFTTrainer` 来做有监督微调（SFT），训练过程只会保存 LoRA 参数到 `LORA_ADAPTER_PATH`。
* 训练参数在 `TrainingArguments` 中设置：如果显存有限，可调低 `per_device_train_batch_size` 并适当增加 `gradient_accumulation_steps`。

**LoRA 优势小结**：

* 只保存少量权重（MB 级别而非 GB），便于迭代与版本控制。
* 在低资源场景下也能显著捕捉风格与偏好（如“户晨风”的说话风格）。

---

## 🧭 构建 FAISS 索引（RAG）说明

* 使用 `langchain_community` 的 `HuggingFaceEmbeddings` 将分块文本向量化。
* 使用 `RecursiveCharacterTextSplitter` 将文档切分为 `chunk_size=500` 的块，重叠 `chunk_overlap=50`。
* 代码里有默认采样（`chunks = chunks[:min(int(len(chunks)*1/50), len(chunks))]`）用于原型/减小索引大小，正式使用请酌情调整或移除采样逻辑。
* 最终索引通过 `FAISS.from_documents()` 构建并 `save_local()` 到 `FAISS_INDEX_PATH`。

---

## 🗣️ 推理流程与交互

* 推理阶段会：

  1. 从 `LORA_ADAPTER_PATH` 加载 LoRA 适配器并挂载到基座模型（支持 4-bit 量化加载基座模型）。
  2. 加载 FAISS 索引并创建 `retriever`，检索 top-k 文本作为上下文。
  3. 使用 `langchain` 的 `create_retrieval_chain`/`create_stuff_documents_chain` 把检索结果与用户输入拼接到 Prompt，然后调用 HF pipeline 生成回答。
* 启动后脚本进入交互式循环，输入 `停止/exit/退出` 或者Ctrl C 退出。

---

## 🧾 常见问题 & 排查建议

* ❌ **找不到 Token / 下载失败**：确认环境变量 `FUCHENFENG_TOKEN` 已设置，或在交互式中输入 Token。CI 中请使用 Secret 管理。
* ❌ **显存不足**：尝试启用 4-bit 加载（脚本示例已包含），降低 `per_device_train_batch_size`，增大 `gradient_accumulation_steps`，或使用更小的基础模型。
* ❌ **没有检索到有用的上下文**：检查分块器参数（chunk_size / overlap）与索引采样设置，确保语料被正确加载并包含足够上下文。
* ❌ **LoRA 权重或索引不存在**：检查 `LORA_ADAPTER_PATH` 与 `FAISS_INDEX_PATH` 路径是否正确，并查看脚本控制台输出以定位失败步骤。

---

## ✅ 实践建议
* 在开发中尽量先使用小比例数据（`TRAIN_SCALE_RATIO`）进行快速迭代，再放大数据量做最终训练。


## 🧾 致谢

特别感谢提供户晨风直播素材的仓库与作者提供的数据集/素材支持：

> [https://github.com/Olcmyk/HuChenFeng](https://github.com/Olcmyk/HuChenFeng)

---

## 📜 许可证 & 声明

* 请确保您有权使用并发布原始语料（直播片段、对话文本等）。
* 本项目示例代码仅用于研究与学习，请遵守相应模型与数据许可条款。

---

## 🙋 联系与贡献

欢迎提交 Issues / PR：

* 如果你发现脚本中的安全问题（例如硬编码 Token），请优先提出。
* 如果你做了改进（数据解析、训练超参、索引方式），欢迎 PR 并在 README 中增加你的贡献说明。

