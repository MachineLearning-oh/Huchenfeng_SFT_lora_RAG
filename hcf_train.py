

# ====================================================================
# 1. 统一导入 & 全局缓存
# ====================================================================
import os, glob, re, random, json
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login

# LLM/SFT 
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline, BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_from_disk # 用于数据集持久化
from trl import SFTTrainer

# NEW：0.3.x 核心导入和 RAG 链
from langchain_core.prompts import PromptTemplate 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader 
# 全局模型和组件缓存：用于 run_inference 避免重复加载
MODEL_CACHE = {
    "tokenizer": None,
    "embeddings": None,
    "inference_model": None,
}


# ====================================================================
# 2. 常量 (已修正：将比例纳入路径)
# ====================================================================
BASE_MODEL_ID = "Qwen/Qwen1.5-4B-Chat"

# 修正 1：定义训练比例，您可以在这里修改（例如 1/1000, 1/10）
TRAIN_SCALE_RATIO = 1/10

# 修正 2：生成路径安全的比例字符串 (例如 0_1)
RATIO_STR = str(round(TRAIN_SCALE_RATIO, 3)).replace('.', '_') 

# 修正 3 & 4：将比例字符串加入到 LoRA 和 Dataset 的缓存路径中
LORA_ADAPTER_PATH = f"./hu_chen_feng_style_adapter_qwen4b_ratio_{RATIO_STR}"
FAISS_INDEX_PATH = "./hu_chen_feng_faiss_index"
# 处理后的数据集缓存路径
PROCESSED_DATASET_PATH = f"./processed_sft_dataset_ratio_{RATIO_STR}" 

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DATA_DIR = "/root/IRLbiddingPT/HuChenFeng"
HF_TOKEN_SECRET_NAME = "Fuchenfeng"


# ====================================================================
# 3. HuggingFace 登录 
# ====================================================================
def hf_login(secret_name):
    # 替换 YOUR_HUGCING_FACE_TOKEN
    MY_TOKEN = "YOUR_HUGCING_FACE_TOKEN" 
    
    if MY_TOKEN != "YOUR_HUGCING_FACE_TOKEN":
        try:
            login(token=MY_TOKEN, skip_if_logged_in=True)
            print("Hugging Face 登录成功")
            return
        except:
            print("硬编码 Token 登录失败。")
            
    token = os.getenv(f"{secret_name.upper()}_TOKEN")
    if token:
        try:
            login(token=token, skip_if_logged_in=True)
            print("Hugging Face 登录成功 (通过环境变量)")
        except Exception as e:
            print("登录失败，可能下载受限:", e)
    else:
        print("警告：未找到 Token,可能触发交互式输入或下载受限。")
        login() 


# ====================================================================
# 4. SFT 训练
# ====================================================================
def train_sft_model(data_dir=BASE_DATA_DIR):
    
    # 检查点 1: LoRA 权重持久化
    if os.path.exists(LORA_ADAPTER_PATH) and glob.glob(os.path.join(LORA_ADAPTER_PATH, 'adapter_model.safetensors')):
        print(f"LoRA 权重已存在于磁盘 ({LORA_ADAPTER_PATH})，跳过 SFT 训练步骤。")
        return

    hf_login(HF_TOKEN_SECRET_NAME)
    
    # 检查点 2: 数据集持久化
    if os.path.exists(PROCESSED_DATASET_PATH):
        print(f"从缓存加载处理后的数据集: {PROCESSED_DATASET_PATH}")
        # 加载的 dataset 已经是 map 后的格式
        dataset = load_from_disk(PROCESSED_DATASET_PATH)
        print(f"采样后的训练数据量: {len(dataset)}")
    else:
        # --- 数据解析和处理逻辑（仅在缓存不存在时运行） ---
        print("缓存数据集不存在，开始解析原始 Markdown 文件...")
        
        # 定义格式化函数（在数据处理前定义，确保作用域正确）
        def fmt(e):
            text = f"<|im_start|>user\n{e['instruction']}<|im_end|>\n<|im_start|>assistant\n{e['response']}<|im_end|>"
            return {"text": text}
        
        def parse_dialogue_to_qa(text):
            text = text.replace('\n', ' ').strip()
            # 简化并修正正则表达式，匹配 "某网友：" 和 "户晨风：" 之间的内容
            pattern = re.compile(r'(某网友：)(.*?)(?=户晨风：|$)', re.DOTALL)
            qa_pairs = []
            for m in pattern.finditer(text):
                q = m.group(2).strip()
                r_match = re.search(r'户晨风：(.*?)(?=\s*(某网友：|户晨风：)|$)', text[m.end():], re.DOTALL)
                if r_match:
                    a = r_match.group(1).strip()
                    if q and a:
                        qa_pairs.append({"instruction": q, "response": a})
            return qa_pairs

        ALL_MD_FILES = glob.glob(os.path.join(data_dir, '**', '*.md'), recursive=True)
        EXCLUDE = ["readme", "summary", "preface", "acknowledgements", "videos"]
        VALID = [f for f in ALL_MD_FILES if not any(os.path.basename(f).lower().startswith(p) for p in EXCLUDE)]

        style_data = []
        if VALID:
            all_text = "\n\n".join(open(f, encoding='utf-8').read() for f in VALID)
            style_data = parse_dialogue_to_qa(all_text)
        if not style_data: 
            style_data = [
                {"instruction": "你对找女助理这个事怎么看？", "response": "哎呀，女助理这个事我跟你们讲啊，找不到人，最近人都不在成都啊，我找了六七个人了！"},
                {"instruction": "你的直播收入怎么样？", "response": "不怎么方便说出来，太丢人了。不过我饿了就啃树皮呀，哈哈哈！"},
            ]

        df = pd.DataFrame(style_data)
        print(f"原始对话数据量 (style_data): {len(df)}")
        dataset = Dataset.from_pandas(df).shuffle(seed=42)
        
        # 采样 (使用 TRAIN_SCALE_RATIO 常量)
        dataset = dataset.select(range(min(int(len(dataset)*TRAIN_SCALE_RATIO), len(dataset))))
        
        
        print(f"采样后的训练数据量: {len(dataset)}")
        
        # 格式化 (这个 .map() 操作会强制加载数据到内存，解决 StopIteration)
        dataset = dataset.map(fmt)
        
        # NEW：保存数据集到磁盘 (保存格式化后的结果)
        dataset.save_to_disk(PROCESSED_DATASET_PATH)
        print(f"处理后的数据集已保存到缓存: {PROCESSED_DATASET_PATH}")
    
    # --- 模型加载和训练逻辑 ---
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA & Trainer
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    
    args = TrainingArguments(
        output_dir="./sft_results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=True,
        report_to="none",
        save_strategy="epoch",
        # tokenizer=tokenizer # 兼容性修正：不传入 tokenizer
    )
    model.enable_input_require_grads()
    
    # SFTTrainer 兼容性修正：不传入 dataset_text_field 和 tokenizer
    trainer = SFTTrainer(
        model=model, 
        args=args, 
        train_dataset=dataset,
        peft_config=lora_config,
    )
    print("开始训练...")
    trainer.train()
    os.makedirs(LORA_ADAPTER_PATH, exist_ok=True)
    trainer.model.save_pretrained(LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(LORA_ADAPTER_PATH)
    print("LoRA 权重已保存至", LORA_ADAPTER_PATH)

# ====================================================================
# 5. 构建 FAISS 索引
# ====================================================================
def build_rag_index(data_dir=BASE_DATA_DIR):
    
    # 步骤 1: 确保 embeddings 实例被加载或创建
    global MODEL_CACHE
    
    # 首次加载嵌入模型，或从缓存中获取实例
    if MODEL_CACHE.get("embeddings") is None:
        print("首次加载嵌入模型...")
        MODEL_CACHE["embeddings"] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': device}
        )
    
    # 将实例从缓存中取出，赋值给局部变量 'embeddings'
    embeddings = MODEL_CACHE["embeddings"] 

    # 步骤 2: 检查点逻辑：FAISS 索引持久化
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print("FAISS 索引已存在，跳过构建")
        return 
    
    print("构建知识库中...")
    
    # 步骤 3: 文件过滤
    files = [f for f in glob.glob(os.path.join(data_dir, '**', '*.md'), recursive=True)
             if not any(os.path.basename(f).upper().startswith(p) for p in
                         ("README", "SUMMARY", "PREFACE", "ACKNOWLEDGEMENTS", "VIDEOS"))]
    docs = []
    
    # 步骤 4: 数据加载（使用 TextLoader 绕过 Unstructured 冲突）
    for f in files:
        # 使用 TextLoader 加载 Markdown 文件
        try:
            loader = TextLoader(f, encoding='utf-8')
            current_docs = loader.load()
            
            # TextLoader 不会自动添加元数据，手动添加源文件名
            for doc in current_docs:
                doc.metadata['source'] = f
            docs.extend(current_docs)
            
        except Exception as e:
            # 捕获文件读取错误
            print(f"使用 TextLoader 加载文件 {f} 失败: {e}")
            continue

    # 步骤 5: 分块、采样和向量化
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    chunks = splitter.split_documents(docs)
    
    # 采样逻辑
    random.seed(42)
    random.shuffle(chunks)
    chunks = chunks[:min(int(len(chunks)*1/50), len(chunks))] 
    
    print(f"将使用 {len(chunks)} 个文本块构建 FAISS 索引...")
    
    # 核心执行：embeddings 已被正确定义
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    print("FAISS 索引已保存至", FAISS_INDEX_PATH)

# ====================================================================
# 6. RAG + LoRA 推理（带模型缓存）
# ====================================================================
def run_inference():
    hf_login(HF_TOKEN_SECRET_NAME)
    
    # 检查依赖文件
    if not os.path.exists(LORA_ADAPTER_PATH) or not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print("缺少 LoRA 权重或 FAISS 索引文件，请确保前两步已成功运行。"); return
    
    # 缓存检查：只加载一次推理模型和 Tokenizer
    global MODEL_CACHE
    if MODEL_CACHE["inference_model"] is None or MODEL_CACHE["tokenizer"] is None:
        print("首次加载推理模型和 Tokenizer...")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        # 基础模型加载（4-bit 量化）
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            device_map="auto"
        )
        
        # 挂载 LoRA 适配器
        style_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).eval()
        
        MODEL_CACHE["tokenizer"] = tokenizer
        MODEL_CACHE["inference_model"] = style_model
    
    tokenizer = MODEL_CACHE["tokenizer"]
    style_model = MODEL_CACHE["inference_model"]

    # 缓存检查：只加载一次嵌入模型
    if MODEL_CACHE["embeddings"] is None:
        print("首次加载嵌入模型...")
        MODEL_CACHE["embeddings"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})
    embeddings = MODEL_CACHE["embeddings"]

    # 4. 加载 FAISS 索引和 Retriever
    vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    # 5. 设置 HuggingFace Pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=style_model,
        tokenizer=tokenizer,
        max_new_tokens=1024, 
        torch_dtype=torch.float16,
        device_map="auto",
        return_full_text=False, # 设为 False，让 chain 处理输出
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1 # 略微降低惩罚，鼓励生成
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # 6. LCEL 链构建 - 修正 Qwen Chat 格式
    # 修正：使用 Qwen 要求的对话格式来包裹整个 Prompt 模板
    QWEN_PROMPT_TEMPLATE = """<|im_start|>system
你是一位著名主播、大型直播机“户晨风”。你的回答必须模仿他的说话风格、口头禅，要幽默、接地气、有点跑火车但充满自信。

<|im_end|>
<|im_start|>user
请使用提供的【上下文信息】来回答用户的问题。如果信息不足，也要用你的风格（户晨风的风格）来应对。

【上下文信息】:
{context}

问题：{input}
<|im_end|>
<|im_start|>assistant
"""
    PROMPT = PromptTemplate.from_template(QWEN_PROMPT_TEMPLATE) 
    document_chain = create_stuff_documents_chain(llm, PROMPT)
    qa_chain = create_retrieval_chain(retriever, document_chain)


    print("\n--- 启动交互式问答循环 ---")
    print("输入 停止/exit 退出")
    while True:
        try:
            q = input("请提问: ")
        except (EOFError, KeyboardInterrupt):
            print("\n再聊"); break
        if q.lower() in {"停止", "exit", "退出"}:
            print("\n再聊"); break
        if not q.strip():
            continue
        print("思考中...")
        
        response = qa_chain.invoke({"input": q})
        answer = response["answer"]
        
        print(f"户晨风: {answer.strip()}\n")

# ====================================================================
# 7. 主入口 
# ====================================================================
if __name__ == "__main__":
    if BASE_DATA_DIR == "/tf/irl_0906/HuChenFeng":
        print("BASE_DATA_DIR 路径需要确认是否正确，当前是：/root/IRLbiddingPT/HuChenFeng")
    
    # 1. 风格微调（带持久化检查和数据集缓存）
    train_sft_model()
    
    # 2. 知识库构建（带持久化检查和嵌入模型缓存）
    build_rag_index()
    
    # 3. 推理（带模型/tokenizer/嵌入模型缓存）
    run_inference()