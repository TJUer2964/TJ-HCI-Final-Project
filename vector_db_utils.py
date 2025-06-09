import os
import torch
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
# 加载环境变量

load_dotenv(os.path.expanduser("~/dot_env/openai.env"))

# 设置路径
VECTOR_DB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_databases")
os.makedirs(VECTOR_DB_ROOT, exist_ok=True)
KNOWLEDGE_BASE_CHUNK_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledgeBase")

# 设置 DashScope key
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

# 通义 text-embedding-v3 接口
def qwen_text_embedding(texts: list[str]) -> torch.Tensor:
    """
    使用通义 Qwen 接口获取文本嵌入。
    返回 tensor (n_texts, 1536)
    """
    client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    embeddings = []
    for text in texts:
        completion = client.embeddings.create(
            model="text-embedding-v3",
            input=text,
            dimensions=1024,
            encoding_format="float"
        )
        embeddings.append(completion.data[0].embedding)
    
    return torch.tensor(embeddings, dtype=torch.float32)

def load_chunks_from_file(chunk_file_path):
    """从指定路径加载文本块。每个块为 {'text': ...} 的字典。"""
    chunks_text_list = []
    current_chunk_str = ""
    try:
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            chunks_text_list = [chunk['text'] for chunk in data]
        return chunks_text_list
    except Exception as e:
        print(f"读取失败: {e}")
        return []

def embed_texts_and_store(kb_name, chunk_file_path):
    """嵌入文本块并用 torch 存储 (向量 .pt + 文本 .texts)，支持追加更新。"""
    if not os.path.exists(chunk_file_path):
        return f"未找到切块文件 {chunk_file_path}", None

    new_texts_list = load_chunks_from_file(chunk_file_path)
    if not new_texts_list:
        return f"未能从 {chunk_file_path} 加载任何文本块。", None

    vector_path = os.path.join(VECTOR_DB_ROOT, kb_name, f"{kb_name}.pt")
    texts_path = os.path.join(VECTOR_DB_ROOT, kb_name,f"{kb_name}.texts")

    try:
        print(f"对 {kb_name} 的 {len(new_texts_list)} 个新文本块（来自 {chunk_file_path}）生成向量嵌入...")
        new_embeddings = qwen_text_embedding(new_texts_list)  # (n_new, dim)

        all_embeddings_list = []
        all_texts_list = []

        if os.path.exists(vector_path) and os.path.exists(texts_path):
            print(f"加载现有的 {kb_name} 向量和文本...")
            try:
                existing_embeddings_tensor = torch.load(vector_path)
                with open(texts_path, 'r', encoding='utf-8') as f:
                    existing_texts_list = [line.strip() for line in f.readlines()]
                
                if existing_embeddings_tensor.shape[0] != len(existing_texts_list):
                    return f"警告：现有向量数量 ({existing_embeddings_tensor.shape[0]}) 与文本数量 ({len(existing_texts_list)}) 不匹配。将重新创建索引。", None
                
                # 检查维度是否匹配
                if existing_embeddings_tensor.shape[1] != new_embeddings.shape[1]:
                    return f"错误：新向量维度 ({new_embeddings.shape[1]}) 与现有向量维度 ({existing_embeddings_tensor.shape[1]}) 不匹配。", None

                all_embeddings_list.append(existing_embeddings_tensor)
                all_texts_list.extend(existing_texts_list)
            except Exception as load_err:
                 return f"加载现有向量/文本失败: {load_err}. 请检查文件或考虑删除它们以重新开始。", None
        
        all_embeddings_list.append(new_embeddings)
        all_texts_list.extend([t.replace('\n', ' ') for t in new_texts_list]) #确保新文本也处理换行符

        final_embeddings_tensor = torch.cat(all_embeddings_list, dim=0)
        
        torch.save(final_embeddings_tensor, vector_path)
        with open(texts_path, 'w', encoding='utf-8') as f: # 'w' to overwrite with all_texts
            for t_processed in all_texts_list:
                f.write(t_processed + '\n')

        return f"成功为 {kb_name} 更新向量索引。新增 {new_embeddings.shape[0]} 个文本块。知识库现有 {final_embeddings_tensor.shape[0]} 个文本块。", vector_path
    except Exception as e:
        return f"嵌入或保存失败: {e}", None

def search_torch_index(kb_name, query_text, k=5):
    """使用 PyTorch cosine 相似度查询向量数据库"""
    vector_path = os.path.join(VECTOR_DB_ROOT, f"{kb_name}.pt")
    texts_path = os.path.join(VECTOR_DB_ROOT, f"{kb_name}.texts")
    if not os.path.exists(vector_path) or not os.path.exists(texts_path):
        return f"未找到 {kb_name} 的向量或文本文件", []

    try:
        db_vectors = torch.load(vector_path)  # shape: (n, dim)
        with open(texts_path, 'r', encoding='utf-8') as f:
            all_texts = [line.strip() for line in f.readlines()]

        n = db_vectors.shape[0]
        if n == 0:
            return "知识库为空", []

        k = min(k, n)  # ⚠️ 限制 k 不超过向量数量

        query_vec = qwen_text_embedding([query_text])  # shape: (1, dim)
        query_vec = torch.nn.functional.normalize(query_vec, dim=1)
        db_vectors = torch.nn.functional.normalize(db_vectors, dim=1)

        scores = torch.matmul(db_vectors, query_vec.T).flatten()  # (n,)
        topk_scores, topk_indices = torch.topk(scores, k)

        results = []
        for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()):
            if idx < len(all_texts):
                results.append({
                    "id": idx,
                    "score": float(score),
                    "text": all_texts[idx]
                })
        return f"共找到 {len(results)} 条结果", results
    except Exception as e:
        return f"查询失败: {e}", []

    
