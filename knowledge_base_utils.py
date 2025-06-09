import os
import json
import shutil
from datetime import datetime
from llm_seg import file_process
from vector_db_utils import *

KNOWLEDGE_BASES_JSON = "knowledge_bases.json"
UPLOAD_DIR = "uploads"
KNOWLEDGE_BASES = "knowledgeBase"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASES, exist_ok=True)

def load_kbs():
    if not os.path.exists(KNOWLEDGE_BASES_JSON):
        with open(KNOWLEDGE_BASES_JSON, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    with open(KNOWLEDGE_BASES_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_kbs(kbs):
    with open(KNOWLEDGE_BASES_JSON, 'w', encoding='utf-8') as f:
        json.dump(kbs, f, ensure_ascii=False, indent=2)

def create_knowledge_base(kb_name):
    kbs = load_kbs()
    if kb_name in kbs:
        return False, f"知识库 '{kb_name}' 已存在"
    kbs[kb_name] = {"created_at": datetime.now().strftime("%Y-%m-%d"), "files": []}
    os.makedirs(os.path.join(KNOWLEDGE_BASES, kb_name), exist_ok=True)
    save_kbs(kbs)
    return True, f"知识库 '{kb_name}' 创建成功"

def delete_knowledge_base(kb_name):
    kbs = load_kbs()
    if kb_name not in kbs:
        return False, f"知识库 '{kb_name}' 不存在"
    shutil.rmtree(os.path.join(KNOWLEDGE_BASES, kb_name), ignore_errors=True)
    del kbs[kb_name]
    save_kbs(kbs)
    return True, f"知识库 '{kb_name}' 已删除"

def list_knowledge_bases():
    return load_kbs()

def list_kb_files(kb_name):
    kbs = load_kbs()
    return kbs.get(kb_name, {}).get("files", [])

def add_file_to_kb(kb_name, file_name):
    kbs = load_kbs()
    if kb_name not in kbs:
        return False, f"知识库 '{kb_name}' 不存在"
    if file_name not in kbs[kb_name]['files']:
        kbs[kb_name]['files'].append(file_name)
        # 构建上传文件的完整路径
        uploaded_file_path = os.path.join(UPLOAD_DIR, kb_name, file_name) # 假设文件已上传到 UPLOAD_DIR/kb_name/file_name
        if not os.path.exists(uploaded_file_path):
            return False, f"上传的文件 {uploaded_file_path} 未找到"

        # 1. 调用 llm_seg 的 file_process 进行切块
        #    file_process(file_path, kb_name, kb_path=KNOWLEDGE_BASES)
        #    它会将切块结果保存到 KNOWLEDGE_BASES/kb_name/file_name_without_ext2chunks.txt
        chunk_file_name_expected = f"{os.path.splitext(file_name)[0]}2chunks.txt"
        chunk_file_path_expected = os.path.join(KNOWLEDGE_BASES, kb_name, chunk_file_name_expected)
        
        # 确保目标目录存在
        os.makedirs(os.path.join(KNOWLEDGE_BASES, kb_name), exist_ok=True)

        try:
            # 写入 konwledgeBase/kb_name/ 目录
            # file_process 返回的是切块后文件的路径
            processed_chunk_file_path = file_process(uploaded_file_path, kb_name=kb_name)
            if not processed_chunk_file_path or not os.path.exists(processed_chunk_file_path):
                 return False, f"文件 '{file_name}' 切块失败或未找到切块结果。预期路径：{processed_chunk_file_path}"
        except Exception as e:
            return False, f"文件 '{file_name}' 切块处理失败: {e}"

        # 2. 调用 vector_db_utils 的 embed_texts_and_store 进行向量化
        try:
            # embed_texts_and_store(kb_name, chunk_file_name)
            # chunk_file_name 是相对于 KNOWLEDGE_BASE_CHUNK_ROOT (即 konwledgeBase) 下的 kb_name 目录的文件名
            msg, vector_path = embed_texts_and_store(kb_name, processed_chunk_file_path)
            if not vector_path:
                # 如果向量化失败，可能需要回滚文件添加操作或至少记录错误
                kbs[kb_name]['files'].remove(file_name) # 简单回滚
                save_kbs(kbs)
                return False, f"文件 '{file_name}' 向量化失败: {msg}"
        except Exception as e:
            kbs[kb_name]['files'].remove(file_name) # 简单回滚
            save_kbs(kbs)
            return False, f"文件 '{file_name}' 向量化时发生意外错误: {e}"
        
        save_kbs(kbs) # 确保在所有操作成功后保存
    return True, f"文件 '{file_name}' 已成功添加到知识库 '{kb_name}' 并完成向量化。"
