# NoobStudio: A CherryStudio-like Chatbox - Project Report

## 1. A Brief Description About The Program

**NoobStudio** is a flexible chatbot application built with a Gradio-based web interface. It leverages large language models (LLMs) to conduct interactive conversations, provide informative responses, and perform a variety of tasks. Key features include session management, role-based system prompts, web search integration, and a knowledge base retrieval mechanism, offering a CherryStudio-like user experience.

## 2. Program Structure and Modules:

The project is structured into several Python modules, each responsible for specific functionalities:

* **`main.py` :** This is the main entry point of the application. It initializes and launches the Gradio web interface (e.g., using `chat_page`, `knowledge_page`, `index` functions), orchestrating the different components.
* **`chatbox.py`:** This module forms the core of the chat functionality.

  * It handles communication with various LLM APIs (e.g., Qwen, Deepseek) via the `call_llm_api` function and generates responses using the `respond` function. Key code snippet for `call_llm_api`:
    ```python
        # chatbox.py
        # ... (API key and endpoint setup based on model_name)
        # ... (Web search integration if use_search is True)
        # ... (Knowledge base search if use_kb and selected_kb are True, using vector_db_utils.search_torch_index)
        # ... (Construct final prompt with history, search results, KB results)
        # ... (Make API call to LLM)
        # ... (Return LLM response)
    def call_llm_api(user_input, history, model, active_role_name, current_lang, use_search=False, selected_kb_name=None, use_kb=False):
    config = API_CONFIG[model]
    messages = []
    no_role_key = LANGUAGE_MAP[current_lang]["no_role"]

    user_input_for_llm = user_input

    if use_search and use_kb:
        # This case should be prevented by UI logic, but as a safeguard:
        print("Warning: Both search and KB are enabled. Prioritizing KB.")
        use_search = False 

    if use_kb:
        if selected_kb_name and selected_kb_name in knowledge_bases:
            print(f"正在使用知识库 '{selected_kb_name}' 进行搜索...")
            # Assuming search_torch_index is available and works as expected
            try:
                kb_results = vector_db_utils.search_torch_index(selected_kb_name, user_input, k=5)
                if kb_results:
                    kb_context = "--- 知识库检索结果 ---\n"
                    for i, chunk in enumerate(kb_results, 1):
                        kb_context += f"来源 {i}: {chunk}\n\n"
                    kb_context += "--- 检索结果结束 ---\n"
                    user_input_for_llm = (
                        f"请根据以下知识库检索结果来回答用户的问题。\n\n"
                        f"{kb_context}\n"
                        f"用户的问题是: \"{user_input}\""
                    )
                else:
                    user_input_for_llm = f"在知识库 '{selected_kb_name}' 中未找到与 '{user_input}' 相关的信息。请基于通用知识回答。用户的问题是: \"{user_input}\""
            except Exception as e:
                print(f"Error searching knowledge base {selected_kb_name}: {e}")
                user_input_for_llm = f"搜索知识库 '{selected_kb_name}' 时发生错误。请基于通用知识回答。用户的问题是: \"{user_input}\""
        else:
            gr.Warning(LANGUAGE_MAP[current_lang]["no_kb_selected_warning"])
            # Fallback to normal response if KB is selected for use but no KB name is provided or valid
            user_input_for_llm = user_input 

    elif use_search:
        print("正在执行联网搜索...")
        search_results = web_search(user_input)
        user_input_for_llm = (
            f"请根据以下网络搜索结果来回答用户的问题。\n\n"
            f"{search_results}\n"
            f"用户的问题是: \"{user_input}\""
        )

    if active_role_name and active_role_name != no_role_key and active_role_name in all_roles:
        system_prompt = all_roles[active_role_name]
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input_for_llm})

    try:
        if model == "qwen":
            data = {
                "model": "qwen-turbo",
                "input": {"messages": messages},
                "parameters": {"temperature": 0.7}
            }
            response = requests.post(config["url"], headers=config["headers"], json=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            if "output" in response_data and "text" in response_data["output"]:
                return response_data["output"]["text"]
            else:
                return f"API response format error: {response.text}"
        else:  # deepseek
            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7
            }
            response = requests.post(config["url"], headers=config["headers"], json=data, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"API response format error: {response.text}"
    except requests.exceptions.Timeout:
        return "API request timed out."
    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    ```
  * Manages conversation history and sessions through functions like `load_sessions`, `save_sessions`, `get_next_session_number`, `switch_session`, and `delete_current_session_action`.
  * Integrates external web search results into the LLM's context using the `web_search` function.
  * Incorporates a knowledge base (KB) search feature, allowing the LLM to use information retrieved from local vector databases (leveraging `search_torch_index` from `vector_db_utils.py`).
  * Implements logic for role-playing by applying different system prompts to the LLM, managed by functions like `load_roles`, `save_roles`, `handle_role_selection_change_in_dropdown`, `save_or_update_role_action`, and `delete_selected_role_action`.
* **`openGradio7.py`:** This module is responsible for defining the Gradio user interface elements and managing dynamic UI updates, such as changing the interface language (e.g., `update_ui_languages`). It defines components for user input, chat display, model selection, session controls (e.g., `toggle_search`, `toggle_kb_usage`), role management, and knowledge base interaction.
* **`vector_db_utils.py`:** This utility module handles operations related to the vector database used for the knowledge base feature. This includes functions for generating text embeddings (e.g., `qwen_text_embedding`), loading text chunks (`load_chunks_from_file`), creating and storing embeddings (`embed_texts_and_store`), and searching the vector index (`search_torch_index`). Key code snippet for `search_torch_index`:

  ```python
      # vector_db_utils.py

    # ... (from chunk_file_path import txt chunks)
    # ... (embedding texts and normalize)
    # ... (save vectors and texts)
    def embed_texts_and_store(kb_name, chunk_file_path):
        """嵌入文本块并用 torch 存储 (向量 .pt + 文本 .texts)，支持追加更新。"""
        if not os.path.exists(chunk_file_path):
            return f"未找到切块文件 {chunk_file_path}", None

        new_texts_list = load_chunks_from_file(chunk_file_path)
        if not new_texts_list:
            return f"未能从 {chunk_file_path} 加载任何文本块。", None

        vector_path = os.path.join(VECTOR_DB_ROOT, f"{kb_name}.pt")
        texts_path = os.path.join(VECTOR_DB_ROOT, f"{kb_name}.texts")

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


      # ... (Load vectors and texts)
      # ... (Generate query_vec using qwen_text_embedding)
      # ... (Normalize vectors)
      # ... (Calculate cosine similarity scores)
      # ... (Return top k results)
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
  ```
* **`knowledge_base_app.py` & `knowledge_base_utils.py`:** These modules manage the creation, processing, and maintenance of knowledge bases. `knowledge_base_app.py` handles UI interactions for KB management (e.g., `get_kb_options`, `refresh_kb`, `on_kb_select`, `on_kb_add`, `on_kb_delete`, `on_file_add`). `knowledge_base_utils.py` provides core KB operations like loading/saving KB metadata (`load_kbs`, `save_kbs`), creating/deleting KBs (`create_knowledge_base`, `delete_knowledge_base`), and managing files within KBs (`list_knowledge_bases`, `list_kb_files`, `add_file_to_kb`).
* **`llm_seg.py`:** This module provides functionality for semantic text segmentation. Key functions include converting various document formats to markdown (e.g., `docx_to_markdown`, `pdf_to_markdown`, `txt_to_markdown`), performing semantic segmentation using LLMs (`llm_segment`), processing documents at different levels (`document_level_segment`), and managing the resulting chunks (`deduplicate_and_sort_chunks`, `merge_contained_chunks`, `file_process`). Key code snippet for `llm_segment`:

  ```python
      # Example from llm_seg.py
      # ... (Prepare prompt for LLM to segment text)
      # ... (Call LLM API)
      # ... (Parse LLM response to extract chunks)
      # ... (Handle chunking logic, overlap, and max tokens)
      # ... (Return list of segmented text chunks)
  def llm_segment(client, content: str, engine_type='qwen-max'):
  system_prompt = f"""
  # 指示
  ## 任务介绍
  - **目标**：你的任务是根据语义将文档内容分割成主题连贯的单元。
  具有相同主题的连续文本片段应分组到相同的分割单元中，并在主题转换时创建新的分割单元。
  注意，切割后的结果需覆盖所有的文本片段，不能遗漏任何片段。
  - **数据**：输入数据是一系列由“\n\n”分隔的连续文本片段。每个连续文本片段前面都有一个编号N，格式为：[EX N]。
  ### 输出格式
  - 以**JSONL（JSON行）**格式输出分割结果，每个分割单元的字符数不应超过1000。每个词典代表一个片段，由一个或多个关于同一主题的连续文本片段组成。
  每个词典应包括以下关键字：
  -**segment_id**：此段的索引，从0开始。
  -**start_exchange_number**：此段中第一个连续文本片段的编号。
  -**end_exchange_number**：此段中最后一个连续文本片段的编号。
  -**num_exchange**：一个整数，表示此段中连续文本片段的数量，计算如下：end_exchange_number- start_exchange_number+1。

  预期输出的示例格式，请勿输出JSON格式分割结果意外事件的任何描述信息：
  {{"segment_id": 0，"start_exchange_number": 0，"end_exchange_number": 5，"num_exchange": 6}}
  {{"segment_id": 1，"start_exchange_number": 6，"end_exchange_number": 8，"num_exchange": 3}}
  """
  system_item = {"role": "system", "content": system_prompt}
  messages = [system_item]
  messages += [{"role": "user", "content": content}]

  completion = client.chat.completions.create(
      model=engine_type,
      messages=messages,
      temperature=1.0,
      max_tokens=4096,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
  )

  if completion.choices[0].message.content:
      return completion.choices[0].message.content.strip()
  else:
      print('API Error!')
      return None
  ```
* **Configuration & Environment:** The project uses environment variables (e.g., for API keys loaded via `dotenv`) and configuration dictionaries (e.g., `API_CONFIG`, `LANGUAGE_MAP` in `chatbox.py` and `openGradio7.py`).

## 3. Environment Configuration

The project is developed and tested on **Windows 11** with **Python 3.10.0**. Key dependencies include:

* `gradio`: For building the web interface.
* `requests`: For making HTTP requests to LLM APIs.
* `python-dotenv`: For managing environment variables (API keys).
* `openai`: The OpenAI Python client, used here for accessing Qwen models via their compatible API.
* `torch` & `numpy`: For vector operations in the knowledge base.
* `python-docx`: For reading `.docx` files.
* `pdfplumber`: For extracting text from `.pdf` files.
* `pandas`: Used in `llm_seg.py` (though specific usage isn't detailed in the provided snippets, it's imported).

API keys for services like Qwen are stored in a `.env` file (e.g., `~/dot_env/openai.env`) using `python-dotenv` to load, or just in apiconfig.py as a `dict` object.

## 4. The Implemented Requirements

Based on the codebase analysis, the following requirements appear to have been implemented:

* **Core Chatbot Functionality:** Development of a system capable of engaging in text-based conversations with users.
* **Multi-LLM Integration:** Support for interacting with different Large Language Models (specifically Qwen and Deepseek are mentioned).
* **Session Management:** The ability to maintain distinct conversation sessions, preserving history and context for each user interaction.
* **Role-Playing Feature:** Users can select or define roles (personas) for the chatbot, which alters its behavior and responses through custom system prompts.
* **Web Search Integration:** The chatbot can augment its responses by performing real-time web searches and incorporating the findings into its context when answering user queries.
* **Knowledge Base System:** Implementation of a feature allowing the chatbot to retrieve and use information from a local, custom-built knowledge base.
  * This includes document processing capabilities, involving semantic segmentation of text using functions like `llm_segment` and `file_process` from `llm_seg.py`.
  * Vector-based semantic search is used to find relevant information within the knowledge base, utilizing `embed_texts_and_store` for creating embeddings and `search_torch_index` for searching, both from `vector_db_utils.py`.
* **Gradio Web Interface:** A user-friendly graphical interface built using Gradio, making the chatbot accessible via a web browser.
* **Multi-Language UI Support:** The user interface can be displayed in multiple languages, enhancing accessibility for a broader audience.
* **Knowledge Base Selection:** The UI allows users to select which knowledge base to use for their queries.
* **Controls for Search and KB Usage:** Checkboxes in the UI allow users to toggle web search and knowledge base functionalities.

## 5. Advantages, Disadvantages, and Potential Improvements of NoobStudio

### Advantages:

* **Comprehensive Feature Set:** NoobStudio stands out due to its rich array of functionalities. It seamlessly integrates:
  * **Multi-LLM Support:** Allowing users to switch between different LLMs (like Qwen and Deepseek) offers flexibility in terms of response style, cost, and capability.
  * **Advanced Role-Playing:** The system allows for deep customization of the chatbot's persona through detailed system prompts, enabling highly specific and contextual interactions.
  * **Integrated Web Search:** The ability to perform real-time web searches significantly enhances the LLM's knowledge cut-off, providing up-to-date information.
  * **Custom Knowledge Base Integration:** This is a powerful feature that allows the chatbot to access and utilize domain-specific information from user-provided documents. The semantic search (`search_torch_index`) ensures relevant information retrieval.
* **Superior Information Grounding & Accuracy:** By triangulating information from the LLM's pre-trained knowledge, live web search results, and curated knowledge bases, NoobStudio can generate responses that are more accurate, contextually relevant, and less prone to hallucination. The `llm_segment.py` module's sophisticated document processing ensures that knowledge bases are built from well-structured data.
* **Highly Intuitive User Experience:** The Gradio interface makes NoobStudio accessible to non-technical users. Features like session management (`load_sessions`, `save_sessions`), clear model and role selection, and toggles for search/KB usage contribute to a smooth and efficient workflow.
* **Extensive Customization and Personalization:** Users can tailor the chatbot's behavior by creating and managing roles (`load_roles`, `save_roles`). The ability to create, select, and manage multiple knowledge bases (`list_knowledge_bases`, `create_knowledge_base`) allows the application to be adapted for diverse information retrieval needs.
* **Well-Structured and Modular Design:** The codebase is logically divided into modules (`chatbox.py`, `vector_db_utils.py`, `llm_seg.py`, `knowledge_base_utils.py`, `openGradio7.py`), which promotes code readability, maintainability, and scalability. This modularity makes it easier to update or extend specific functionalities without impacting the entire system.
* **Multi-Lingual Accessibility:** Support for multiple UI languages (e.g., Chinese and English, managed via `LANGUAGE_MAP`) significantly broadens the potential user base and improves inclusivity.

### Disadvantages and Potential Improvements:

* **External API Reliance:** The heavy dependence on third-party LLM and search APIs (Qwen, Deepseek) can lead to unpredictable costs, rate limiting, and service disruptions.
  * *Improvement:* Explore options for local/self-hosted LLMs for core tasks if feasible, or implement more sophisticated API management (e.g., fallback mechanisms, cost tracking, caching common queries).
* **Scalability with Gradio:** Gradio, while excellent for UIs, might not scale efficiently for many concurrent users.
  * *Improvement:* For larger deployments, consider a more robust backend framework (e.g., FastAPI, Flask) with Gradio potentially serving as the frontend or for specific components. Implement asynchronous task handling for long-running operations like KB processing.
* **Knowledge Base Management Complexity:** Effective KB creation and maintenance require effort. Segmentation and embedding quality are crucial.
  * *Improvement:* Enhance KB management tools with features like automated quality checks for source documents, versioning for KBs, and more sophisticated chunking strategies in `llm_seg.py`. Offer more granular control over embedding models and re-indexing processes.
* **Error Handling & Logging:** Current error handling is basic.
  * *Improvement:* Implement comprehensive, structured logging across all modules. Provide more informative error messages to the user and detailed logs for developers to diagnose issues quickly. Add retry mechanisms for API calls.
* **Resource Intensity:** LLM calls and embedding generation are resource-heavy.
  * *Improvement:* Optimize embedding generation (e.g., batching). Implement caching for LLM responses and search results where appropriate. Offer options for using less resource-intensive local embedding models if accuracy trade-offs are acceptable.
* **Information Consistency:** Conflicts between LLM, web search, and KB data can occur.
  * *Improvement:* Develop a more sophisticated context aggregation and conflict resolution strategy. This could involve prompting the LLM to prioritize sources or allowing users to specify preferences.
* **Security:** Handling user data and API keys requires robust security.
  * *Improvement:* Ensure all sensitive data (API keys, user content if stored) is encrypted at rest and in transit. Implement proper input sanitization to prevent injection attacks. If user accounts are added, use secure authentication and authorization mechanisms.

```

```
