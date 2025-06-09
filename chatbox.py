import gradio as gr
import requests
import uuid
import json
import os
from typing import List, Tuple, Dict, Any
import vector_db_utils # Import vector_db_utils
from knowledge_base_utils import list_knowledge_bases # Import load_knowledge_base
from apiconfig import API_CONFIG

SESSIONS_FILE = "chat_sessions.json"
ROLES_FILE = "roles.json"

knowledge_bases = list_knowledge_bases()

# 语言映射
LANGUAGE_MAP = {
    "中文": {
        "title": "与LLM对话",
        "input_placeholder": "请输入...",
        "input_label": "输入",
        "history_label": "历史记录",
        "model_label": "选择模型",
        "new_session": "新建会话",
        "default_session": "default chat",
        "session_prefix": "chat",
        "send_button": "发送",
        "language_label": "Language",
        "search_label": "联网搜索",
        "delete_session": "删除会话",
        "delete_confirm": "确定要删除当前会话吗？",
        "role_label": "选择角色",
        "role_management_header": "角色管理",
        "role_name_label": "角色名称",
        "role_prompt_label": "角色提示词",
        "role_prompt_placeholder": "输入此角色的系统提示词...",
        "save_role_button": "保存角色",
        "delete_role_button_text": "删除选中角色",
        "no_role": "无角色",
        "role_name_empty_warning": "角色名称不能为空！",
        "role_saved_info": "角色 '{name}' 已保存。",
        "cannot_delete_no_role_warning": "不能删除'{no_role}'或未选择角色。",
        "role_deleted_info": "角色 '{name}' 已删除。",
        "role_not_found_warning": "角色 '{name}' 未找到。",
        "kb_label": "选择知识库", # Add kb_label
        "use_kb_label": "使用知识库", # Add use_kb_label
        "kb_and_search_warning": "知识库和联网搜索不能同时使用！", # Add warning
        "no_kb_selected_warning": "请先选择一个知识库！" # Add no kb selected warning
    },
    "English": {
        "title": "Chat with LLM",
        "input_placeholder": "Please input...",
        "input_label": "Input",
        "history_label": "History",
        "model_label": "Select Model",
        "new_session": "New Chat",
        "default_session": "default chat",
        "session_prefix": "chat",
        "send_button": "Send",
        "language_label": "Language",
        "search_label": "Web Search",
        "delete_session": "Delete Chat",
        "delete_confirm": "Are you sure you want to delete the current chat?",
        "role_label": "Select Role",
        "role_management_header": "Role Management",
        "role_name_label": "Role Name",
        "role_prompt_label": "Role Prompt",
        "role_prompt_placeholder": "Enter the system prompt for this role...",
        "save_role_button": "Save Role",
        "delete_role_button_text": "Delete Selected Role",
        "no_role": "No Role",
        "role_name_empty_warning": "Role name cannot be empty!",
        "role_saved_info": "Role '{name}' saved.",
        "cannot_delete_no_role_warning": "Cannot delete '{no_role}' or no role is selected.",
        "role_deleted_info": "Role '{name}' deleted.",
        "role_not_found_warning": "Role '{name}' not found.",
        "kb_label": "Select Knowledge Base", # Add kb_label
        "use_kb_label": "Use Knowledge Base", # Add use_kb_label
        "kb_and_search_warning": "Knowledge base and web search cannot be used simultaneously!", # Add warning
        "no_kb_selected_warning": "Please select a knowledge base first!" # Add no kb selected warning
    }
}



# 角色管理
DEFAULT_ROLES = {
    LANGUAGE_MAP["中文"]["no_role"]: "",
    "商家运营专家": "你现在是一名经验丰富的商家运营专家，你擅长管理商家关系，优化商家业务流程，提高商家满意度。你对电商行业有深入的了解，并有优秀的商业洞察力。请在这个角色下为我解答以下问题。",
    "内容运营": "你现在是一名专业的内容运营人员，你精通内容创作、编辑、发布和优化。你对读者需求有敏锐的感知，擅长通过高质量的内容吸引和保留用户。请在这个角色下为我解答以下问题。",
    "数据分析师": "你现在是一名数据分析师，你精通各种统计分析方法，懂得如何清洗、处理和解析数据以获得有价值的洞察。你擅长利用数据驱动的方式来解决问题和提升决策效率。请在这个角色下为我解答以下问题。"
}
all_roles: Dict[str, str] = {}

def load_roles():
    global all_roles
    if os.path.exists(ROLES_FILE):
        try:
            with open(ROLES_FILE, 'r', encoding='utf-8') as f:
                all_roles = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Roles file {ROLES_FILE} is corrupted. Loading default roles.")
            all_roles = DEFAULT_ROLES.copy()
    else:
        all_roles = DEFAULT_ROLES.copy()

    no_role_key_cn = LANGUAGE_MAP["中文"]["no_role"]
    if no_role_key_cn not in all_roles:
        all_roles[no_role_key_cn] = ""

    save_roles()

def save_roles():
    global all_roles
    with open(ROLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_roles, f, ensure_ascii=False, indent=2)

load_roles()

# 会话管理
sessions: Dict[str, Dict[str, Any]] = {}

def get_next_session_number():
    existing_numbers = set()
    for session_id, session_data in sessions.items():
        if session_id != "default":
            try:
                num = int(session_data["title"].split()[-1])
                existing_numbers.add(num)
            except (ValueError, IndexError):
                continue
    next_num = 1
    while next_num in existing_numbers:
        next_num += 1
    return next_num

def load_sessions():
    global sessions
    no_role_key = LANGUAGE_MAP["中文"]["no_role"]
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
                loaded_sessions = json.load(f)

            if "default" not in loaded_sessions:
                loaded_sessions["default"] = {
                    "title": LANGUAGE_MAP["中文"]["default_session"],
                    "history": [],
                    "model": "qwen",
                    "active_role": no_role_key,
                    "selected_kb": None, # Add selected_kb
                    "use_kb": False # Add use_kb
                }

            valid_sessions = {}
            for session_id, session_data in loaded_sessions.items():
                if isinstance(session_data, dict) and "title" in session_data and "history" in session_data:
                    if "model" not in session_data:
                        session_data["model"] = "qwen"
                    if "active_role" not in session_data:
                        session_data["active_role"] = no_role_key
                    if "selected_kb" not in session_data: # Add selected_kb initialization
                        session_data["selected_kb"] = None
                    if "use_kb" not in session_data: # Add use_kb initialization
                        session_data["use_kb"] = False
                    if session_id != "default" and not session_data["title"].startswith(
                            LANGUAGE_MAP["中文"]["session_prefix"] + " "):
                        try:
                            num_part = session_data["title"].replace(LANGUAGE_MAP["English"]["session_prefix"],
                                                                     " ").replace(LANGUAGE_MAP["中文"]["session_prefix"],
                                                                                 " ").strip()
                            if num_part.isdigit():
                                session_data["title"] = f"{LANGUAGE_MAP['中文']['session_prefix']} {num_part}"
                            else:
                                next_num_val = get_next_session_number()
                                session_data["title"] = f"{LANGUAGE_MAP['中文']['session_prefix']} {next_num_val}"
                        except:
                            next_num_val = get_next_session_number()
                            session_data["title"] = f"{LANGUAGE_MAP['中文']['session_prefix']} {next_num_val}"

                    valid_sessions[session_id] = session_data
            sessions = valid_sessions
            if valid_sessions != loaded_sessions:
                save_sessions()

        except Exception as e:
            print(f"Error loading sessions: {e}") # Add error logging
            sessions = {
                "default": {
                    "title": LANGUAGE_MAP["中文"]["default_session"],
                    "history": [],
                    "model": "qwen",
                    "active_role": no_role_key,
                    "selected_kb": None, # Add selected_kb
                    "use_kb": False # Add use_kb
                }
            }
            save_sessions()
    else:
        sessions = {
            "default": {
                "title": LANGUAGE_MAP["中文"]["default_session"],
                "history": [],
                "model": "qwen",
                "active_role": no_role_key,
                "selected_kb": None, # Add selected_kb
                "use_kb": False # Add use_kb
            }
        }
        save_sessions()

def save_sessions():
    global sessions
    no_role_key = LANGUAGE_MAP["中文"]["no_role"]
    if "default" not in sessions:
        sessions["default"] = {
            "title": LANGUAGE_MAP["中文"]["default_session"],
            "history": [],
            "model": "qwen",
            "active_role": no_role_key,
            "selected_kb": None, # Add selected_kb
            "use_kb": False # Add use_kb
        }
    with open(SESSIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

load_sessions()

def convert_history_to_chatbot_format(history: List[Dict[str, str]]) -> List[Tuple[str | None, str | None]]:
    chatbot_messages = []
    user_msg = None
    for msg in history:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None:
            chatbot_messages.append((user_msg, msg["content"]))
            user_msg = None
        elif msg["role"] == "assistant" and user_msg is None:
            chatbot_messages.append((None, msg["content"]))
    if user_msg is not None:
        chatbot_messages.append((user_msg, None))
    return chatbot_messages


# 联网搜索
def web_search(query):
    api_key = API_CONFIG["tavily"].get("api_key")
    if not api_key or "tvly-" not in api_key:
        return "API密钥无效"

    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "max_results": 5
    }
    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        results = response.json().get('results', [])
        if not results:
            return "未能从网络上找到相关信息。"

        search_context = "--- 网络搜索结果 ---\n"
        for i, result in enumerate(results, 1):
            search_context += f"来源 {i}: {result.get('title', '')}\nURL: {result.get('url', '')}\n内容: {result.get('content', '')}\n\n"
        search_context += "--- 搜索结果结束 ---\n"
        return search_context
    except requests.exceptions.Timeout:
        return "网络搜索请求超时。"
    except requests.exceptions.RequestException as e:
        return f"网络搜索请求失败: {e}"
    except Exception as e:
        return f"网络搜索时发生未知错误: {e}"


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
                kb_results = vector_db_utils.search_torch_index(selected_kb_name, user_input, top_n=5)
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


def respond(user_input,
            chat_history_tuples,
            session_id,
            model,
            active_role_name,
            current_lang,
            use_search,
            selected_kb_name, # Add selected_kb_name
            use_kb): # Add use_kb
    if not user_input.strip():
        return "", chat_history_tuples, model, session_id, active_role_name, selected_kb_name, use_kb

    if session_id not in sessions:
        session_id = "default"

    current_session = sessions[session_id]
    internal_history_dicts = current_session["history"]

    # Mutual exclusion logic (UI should also handle this, but good to have a backend check)
    if use_search and use_kb:
        gr.Warning(LANGUAGE_MAP[current_lang]["kb_and_search_warning"])
        # Default to not using KB if both are somehow true, or could prioritize one.
        # For now, let's say we disable KB if both are true, to simplify.
        # This state should ideally be prevented from happening by the UI logic.
        use_kb = False 

    bot_response = call_llm_api(user_input, internal_history_dicts, model, active_role_name, current_lang, use_search, selected_kb_name, use_kb)

    internal_history_dicts.append({"role": "user", "content": user_input})
    internal_history_dicts.append({"role": "assistant", "content": bot_response})

    current_session["history"] = internal_history_dicts
    current_session["model"] = model
    current_session["active_role"] = active_role_name
    current_session["selected_kb"] = selected_kb_name # Save selected_kb
    current_session["use_kb"] = use_kb # Save use_kb
    save_sessions()

    updated_chatbot_display_history = convert_history_to_chatbot_format(internal_history_dicts)
    return "", updated_chatbot_display_history, model, session_id, active_role_name


def switch_session(session_id_to_switch, lang):
    if session_id_to_switch is None or session_id_to_switch not in sessions:
        session_id_to_switch = "default"

    session = sessions[session_id_to_switch]
    chatbot_display = convert_history_to_chatbot_format(session.get("history", []))
    active_model = session.get("model", "qwen")
    active_role = session.get("active_role", LANGUAGE_MAP[lang]["no_role"])
    selected_kb = session.get("selected_kb", None) # Get selected_kb
    use_kb_val = session.get("use_kb", False) # Get use_kb

    return chatbot_display, active_model, session_id_to_switch, active_role, selected_kb, use_kb_val


def create_new_session(language, current_selected_role, selected_kb_name, use_kb_val):
    load_sessions()
    session_id = str(uuid.uuid4())
    next_num = get_next_session_number()

    no_role_key = LANGUAGE_MAP[language]["no_role"]
    sessions[session_id] = {
        "title": f"{LANGUAGE_MAP[language]['session_prefix']} {next_num}",
        "history": [],
        "model": "qwen",
        "active_role": current_selected_role if current_selected_role else no_role_key,
        "selected_kb": selected_kb_name, # Add selected_kb
        "use_kb": use_kb_val # Add use_kb
    }
    save_sessions()
    new_options = [(s_data["title"], s_id) for s_id, s_data in sessions.items()]

    return (
        gr.update(choices=new_options, value=session_id),
        session_id,
        [],
        "qwen",
        sessions[session_id]["active_role"],
        sessions[session_id]["selected_kb"], # Return selected_kb
        sessions[session_id]["use_kb"] # Return use_kb
    )


def delete_current_session_action(session_id_to_delete, lang):
    load_sessions()
    no_role_key = LANGUAGE_MAP[lang]["no_role"]

    if session_id_to_delete == "default":
        sessions["default"]["history"] = []
        sessions["default"]["active_role"] = no_role_key
        sessions["default"]["selected_kb"] = None # Reset KB for default
        sessions["default"]["use_kb"] = False # Reset KB for default
        save_sessions()
        default_session_data = sessions["default"]
        return (
            gr.update(choices=[(s_data["title"], s_id) for s_id, s_data in sessions.items()], value="default"),
            "default",
            [],
            default_session_data.get("model", "qwen"),
            default_session_data.get("active_role", no_role_key),
            default_session_data.get("selected_kb", None), # Return selected_kb
            default_session_data.get("use_kb", False), # Return use_kb
            gr.update(visible=True)
        )

    if session_id_to_delete in sessions:
        del sessions[session_id_to_delete]
        save_sessions()

    default_session_data = sessions["default"]
    return (
        gr.update(choices=[(s_data["title"], s_id) for s_id, s_data in sessions.items()], value="default"),
        "default",
        convert_history_to_chatbot_format(default_session_data.get("history", [])),
        default_session_data.get("model", "qwen"),
        default_session_data.get("active_role", no_role_key),
        default_session_data.get("selected_kb", None), # Return selected_kb
        default_session_data.get("use_kb", False), # Return use_kb
        gr.update(visible=True)
    )


def initialize_session_dropdown_options(lang):
    load_sessions()
    options = [(s_data["title"], s_id) for s_id, s_data in sessions.items()]
    if not options:
        default_title = LANGUAGE_MAP[lang]["default_session"]
        options = [(default_title, "default")]
    return gr.update(choices=options, value="default", label=LANGUAGE_MAP[lang]["history_label"])


def load_initial_state(lang):
    load_sessions()
    load_roles()

    default_session_id = "default"
    if default_session_id not in sessions:
        no_role_key = LANGUAGE_MAP[lang]["no_role"]
        sessions[default_session_id] = {
            "title": LANGUAGE_MAP[lang]["default_session"],
            "history": [],
            "model": "qwen",
            "active_role": no_role_key,
            "selected_kb": None, # Add selected_kb
            "use_kb": False # Add use_kb
        }
        save_sessions()

    default_s = sessions[default_session_id]
    session_options = [(s_data["title"], s_id) for s_id, s_data in sessions.items()]

    default_chatbot_display = convert_history_to_chatbot_format(default_s.get("history", []))
    default_model = default_s.get("model", "qwen")
    default_role = default_s.get("active_role", LANGUAGE_MAP[lang]["no_role"])
    default_selected_kb = default_s.get("selected_kb", None) # Get default selected_kb
    default_use_kb = default_s.get("use_kb", False) # Get default use_kb

    role_choices = list(all_roles.keys())
    kb_choices = list(knowledge_bases.keys()) # Get kb_choices

    return (
        gr.update(choices=session_options, value=default_session_id),
        default_chatbot_display,
        default_model,
        default_session_id,
        gr.update(choices=role_choices, value=default_role),
        default_role,
        all_roles.get(default_role, ""),
        default_role,
        gr.update(choices=kb_choices, value=default_selected_kb), # Update kb_selector_dd
        default_selected_kb, # Return default_selected_kb for state
        gr.update(value=default_use_kb) # Update use_kb_checkbox
    )


def handle_role_selection_change_in_dropdown(selected_role_name_from_dropdown, current_session_id_val, lang):
    prompt_for_selected_role = all_roles.get(selected_role_name_from_dropdown, "")

    if current_session_id_val in sessions:
        sessions[current_session_id_val]["active_role"] = selected_role_name_from_dropdown
        save_sessions()

    return selected_role_name_from_dropdown, prompt_for_selected_role, selected_role_name_from_dropdown


def save_or_update_role_action(role_name_to_save, role_prompt_to_save, lang):
    no_role_key = LANGUAGE_MAP[lang]["no_role"]
    if not role_name_to_save.strip():
        gr.Warning(LANGUAGE_MAP[lang]["role_name_empty_warning"])
        return gr.update(choices=list(all_roles.keys())), gr.update(), role_name_to_save, role_prompt_to_save

    if role_name_to_save == no_role_key and role_prompt_to_save.strip() != "":
        gr.Warning(f"Cannot assign a prompt to the '{no_role_key}' role. It must be empty.")
        return gr.update(choices=list(all_roles.keys()), value=no_role_key), no_role_key, "", no_role_key

    all_roles[role_name_to_save] = role_prompt_to_save
    save_roles()
    gr.Info(LANGUAGE_MAP[lang]["role_saved_info"].format(name=role_name_to_save))

    return gr.update(choices=list(all_roles.keys()),
                     value=role_name_to_save), role_name_to_save, role_prompt_to_save, role_name_to_save


def delete_selected_role_action(role_name_to_delete, current_session_id_val, lang):
    no_role_key = LANGUAGE_MAP[lang]["no_role"]

    if not role_name_to_delete or role_name_to_delete == no_role_key:
        gr.Warning(LANGUAGE_MAP[lang]["cannot_delete_no_role_warning"].format(no_role=no_role_key))
        return gr.update(value=role_name_to_delete), role_name_to_delete, all_roles.get(role_name_to_delete,
                                                                                        ""), role_name_to_delete

    if role_name_to_delete in all_roles:
        del all_roles[role_name_to_delete]
        save_roles()
        gr.Info(LANGUAGE_MAP[lang]["role_deleted_info"].format(name=role_name_to_delete))

        role_updated_in_sessions = False
        for s_id in sessions:
            if sessions[s_id].get("active_role") == role_name_to_delete:
                sessions[s_id]["active_role"] = no_role_key
                role_updated_in_sessions = True
        if role_updated_in_sessions:
            save_sessions()

        new_active_role_for_ui = no_role_key
        if sessions[current_session_id_val].get("active_role") == no_role_key:
            new_active_role_for_ui = no_role_key

        current_role_for_state = no_role_key if role_name_to_delete == sessions[current_session_id_val].get(
            "active_role") else sessions[current_session_id_val].get("active_role")

        return gr.update(choices=list(all_roles.keys()), value=new_active_role_for_ui), \
            new_active_role_for_ui, \
            "", \
            ""
    else:
        gr.Warning(LANGUAGE_MAP[lang]["role_not_found_warning"].format(name=role_name_to_delete))
        return gr.update(value=role_name_to_delete), role_name_to_delete, all_roles.get(role_name_to_delete,
                                                                                        ""), role_name_to_delete