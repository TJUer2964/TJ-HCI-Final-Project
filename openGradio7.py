import gradio as gr
import requests
import uuid
import json
import os
from typing import List, Tuple, Dict, Any


SESSIONS_FILE = "chat_sessions.json"
ROLES_FILE = "roles.json"

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
        "role_not_found_warning": "角色 '{name}' 未找到。"
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
        "role_not_found_warning": "Role '{name}' not found."
    }
}

#API 配置
API_CONFIG = {
    "qwen": {
        "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
        "headers": {
            "Authorization": "Bearer sk-bca85acc48e84d2bb4613dbba2b774a6",
            "Content-Type": "application/json"
        }
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer sk-24de5ef58c4145518631883768cbe6af",
            "Content-Type": "application/json"
        }
    }
}

#角色管理
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
                    "active_role": no_role_key
                }

            valid_sessions = {}
            for session_id, session_data in loaded_sessions.items():
                if isinstance(session_data, dict) and "title" in session_data and "history" in session_data:
                    if "model" not in session_data:
                        session_data["model"] = "qwen"
                    if "active_role" not in session_data:
                        session_data["active_role"] = no_role_key
                    if session_id != "default" and not session_data["title"].startswith(LANGUAGE_MAP["中文"]["session_prefix"] + " "):
                        try:
                            num_part = session_data["title"].replace(LANGUAGE_MAP["English"]["session_prefix"], "").replace(LANGUAGE_MAP["中文"]["session_prefix"], "").strip()
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
            sessions = {
                "default": {
                    "title": LANGUAGE_MAP["中文"]["default_session"],
                    "history": [],
                    "model": "qwen",
                    "active_role": no_role_key
                }
            }
            save_sessions()
    else:
        sessions = {
            "default": {
                "title": LANGUAGE_MAP["中文"]["default_session"],
                "history": [],
                "model": "qwen",
                "active_role": no_role_key
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
            "active_role": no_role_key
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

def call_llm_api(user_input, history, model, active_role_name, current_lang):
    config = API_CONFIG[model]
    messages = []
    no_role_key = LANGUAGE_MAP[current_lang]["no_role"]

    if active_role_name and active_role_name != no_role_key and active_role_name in all_roles:
        system_prompt = all_roles[active_role_name]
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

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
            current_lang) :
    if not user_input.strip():
        return "", chat_history_tuples, model, session_id, active_role_name

    if session_id not in sessions:
        session_id = "default"

    current_session = sessions[session_id]
    internal_history_dicts = current_session["history"]

    bot_response = call_llm_api(user_input, internal_history_dicts, model, active_role_name, current_lang)

    internal_history_dicts.append({"role": "user", "content": user_input})
    internal_history_dicts.append({"role": "assistant", "content": bot_response})
    
    current_session["history"] = internal_history_dicts
    current_session["model"] = model
    current_session["active_role"] = active_role_name
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
    
    return chatbot_display, active_model, session_id_to_switch, active_role

def create_new_session(language, current_selected_role):
    load_sessions()
    session_id = str(uuid.uuid4())
    next_num = get_next_session_number()
    
    no_role_key = LANGUAGE_MAP[language]["no_role"]
    sessions[session_id] = {
        "title": f"{LANGUAGE_MAP[language]['session_prefix']} {next_num}",
        "history": [],
        "model": "qwen",
        "active_role": current_selected_role if current_selected_role else no_role_key
    }
    save_sessions()
    new_options = [(s_data["title"], s_id) for s_id, s_data in sessions.items()]
    
    return (
        gr.update(choices=new_options, value=session_id),
        session_id,
        [],
        "qwen",
        sessions[session_id]["active_role"]
    )

def delete_current_session_action(session_id_to_delete, lang):
    load_sessions()
    no_role_key = LANGUAGE_MAP[lang]["no_role"]

    if session_id_to_delete == "default":
        sessions["default"]["history"] = []
        sessions["default"]["active_role"] = no_role_key
        save_sessions()
        default_session_data = sessions["default"]
        return (
            gr.update(choices=[(s_data["title"], s_id) for s_id, s_data in sessions.items()], value="default"),
            "default",
            [],
            default_session_data.get("model", "qwen"),
            default_session_data.get("active_role", no_role_key),
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
        gr.update(visible=True)
    )

def update_ui_languages(language):
    lang_map = LANGUAGE_MAP[language]
    no_role_key_for_lang = lang_map["no_role"]
    
    return {
        title_md: gr.Markdown(f"# {lang_map['title']}"),
        user_input_tb: gr.Textbox(
            placeholder=lang_map["input_placeholder"],
            label=lang_map["input_label"]
        ),
        session_dropdown: gr.Dropdown(
            label=lang_map["history_label"]
        ),
        model_selector_dd: gr.Dropdown(
            label=lang_map["model_label"]
        ),
        new_session_btn: gr.Button(
            value=lang_map["new_session"]
        ),
        submit_btn: gr.Button(
            value=lang_map["send_button"]
        ),
        delete_session_btn: gr.Button(
            value=lang_map["delete_session"]
        ),
        role_selector_dd: gr.Dropdown(label=lang_map['role_label']),
        manage_roles_accordion: gr.Accordion(label=lang_map['role_management_header']),
        role_name_input_tb: gr.Textbox(label=lang_map['role_name_label']),
        role_prompt_input_tb: gr.Textbox(label=lang_map['role_prompt_label'], placeholder=lang_map['role_prompt_placeholder']),
        save_role_btn: gr.Button(value=lang_map['save_role_button']),
        delete_role_in_management_btn: gr.Button(value=lang_map['delete_role_button_text'])
    }

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
            "active_role": no_role_key
        }
        save_sessions()

    default_s = sessions[default_session_id]
    session_options = [(s_data["title"], s_id) for s_id, s_data in sessions.items()]
    
    default_chatbot_display = convert_history_to_chatbot_format(default_s.get("history", []))
    default_model = default_s.get("model", "qwen")
    default_role = default_s.get("active_role", LANGUAGE_MAP[lang]["no_role"])

    role_choices = list(all_roles.keys())

    return (
        gr.update(choices=session_options, value=default_session_id),
        default_chatbot_display,
        default_model,
        default_session_id,
        gr.update(choices=role_choices, value=default_role),
        default_role,
        all_roles.get(default_role, ""),
        default_role
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
    
    return gr.update(choices=list(all_roles.keys()), value=role_name_to_save), role_name_to_save, role_prompt_to_save, role_name_to_save

def delete_selected_role_action(role_name_to_delete, current_session_id_val, lang):
    no_role_key = LANGUAGE_MAP[lang]["no_role"]

    if not role_name_to_delete or role_name_to_delete == no_role_key:
        gr.Warning(LANGUAGE_MAP[lang]["cannot_delete_no_role_warning"].format(no_role=no_role_key))
        return gr.update(value=role_name_to_delete), role_name_to_delete, all_roles.get(role_name_to_delete, ""), role_name_to_delete

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

        current_role_for_state = no_role_key if role_name_to_delete == sessions[current_session_id_val].get("active_role") else sessions[current_session_id_val].get("active_role")

        return gr.update(choices=list(all_roles.keys()), value=new_active_role_for_ui), \
               new_active_role_for_ui, \
               "", \
               ""
    else:
        gr.Warning(LANGUAGE_MAP[lang]["role_not_found_warning"].format(name=role_name_to_delete))
        return gr.update(value=role_name_to_delete), role_name_to_delete, all_roles.get(role_name_to_delete, ""), role_name_to_delete


with gr.Blocks(title="LLM Chat", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:

    current_session_id_state = gr.State("default")
    current_language_state = gr.State("中文")
    current_selected_role_name_state = gr.State(LANGUAGE_MAP["中文"]["no_role"])

    title_md = gr.Markdown(f"# {LANGUAGE_MAP['中文']['title']}")

    with gr.Row():
        with gr.Column(scale=3, min_width=300):
            with gr.Group():
                language_selector_rd = gr.Radio(
                    choices=["中文", "English"],
                    value="中文",
                    label=LANGUAGE_MAP["中文"]["language_label"],
                    interactive=True
                )
                new_session_btn = gr.Button(LANGUAGE_MAP["中文"]["new_session"], min_width=100)
                session_dropdown = gr.Dropdown(
                    label=LANGUAGE_MAP["中文"]["history_label"],
                    choices=[(s_data["title"], s_id) for s_id, s_data in sessions.items()],
                    value="default",
                    interactive=True
                )
                delete_session_btn = gr.Button(
                    LANGUAGE_MAP["中文"]["delete_session"],
                    variant="stop",
                    min_width=100
                )
            
            with gr.Group():
                model_selector_dd = gr.Dropdown(
                    label=LANGUAGE_MAP["中文"]["model_label"],
                    choices=["qwen", "deepseek"],
                    value="qwen",
                    interactive=True
                )
                role_selector_dd = gr.Dropdown(
                    label=LANGUAGE_MAP["中文"]["role_label"],
                    choices=list(all_roles.keys()),
                    value=LANGUAGE_MAP["中文"]["no_role"],
                    interactive=True
                )

            manage_roles_accordion = gr.Accordion(LANGUAGE_MAP["中文"]["role_management_header"], open=False)
            with manage_roles_accordion:
                role_name_input_tb = gr.Textbox(label=LANGUAGE_MAP["中文"]["role_name_label"])
                role_prompt_input_tb = gr.Textbox(
                    label=LANGUAGE_MAP["中文"]["role_prompt_label"],
                    lines=5,
                    placeholder=LANGUAGE_MAP["中文"]["role_prompt_placeholder"]
                )
                with gr.Row():
                    save_role_btn = gr.Button(LANGUAGE_MAP["中文"]["save_role_button"])
                    delete_role_in_management_btn = gr.Button(LANGUAGE_MAP["中文"]["delete_role_button_text"], variant="secondary")
        
        with gr.Column(scale=7):
            chatbot_display_component = gr.Chatbot(
                scale=10,
                height=700,
                show_copy_button=True,
                bubble_full_width=False,
            )
            with gr.Row(equal_height=True):
                user_input_tb = gr.Textbox(
                    placeholder=LANGUAGE_MAP["中文"]["input_placeholder"],
                    scale=15,
                    label=LANGUAGE_MAP["中文"]["input_label"],
                    autofocus=True
                )
                submit_btn = gr.Button(
                    LANGUAGE_MAP["中文"]["send_button"],
                    variant="primary",
                    scale=1,
                    min_width=100
                )

    language_selector_rd.change(
        fn=update_ui_languages,
        inputs=language_selector_rd,
        outputs=[title_md, user_input_tb, session_dropdown, model_selector_dd, 
                 new_session_btn, submit_btn, delete_session_btn,
                 role_selector_dd, manage_roles_accordion, role_name_input_tb,
                 role_prompt_input_tb, save_role_btn, delete_role_in_management_btn]
    ).then(
        lambda lang: lang,
        inputs=language_selector_rd,
        outputs=current_language_state
    ).then(
        fn=initialize_session_dropdown_options,
        inputs=language_selector_rd,
        outputs=session_dropdown
    )
    
    demo.load(
        fn=load_initial_state,
        inputs=[current_language_state],
        outputs=[session_dropdown, chatbot_display_component, model_selector_dd, current_session_id_state, 
                 role_selector_dd, current_selected_role_name_state, role_prompt_input_tb, role_name_input_tb]
    )

    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, current_language_state],
        outputs=[chatbot_display_component, model_selector_dd, current_session_id_state, role_selector_dd]
    ).then(
        lambda role_name_from_session: (all_roles.get(role_name_from_session, ""), role_name_from_session, role_name_from_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    )

    new_session_btn.click(
        fn=create_new_session,
        inputs=[current_language_state, current_selected_role_name_state],
        outputs=[session_dropdown, current_session_id_state, chatbot_display_component, model_selector_dd, role_selector_dd]
    ).then(
        lambda role_name_from_new_session: (all_roles.get(role_name_from_new_session, ""), role_name_from_new_session, role_name_from_new_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    )
    
    delete_session_btn.click(
        fn=delete_current_session_action,
        inputs=[current_session_id_state, current_language_state],
        outputs=[session_dropdown, current_session_id_state, chatbot_display_component, model_selector_dd, role_selector_dd, delete_session_btn]
    ).then(
        lambda role_name_from_default_session: (all_roles.get(role_name_from_default_session, ""), role_name_from_default_session, role_name_from_default_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    )

    user_input_tb.submit(
        fn=respond,
        inputs=[user_input_tb, chatbot_display_component, current_session_id_state, model_selector_dd, current_selected_role_name_state, current_language_state],
        outputs=[user_input_tb, chatbot_display_component, model_selector_dd, current_session_id_state, current_selected_role_name_state]
    )
    submit_btn.click(
        fn=respond,
        inputs=[user_input_tb, chatbot_display_component, current_session_id_state, model_selector_dd, current_selected_role_name_state, current_language_state],
        outputs=[user_input_tb, chatbot_display_component, model_selector_dd, current_session_id_state, current_selected_role_name_state]
    )

    role_selector_dd.change(
        fn=handle_role_selection_change_in_dropdown,
        inputs=[role_selector_dd, current_session_id_state, current_language_state],
        outputs=[role_name_input_tb, role_prompt_input_tb, current_selected_role_name_state]
    )

    save_role_btn.click(
        fn=save_or_update_role_action,
        inputs=[role_name_input_tb, role_prompt_input_tb, current_language_state],
        outputs=[role_selector_dd, current_selected_role_name_state, role_prompt_input_tb, role_name_input_tb]
    )
    
    delete_role_in_management_btn.click(
        fn=delete_selected_role_action,
        inputs=[role_name_input_tb, current_session_id_state, current_language_state],
        outputs=[role_selector_dd, current_selected_role_name_state, role_prompt_input_tb, role_name_input_tb]
    )
    
    role_selector_dd.change(lambda x: x, inputs=role_selector_dd, outputs=role_name_input_tb)

if __name__ == "__main__":
    demo.launch()