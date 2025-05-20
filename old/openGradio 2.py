# import gradio as gr
import requests
import uuid
import json
# from typing import List, Tuple, Dict

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
        "language_label": "Language"
    },
    "日本語": {
        "title": "LLMと話しましょう",
        "input_placeholder": "質問してみましょう",
        "input_label": "入力",
        "history_label": "履歴",
        "model_label": "モデルを選ぶ",
        "new_session": "新しいチャットを作成",
        "default_session": "default chat",
        "session_prefix": "chat",
        "send_button": "発信する",
        "language_label": "Language"
    }
}

# API配置
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

# 初始化会话存储
sessions = {
    "default": {
        "title": "default chat",
        "history": [],
        "model": "qwen"
    }
}

def convert_history_to_messages(history):
    """
    将历史记录转换为API所需的格式
    """
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages

def call_llm_api(user_input, history, model):
    """
    调用LLM API
    """
    config = API_CONFIG[model]

    try:
        if model == "qwen":
            messages = convert_history_to_messages(history)
            messages.append({"role": "user", "content": user_input})

            data = {
                "model": "qwen-turbo",
                "input": {"messages": messages},
                "parameters": {"temperature": 0.7}
            }
            
            response = requests.post(config["url"], headers=config["headers"], json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                if "output" in response_data and "text" in response_data["output"]:
                    return response_data["output"]["text"]
                else:
                    return f"API响应格式错误: {response.text}"
            else:
                return f"API请求失败，状态码：{response.status_code}"
                
        else:  # deepseek
            messages = convert_history_to_messages(history)
            messages.append({"role": "user", "content": user_input})

            data = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7
            }
            
            response = requests.post(config["url"], headers=config["headers"], json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                if "choices" in response_data:
                    return response_data["choices"][0]["message"]["content"]
                else:
                    return f"API响应格式错误: {response.text}"
            else:
                return f"API请求失败，状态码：{response.status_code}"
                
    except Exception as e:
        return f"请求发生异常：{str(e)}"

def switch_session(session_id):
    session = sessions[session_id]
    # 将历史记录转换为Chatbot所需的格式
    chat_history = []
    for msg in session["history"]:
        chat_history.append({"role": msg["role"], "content": msg["content"]})
    return chat_history, session["model"], session_id

def respond(user_input, chat_history, session_id, model):
    """
    处理用户输入，生成模型响应并更新对话历史
    """
    if session_id not in sessions:
        session_id = "default"
    
    current_session = sessions[session_id]
    current_history = current_session["history"]
    
    # 调用API获取回复
    bot_response = call_llm_api(user_input, current_history, model)
    
    # 更新历史记录
    current_history.append({"role": "user", "content": user_input})
    current_history.append({"role": "assistant", "content": bot_response})
    current_session["history"] = current_history
    current_session["model"] = model
    
    # 更新当前聊天显示
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": bot_response})
    
    return "", chat_history, model, session_id

def create_new_session(language):
    """
    创建新会话
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "title": f"chat {len(sessions)}",
        "history": [],
        "model": "qwen"
    }
    
    # 生成新的选项列表，使用(title, id)格式
    new_options = [(v["title"], k) for k, v in sessions.items()]
    
    # 返回空的历史记录列表
    empty_history = []
    
    return (
        gr.update(choices=new_options, value=session_id),
        session_id,
        empty_history,
        "qwen"
    )

def update_ui(language):

    return {
        title: gr.Markdown(f"# {LANGUAGE_MAP[language]['title']}"),
        user_input: gr.Textbox(
            placeholder=LANGUAGE_MAP[language]["input_placeholder"],
            label=LANGUAGE_MAP[language]["input_label"]
        ),
        session_dropdown: gr.Dropdown(
            label=LANGUAGE_MAP[language]["history_label"]
        ),
        model_selector: gr.Dropdown(
            label=LANGUAGE_MAP[language]["model_label"]
        ),
        new_session_btn: gr.Button(
            value=LANGUAGE_MAP[language]["new_session"]
        ),
        submit_btn: gr.Button(
            value=LANGUAGE_MAP[language]["send_button"]
        )
    }

with gr.Blocks(title="LLM", fill_height=True,theme=gr.themes.Soft()) as demo:
    title = gr.Markdown("# Chat With LLM")
    current_session_id = gr.State("default")
    current_language = gr.State("中文")
    
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                new_session_btn = gr.Button(LANGUAGE_MAP["中文"]["new_session"], scale=1)
                model_selector = gr.Dropdown(
                    scale=1,
                    label=LANGUAGE_MAP["中文"]["model_label"],
                    choices=["qwen", "deepseek"],
                    value="qwen"
                )
                session_dropdown = gr.Dropdown(
                    scale=1,
                    label=LANGUAGE_MAP["中文"]["history_label"],
                    choices=[(v["title"], k) for k, v in sessions.items()],
                    value="default"
                )
                language_selector = gr.Radio(
                    choices=["中文", "日本語"],
                    value="中文",
                    label="Language"
                )
            with gr.Column(scale=15):
                chatbot = gr.Chatbot(
                    scale=10,
                    type="messages",
                    height=800,
                    show_copy_button=True
                )
                with gr.Row(equal_height=True):
                    user_input = gr.Textbox(
                        placeholder=LANGUAGE_MAP["中文"]["input_placeholder"],
                        scale=15,
                        label=LANGUAGE_MAP["中文"]["input_label"]
                    )
                    submit_btn = gr.Button(
                        LANGUAGE_MAP["中文"]["send_button"],
                        variant="secondary",
                        scale=1,
                        min_width=100
                    )
    
    # 事件
    language_selector.change(
        fn=update_ui,
        inputs=language_selector,
        outputs=[title, user_input, session_dropdown, model_selector, new_session_btn, submit_btn]
    )

    session_dropdown.change(
        fn=switch_session,
        inputs=session_dropdown,
        outputs=[chatbot, model_selector, current_session_id]
    )

    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, current_session_id, model_selector],
        outputs=[user_input, chatbot, model_selector, current_session_id]
    )
    
    submit_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, current_session_id, model_selector],
        outputs=[user_input, chatbot, model_selector, current_session_id]
    )
    
    new_session_btn.click(
        fn=create_new_session,
        inputs=language_selector,
        outputs=[session_dropdown, current_session_id, chatbot, model_selector]
    )

if __name__ == "__main__":
    demo.launch()