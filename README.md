# 同济大学 软件工程 2025春 用户交互技术 期末项目

A cherrystudio-like chatbox, you can chat with robots and implement knowledgebase.

Dependency:

*  python==3.10.0

```bash
pip install python-dotenv openai fastapi uvicorn gradio==3.50.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For llm_seg, specifiy your  API_KEY and API_BASE in `~/dot_env/<your_api>.env`

```text
EXAMPLE_API_KEY=""
EXAMPLE_API_BASE=""
```

For the chatbox.py,  the api_config is defined in another file:

```python
API_CONFIG = {
    <llm_model1>: {
        "url": BASE_URL,
        "headers": {
            "Authorization": "Bearer <API_KEY>",
            "Content-Type": "application/json"
        }
    },
    <llm_model2>: {
        "url": BASE_URL,
        "headers": {
            "Authorization": "Bearer <API_KEY>",
            "Content-Type": "application/json"
        }
    },
    "tavily": {
        "api_key": API_KEY
    }
}
```


