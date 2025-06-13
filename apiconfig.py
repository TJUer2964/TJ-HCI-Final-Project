# API 配置
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
    },
    "tavily": {
        "api_key": "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"
    }
}