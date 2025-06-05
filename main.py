# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from openGradio7 import demo as chat_demo

app = FastAPI()

app.mount("/chat/app", chat_demo.app, name="chat_app")

NAV_BAR = """
<nav style="padding:10px; background:#f0f0f0;">
  <a href="/" style="margin-right:1rem;">首页</a>
  <a href="/chat" style="margin-right:1rem;">chatbox</a>
</nav>
"""

def make_wrapper(path_to_iframe: str) -> str:
    return NAV_BAR + f"""
<iframe
  src="{path_to_iframe}"
  style="width:100%; height:calc(100vh - 50px); border:none; margin:0; padding:0; display:block;"
  allow="clipboard-read; clipboard-write"
></iframe>
"""

@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    """
    提供 Gradio 应用的包装页面。
    Gradio 应用将通过 iframe 从 "/chat/app/" 加载。
    """
    return make_wrapper("/chat/app/")

@app.get("/", response_class=HTMLResponse)
def index():
    return NAV_BAR + """
    <div style="padding: 20px;">
        <h1>欢迎来到chatbox</h1>
    </div>
    """

if __name__ == "__main__":
    print("请在浏览器中打开 http://localhost:5001")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="debug"
    )