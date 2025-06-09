# main.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from openGradio7 import demo as chat_demo
from knowledge_base_app import demo as knowledge_demo

app = FastAPI()

app.mount("/chat/app", chat_demo.app, name="chat_app")
app.mount("/knowledge/app", knowledge_demo.app, name="knowledge_app")


NAV_BAR = """
<nav style="display: flex; align-items: center; padding: 12px 20px; background-color: #333; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
  <a href="/chat" style="color: white; text-decoration: none; margin-right: 20px; font-size: 16px; font-weight: 500; padding: 8px 12px; border-radius: 4px; transition: background-color 0.3s ease;">ChatBox</a>
  <a href="/knowledge" style="color: white; text-decoration: none; margin-right: 20px; font-size: 16px; font-weight: 500; padding: 8px 12px; border-radius: 4px; transition: background-color 0.3s ease;">KnowledgeBase</a>
</nav>
<script>
  // Add hover effect for links
  const navLinks = document.querySelectorAll('nav a');
  navLinks.forEach(link => {
    link.onmouseover = () => link.style.backgroundColor = '#555';
    link.onmouseout = () => link.style.backgroundColor = 'transparent';
  });
</script>
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

@app.get("/knowledge", response_class=HTMLResponse)
def knowledge_page():
    """
    提供 Gradio 应用的包装页面。
    Gradio 应用将通过 iframe 从 "/knowledge/app/" 加载。
    """
    return make_wrapper("/knowledge/app/")

@app.get("/", response_class=HTMLResponse)
def index():
    return make_wrapper("/chat/app/")

if __name__ == "__main__":
    print("请在浏览器中打开 http://localhost:5001")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="debug"
    )