# import gradio as gr
import numpy as np
from PIL import Image
import torch
import clip 
from upstash_vector import Index 
from upstash_env import UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN

# --- Actual Implementations ---

def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads the CLIP model and preprocessing function."""
    print(f"Loading CLIP model on device: {device}")
    try:
        model, preprocess = clip.load("ViT-L/14", device=device)
        print("CLIP model loaded successfully.")
        # text_input = clip.tokenize(["a test sentence"]).to(device)
        # with torch.no_grad():
        #     text_features = model.encode_text(text_input)
        # print(f"文本特征向量形状: {text_features.shape}")
        return model, preprocess
    except Exception as e:
        print(f"Error loading CLIP model: {e}\nMaybe you need to clear the cache.")
        return None, None

def connect_vector_db():
    """Connects to the Upstash Vector index using environment variables."""
    print("Connecting to Upstash Vector...")
    try:
        url = UPSTASH_VECTOR_REST_URL
        token = UPSTASH_VECTOR_REST_TOKEN
        index = Index(url=url, token=token)
        # Test connection (optional, but recommended)
        index.info()
        print("Successfully connected to Upstash Vector.")
        return index
    except Exception as e:
        print(f"Error connecting to Upstash Vector: {e}\nMaybe you need to shut the vpn.")
        return None

def get_text_features(text, model, preprocess, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Extracts features from text using CLIP."""
    if not model or not preprocess:
        print("Error: CLIP model not loaded.")
        return None
    try:
        text_inputs = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        # Normalize features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting text features: {e}")
        return None

def get_image_features(image, model, preprocess, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Extracts features from an image using CLIP."""
    if not model or not preprocess:
        print("Error: CLIP model not loaded.")
        return None
    if image is None:
        print("Error: Input image is None.")
        return None
    try:
        # Ensure image is PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            print(f"Error: Unexpected image type: {type(image)}")
            return None
            
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting image features: {e}")
        return None

def search_vectors(query_vector, index, top_k=10):
    """Searches for similar vectors in Upstash Vector."""
    if index is None:
        print("Error: Not connected to Vector DB.")
        return []
    if query_vector is None:
        print("Error: Query vector is None.")
        return []
    try:
        print(f"Searching for top {top_k} similar vectors...")
        results = index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)
        print(f"Found {len(results)} results from vector search.")
        return results
    except Exception as e:
        print(f"Error searching vectors: {e}")
        return []



# --- Gradio Interface --- 

# Load models and connect to DB (using actual implementations)
clip_model, clip_preprocess = load_clip_model()
vector_index = connect_vector_db()

def search_by_text(text_query, top_k):
    if not text_query:
        return [], "请输入搜索文本。"
    if not clip_model or not vector_index:
         return [], "错误：模型或数据库未初始化。请检查启动日志。"
    query_vector = get_text_features(text_query, clip_model, clip_preprocess)
    if query_vector is None:
        return [], "错误：无法提取文本特征。"
        
    search_results = search_vectors(query_vector, vector_index, top_k=int(top_k))
    
    # Format results for Gradio Gallery
    gallery_output = []
    if search_results:
        gallery_output = [(res.metadata['image_path'], res.metadata.get('label', 'No Label') + f" (Score: {res.score:.2f})") for res in search_results if res.metadata]
    
    return gallery_output, f"找到 {len(gallery_output)} 个结果。"

def search_by_image(image_query, top_k):
    if image_query is None:
        return [], "请上传或绘制查询图片。"
    if not clip_model or not vector_index:
         return [], "错误：模型或数据库未初始化。请检查启动日志。"
    query_vector = get_image_features(image_query, clip_model, clip_preprocess)
    if query_vector is None:
        return [], "错误：无法提取图像特征。"
        
    search_results = search_vectors(query_vector, vector_index, top_k=int(top_k))
    
    # Format results for Gradio Gallery
    gallery_output = []
    if search_results:
        gallery_output = [(res.metadata['image_path'], res.metadata.get('label', 'No Label') + f" (Score: {res.score:.2f})") for res in search_results if res.metadata]
    
    return gallery_output, f"找到 {len(gallery_output)} 个结果。"

with gr.Blocks(css=".gradio-container {max-width: 90% !important;}") as demo:
    gr.Markdown("## 图像搜索")
    gr.Markdown("使用文本或图像进行搜索。")

    top_k_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="返回结果数量 (Top K)")
    status_output = gr.Textbox(label="状态", interactive=False)

    with gr.Tabs():
        with gr.TabItem("文本搜索"):
            with gr.Row():
                text_input = gr.Textbox(label="输入搜索文本", placeholder="例如：'red apple' 或 'yogurt carton'")
                text_search_button = gr.Button("搜索", variant="primary")
            text_gallery = gr.Gallery(label="搜索结果", show_label=True, elem_id="gallery_text", columns=5, object_fit="contain", height="auto") # Changed elem_id

        with gr.TabItem("图像搜索"):
            with gr.Row():
                # Allow upload, paste, or drawing
                image_input = gr.Image(type="pil", label="上传、粘贴或绘制查询图像", sources=['upload', 'clipboard', 'webcam'])
                image_search_button = gr.Button("搜索", variant="primary")
            image_gallery = gr.Gallery(label="搜索结果", show_label=True, elem_id="gallery_image", columns=5, object_fit="contain", height="auto") # Changed elem_id

    # Event Handlers
    text_search_button.click(
        search_by_text,
        inputs=[text_input, top_k_slider],
        outputs=[text_gallery, status_output]
    )
    
    text_input.submit(
        search_by_text,
        inputs=[text_input, top_k_slider],
        outputs=[text_gallery, status_output]
    )

    image_search_button.click(
        search_by_image,
        inputs=[image_input, top_k_slider],
        outputs=[image_gallery, status_output]
    )

if __name__ == "__main__":
    # Note: You might need to install required libraries:
    # pip install gradio numpy Pillow requests upstash-vector-py torch
    # For actual CLIP:
    # pip install ftfy regex tqdm
    # pip install git+https://github.com/openai/CLIP.git
    # Or use transformers (if using a transformers-based CLIP model):
    # pip install transformers torchvision torchaudio
    
    # IMPORTANT: Set Upstash credentials as environment variables before running:
    # export UPSTASH_VECTOR_REST_URL='YOUR_URL'
    # export UPSTASH_VECTOR_REST_TOKEN='YOUR_TOKEN'
    # (On Windows, use 'set' instead of 'export', e.g., set UPSTASH_VECTOR_REST_URL=YOUR_URL)
    
    # Check if model and index loaded successfully before launching
    if clip_model and vector_index:
        demo.launch()
    else:
        print("some error occurred, please check the log.")  
        
        demo.launch()
    