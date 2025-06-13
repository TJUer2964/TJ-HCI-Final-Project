import gradio as gr
import os
from knowledge_base_utils import (
    list_knowledge_bases,
    create_knowledge_base,
    delete_knowledge_base,
    list_kb_files,
    add_file_to_kb,
)

# 状态全局变量 - 使用gr.State替代全局变量
def get_kb_options():
    return list_knowledge_bases()

# 自定义CSS，用于调整间距和可能的字体大小
custom_css = """
.gradio-container { padding: 20px !important; }
.gr-panel { padding: 15px !important; border-radius: 10px !important; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1) !important; }
.gr-button { min-width: 120px !important; margin-top: 5px !important; margin-bottom: 5px !important; }
.gr-input { margin-bottom: 10px !important; }
.gr-dropdown { margin-bottom: 10px !important; }
"""

with gr.Blocks(title="知识库管理系统", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), css=custom_css) as demo:
    gr.Markdown("## 知识库管理系统")
    # 使用gr.State替代全局变量保存状态
    current_kb = gr.State("")  # 当前选中的知识库名
    
    with gr.Row():
        with gr.Column(scale=1): # 增大左侧列的比例
            with gr.Blocks(): # 使用Blocks包裹增加视觉区块感
                gr.Markdown("### 知识库操作")
                kb_list = gr.Dropdown(choices=get_kb_options(), label="选择知识库", interactive=True, elem_id="kb_list_dropdown")
                kb_refresh_btn = gr.Button("🔄 刷新列表", elem_id="kb_refresh_button")
                with gr.Accordion("创建与删除知识库", open=False):
                    new_kb_name = gr.Textbox(label="新知识库名称", placeholder="输入新知识库名称...", elem_id="new_kb_name_textbox")
                    kb_add_btn = gr.Button("➕ 添加知识库", variant="primary", elem_id="kb_add_button")
                    kb_delete_btn = gr.Button("🗑️ 删除当前知识库", variant="stop", interactive=False, elem_id="kb_delete_button")

        with gr.Column(scale=3): # 增大右侧列的比例
            with gr.Blocks(): # 使用Blocks包裹增加视觉区块感
                gr.Markdown("### 文件管理")
                file_list = gr.Dropdown(choices=[], label="知识库内文件列表", interactive=True, elem_id="file_list_dropdown")
                with gr.Accordion("上传与处理文件", open=True):
                    file_upload = gr.File(label="选择或拖拽文件上传", file_count="multiple", elem_id="file_upload_area")
                    file_add_btn = gr.Button("📤 添加文件到当前知识库", variant="primary", elem_id="file_add_button")
            
            with gr.Blocks():
                gr.Markdown("### 操作状态与日志")
                status_box = gr.Textbox(label="操作日志", lines=5, max_lines=10, interactive=False, placeholder="此处显示操作结果...", elem_id="status_textbox")
    
    # 逻辑绑定
    def refresh_kb():
        """刷新知识库列表"""
        kb_options = get_kb_options()
        return gr.update(choices=kb_options, value=None)

    def on_kb_select(kb_name, request: gr.Request):
        """选择知识库时的处理"""
        # print(f"Knowledge base selected: {kb_name} by {request.client.host}")
        if kb_name:
            files = list_kb_files(kb_name) or []
            return (
                kb_name,  # 更新当前知识库状态
                gr.update(choices=files, value=None),  # 更新文件列表
                gr.update(interactive=True)  # 启用删除按钮
            )
        return (
            "",  # 清空当前知识库状态
            gr.update(choices=[], value=None),  # 清空文件列表
            gr.update(interactive=False)  # 禁用删除按钮
        )

    def on_kb_add(kb_name_to_add):
        """添加新知识库"""
        if not kb_name_to_add.strip():
            gr.Warning("知识库名称不能为空！")
            return gr.update(), gr.update(), gr.update(value="")
        
        success, msg = create_knowledge_base(kb_name_to_add)
        if success:
            gr.Info(f"知识库 '{kb_name_to_add}' 创建成功！")
            kb_options = get_kb_options()
            return (
                gr.update(choices=kb_options, value=kb_name_to_add),
                gr.update(value=""),  # 清空输入框
                msg # 更新状态栏
            )
        else:
            gr.Error(f"创建知识库 '{kb_name_to_add}' 失败: {msg}")
            return gr.update(), gr.update(value=kb_name_to_add), msg

    def on_kb_delete(kb_name_to_delete):
        """删除当前知识库"""
        if not kb_name_to_delete:
            gr.Warning("请先选择一个知识库进行删除！")
            return gr.update(), gr.update(), gr.update(), ""
        
        # 添加一个确认步骤会更好，但Gradio直接实现较为复杂，此处简化
        success, msg = delete_knowledge_base(kb_name_to_delete)
        if success:
            gr.Info(f"知识库 '{kb_name_to_delete}' 已删除！")
            kb_options = get_kb_options()
            return (
                gr.update(choices=kb_options, value=None),
                gr.update(choices=[], value=None),
                "",  # 清空当前知识库
                msg # 更新状态栏
            )
        else:
            gr.Error(f"删除知识库 '{kb_name_to_delete}' 失败: {msg}")
            return gr.update(), gr.update(), kb_name_to_delete, msg

    def on_file_add(file_objs, kb_name_current):
        """添加文件到知识库"""
        if not kb_name_current:
            gr.Warning("请先选择一个知识库！")
            return gr.update(), "请先选择知识库"
        
        if not file_objs:
            gr.Warning("请选择要上传的文件！")
            return gr.update(), "请选择要上传的文件"
        
        results_log = []
        all_successful = True
        for file_obj in file_objs:
            # 注意：add_file_to_kb 需要的是文件路径，gradio File 组件直接给出的是临时文件路径
            # 我们需要将 file_obj.name (临时路径) 传递给 add_file_to_kb
            success, msg = add_file_to_kb(kb_name_current, file_obj.name) 
            results_log.append(f"文件 '{os.path.basename(file_obj.name)}': {'✅ 处理成功' if success else '❌ 处理失败'} - {msg}")
            if not success:
                all_successful = False
        
        # 更新文件列表
        files_in_kb = list_kb_files(kb_name_current) or []
        final_message = "\n".join(results_log)
        if all_successful:
            gr.Info("所有文件处理完成！")
        else:
            gr.Warning("部分文件处理失败，请查看日志。")
        return gr.update(choices=files_in_kb, value=None), final_message

    # 事件绑定
    kb_list.change(
        fn=on_kb_select,
        inputs=kb_list,
        outputs=[current_kb, file_list, kb_delete_btn],
        show_progress="minimal"
    )
    
    kb_add_btn.click(
        fn=on_kb_add,
        inputs=new_kb_name,
        outputs=[kb_list, new_kb_name, status_box],
        show_progress="full"
    )
    
    kb_refresh_btn.click(
        fn=refresh_kb,
        inputs=[],
        outputs=kb_list,
        show_progress="minimal"
    ).then(
        lambda: gr.Info("知识库列表已刷新"), None, None
    )
    
    kb_delete_btn.click(
        fn=on_kb_delete,
        inputs=current_kb,
        outputs=[kb_list, file_list, current_kb, status_box],
        show_progress="full"
    )
    
    file_add_btn.click(
        fn=on_file_add,
        inputs=[file_upload, current_kb],
        outputs=[file_list, status_box],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch()


