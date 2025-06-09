import gradio as gr

from chatbox import (
    LANGUAGE_MAP, all_roles, sessions,knowledge_bases,respond,
    switch_session, create_new_session, delete_current_session_action,
    initialize_session_dropdown_options,
    load_initial_state, handle_role_selection_change_in_dropdown,
    save_or_update_role_action, delete_selected_role_action
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
        delete_role_in_management_btn: gr.Button(value=lang_map['delete_role_button_text']),
        search_checkbox: gr.Checkbox(label=lang_map['search_label']),
        kb_selector_dd: gr.Dropdown(label=lang_map['kb_label']), # Add kb_selector_dd
        use_kb_checkbox: gr.Checkbox(label=lang_map['use_kb_label']) # Add use_kb_checkbox
    }




with gr.Blocks(title="LLM Chat", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:

    current_session_id_state = gr.State("default")
    current_language_state = gr.State("中文")
    current_selected_role_name_state = gr.State(LANGUAGE_MAP["中文"]["no_role"])
    selected_kb_name_state = gr.State(None) # Add selected_kb_name_state
    use_kb_state = gr.State(False) # Add use_kb_state

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
                kb_selector_dd = gr.Dropdown( # Define kb_selector_dd
                    label=LANGUAGE_MAP["中文"]["kb_label"], 
                    choices=list(knowledge_bases.keys()), 
                    value=None, 
                    interactive=True,
                    elem_id="kb_selector_dropdown"
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
                with gr.Column(scale=3,min_width=150):
                    search_checkbox = gr.Checkbox(label=LANGUAGE_MAP["中文"]["search_label"], value=False, interactive=True, elem_id="search_checkbox")
                    use_kb_checkbox = gr.Checkbox(label=LANGUAGE_MAP["中文"]["use_kb_label"], value=False, interactive=True, elem_id="use_kb_checkbox") # Define use_kb_checkbox
                    submit_btn = gr.Button(
                        LANGUAGE_MAP["中文"]["send_button"],
                        variant="primary",
                    )

    # Event handling for mutual exclusivity of search_checkbox and use_kb_checkbox
    def toggle_search(use_kb_checked, current_lang):
        if use_kb_checked:
            gr.Info(LANGUAGE_MAP[current_lang]["kb_and_search_warning"])
            return gr.Checkbox.update(value=False, interactive=False)
        return gr.Checkbox.update(interactive=True)

    def toggle_kb_usage(search_checked, current_lang):
        if search_checked:
            gr.Info(LANGUAGE_MAP[current_lang]["kb_and_search_warning"])
            return gr.Checkbox.update(value=False, interactive=False)
        return gr.Checkbox.update(interactive=True)

    use_kb_checkbox.change(
        fn=toggle_search,
        inputs=[use_kb_checkbox, current_language_state],
        outputs=[search_checkbox]
    ).then(lambda val: val, inputs=use_kb_checkbox, outputs=use_kb_state)

    search_checkbox.change(
        fn=toggle_kb_usage,
        inputs=[search_checkbox, current_language_state],
        outputs=[use_kb_checkbox]
    )
    
    kb_selector_dd.change(lambda val: val, inputs=kb_selector_dd, outputs=selected_kb_name_state)

    language_selector_rd.change(
        fn=update_ui_languages,
        inputs=language_selector_rd,
        outputs=[title_md, user_input_tb, session_dropdown, model_selector_dd, 
                 new_session_btn, submit_btn, delete_session_btn,
                 role_selector_dd, manage_roles_accordion, role_name_input_tb,
                 role_prompt_input_tb, save_role_btn, delete_role_in_management_btn, search_checkbox,
                 kb_selector_dd, use_kb_checkbox]
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
                 role_selector_dd, current_selected_role_name_state, role_prompt_input_tb, role_name_input_tb,
                 kb_selector_dd, selected_kb_name_state, use_kb_checkbox] # Add kb outputs
    )

    session_dropdown.change(
        fn=switch_session,
        inputs=[session_dropdown, current_language_state],
        outputs=[chatbot_display_component, model_selector_dd, current_session_id_state, role_selector_dd, kb_selector_dd, use_kb_checkbox]
    ).then(
        lambda role_name_from_session: (all_roles.get(role_name_from_session, ""), role_name_from_session, role_name_from_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    ).then(
        lambda kb_name_from_session: kb_name_from_session, # Update selected_kb_name_state when session changes
        inputs=kb_selector_dd, 
        outputs=selected_kb_name_state
    ).then(
        lambda use_kb_from_session: use_kb_from_session, # Update use_kb_state when session changes
        inputs=use_kb_checkbox,
        outputs=use_kb_state
    )

    new_session_btn.click(
        fn=create_new_session,
        inputs=[current_language_state, current_selected_role_name_state, selected_kb_name_state, use_kb_state], # Add KB states as inputs
        outputs=[session_dropdown, current_session_id_state, chatbot_display_component, model_selector_dd, role_selector_dd, kb_selector_dd, use_kb_checkbox]
    ).then(
        lambda role_name_from_new_session: (all_roles.get(role_name_from_new_session, ""), role_name_from_new_session, role_name_from_new_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    )
    
    delete_session_btn.click(
        fn=delete_current_session_action,
        inputs=[current_session_id_state, current_language_state],
        outputs=[session_dropdown, current_session_id_state, chatbot_display_component, model_selector_dd, role_selector_dd, kb_selector_dd, use_kb_checkbox, delete_session_btn]
    ).then(
        lambda role_name_from_default_session: (all_roles.get(role_name_from_default_session, ""), role_name_from_default_session, role_name_from_default_session),
        inputs=role_selector_dd,
        outputs=[role_prompt_input_tb, role_name_input_tb, current_selected_role_name_state]
    )

    user_input_tb.submit(
        fn=respond,
        inputs=[user_input_tb, chatbot_display_component, current_session_id_state, model_selector_dd, current_selected_role_name_state, current_language_state, search_checkbox, selected_kb_name_state, use_kb_state],
        outputs=[user_input_tb, chatbot_display_component, model_selector_dd, current_session_id_state, current_selected_role_name_state]
    )
    submit_btn.click(
        fn=respond,
        inputs=[user_input_tb, chatbot_display_component, current_session_id_state, model_selector_dd, current_selected_role_name_state, current_language_state, search_checkbox, selected_kb_name_state, use_kb_state],
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
    demo.launch(share = True)