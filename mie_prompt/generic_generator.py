from mie_prompt.prompts.loader import load_prompt_dict, load_prompt_text

HYVIDEO_T2V_SYSTEM_PROMPT = load_prompt_text("hunyuan/t2v")

HYVIDEO_I2V_SYSTEM_PROMPT = load_prompt_text("hunyuan/i2v")

ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE = load_prompt_text("zimage/t2i")

FLUX2_T2I_SYSTEM_PROMPT = load_prompt_text("flux2/t2i")

LTX2_SYSTEM_PROMPT = load_prompt_text("ltx2/system")

KONTEXT_PRESETS = load_prompt_dict("kontext/presets")

FRAME_TRANSITION_SYSTEM_PROMPTS = load_prompt_dict("frame_transition/system_prompts")

FLUX_KLEIN_T2V_SYSTEM_PROMPT = load_prompt_text("flux_klein/t2v")

def generate_prompt(llm_service_connector, input_text, mode, seed=None):
    if not input_text.strip():
        if mode == 'advanced':
            system_msg = load_prompt_text('prompt_generator/random_advanced')
        else:
            system_msg = load_prompt_text('prompt_generator/random_simple')
        messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': 'Generate a random prompt.'}]
    elif mode == 'simple':
        system_msg = load_prompt_text('prompt_generator/translate_simple')
        messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': input_text}]
    else:
        system_msg = load_prompt_text('prompt_generator/expand_advanced')
        messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': input_text}]
    prompt = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (prompt.strip(),)

def generate_kontext_prompt(llm_service_connector, image1_description, image2_description, edit_instruction, preset, seed=None, *, presets=None):
    all_presets = presets if presets is not None else KONTEXT_PRESETS
    preset_data = all_presets.get(preset)
    if not preset_data:
        raise ValueError(f'Unknown preset: {preset}')
    user_content = ''
    if image1_description.strip():
        user_content += f'Image 1 (person) description: {image1_description.strip()}\n'
    if image2_description.strip():
        user_content += f'Image 2 (clothing) description: {image2_description.strip()}\n'
    if edit_instruction.strip():
        user_content += f'Edit instruction: {edit_instruction.strip()}'
    if not user_content.strip():
        user_content = 'No additional image description or edit instruction provided.'
    messages = [{'role': 'system', 'content': preset_data['system']}, {'role': 'user', 'content': user_content}]
    kontext_prompt = llm_service_connector.invoke(messages)
    return (kontext_prompt.strip(),)

def generate_transition_prompt(llm_service_connector, start_image_description, end_image_description, model, seed=None):
    system_prompt = FRAME_TRANSITION_SYSTEM_PROMPTS.get(model, FRAME_TRANSITION_SYSTEM_PROMPTS['wan'])
    user_content = f'Start frame description: {start_image_description.strip()}\nEnd frame description: {end_image_description.strip()}'
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)

def generate_hyvideo_t2v_prompt(llm_service_connector, input_text, seed=None):
    system_msg = HYVIDEO_T2V_SYSTEM_PROMPT
    user_msg = input_text.strip() or 'Generate a random cinematic text-to-video prompt.'
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)

def generate_hyvideo_i2v_prompt(llm_service_connector, image_description, input_text, seed=None):
    combined = ''.join(['参考图像描述：' + image_description.strip() if image_description.strip() else '', '\n文本指令：' + input_text.strip() if input_text.strip() else '']).strip()
    system_msg = HYVIDEO_I2V_SYSTEM_PROMPT.replace('{}', combined)
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': combined or '请根据系统提示生成结果。'}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)

def generate_zimage_prompt(llm_service_connector, prompt, seed=None):
    system_msg = ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE.replace('{prompt}', prompt.strip() or '')
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': prompt.strip() or '请根据系统提示生成结果。'}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)

def generate_zimage_prompt_with_image(llm_service_connector, image, prompt, seed=None, image_detail='auto'):
    system_msg = ZIMAGE_T2I_SYSTEM_PROMPT_TEMPLATE.replace('{prompt}', (prompt or '').strip()) + '\n参考图像是主要信息源。若提供了文本指令，仅作为辅助约束，最终输出以图片内容为主。'
    url = image
    parts = []
    if url:
        parts.append({'type': 'image_url', 'image_url': {'url': url, 'detail': image_detail}})
    if isinstance(prompt, str) and prompt.strip():
        parts.append({'type': 'text', 'text': prompt.strip()})
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': parts if parts else [{'type': 'text', 'text': ''}]}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)

def generate_flux2_prompt(llm_service_connector, input_text, seed=None):
    system_msg = FLUX2_T2I_SYSTEM_PROMPT
    user_msg = input_text.strip()
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.7, top_p=0.9)
    return (out.strip(),)

def generate_flux_klein_t2v_prompt(llm_service_connector, input_text, seed=None):
    system_msg = FLUX_KLEIN_T2V_SYSTEM_PROMPT
    user_msg = input_text.strip() or 'Generate a random high-quality prompt for FLUX.2 [klein].'
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.7, top_p=0.9)
    return (out.strip(),)

def generate_ltx2_prompt(llm_service_connector, input_text, seed=None):
    system_msg = LTX2_SYSTEM_PROMPT
    user_msg = input_text.strip() or 'Generate a random cinematic video prompt.'
    messages = [{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}]
    out = llm_service_connector.invoke(messages, seed=seed, temperature=0.8, top_p=0.9)
    return (out.strip(),)
