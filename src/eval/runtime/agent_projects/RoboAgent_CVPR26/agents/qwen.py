from qwen_vl_utils import process_vision_info

# @title inference function
def inference(processor, model, image_paths, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False, more_args={}, log_file=None):
    content = []
    for image_path in image_paths:
        image_local_path = "file://" + image_path
        content.append({"type": "image", "image": image_local_path})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("text:", text)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        padding=True, 
        return_tensors="pt")
    inputs = inputs.to('cuda')

    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **more_args)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    if log_file:
        with open(log_file, "a") as f:
            for image_path in image_paths:
                f.write(image_path + "\n")
            f.write(text)
            f.write("--------------------------------------------\n")
            f.write(output_text[0])
            f.write("\n\n=============================================\n\n")
    
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
