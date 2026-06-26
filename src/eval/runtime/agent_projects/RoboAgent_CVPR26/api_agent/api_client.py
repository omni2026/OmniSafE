import base64
import re
from openai import OpenAI


class APIClient:
    def __init__(self, model_name="gpt-5.5", base_url=None, api_key=None,
                 default_sys_prompt="You are a helpful assistant."):
        self.model_name = model_name
        self.default_sys_prompt = default_sys_prompt
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)

    def inference(self, image_paths=[], prompt="", sys_prompt=None,
                  max_new_tokens=4096, temperature=0.0, top_p=1.0,
                  log_file=None):
        if sys_prompt is None:
            sys_prompt = self.default_sys_prompt

        prompt_text = self._strip_image_tags(prompt)

        content = []
        for image_path in image_paths:
            b64 = self._encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high"
                }
            })
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output_text = response.choices[0].message.content

        if log_file:
            with open(log_file, "a") as f:
                for image_path in image_paths:
                    f.write(image_path + "\n")
                f.write(prompt_text)
                f.write("--------------------------------------------\n")
                f.write(output_text)
                f.write("\n\n=============================================\n\n")

        return output_text

    @staticmethod
    def _strip_image_tags(prompt):
        prompt = re.sub(r'<image>\s*', '', prompt)
        return prompt

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
