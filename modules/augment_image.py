import os
import json
import base64

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from PIL import Image


class FeatureManipulation:
    dp_augment_image = 'data/augment_image'
    fp_image = os.path.join(dp_augment_image, "original.jpeg")
    fp_api_key = 'api_key.json'

    def __init__(self, model, instruction):
        self.model = model  # "qwen", "openai" or "gemini"
        self.instruction = instruction
        self.output_path = self.unique_path(os.path.join(self.dp_augment_image, f"remove_median({model})_1.png"))

    @staticmethod
    def unique_path(path):
        root, ext = os.path.splitext(path)
        name, n = root.rsplit("_", 1)
        assert n.isdigit(), "Filename must end with an underscore followed by a number"
        n = int(n)
        while os.path.exists(path):
            n += 1
            path = f"{name}_{n}{ext}"
        return path

    def api_key(self, name):
        with open(self.fp_api_key) as f:
            return json.load(f)[name]['key']

    def edit_qwen(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        image = Image.open(self.fp_image).convert("RGB")
        pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", dtype=torch.bfloat16, device_map="balanced")
        result = pipe(
            image=[image],
            prompt=self.instruction,
            negative_prompt=" ",
            num_inference_steps=40,
            true_cfg_scale=4.0,
            generator=torch.manual_seed(0),
        ).images[0]
        result.save(self.output_path)

    def edit_openai(self):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key('openai'))
        with open(self.fp_image, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        response = client.responses.create(
            model="gpt-6-astra",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": self.instruction},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ]}],
            tools=[{"type": "image_generation", "action": "edit"}],
        )
        image_calls = [o.result for o in response.output if o.type == "image_generation_call"]
        with open(self.output_path, 'wb') as f:
            f.write(base64.b64decode(image_calls[0]))

    def edit_gemini(self):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key('google'))
        response = client.models.generate_content(
            model="gemini-nano-banana-2.1",
            contents=[self.instruction, Image.open(self.fp_image).convert("RGB")],
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with open(self.output_path, 'wb') as f:
                    f.write(part.inline_data.data)
                return
        raise RuntimeError(f"No image returned: {response.text}")
        
    def run(self):
        {"qwen": self.edit_qwen, "openai": self.edit_openai, "gemini": self.edit_gemini}[self.model]()
        print("saved", self.output_path)


if __name__ == "__main__":
    model = 'gemini'
    instruction = "Remove the median and draw double yellow center lines as the shape of the original median."
    FeatureManipulation(model=model, instruction=instruction).run()
