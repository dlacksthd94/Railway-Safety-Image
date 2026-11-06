import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import torch
from diffusers import QwenImageEditPlusPipeline, PipelineQuantizationConfig

class Augmentor:
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = self.load_model()
    
    def load_model(self):
        quantization_config = PipelineQuantizationConfig(quant_backend="bitsandbytes_8bit", quant_kwargs={"load_in_8bit": True,})
        pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16, device_map="balanced")
        print("pipeline loaded")
        pipeline.set_progress_bar_config(disable=None)
        return pipeline
    
    def save_image(self, image, fp):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        image.save(fp)

    def run_model(self, prompt, image_src, image_tgt, fp_image_output):
        inputs = {
            # "image": [image_src, image_tgt],
            "image": [image_src],
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }
        with torch.inference_mode():
            output = self.model(**inputs)
            output_image = output.images[0]
            self.save_image(output_image, fp_image_output)
    
    def to_sunlight(self, prompt, image_src, image_tgt, fp_image_output):
        self.run_model(prompt, image_src, image_tgt, fp_image_output)

    def to_shadow(self, prompt, image_src, image_tgt, fp_image_output):
        self.run_model(prompt, image_src, image_tgt, fp_image_output)
    
    def to_no_shadow(self, prompt, image_src, image_tgt, fp_image_output):
        self.run_model(prompt, image_src, image_tgt, fp_image_output)

    def to_night(self, prompt, image_src, image_tgt, fp_image_output):
        self.run_model(prompt, image_src, image_tgt, fp_image_output)

    def to_rainy(self, prompt, image_src, image_tgt, fp_image_output):
        self.run_model(prompt, image_src, image_tgt, fp_image_output)

def augment_image(cfg):
    aug = Augmentor(cfg)

    # df_image_preprocess = pd.read_csv(cfg.path.df_image_preprocess)
    df_image_seq = pd.read_csv(cfg.path.df_image_seq)
    df_image_seq = df_image_seq.sort_values(by=['crossing_id', 'seq_id', 'img_pos'])
    df_image_source = df_image_seq
    for idx, row in tqdm(df_image_source.iterrows(), total=len(df_image_source)):
        fn_image = f"{str(row['img_pos']).zfill(4)}_{row['img_id']}.jpg"
        fp_image = os.path.join(cfg.path.dir_image_seq, row['crossing_id'], row['seq_id'], fn_image)
        image = Image.open(fp_image)

        image_tgt_sunlight = Image.open(os.path.join(cfg.path.dir_reference_image, 'sunlight_1.jpg'))
        image_tgt_night = Image.open(os.path.join(cfg.path.dir_reference_image, 'night_1.jpg'))
        image_tgt_rainy = Image.open(os.path.join(cfg.path.dir_reference_image, 'rainy_1.jpg'))

        # prompt = f"Make the first image look like it is taken in strong sunlight condition, like the second image."
        prompt = f"Make the image look like it is taken in sunlight condition. The sunlight is so strong and large, hindering the driver's sight."
        fp_image_sunlight = os.path.join(cfg.path.dir_image_augmented, row['crossing_id'], row['seq_id'], 'sunlight', fn_image)
        aug.to_sunlight(prompt, image, image_tgt_sunlight, fp_image_sunlight)

        prompt = f"Make the image look like it is taken in shadow condition."
        fp_image_shadow = os.path.join(cfg.path.dir_image_augmented, row['crossing_id'], row['seq_id'], 'shadow', fn_image)
        aug.to_shadow(prompt, image, image_tgt_sunlight, fp_image_shadow)
        
        prompt = f"Make the image look like it is taken in no shadow condition."
        fp_image_no_shadow = os.path.join(cfg.path.dir_image_augmented, row['crossing_id'], row['seq_id'], 'no_shadow', fn_image)
        aug.to_no_shadow(prompt, image, image_tgt_sunlight, fp_image_no_shadow)

        prompt = f"Make the image look like it is taken at night. The FLS(Flashing Light Signal) must be turned off and the headlight of the cars (including from the driver's car) must be turned on."
        fp_image_night = os.path.join(cfg.path.dir_image_augmented, row['crossing_id'], row['seq_id'], 'night', fn_image)
        aug.to_night(prompt, image, image_tgt_night, fp_image_night)

        prompt = f"Make the image look like it is taken in rainy condition."
        fp_image_rainy = os.path.join(cfg.path.dir_image_augmented, row['crossing_id'], row['seq_id'], 'rainy', fn_image)
        aug.to_rainy(prompt, image, image_tgt_rainy, fp_image_rainy)
    