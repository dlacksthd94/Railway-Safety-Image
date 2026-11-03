from PIL import Image
# image_path = 'data/FRA F 6180.57 (Form 57) form only (cropped).jpg'
image_path = 'data/FRA F 6180.57 (Form 57) form only.jpg'
image = Image.open(image_path).convert("RGB")

generation_config_base = {'max_new_tokens': 4096}
generation_config_sample = {**generation_config_base, 'do_sample': True}#, 'temperature': 1, 'top_p': 0.95} # sample or beam sample
generation_config_search = {**generation_config_base, 'do_sample': False} # greedy search or beam search

# ############### LayoutLMv3
# model_path = 'microsoft/layoutlmv3-large'

# from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
# from transformers import DonutProcessor, VisionEncoderDecoderModel
# import torch

# question = "Transcribe all fields"

# processor = LayoutLMv3Processor.from_pretrained(
#     model_path,
#     apply_ocr=True         # uses pytesseract to get words + boxes
# )

# encoding = processor(
#     images=image,
#     text=question,
#     return_tensors="pt",
#     truncation=True,
#     max_length=512,
#     padding="max_length"
# )  # includes input_ids (question+words), bbox, pixel_values, attention_mask
# processor.tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=True)

# model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_path)
# outputs = model(**encoding)

# start, end = outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1)
# # Map predicted token span back to string:
# answer = processor.tokenizer.decode(
#     encoding["input_ids"][0][start:end+1], skip_special_tokens=True
# )
# print(answer)

# ############### Donut
# # ft_ver = 'zhtrainticket'
# # ft_ver = 'docvqa'
# ft_ver = 'cord-v2'
# model_path = f'naver-clova-ix/donut-base-finetuned-{ft_ver}'

# import re
# from transformers import DonutProcessor, VisionEncoderDecoderModel
# from datasets import load_dataset
# import torch

# processor = DonutProcessor.from_pretrained(model_path)
# model = VisionEncoderDecoderModel.from_pretrained(model_path)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)  # doctest: +IGNORE_RESULT

# # load document image
# dataset = load_dataset("hf-internal-testing/example-documents", split="test")
# image = dataset[2]["image"]

# # prepare decoder inputs
# ft_ver = 'zhtrainticket'
# # ft_ver = 'docvqa'
# # ft_ver = 'cord-v2'
# task_prompt =f"<s_{ft_ver}>"
# decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

# pixel_values = processor(image, return_tensors="pt").pixel_values

# outputs = model.generate(
#     pixel_values.to(device),
#     decoder_input_ids=decoder_input_ids.to(device),
#     max_length=model.decoder.config.max_position_embeddings,
#     pad_token_id=processor.tokenizer.pad_token_id,
#     eos_token_id=processor.tokenizer.eos_token_id,
#     use_cache=True,
#     bad_words_ids=[[processor.tokenizer.unk_token_id]],
#     return_dict_in_generate=True,
# )

# sequence = processor.batch_decode(outputs.sequences)[0]
# sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
# sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
# print(processor.token2json(sequence))

# ############### 
# model_path = 'google/pix2struct-docvqa-large'
# # model_path = 'google/pix2struct-infographics-vqa-large'

# import requests
# from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


# model = Pix2StructForConditionalGeneration.from_pretrained(model_path).to("cuda")
# processor = Pix2StructProcessor.from_pretrained(model_path)

# question = "What is the name of field 1?"

# inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")

# predictions = model.generate(**inputs)
# print(processor.decode(predictions[0], skip_special_tokens=True))

# ############### BLIP-2
# # model_path = 'Salesforce/blip2-opt-2.7b' # 4B
# # model_path = 'Salesforce/blip2-flan-t5-xl' # 4B
# # model_path = 'Salesforce/blip2-opt-6.7b-coco' # 8B
# model_path = 'Salesforce/blip2-flan-t5-xxl' # 12B

# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = Blip2Processor.from_pretrained(model_path)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     model_path,
#     # load_in_8bit=True,
#     device_map={"": 0},
#     # torch_dtype=torch.float16
# )  # doctest: +IGNORE_RESULT

# prompt = "Question: What is the name of entry 1. Answer:"
# # prompt = "Question: What is the title. Answer:"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda")#, dtype=torch.float16)
# generated_ids = model.generate(**inputs)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# print(generated_text)

# ############### InstructBLIP
# # model_path = 'Salesforce/instructblip-flan-t5-xl' # 4B
# # model_path = 'Salesforce/instructblip-vicuna-7b' # 8B
# model_path = 'Salesforce/instructblip-flan-t5-xxl' # 12B
# # model_path = 'Salesforce/instructblip-vicuna-13b' # 14B

# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# import torch
# from PIL import Image

# model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
# processor = InstructBlipProcessor.from_pretrained(model_path)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# _ = model.to(device)

# prompt = "Extract key-value"
# # prompt = "What is the title?"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
# outputs = model.generate(
#     **inputs,
#     # do_sample=False,
#     # num_beams=5,
#     # max_length=256,
#     # min_length=1,
#     # top_p=0.9,
#     # repetition_penalty=1.5,
#     # length_penalty=1.0,
#     # temperature=1,
# )
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)


# ############### replace hash_id with report key
# import pandas as pd
# import utils_scrape
# df_record = pd.read_csv(path_df_record)
# df_record = df_record[df_record['State Name'] == 'CALIFORNIA']
# df_record['hash_id'] = df_record.apply(utils_scrape.hash_row, axis=1)
# df_record['Date'] = pd.to_datetime(df_record['Date'])
# df_record['Report Key']

# df_record_news = pd.read_csv(path_df_record_news)
# df_record_news

# df_record_news_merge = pd.merge(df_record_news, df_record[['hash_id', 'Report Key']], left_on='incident_id', right_on='hash_id', how='left')
# df_record_news_merge = df_record_news_merge.drop(['incident_id', 'hash_id'], axis=1)
# df_record_news_merge = df_record_news_merge.rename(columns={'Report Key': 'report_key'})
# df_record_news_merge.to_csv(path_df_record_news, index=False)

############### OmniParser V2 setup and test
# git clone https://github.com/microsoft/OmniParser
# cd OmniParser
# conda create -n omni python==3.12 -y
# conda activate omni
# pip install -r requirements.txt
#    (needs huggingface-cli: pip install -U "huggingface_hub[cli]")
# for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do \
#   huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; \
# done
# mv weights/icon_caption weights/icon_caption_florence
# python gradio_demo.py
# jupyter notebook demo.ipynb

# import replicate

# model = replicate.models.get("microsoft/omniparser-v2")
# output = model.predict(image=image_path)
# print(output)


# ############### test parsing models
# from transformers import pipeline
# import utils
# from convert_report_to_json import select_generate_func

# api = 'Huggingface'
# # model_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
# model_path = 'Qwen/Qwen2.5-VL-72B-Instruct'
# # model_path = 'OpenGVLab/InternVL3_5-38B-HF'
# # model_path = 'google/gemma-3-27b-it'
# # model_path = 'llava-hf/llava-v1.6-34b-hf'

# dict_model_config = {
#     # 'Qwen/Qwen2.5-VL-7B-Instruct': {},
#     # 'Qwen/Qwen2.5-VL-32B-Instruct': {'load_in_4bit': True},
#     # 'Qwen/Qwen2.5-VL-32B-Instruct': {'load_in_8bit': True},
#     'Qwen/Qwen2.5-VL-32B-Instruct': {},
#     'Qwen/Qwen2.5-VL-72B-Instruct': {'load_in_8bit': True},
#     # 'OpenGVLab/InternVL3-38B-hf': {'load_in_8bit': True},
#     'OpenGVLab/InternVL3_5-38B-HF': {},
#     'google/gemma-3-27b-it': {},
#     'llava-hf/llava-v1.6-34b-hf': {},
# }
# quant_config = dict_model_config[model_path]
# device = 'auto'

# pipe = pipeline(model=model_path, device_map=device, model_kwargs=quant_config)
# generate_func = select_generate_func(api)

# # # field = '5. Date of Accident/Incident'
# # # field = '6. Time of Accident/Incident'
# # # field = '12. Highway Name or Number'
# # # field = '46. Highway-Rail Crossing Users'
# # field = '49. Railroad Employees'
# # # field = '52. Passengers on Train'
# # prompt = f"""Each field of the form has areas to write or mark the answers.
# # Identify every answer place associated with the "{field}" field."""
# # # prompt = f"""Break down every answer place required for the "{field}" field, based in the image."""
# # # prompt = f"""Briefly describe "{field}" field in the context of its adjecent fields."""
# # with utils.Timer(model_path):
# #     output = generate_func(pipe, model_path, [{"type": "image", "image": image}, {"type": "text", "text": prompt},], generation_config_search)
# #     print(output)

# # prompt = f"""Break down every answer place of "{field}" that is required to be written or marked.
# # Include choices only when it is provided in the form.
# # Strictly follow the JSON format:""" + \
# # """
# # ```
# # "answer_places": {
# #     "<answer place name>": {
# #         "type": "<free-text/digit/single choice/multiple choice>",
# #         "choices": {
# #             "<choice code>": "<choice name>",
# #         },
# #     },
# # }
# # ```
# # """
# # with utils.Timer(model_path):
# #     output = generate_func(pipe, model_path, [{"type": "image", "image": image}, {"type": "text", "text": prompt},], generation_config_search)
# #     print(output)

# prompt = """Identify the indices and names of all entries.
# For each entry, break down every answer place that is required to write or mark.
# Strictly follow the JSON format:
# ```json
# {
#     "<entry idx>": {
#         "name": "<entry name>",
#         "answer_places": [
#             "<answer place name>": {
#                 "type": "<free-text/digit/single choice/multiple choice>",
#                 "choices": {
#                     "<choice code>": "<choice name>",
#                 },
#             },
#         ]
#     },
# }
# ```
# """
# with utils.Timer(model_path):
#     output = generate_func(pipe, model_path, [{"type": "image", "image": image}, {"type": "text", "text": prompt},], generation_config_search)
#     print(output)

# with open('transcription.txt', 'w') as f:
#     f.write(str(output))

############### calculating transcription error rate

import numpy as np
array = np.array([1, 3, 3, 4]) # both
# array = np.array([3, 6, 7, 6]) # sample aggregation only
array = np.array([3, 3, 3, 3]) # human-centric schema only
array.mean()
array.std()

############### calculating acc + cov

import numpy as np

# intern - single
array = np.array([47.70, 45.90, 44.17, 38.33]) # acc
array.mean()
array.std()
array = np.array([77.05, 79.53, 77.18, 74.64]) # cov
array.mean()
array.std()

# intern - group
array = np.array([55.26, 56.59, 54.53, 51.47]) # acc
array.mean()
array.std()
array = np.array([81.92, 96.29, 82.38, 85.63]) # cov
array.mean()
array.std()

# intern - all
array = np.array([56.87, 59.47, 53.96, 53.36]) # acc
array.mean()
array.std()
array = np.array([73.32, 86.01, 72.40, 73.55]) # cov
array.mean()
array.std()

# qwen3 - single
array = np.array([55.58, 54.68, 55.44, 56.03]) # acc
array.mean()
array.std()
array = np.array([89.37, 88.80, 89.14, 88.97]) # cov
array.mean()
array.std()

# qwen3 - group
array = np.array([68.49, 66.82, 65.21, 67.34]) # acc
array.mean()
array.std()
array = np.array([93.85, 94.24, 95.03, 95.44]) # cov
array.mean()
array.std()

# qwen3 - all
array = np.array([70.65, 69.33, 70.93, 72.35]) # acc
array.mean()
array.std()
array = np.array([90.60, 89.99, 90.07, 91.13]) # cov
array.mean()
array.std()

# o4 - single
array = np.array([59.13, 55.86, 54.85, 52.34]) # acc
array.mean()
array.std()
array = np.array([88.14, 91.56, 89.65, 82.37]) # cov
array.mean()
array.std()

# o4 - group
array = np.array([67.92, 65.50, 63.92, 61.05]) # acc
array.mean()
array.std()
array = np.array([97.18, 91.61, 92.09, 77.01]) # cov
array.mean()
array.std()

# o4 - all
array = np.array([68.80, 69.39, 68.34, 61.25]) # acc
array.mean()
array.std()
array = np.array([92.61, 86.66, 86.07, 74.87]) # cov
array.mean()
array.std()


############### mapillary API
