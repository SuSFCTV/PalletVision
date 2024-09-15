from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import requests
import copy
import torch
import os
import json
import random
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

from utils import *
#%matplotlib inline


def annotate_dataset(image_directory, output_path):
    annotations = []

    # Проход по всем файлам в директории
    for root, _, files in os.walk(image_directory):
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Полный путь к изображению
                file_path = os.path.join(root, file)
                image = Image.open(file_path)

                # Выполняем детекцию объектов и аннотируем изображения
                task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
                text_input = "The pallet. Damaged parts of pallet. Sticking out nails. Lost parts of pallet"  #
                # Начальный текст

                results = run_example(image, task_prompt, text_input)
                bounding_boxes = results.get('bboxes', [])
                labels = results.get('labels', [])

                # Создаем аннотацию для каждого изображения
                annotation = {
                    "image_path": file_path,
                    "bounding_boxes": bounding_boxes,
                    "labels": labels
                }

                # Добавляем аннотацию к списку
                annotations.append(annotation)

    # Сохраняем аннотации в файл JSON
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)

    print(f"Annotations saved to {output_path}")

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'

# Указываем директорию с изображениями и путь для сохранения аннотаций
image_directory = "/path/to/your/image/directory"
output_path = "/path/to/output/annotations.json"

# Запускаем аннотацию датасета
annotate_dataset(image_directory, output_path)