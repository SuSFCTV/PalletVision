import json
import os
import random
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

def run_example(image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

def annotate_dataset(image_directory):
    # Проход по всем файлам в директории train, val, test
    for dataset_type in ['train', 'val', 'test']:
        image_dir = os.path.join(image_directory, dataset_type, 'images')
        label_dir = os.path.join(image_directory, dataset_type, 'labels')

        # Создаем директорию labels, если она не существует
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc=f"Processing {dataset_type} images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Полный путь к изображению
                    file_path = os.path.join(root, file)
                    image = Image.open(file_path)

                    # Выполняем детекцию объектов и аннотируем изображения
                    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
                    text_input = "The pallet. Damaged parts of pallet. Sticking out nails. Lost parts of pallet"  # Начальный текст

                    results = run_example(image, task_prompt, text_input)
                    bounding_boxes = results.get('bboxes', [])
                    labels = results.get('labels', [])

                    # Преобразуем bounding boxes и labels в необходимый формат
                    bboxes = {}
                    for label, bbox in zip(labels, bounding_boxes):
                        x1, y1, x2, y2 = bbox
                        bboxes[label] = {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }

                    # Создаем аннотацию для изображения
                    annotation = {
                        "is_damaged": 1 if any('damaged' in label for label in labels) else 0,
                        "bboxes": bboxes
                    }

                    # Сохраняем аннотацию в файл JSON в директории labels
                    annotation_filename = os.path.splitext(file)[0] + '.json'
                    annotation_path = os.path.join(label_dir, annotation_filename)
                    with open(annotation_path, 'w') as f:
                        json.dump(annotation, f, indent=4)

                    print(f"Annotation saved for {file} at {annotation_path}")

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Указываем директорию с данными
image_directory = "dataset"

annotate_dataset(image_directory)
