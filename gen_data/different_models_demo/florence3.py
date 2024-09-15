import os
import torch
import PIL.Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from roboflow import Roboflow
import shutil

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
    #print(parsed_answer)
    return parsed_answer


def normalize_bbox(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def annotate_dataset(image_directory):
    # Проход по всем файлам в директории train, val, test
    for dataset_type in ['train', 'valid', 'test']:
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
                    text_input = "The pallet. Nails. Wood splits. Wooden cracks. Holes in wood"  #
                    # Начальный текст

                    results = run_example(image, task_prompt, text_input)
                    bounding_boxes = results[task_prompt].get('bboxes', [])
                    labels = results[task_prompt].get('labels', [])

                    # Получаем размеры изображения для нормализации
                    img_width, img_height = image.size

                    # Преобразуем bounding boxes и labels в необходимый формат
                    annotation_lines = []
                    #print("labels:", labels)
                    #print("bbox:", bounding_boxes)
                    for label, bbox in zip(labels, bounding_boxes):
                        x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)
                        annotation_line = f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        print("annot_line", annotation_line)
                        annotation_lines.append(annotation_line)

                    # Сохраняем аннотацию в текстовый файл в директории labels
                    annotation_filename = os.path.splitext(file)[0] + '.txt'
                    annotation_path = os.path.join(label_dir, annotation_filename)
                    with open(annotation_path, 'w') as f:
                        f.write("\n".join(annotation_lines))

                    #print(f"Annotation saved for {file} at {annotation_path}")


def resize_images(image_directory, target_size=(640, 640)):
    for dataset_type in ['train', 'valid', 'test']:
        image_dir = os.path.join(image_directory, dataset_type, 'images')

        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc=f"Resizing {dataset_type} images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Полный путь к изображению
                    file_path = os.path.join(root, file)
                    image = PIL.Image.open(file_path)

                    if image.size != target_size:
                        # Ресайз изображения
                        resized_image = image.resize(target_size, PIL.Image.Resampling.LANCZOS)

                        # Сохранение ресайзнутого изображения
                        resized_image.save(file_path)
                        print(f"Resized image saved for {file_path}")
                    else:
                        print(f"Image {file_path} already has the target size and was not resized.")


if __name__ == "__main__":
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Указываем директорию с данными
    image_directory = "dataset"

    rf = Roboflow(api_key="jcjaiBh7FcKpC7J7GmDG")

    dataset = rf.workspace("ranjitha-p-fv0dt").project("pallet_detection-qxb1d").version(2).download("yolov8",
                                                                                                     location=image_directory)

    print("dataset downloaded")
    for dataset_type in ['train', 'valid', 'test']:
        dir_path = os.path.join(image_directory, dataset_type, "labels")
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    # Запускаем аннотацию датасета
    annotate_dataset(image_directory)
    resize_images(image_directory)

