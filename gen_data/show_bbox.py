import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

names = [
    "pallets", "pallet", "broken pallet", "broken pallets", "nails", "wood splits",
    "wood cracks", "plank cracks", "holes in wood", "asymmetric structure",
    "wood damage", "damage", "defected pallet"
]

def load_annotations_str_label(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()

            # Предположим, что координаты - это 4 последних значения
            label_parts = parts[:-4]
            label = " ".join(label_parts)
            bbox = [float(coord) for coord in parts[-4:]]

            annotations.append((label, bbox))
    return annotations

def load_annotantion_int_label(annotation_filepath):
    annotations = []
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()

            # Предположим, что координаты - это 4 последних значения
            label_parts = parts[:-4]
            label = " ".join(label_parts)
            bbox = [float(coord) for coord in parts[-4:]]

            annotations.append((label, bbox))
    return annotations




def denormalize_bbox(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2


def visualize(image_path, annotation_path, output_path=os.path.join("dataset", "output.jpg"), label_is_str=False):
    # Загружаем изображение
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Загружаем аннотации
    if label_is_str:
        annotations = load_annotations_str_label(annotation_path)
    else:
        annotations = load_annotantion_int_label(annotation_path)

    image_width, image_height = image.size

    for label, bbox in annotations:
        # Декодируем bounding box
        x1, y1, x2, y2 = denormalize_bbox(bbox, image_width, image_height)

        # Рисуем bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Добавляем текст label
        if not label_is_str:
            draw.text((x1, y1), names[int(label)], fill="red")
        else:
            draw.text((x1, y1), label, fill="red")

    # Показать изображение
    #image.show()
    image.save("output1.png")
    print(f"Annotated image saved at output1.png")


if __name__ == "__main__":
    # Указываем путь к директории с изображениями и аннотациями
    image_directory = "dataset_jayaraju"
    dataset_type = 'train'

    # Указываем имя изображения (без расширения)
    #image_name = "2cb-1-jpeg_jpg.rf.9b446919a802d37a09622e33fb4ee8e2"
    image_name = "VID20231013114757_mp4-212_jpg.rf.533d3c5fb2c19e894a755751d0a5ba04"
    # Пути к изображению и аннотации
    image_path = os.path.join(image_directory, dataset_type, 'images', f"{image_name}.jpg")
    annotation_path = os.path.join(image_directory, dataset_type, 'labels', f"{image_name}.txt")

    # Вызываем функцию визуализации
    visualize(image_path, annotation_path)