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

# @title Функции
def get_unique_labels(json_data):
    unique_labels = set()
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == 'label':
                unique_labels.add(value)
            elif isinstance(value, (dict, list)):
                unique_labels.update(get_unique_labels(value))
    elif isinstance(json_data, list):
        for item in json_data:
            unique_labels.update(get_unique_labels(item))
    return unique_labels

def process_json_files(directory):
    result = {}
    files_to_process = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                files_to_process.append(os.path.join(root, file))

    for file_path in tqdm(files_to_process, desc="Processing JSON files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                unique_labels = get_unique_labels(json_data)
                if unique_labels:
                    result[file_path] = list(unique_labels)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing file {file_path}: {e}")
    return result

def get_all_unique_labels(result):
    all_unique_labels = set()
    for labels in result.values():
        all_unique_labels.update(labels)
    return all_unique_labels

def get_filename_without_extension(file_path):
    # Получаем название файла с расширением
    filename_with_extension = os.path.basename(file_path)
    # Удаляем расширение файла
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

def find_files_with_labels(result, labels):
    matching_files = []
    labels_set = set(labels)
    for file_path, file_labels in result.items():
        if labels_set.issubset(file_labels):
            matching_files.append(file_path)
    return [get_filename_without_extension(i) for i in matching_files]

def find_files_in_directory(directory, filenames):
    result_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Получаем название файла без расширения
            file_without_extension = os.path.splitext(file)[0]
            if file_without_extension in filenames:
                result_paths.append(os.path.join(root, file))
    return result_paths


def get_random_file_paths(directory='/content/Mapillary Vistas/training/images', n=10):
    # Список для хранения всех путей до файлов в папке
    all_files = []

    # Проходимся по всем файлам в папке и добавляем их в список
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    # Если файлов меньше, чем запрашиваемое количество N, возвращаем все файлы
    if len(all_files) <= n:
        return all_files

    # Возвращаем N случайных путей до файлов
    return random.sample(all_files, n)

def show_images(image_paths, column_width=16):
    # Количество изображений
    num_images = len(image_paths)

    # Количество строк (по 2 изображения на строку)
    num_rows = ceil(num_images / 2)

    # Создаем фигуру с динамической высотой
    fig, axes = plt.subplots(num_rows, 2, figsize=(column_width, num_rows * column_width / 2))

    # Убираем оси для всех подграфиков
    for ax in axes.flatten():
        ax.axis('off')

    # Отображаем изображения
    for idx, image_path in enumerate(image_paths):
        # Открываем изображение
        img = mpimg.imread(image_path)

        # Вычисляем позицию подграфика
        row = idx // 2
        col = idx % 2

        # Отображаем изображение и подписываем его
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{idx}: {os.path.basename(image_path)}")

    plt.tight_layout()
    plt.show()

def filter_images(labels):
    filtered_results = find_files_with_labels(result, labels)
    file_paths = find_files_in_directory(directory, filtered_results)

    return file_paths