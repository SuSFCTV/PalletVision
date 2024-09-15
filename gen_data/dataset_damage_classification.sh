#!/bin/bash

# Убедитесь, что у вас есть права на выполнение этого скрипта: chmod +x run_annotation.sh

# Определите значения параметров для нового датасета
MODEL_PATH="yolow-l3.onnx"
CUDA_DEVICE="0"
DATASET_DIR="dataset_damage_classification"
WORKSPACE="jayaraju"
PROJECT="pallets-cwxmr"
VERSION=2
FORMAT="yolov8"
PYTHON_SCRIPT="yolo_world.py"

# Переменная указывающая на использование CUDA
USE_CUDA=true

# Запуск Python-скрипта с заранее определенными параметрами
python3 $PYTHON_SCRIPT \
    --model-path "$MODEL_PATH" \
    --cuda "$USE_CUDA" \
    --cuda-device "$CUDA_DEVICE" \
    --dataset-dir "$DATASET_DIR" \
    --workspace "$WORKSPACE" \
    --project "$PROJECT" \
    --version $VERSION \
    --format "$FORMAT"