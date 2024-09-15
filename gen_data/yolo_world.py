import argparse

import onnxruntime as ort
import numpy as np
import cv2
import os

import PIL.Image
from roboflow import Roboflow
from tqdm import tqdm
import shutil

# Определяем список имен классов
names = [
    "pallets", "pallet", "broken pallet", "broken pallets", "nails", "wood splits",
    "wood cracks", "plank cracks", "holes in wood", "asymmetric structure",
    "wood damage", "damage", "defected pallet"
]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def normalize_bbox(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def run_example(image, session, input_name, output_names, score_threshold=0.2, target_size=(640, 640), iou_threshold=0.5):
    # Resize and normalize the image
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    outputs = session.run(output_names, {input_name: image})

    # class_ids = outputs[0][0]
    bboxes = outputs[1][0]
    scores = outputs[2][0]
    additional_info = outputs[3][0]

    detection_results = []

    for i in range(len(scores)):
        if scores[i] > score_threshold and additional_info[i] != -1:
            x_min, y_min, x_max, y_max = bboxes[i]
            bbox = [x_min, y_min, x_max, y_max]
            detection_results.append({
                'bbox': bbox,
                'score': scores[i],
                'label': int(additional_info[i])
            })

    # Combine boxes with high IoU
    final_bboxes = []
    final_labels = []

    for i, det in enumerate(detection_results):
        if det is None:
            continue
        best_bbox = det['bbox']
        best_score = det['score']
        best_label = det['label']

        for j, next_det in enumerate(detection_results[i + 1:], start=i + 1):
            if next_det is None:
                continue
            if iou(best_bbox, next_det['bbox']) > iou_threshold:
                if next_det['score'] > best_score:
                    best_bbox = next_det['bbox']
                    best_score = next_det['score']
                    best_label = next_det['label']
                detection_results[j] = None

        final_bboxes.append(best_bbox)
        final_labels.append(best_label)

    return final_bboxes, final_labels


def annotate_dataset(image_directory, session, input_name, output_names):
    for dataset_type in ['train', 'valid', 'test']:
        image_dir = os.path.join(image_directory, dataset_type, 'images')
        label_dir = os.path.join(image_directory, dataset_type, 'labels')

        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc=f"Processing {dataset_type} images"):
                good_pallet_flag = False
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    annotation_filename = os.path.splitext(file)[0] + '.txt'
                    annotation_path = os.path.join(label_dir, annotation_filename)
                    with open(annotation_path, 'r') as f:
                        orig_annotation = f.readline()

                    annot_list = orig_annotation.split()

                    if annot_list and annot_list[0] == '2':
                        good_pallet_flag = True


                    image = cv2.imread(file_path)

                    bounding_boxes, labels = run_example(image, session, input_name, output_names)

                    img_width, img_height = image.shape[1], image.shape[0]
                    annotation_lines = []

                    for label, bbox in zip(labels, bounding_boxes):
                        if label in {0, 1} and not good_pallet_flag:
                            label = 2
                        elif label in {2, 3} and good_pallet_flag:
                            label = 0
                        x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)
                        annotation_line = f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        annotation_lines.append(annotation_line)

                    with open(annotation_path, 'w') as f:
                        f.write("\n".join(annotation_lines))


def resize_images(image_directory, target_size=(640, 640)):
    for dataset_type in ['train', 'valid', 'test']:
        image_dir = os.path.join(image_directory, dataset_type, 'images')

        for root, _, files in os.walk(image_dir):
            for file in tqdm(files, desc=f"Resizing {dataset_type} images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    image = PIL.Image.open(file_path)

                    if image.size != target_size:
                        resized_image = image.resize(target_size, PIL.Image.Resampling.LANCZOS)
                        resized_image.save(file_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO inference and annotate dataset.")
    parser.add_argument('--model-path', type=str, default="yolow-l3.onnx", help="Path to the ONNX model.")
    parser.add_argument('--cuda', type=bool, default=True, help="Whether to use CUDA if available.")
    parser.add_argument('--dataset-dir', type=str, default="dataset",
                        help="Directory where the dataset will be downloaded and saved.")
    parser.add_argument('--cuda-device', type=str, default="0", help="CUDA device to use.")
    parser.add_argument('--workspace', type=str, default="ranjitha-p-fv0dt", help="Roboflow workspace name.")
    parser.add_argument('--project', type=str, default="pallet_detection-qxb1d", help="Roboflow project name.")
    parser.add_argument('--version', type=int, default=2, help="Roboflow dataset version number.")
    parser.add_argument('--format', type=str, default="yolov8", help="Format for downloading the dataset.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    session = ort.InferenceSession(args.model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    if os.path.exists(args.dataset_dir) and os.path.isdir(args.dataset_dir):
        shutil.rmtree(args.dataset_dir)

    rf = Roboflow(api_key="jcjaiBh7FcKpC7J7GmDG")
    dataset = rf.workspace(args.workspace).project(args.project).version(args.version).download(args.format,
                                                                                                location=args.dataset_dir)

    annotate_dataset(args.dataset_dir, session, input_name, output_names)
    resize_images(args.dataset_dir)


if __name__ == "__main__":
    main()