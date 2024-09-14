import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import numpy as np
import cv2
from gradio.themes.base import Base
from typing import Iterable
from gradio.themes.utils import fonts, sizes
import matplotlib.pyplot as plt
from style import Seafoam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# current model
session = ort.InferenceSession("weights/fpn-rcnn.onnx")

BOX_COLOR = "red"
TEXT_COLOR = "white"
LABEL_BG_COLOR = "black"

classes = [
    "not broken pallet",
    "clean pallet",
    "nails",
    "damaged wood splits",
    "damaged wood cracks",
    "damaged plank cracks",
    "holes in wood",
    "asymmetric structure",
    "wood damage",
]


def generate_colors(n):
    cmap = plt.get_cmap("tab10")
    return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]


colors = {cls: color for cls, color in zip(classes, generate_colors(len(classes)))}

seafoam = Seafoam()


def detect_objects(image):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_resized = cv2.resize(image, (640, 640))
    image_resized = image_resized.astype(np.float32) / 255.0
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized = np.expand_dims(image_resized, axis=0)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: image_resized})

    bboxes = outputs[1][0]
    scores = outputs[2][0]
    additional_info = outputs[3][0]

    box_threshold = 0.3
    score_threshold = 0.43

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default()

    good_boxes = []
    bad_boxes = []
    not_very_bad = True

    for i in range(len(scores)):
        if scores[i] > box_threshold and additional_info[i] != -1:
            if additional_info[i] < len(classes):
                x_min, y_min, x_max, y_max = bboxes[i]
                class_name = classes[additional_info[i]]

                if class_name in ["not broken pallet", "clean pallet"] and scores[i] > score_threshold:
                    good_boxes.append((i, scores[i], (x_min, y_min, x_max, y_max)))
                else:
                    if scores[i] > 0.7:
                        not_very_bad = False
                    bad_boxes.append((i, scores[i], (x_min, y_min, x_max, y_max)))
            else:
                print(
                    f"Warning: Detected object with invalid class index {additional_info[i]}. Skipping."
                )

    if good_boxes and not_very_bad and len(bad_boxes) < 4:
        best_box = max(good_boxes, key=lambda x: x[1])
        label = "Good"
    else:
        best_box = max(bad_boxes, key=lambda x: x[1])
        label = "Bad"

    i, score, (x_min, y_min, x_max, y_max) = best_box
    start_point = (int(x_min), int(y_min))
    end_point = (int(x_max), int(y_max))

    class_name = classes[additional_info[i]]
    color = colors.get(classes[additional_info[i]], (255, 255, 255))
    draw.rectangle([start_point, end_point], outline=color, width=5)

    label_text = f"{label}: {class_name} {score:.2f}"
    text_bbox = draw.textbbox(start_point, label_text, font=font)
    text_background = [
        text_bbox[0],
        text_bbox[1],
        text_bbox[2] + 5,
        text_bbox[3],
    ]
    draw.rectangle(text_background, fill=LABEL_BG_COLOR)
    draw.text(
        (text_bbox[0] + 2, text_bbox[1]),
        label_text,
        fill=TEXT_COLOR,
        font=font,
    )

    text_output = f"{label}: {class_name} {score:.2f}"
    image_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return Image.fromarray(image_result), text_output


image_dir = "images/"

example_files = [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="pil"), gr.Textbox(label="Detected Objects")],
    examples=example_files,
    cache_examples=False,
    live=True,
    theme=seafoam,
)

if __name__ == "__main__":
    demo.launch(share=True)
