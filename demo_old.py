import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from gradio.themes.base import Base
from typing import Iterable
from gradio.themes.utils import colors, fonts, sizes
from style import Seafoam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
onnx_model = YOLO("weights/best.onnx")

BOX_COLOR = "red"
TEXT_COLOR = "white"
LABEL_BG_COLOR = "black"

seafoam = Seafoam()


def detect_objects(image):
    results = onnx_model(image)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        label_name = result.names
        for box in result.boxes:
            print(image)
            print(
                f"Class: {box.cls.item()}, Confidence: {box.conf.item()}, Box: {box.xyxy}"
            )
            xyxy = box.xyxy[0].cpu().numpy()
            label = label_name[box.cls.item()]
            conf = box.conf.item()

            draw.rectangle(
                [xyxy[0], xyxy[1], xyxy[2], xyxy[3]],
                outline=BOX_COLOR,
                width=10,
            )

            label_text = f"{label} {conf:.2f}"

            text_bbox = draw.textbbox((xyxy[0], xyxy[1]), label_text, font=font)
            text_background = [
                text_bbox[0],
                text_bbox[1],
                text_bbox[2] + 5,
                text_bbox[3],
            ]

            draw.rectangle(text_background, fill=LABEL_BG_COLOR, outline=10)

            draw.text(
                (text_bbox[0] + 2, text_bbox[1]), label_text, fill=TEXT_COLOR, font=font
            )

    return img_pil, "ауе"


image_dir = "images/"
example_size = range(1, 4)

example_files = [f"good_{i}.jpg" for i in example_size] + [
    f"bad_{i}.jpg" for i in example_size
]

example_files += [
    os.path.join(image_dir, file)
    for file in os.listdir(image_dir)
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

example_files = example_files[:10]

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
    demo.launch()
