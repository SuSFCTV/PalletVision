import os
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
onnx_model = YOLO("best.onnx")

BOX_COLOR = "red"
TEXT_COLOR = "white"
LABEL_BG_COLOR = "black"


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

    return img_pil


example_size = range(1, 4)

demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="pil"),
    examples=[f"good_{i}.jpg" for i in example_size]
    + [f"bad_{i}.jpg" for i in example_size],
    cache_examples=False,
    live=True,
)


if __name__ == "__main__":
    demo.launch()
