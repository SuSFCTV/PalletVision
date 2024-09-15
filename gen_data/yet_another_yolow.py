import random

import onnxruntime as ort
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
session = ort.InferenceSession('yolow-l3.onnx')

image = cv2.imread('1.jpg')
image = cv2.resize(image, (640, 640))  # Resize to the input dimension expected by the YOLO model
image = image.astype(np.float32) / 255.0  # Normalize the image
image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
image = np.expand_dims(image, axis=0)  # Add batch dimension

input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

outputs = session.run(output_names, {input_name: image})

output_image = cv2.imread('1.jpg')
output_image = cv2.resize(output_image, (640, 640))  # Resize to the input dimension expected by the YOLO model

print("outputs:", outputs)
# class_ids = outputs[0][0]  # Single class assuming one value
bboxes = outputs[1][0]
scores = outputs[2][0]
additional_info = outputs[3][0]  # Adjusted for your outputs' structure

score_threshold = 0.3
# print("class_ids:", class_ids)
print("bboxes:", bboxes)
print("scores:", scores)
print("additional_info:", additional_info)
classes = [
    "pallets", "pallet", "broken pallet", "broken pallets", "nails", "wood splits",
    "wood cracks", "plank cracks", "holes in wood", "asymmetric structure",
    "wood damage", "damage", "defected pallet"
]

# Defining a palette with high-contrast colors
colors = {
    "pallets": (255, 0, 0),  # Red
    "pallet": (0, 255, 0),  # Green
    "broken pallet": (0, 0, 255),  # Blue
    "broken pallets": (255, 255, 0),  # Cyan
    "nails": (255, 0, 255),  # Magenta
    "wood splits": (0, 255, 255),  # Yellow
    "wood cracks": (128, 0, 0),  # Maroon
    "plank cracks": (128, 128, 0),  # Olive
    "holes in wood": (0, 128, 0),  # Dark Green
    "asymmetric structure": (128, 0, 128),  # Purple
    "wood damage": (0, 128, 128),  # Teal
    "damage": (0, 0, 128),  # Navy
    "defected pallet": (128, 128, 128)  # Gray
}

for i in range(len(scores)):
    if scores[i] > score_threshold and additional_info[i] != -1:  # Adjusted the condition
        x_min, y_min, x_max, y_max = bboxes[i]
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))

        color = colors[classes[additional_info[i]]]
        cv2.rectangle(output_image, start_point, end_point, color, 2)

        label = f"{classes[additional_info[i]]}, {scores[i]:.2f}"
        cv2.putText(output_image, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Save the result image
cv2.imwrite('output.jpg', output_image)
print('Output image saved as output_with_detections.jpg')