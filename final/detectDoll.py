import numpy as np
from numpy import random
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import onnx
import onnxruntime as ort

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "best.onnx"
model = ort.InferenceSession(model_path)

input_name = model.get_inputs()[0].name
output_names = [output.name for output in model.get_outputs()]

names = ["Carno", "Melody", "Other"]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def detectDoll(frame):
    image_orig = frame.copy()
    image = letterbox(frame, (960, 720), stride=64, auto=True)[0]
    image = transforms.ToTensor()(image).unsqueeze(0).numpy().astype(np.float32)
    image = image.astype(np.float32)
    
    input_feed = {input_name: image}
    outputs = model.run(output_names, input_feed)

    output = outputs[0]
    output = non_max_suppression_kpt(output, 0.25, 0.65)[0]

    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    dec = 0
    mconf = 0.0
    
    detected_objects = []

    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        if conf > mconf:
            if names[int(cls)] == "Carno":
                dec = 1
                mconf = conf
            elif names[int(cls)] == "Melody":
                dec = 2
                mconf = conf
        plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)

        detected_objects.append(f'{names[int(cls)]} with confidence {conf:.2f}')
    
    print("Detected objects:", detected_objects)

    cv2.imwrite("test_output.jpg", image)
    return image_orig, dec

detectDoll(cv2.imread('test.jpg'))
