import numpy as np
from numpy import random
import cv2
import torch
from torchvision import transforms

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import  plot_one_box

WEIGHT = './best.pt'
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = attempt_load(WEIGHT, map_location=device)
if device == "cuda":
    model = model.half().to(device)
else:
    model = model.float().to(device)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def detectDoll(image):
    image_orig = image.copy()
    image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    if device == "cuda":
        image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    else:
        image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)
    with torch.no_grad():
        output = model(image)[0]
    output = non_max_suppression_kpt(output, 0.25, 0.65)[0]

    ## Draw label and confidence on the image
    output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_orig.shape).round()
    mconf = 0
    mlabel = 0
    label = 0
    for *xyxy, conf, cls in output:
        label = f'{names[int(cls)]} {conf:.2f}'
        if(conf > mconf):
            if(names[int(cls)] == "Carno"):
                label = 1
                mconf = conf
            elif(names[int(cls)] == "Melody"):
                label = 2
                mconf = conf
        # plot_one_box(xyxy, image_orig, label=label, color=colors[int(cls)], line_thickness=1)
    return label