import os
import cv2
import torch
from utils import load_and_resize_image, process_image
from MIDAS.MiDaS.midas.model_loader import load_model

# Load MiDaS model
model_type = "dpt_beit_large_512"
model_path = "./MIDAS/MiDaS/weights/dpt_beit_large_512.pt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, transform, net_w, net_h = load_model(device, model_path, model_type)


def process_images_from_queue(image_queue):
    pass
