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

i = 0

def process_images_from_queue(image_queue):
    global i
    while True:
        frame = image_queue.get()
        if frame is None:
            break

        preferred_height = 384
        square_images = False #Cambia esto si prefieres imágenes cuadradas
        output_path = "output"

        img = load_and_resize_image(frame, preferred_height, square_images)
        depth_map = process_image(img, model, transform, device)

        if depth_map is not None:
            depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map_normalized = (depth_map_normalized * 255).astype("uint8")
            depth_output_name = os.path.join(output_path, "{}.png".format(i))

            cv2.imwrite(depth_output_name, depth_map_normalized)

            # Visualizar la profundidad
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.title('Original Image')
            # plt.imshow(img)
            # plt.axis('off')

            # plt.subplot(1, 2, 2)
            # plt.title('Depth Map')
            # plt.imshow(depth_map, cmap='plasma')
            # plt.colorbar(label='Depth')
            # plt.ax
            i += 1