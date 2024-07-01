import queue
import time
import pygame
from tello_control import FrontEnd
from image_processing import process_image, load_and_resize_image
from MIDAS.MiDaS.midas.model_loader import load_model
import os
import cv2
import torch
import numpy as np

# Cargar el modelo MiDaS
model_type = "dpt_beit_large_512"
model_path = "./MIDAS/MiDaS/weights/dpt_beit_large_512.pt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, transform, net_w, net_h = load_model(device, model_path, model_type)


def main():
    frontend = FrontEnd()

    frontend.tello.connect()
    frontend.tello.set_speed(frontend.speed)
    frontend.tello.streamoff()
    frontend.tello.streamon()

    frame_read = frontend.tello.get_frame_read()
    should_stop = False

    while not should_stop:
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT + 1:
                frontend.update()
            elif event.type == pygame.QUIT:
                should_stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    should_stop = True
                else:
                    frontend.keydown(event.key)
            elif event.type == pygame.KEYUP:
                frontend.keyup(event.key)

        if frame_read.stopped:
            break

        frontend.screen.fill([0, 0, 0])
        frame = frame_read.frame

        current_time = time.time()
        if current_time - frontend.last_process_time >= 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frontend.save_image(frame)
            frontend.last_process_time = current_time

            #Procesar la imagen
            preferred_height = 384
            square_images = False
            output_path = "output"
            img = load_and_resize_image(frame, preferred_height, square_images)
            depth_map = process_image(img, model, transform, device)

            if depth_map is not None:
                depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map_normalized = (depth_map_normalized * 255).astype("uint8")
                depth_output_name = os.path.join(output_path, "depth_{}.png".format(frontend.image_counter))
                cv2.imwrite(depth_output_name, depth_map_normalized)

        # text = "Battery: {}%".format(frontend.tello.get_battery())
        # cv2.putText(frame, text, (5, 720 - 5),
        #                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # frame = np.rot90(frame)
        # frame = np.flipud(frame)
        # frame = pygame.surfarray.make_surface(frame)
        # frontend.screen.blit(frame, (0, 0))
        pygame.display.update()
        time.sleep(1 / frontend.FPS)

    frontend.tello.end()

if __name__ == '__main__':
    main()
