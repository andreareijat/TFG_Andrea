
from DJITelloPy.djitellopy import Tello
import cv2
import pygame
import os
import numpy as np
import time
import torch
import timm
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor
from MIDAS.MiDaS.midas.model_loader import load_model
import warnings
import threading
import queue
warnings.filterwarnings("ignore")

"""
Para separar el procesamiento del dron
del de la red neuronal, se empleand threads.
El dron trabaja en un hilo mientras la red
procesa imagenes en otro. 
Se usa una cola para enviar las imagenes 
capturadas por el dron al hilo de la red.
"""

i=0
# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120

# Load MiDaS model
model_type = "dpt_beit_large_512"
model_path = "./MIDAS/MiDaS/weights/dpt_beit_large_512.pt" 

# Load transforms to resize and normalize the image
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, transform, net_w, net_h = load_model(device, model_path, model_type)

image_queue = queue.Queue()


def resize_image(image, height, square=False):
    h, w = image.shape[:2]
    if square:
        new_size = max(h, w)
        top = (new_size - h) // 2
        bottom = new_size - h - top
        left = (new_size - w) // 2
        right = new_size - w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        image = cv2.resize(image, (height, height))
    else:
        new_width = int(w * (height / h))
        image = cv2.resize(image, (new_width, height))
    return image


def load_and_resize_image(image, height, square):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = resize_image(img, height, square)
    return img


def process_image(image, model, transform, device):

    img_input = transform({"image": image})["image"]
    
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    return prediction

def process_images_from_queue(frame):
    global i

    while True:
        frame = image_queue.get()
        if frame is None: 
            break

        preferred_height = 384  
        square_images = False  #Cambia esto si prefieres imágenes cuadradas
        output_path = "output"

        img = load_and_resize_image(frame, preferred_height, square_images)
        depth_map = process_image(img, model, transform, device)
        
        if depth_map is  not None:
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
            i+=1

class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.

    """

    def __init__(self):
        pygame.init()

        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)
        self.last_process_time = time.time()

    def run(self):

        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame

            current_time = time.time()
            if current_time - self.last_process_time >= 1: 
                self.last_process_time = current_time
                image_queue.put(frame)

            # battery n. 
            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello.
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

def main():
    frontend = FrontEnd()

    processing_thread = threading.Thread(target=process_images_from_queue)
    processing_thread.start()

    frontend.run()

    image_queue.put(None)
    processing_thread.join()


if __name__ == '__main__':
    main()