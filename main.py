import threading
import queue
from tello_control import FrontEnd
from image_processing import process_images_from_queue

"""
Para separar el procesamiento del dron
del de la red neuronal, se empleand threads.
El dron trabaja en un hilo mientras la red
procesa imagenes en otro. 
Se usa una cola para enviar las imagenes 
capturadas por el dron al hilo de la red.
"""

#TODO: el procesamiento del run del MIDAS no debe ser igual que el que se hace 
#aqui ya que las imagenes de output estan en blanco y negro y no saca el fichero
#.map, mirar si son distintos

image_queue = queue.Queue()

def main():
    frontend = FrontEnd(image_queue)

    processing_thread = threading.Thread(target=process_images_from_queue, args=(image_queue,))
    processing_thread.start()

    frontend.run()

    image_queue.put(None)
    processing_thread.join()

if __name__ == '__main__':
    main()
