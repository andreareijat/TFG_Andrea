# TFG_Andrea



PROYECTO DE CONTROL Y PROCESAMIENTO DE IMAGENES MONOCULARES CON DRON TELLO

	Este proyecto utiliza un dron Tello para capturar imágenes monoculares y procesarlas utilizando una red neuronal MiDaS para estimar la profundidad. El objetivo es emplear dicha información para generar un algoritmo de control que permita al dron moverse de manera autónoma. 



ESTADO

	En este momento, se realiza el movimiento del dron mediante teclado:

	    """ Maintains the Tello display and moves it through the keyboard keys.
		Press escape key to quit.
		The controls are:
		    - T: Takeoff
		    - L: Land
		    - Arrow keys: Forward, backward, left and right.
		    - A and D: Counter clockwise and clockwise rotations (yaw)
		    - W and S: Up and down.
	    """
	    
	Se crearon dos hilos con la libreria threading: 
		- Hilo para el movimiento del dron. 
		- Hilo para el procesado de la red neuronal. 
		
	De esta manera, al mismo tiempo que el dron se mueve guarda las imágenes que captura en una Queue. Paralelamente, el hilo de la red neuronal toma las imágenes de la cola y las procesa para obtener la profundidad. 


PENDIENTE

	- Comprobar varias frecuencias de muestreo de imágenes y lectura de la cola.
	
	
INFORMACIÓN


	Se emplea el repositorio de MiDaS y el de DJITelloPy para el control del dron. 
	- Dron: https://github.com/damiafuentes/DJITelloPy
	- Red: https://github.com/isl-org/MiDaS
	
