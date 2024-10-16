import cv2
# Cargar el clasificador Haar para la detección de rostros
cara = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ruta de la imagen
imagen = 'images/top-gun.jpg'

# Cargar la imagen desde el disco
img = cv2.imread(imagen)

# Verificar si la imagen se cargó correctamente
if img is None:
    print(f"Error: No se pudo cargar la imagen desde '{imagen}'. Verifica la ruta.")
else:
    # Convertir la imagen a escala de grises (necesario para la detección de rostros)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = cara.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un rectángulo alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar la imagen con los rostros detectados
    cv2.imshow('Detección de rostros', img)

    # Esperar a que se presione una tecla para cerrar la ventana
    cv2.waitKey(0)
    cv2.destroyAllWindows()

