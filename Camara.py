import cv2
# Iniciar captura de video (webcam)
camara = cv2.VideoCapture(0)  # Usa '0' para webcam

# Cargar el clasificador preentrenado de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = camara.read()  # Leer un frame del video

    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    cara = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in cara:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 255, 10), 2)

    # Mostrar el video con los rectángulos dibujados
    cv2.imshow('Detección de Rostros', frame)
    
    # Cerrar ventana con la tecla 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
# Liberar recursos
camara.release()
cv2.destroyAllWindows()

