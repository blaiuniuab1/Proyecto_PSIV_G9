import cv2

cap = cv2.VideoCapture("Clip 001.mp4")

# Vamos a saltar los primeros 60 frames (aprox 2 segundos) para evitar pantallas grises
for _ in range(60):
    ret, img = cap.read()

cap.release()

puntos = []

def click_event(event, x, y, flags, params):
    # Si haces click izquierdo, guarda y dibuja el punto
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append([x, y])
        print(f"Punto {len(puntos)} guardado: [{x}, {y}]")
        
        # Dibujar un circulito rojo donde hiciste click
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Calibracion', img)
        
        if len(puntos) == 4:
            print("\n¡Listo! Copia esto en tu código principal (nfl_tracker.py):")
            print(f"self.src_pts = np.array({puntos}, dtype=np.float32)")

if ret:
    print("Haz CLICK en 4 puntos que formen un rectángulo en el campo (ej: yardas 10 y 20).")
    print("Sigue este orden: 1.Arriba-Izq, 2.Arriba-Der, 3.Abajo-Izq, 4.Abajo-Der")
    cv2.imshow('Calibracion', img)
    cv2.setMouseCallback('Calibracion', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error al cargar el video.")