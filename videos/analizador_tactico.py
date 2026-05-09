import cv2
import numpy as np
import math
from ultralytics import YOLO
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# VARIABLES GLOBALES PARA LA INTERACCIÓN DEL RATÓN
# ---------------------------------------------------------
pausado = False
frame_actual = None
frame_limpio = None
jugadores_en_pantalla = []
campo = None

# ---------------------------------------------------------
# CLASE DE HOMOGRAFÍA (Matemáticas del campo)
# ---------------------------------------------------------
class FieldHomography:
    def __init__(self):
        self.src_pts = np.array([[159, 120], [226, 114], [568, 430], [721, 360]], dtype=np.float32)
        self.dst_pts = np.array([[10, 0], [20, 0], [10, 53.3], [20, 53.3]], dtype=np.float32)
        self.H, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        self.hsv_low = np.array([43, 46, 81])
        self.hsv_high = np.array([57, 86, 178])

    def image_to_real(self, x, y):
        p = np.array([x, y, 1.0])
        p_real = np.dot(self.H, p)
        return p_real[0] / p_real[2], p_real[1] / p_real[2]

    def get_terrain_mask(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.hsv_low, self.hsv_high)
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.dilate(mask, np.ones((31, 31), dtype=np.uint8))
        _, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
        return mask

# ---------------------------------------------------------
# FUNCIÓN QUE SE EJECUTA CUANDO HACES CLIC
# ---------------------------------------------------------
def clic_raton(event, x, y, flags, param):
    global frame_actual, frame_limpio, jugadores_en_pantalla, campo, pausado

    # Si hacemos clic izquierdo o derecho
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        if not pausado:
            print("¡Primero pausa el vídeo pulsando la tecla 'p'!")
            return

        # Restauramos la imagen limpia para borrar líneas de clics anteriores
        frame_actual = frame_limpio.copy()

        # Convertir los píxeles del clic a yardas reales
        click_real_x, click_real_y = campo.image_to_real(x, y)

        # ¿Qué equipo buscamos? Clic Izq = Oscuro, Clic Der = Blanco
        equipo_buscado = "Oscuro" if event == cv2.EVENT_LBUTTONDOWN else "Blanco"
        
        distancia_minima = float('inf')
        jugador_cercano = None

        # Buscar en la lista de jugadores el más cercano
        for jug in jugadores_en_pantalla:
            j_x, j_y, j_team, j_box = jug
            if j_team == equipo_buscado:
                # Fórmula de distancia euclidiana
                dist = math.sqrt((j_x - click_real_x)**2 + (j_y - click_real_y)**2)
                if dist < distancia_minima:
                    distancia_minima = dist
                    jugador_cercano = jug

        # Si hemos encontrado a un jugador, lo dibujamos
        if jugador_cercano:
            c_x, c_y, c_team, c_box = jugador_cercano
            cx_img = (c_box[0] + c_box[2]) // 2
            cy_img = c_box[3]

            # Color de la línea según el equipo
            color_linea = (255, 50, 50) if equipo_buscado == "Oscuro" else (50, 255, 255)
            
            # Dibujar el punto del clic, el círculo en el jugador y la línea uniéndolos
            cv2.circle(frame_actual, (x, y), 6, (0, 0, 255), -1) 
            cv2.circle(frame_actual, (cx_img, cy_img), 12, (0, 255, 0), 3) 
            cv2.line(frame_actual, (x, y), (cx_img, cy_img), color_linea, 3)

            # Escribir el texto de la distancia
            texto = f"Distancia: {distancia_minima:.1f} yardas"
            cv2.putText(frame_actual, texto, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Analizador Interactivo", frame_actual)


# ---------------------------------------------------------
# BUCLE PRINCIPAL
# ---------------------------------------------------------
def main(video_path):
    global pausado, frame_actual, frame_limpio, jugadores_en_pantalla, campo

    cap = cv2.VideoCapture(video_path)
    for _ in range(5): cap.read() # Saltar frames grises

    print("Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt') 
    campo = FieldHomography()
    
    # Crear la ventana e indicarle que escuche al ratón
    cv2.namedWindow("Analizador Interactivo")
    cv2.setMouseCallback("Analizador Interactivo", clic_raton)

    print("\n" + "="*50)
    print("INSTRUCCIONES:")
    print(" - Pulsa 'p' para PAUSAR / REANUDAR el vídeo.")
    print(" - Con el vídeo pausado, haz CLIC IZQ para buscar equipo OSCURO.")
    print(" - Con el vídeo pausado, haz CLIC DER para buscar equipo BLANCO.")
    print(" - Pulsa 'q' para SALIR.")
    print("="*50 + "\n")

    while cap.isOpened():
        if not pausado:
            ret, frame = cap.read()
            if not ret: break

            terrain_mask = campo.get_terrain_mask(frame)
            results = model(frame, verbose=False)[0]
            
            valid_players = []
            player_colors = []
            jugadores_en_pantalla = [] # Reiniciamos la lista en cada frame
            
            # Detectar jugadores
            for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                if int(cls) == 0: 
                    x1, y1, x2, y2 = map(int, box)
                    x_center = (x1 + x2) // 2
                    
                    if y2 < frame.shape[0] and terrain_mask[y2-5, x_center] == 255:
                        valid_players.append((x1, y1, x2, y2))
                        y_chest = y1 + (y2 - y1) // 4  
                        chest_roi = frame[max(0, y_chest-10):y_chest+10, max(0, x_center-10):x_center+10]
                        if chest_roi.size > 0:
                            player_colors.append(cv2.mean(chest_roi)[:3])
                        else:
                            player_colors.append((0, 0, 0))

            # Clasificar equipos
            if len(valid_players) >= 2:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                teams = kmeans.fit_predict(player_colors)
                brightness = [sum(c) for c in kmeans.cluster_centers_]
                
                dark_team_id = 0 if brightness[0] < brightness[1] else 1
                
                for i, (x1, y1, x2, y2) in enumerate(valid_players):
                    team_id = teams[i]
                    team_label = "Oscuro" if team_id == dark_team_id else "Blanco"
                    box_color = (255, 50, 50) if team_label == "Oscuro" else (50, 255, 255)
                        
                    x_center = (x1 + x2) // 2
                    real_x, real_y = campo.image_to_real(x_center, y2)
                    
                    # Guardar en memoria para que el ratón pueda encontrarlos luego
                    jugadores_en_pantalla.append((real_x, real_y, team_label, (x1, y1, x2, y2)))
                    
                    # Dibujar
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.rectangle(frame, (x1, y1 - 15), (x1 + 100, y1), (0,0,0), -1)
                    cv2.putText(frame, f"{team_label} | {real_x:.1f}y", (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            # Guardamos copias del frame para que el ratón dibuje encima sin ensuciarlo permanentemente
            frame_limpio = frame.copy()
            frame_actual = frame.copy()

        cv2.imshow("Analizador Interactivo", frame_actual)
        
# Lógica de teclado
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado
            if pausado:
                print("Vídeo PAUSADO. Haz clic en el campo o pulsa 'B' para buscar al QB.")
            else:
                print("Reproduciendo...")
                
        # ---------------------------------------------------------
        # NUEVA FUNCIÓN: BUSCAR AL QB Y SU COMPAÑERO MÁS CERCANO
        # ---------------------------------------------------------
        elif tecla == ord('b') or tecla == ord('B'):
            if not pausado:
                print("¡Primero pausa el vídeo (P) para usar esta función!")
            else:
                frame_actual = frame_limpio.copy() # Limpiamos dibujos anteriores
                
                # 1. Filtrar solo a los jugadores del equipo atacante (Oscuro)
                jugadores_oscuros = [j for j in jugadores_en_pantalla if j[2] == "Oscuro"]
                
                if jugadores_oscuros:
                    # 2. El QB es el jugador con la X más pequeña (el más atrasado a la izquierda)
                    qb = min(jugadores_oscuros, key=lambda j: j[0])
                    
                    # 3. Buscar al compañero más cercano al QB
                    dist_minima = float('inf')
                    comp_cercano = None
                    
                    for jug in jugadores_oscuros:
                        if jug != qb: # No calcular la distancia del QB consigo mismo
                            # Fórmula de Pitágoras (Distancia)
                            dist = math.sqrt((jug[0] - qb[0])**2 + (jug[1] - qb[1])**2)
                            if dist < dist_minima:
                                dist_minima = dist
                                comp_cercano = jug
                    
                    # 4. Dibujar al QB (Círculo Verde)
                    qb_cx = (qb[3][0] + qb[3][2]) // 2
                    qb_cy = qb[3][3]
                    cv2.circle(frame_actual, (qb_cx, qb_cy), 18, (0, 255, 0), -1)
                    cv2.putText(frame_actual, "QB", (qb[3][0], qb[3][1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 5. Dibujar al compañero más cercano (Círculo y Línea Rosa)
                    if comp_cercano:
                        comp_cx = (comp_cercano[3][0] + comp_cercano[3][2]) // 2
                        comp_cy = comp_cercano[3][3]
                        
                        cv2.circle(frame_actual, (comp_cx, comp_cy), 15, (255, 0, 255), 3)
                        cv2.line(frame_actual, (qb_cx, qb_cy), (comp_cx, comp_cy), (255, 0, 255), 3)
                        
                        texto = f"Mas cercano: {dist_minima:.1f} yds"
                        cv2.putText(frame_actual, texto, (qb_cx + 20, qb_cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                cv2.imshow("Analizador Interactivo", frame_actual)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("Clip 001.mp4")
