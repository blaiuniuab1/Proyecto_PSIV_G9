import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

class FieldHomography:
    def __init__(self):
        # Tus coordenadas de calibración
        self.src_pts = np.array([[159, 120], [226, 114], [568, 430], [721, 360]], dtype=np.float32)
        self.dst_pts = np.array([
            [10, 0],     
            [20, 0],     
            [10, 53.3],  
            [20, 53.3]   
        ], dtype=np.float32)

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

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Preparar el guardado del vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('resultado_equipos.mp4', fourcc, fps, (width, height))

    # Saltar primeros fotogramas grises
    for _ in range(5):
        ret, frame = cap.read()
        if not ret: return

    print("Cargando modelo YOLO...")
    model = YOLO('yolov8n.pt') 
    field = FieldHomography()
    print("¡Modelo cargado! Procesando clustering de equipos...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        terrain_mask = field.get_terrain_mask(frame)
        results = model(frame, verbose=False)[0]
        
        valid_players = []
        player_colors = []
        
        # 1. FILTRAR JUGADORES Y EXTRAER COLORES DEL PECHO
        for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) == 0: 
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) // 2
                
                # Comprobar que los pies tocan el césped
                if y2 < frame.shape[0] and terrain_mask[y2-5, x_center] == 255:
                    valid_players.append((x1, y1, x2, y2))
                    
                    # MEJORA 1: Mirar solo un pequeño cuadrado en el centro del pecho
                    y_chest = y1 + (y2 - y1) // 4  # Altura aproximada del pecho
                    chest_roi = frame[max(0, y_chest-10):y_chest+10, max(0, x_center-10):x_center+10]
                    
                    if chest_roi.size > 0:
                        avg_color = cv2.mean(chest_roi)[:3] 
                        player_colors.append(avg_color)
                    else:
                        player_colors.append((0, 0, 0))

        # 2. CLASIFICAR Y ESTABILIZAR (EVITAR EL PARPADEO)
        if len(valid_players) >= 2:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            teams = kmeans.fit_predict(player_colors)
            
            # MEJORA 2: Ordenar los clusters por luminosidad
            centers = kmeans.cluster_centers_
            brightness = [sum(c) for c in centers]
            
            # El equipo con menos brillo es el oscuro, el de más brillo es el claro
            if brightness[0] < brightness[1]:
                dark_team_id = 0
                light_team_id = 1
            else:
                dark_team_id = 1
                light_team_id = 0
            
            # 3. DIBUJAR RESULTADOS ESTABLES
            for i, (x1, y1, x2, y2) in enumerate(valid_players):
                team_id = teams[i]
                
                if team_id == dark_team_id:
                    box_color = (255, 50, 50)  # Azul para el equipo oscuro
                    team_label = "Oscuro"
                else:
                    box_color = (50, 255, 255) # Amarillo para el equipo blanco
                    team_label = "Blanco"
                    
                x_center = (x1 + x2) // 2
                real_x, real_y = field.image_to_real(x_center, y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                coord_text = f"{team_label} | {real_x:.1f}y"
                
                cv2.rectangle(frame, (x1, y1 - 15), (x1 + 100, y1), (0,0,0), -1)
                cv2.putText(frame, coord_text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

        out.write(frame)
        cv2.imshow("IA NFL Tracker - Equipos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    print("¡Finalizado! El vídeo se ha guardado como 'resultado_equipos.mp4'.")

if __name__ == "__main__":
    main("Clip 001.mp4")