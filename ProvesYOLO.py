from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
results = model('fotos\Captura de pantalla 2026-05-05 085328.jpg', conf=0.1)
results[0].show()
