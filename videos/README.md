# Análisis Táctico de Fútbol Americano con IA

Este directorio contiene las herramientas de visión artificial desarrolladas para procesar y analizar jugadas de fútbol americano utilizando YOLOv8 y técnicas de homografía.

## Descripción de los Archivos

### Scripts de Python
* **sacar_puntos.py**: Herramienta de calibración para definir los 4 puntos de referencia en el campo y establecer la relación entre píxeles y yardas reales.
* **nfl_tracker.py**: Script principal que realiza la detección de jugadores, el mapeo de coordenadas en el campo y la clasificación automática de equipos por color.
* **analizador_tactico.py**: Herramienta de análisis interactivo. Permite pausar el vídeo para medir distancias manualmente con clics y realizar la detección automática del Quarterback y su compañero más cercano pulsando la tecla B.

### Archivos de Vídeo
* **Clip 001.mp4**: Vídeo original sin procesar utilizado como fuente de datos.
* **resultado_final.mp4**: Primera versión del procesamiento con detecciones básicas y visualización de coordenadas.
* **resultado_equipos.mp4**: Vídeo procesado final con la clasificación de equipos (Oscuro y Blanco) optimizada y estabilizada.

### Carpetas y Otros
* **fotos_analizador_tactico/**: Carpeta destinada a almacenar capturas de pantalla y resultados visuales obtenidos durante el análisis.
