# Análisis Táctico de Fútbol Americano con Inteligencia Artificial

Este repositorio contiene el código fuente y las demostraciones visuales de un sistema de visión por computador diseñado para extraer métricas tácticas avanzadas a partir de vídeos de partidos de fútbol americano. 

El proyecto combina modelos de detección de objetos (YOLOv8), algoritmos de agrupamiento no supervisado (K-Means) y transformaciones geométricas espaciales (Homografía) para convertir píxeles en datos de campo reales (yardas).

## Arquitectura del Proyecto

El sistema está dividido en tres módulos de procesamiento principales:

### 1. Scripts de Procesamiento y Análisis (Código Fuente)

* **sacar_puntos.py**: Módulo de calibración de cámara. Permite al usuario seleccionar manualmente 4 puntos de referencia en el fotograma inicial para calcular la matriz de homografía. Esto es fundamental para corregir la perspectiva de la cámara y traducir las coordenadas de pantalla (2D) a distancias reales en el campo (yardas).
* **nfl_tracker.py**: Módulo central de procesamiento por lotes (Batch Processing). Utiliza YOLOv8 para aislar a los jugadores y aplica K-Means sobre el área del pecho de cada detección para clasificarlos dinámicamente en dos equipos (Oscuro y Blanco), evitando el ruido visual del césped. Genera un vídeo de salida estabilizado.
* **analizador_tactico.py**: Herramienta de evaluación interactiva. Funciona como un reproductor de vídeo con capacidades analíticas en tiempo real:
  * Medición manual: Al pausar el vídeo, el usuario puede hacer clic en cualquier punto del campo para calcular la distancia euclidiana exacta al jugador más cercano.
  * Automatización táctica: Mediante la tecla 'B', el sistema escanea el campo, identifica automáticamente la posición del Quarterback (QB) basándose en coordenadas espaciales, y traza la distancia y línea de pase hacia su compañero de equipo más próximo.

### 2. Recursos Multimedia (Entradas y Salidas)

* **Clip 001.mp4**: Archivo de vídeo original de entrada (perspectiva All-22 o cámara táctica).
* **resultado_final.mp4**: Primera iteración del renderizado, demostrando la detección básica de entidades y el anclaje espacial de sus coordenadas.
* **resultado_equipos.mp4**: Renderizado final avanzado que demuestra la separación de equipos mediante clustering de color en tiempo real.

### 3. Documentación Visual (Pruebas de Concepto)

* **fotos_analizador_tactico_distancias/**: Directorio de capturas que demuestran el funcionamiento de la herramienta interactiva de medición manual de distancias mediante clics en el campo.
* **fotos_analizador_tactico_quarterback/**: Directorio de capturas que validan la función de detección algorítmica del Quarterback y el cálculo de la opción de pase más cercana.
