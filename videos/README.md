# Anàlisi Tàctica de Futbol Americà amb IA

Aquest directori conté el codi i els recursos visuals d'un sistema de visió per computador dissenyat per analitzar partits de futbol americà mitjançant intel·ligència artificial (YOLOv8) i transformacions geomètriques (Homografia).

## Estructura del Projecte

### Scripts de Python (Codi Font)
* **sacar_puntos.py**: Eina de calibratge. Permet fer clic en 4 punts del vídeo per generar la matriu matemàtica que tradueix els píxels de la pantalla a iardes reals en el camp.
* **nfl_tracker.py**: L'script principal del sistema. Detecta els jugadors fent servir IA, els ubica en iardes i separa automàticament els dos equips mitjançant clustering (K-Means) analitzant els colors dels seus uniformes.
* **analizador_tactico.py**: Eina interactiva. Reprodueix el partit i permet pausar-lo (tecla 'P') per fer clics al camp i calcular en temps real la distància exacta als defensors o atacants més propers.

### Vídeos
* **Clip 001.mp4**: Vídeo original utilitzat com a entrada per a l'anàlisi.
* **resultado_final.mp4**: Primer vídeo exportat que mostra la detecció bàsica de jugadors i les seves coordenades (X, Y) mapejades sobre la gespa.
* **resultado_equipos.mp4**: Vídeo exportat avançat on la IA ja classifica correctament i en temps real els jugadors en Equip Fosc i Equip Blanc.

### Carpetes i Altres
* **fotos_analizador_tactico/**: Carpeta que conté les captures de pantalla generades durant l'ús de l'eina interactiva o proves visuals.
