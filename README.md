# Detección de placas vehiculares en video

Proyecto final de Procesamiento de Imágenes y Video (CETYS Universidad).
Equipo: Emilio Fernández Pouget, Paul Cruz Gómez, Luis Mendivil Aguilera.

El pipeline procesa el video cuadro por cuadro: escala de grises → filtro gaussiano → CLAHE →
localización multiescala de la placa con morfología matemática (black-hat, cierre, apertura) y Otsu →
rectificación → binarización de Sauvola → apertura y limpieza de borde → componentes conexas
validadas con esqueletización → OCR (Tesseract) → seguimiento y votación entre cuadros.

El filtro gaussiano (`filtros.py`), la morfología (`morfologia.py`) y la binarización (`binarizacion.py`) están programados desde cero.

## Requisitos

```
pip install opencv-python numpy pillow pytesseract scikit-image
```
Tesseract 5 instalado en el sistema (Windows: instalador de UB Mannheim; macOS: `brew install tesseract`).

## Uso

```bash
python pruebas_vs_opencv.py                 # valida las funciones propias contra OpenCV
python generar_video_prueba.py              # crea video_prueba.mp4 (día, noche, lluvia)
python alpr_video.py --fuente video_prueba.mp4 --salida resultado.mp4 \
       --capturas capturas --cuadros_captura 20,75,110,170,200,260
python resumen.py registro.csv BCX482A 90   # exactitud por tramo

python alpr_video.py --fuente 0 --mostrar --cada 3   # cámara web en tiempo real
python alpr_video.py --fuente mi_video.mp4           # video propio
python alpr_video.py --fuente carpeta_fotos          # fotos: guarda resultado y etapas de cada una
```

Salidas: video anotado, `registro.csv` (cuadro, tiempo, lectura, confianza, lectura votada)
y, con `--capturas`, paneles con todas las etapas intermedias.
