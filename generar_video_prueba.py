"""
Genera un video SINTÉTICO de prueba (no es tráfico real) para validar el
pipeline mientras se graba el video real del equipo.

Tres tramos de 6 s cada uno, que corresponden a los objetivos específicos:
  1) día, auto acercándose      2) baja iluminación      3) lluvia
La placa es ficticia ("BCX-482-A") con formato parecido al de Baja California.

Uso:  python generar_video_prueba.py  [salida.mp4]
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ANCHO, ALTO, FPS, SEG = 1280, 720, 15, 6
TEXTO_PLACA = "BCX-482-A"
_CANDIDATAS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",   # Linux
    "C:/Windows/Fonts/ARIALNB.TTF",                                    # Windows (Arial Narrow Bold)
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",        # macOS
    "/Library/Fonts/Arial Bold.ttf",
]
FUENTE = next((p for p in _CANDIDATAS if os.path.exists(p)), None)
if FUENTE is None:
    raise SystemExit("No se encontró una fuente TrueType; agrega una ruta a _CANDIDATAS.")
rng = np.random.default_rng(7)


def dibujar_placa(w=600, h=300):
    """Placa 2:1 (30 x 15 cm): fondo blanco, estado arriba, número al centro."""
    im = Image.new("RGB", (w, h), (235, 235, 230))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4, 4, w - 5, h - 5], radius=22, outline=(40, 40, 40), width=7)
    d.rectangle([20, 20, w - 21, 70], fill=(150, 30, 40))
    f1 = ImageFont.truetype(FUENTE, 38)
    d.text((w // 2, 45), "BAJA CALIFORNIA", font=f1, fill=(245, 245, 245), anchor="mm")
    f2 = ImageFont.truetype(FUENTE, 96)
    d.text((w // 2, 168), TEXTO_PLACA, font=f2, fill=(20, 20, 25), anchor="mm")
    f3 = ImageFont.truetype(FUENTE, 30)
    d.text((w // 2, 262), "MEXICO   FRONTERA", font=f3, fill=(60, 60, 60), anchor="mm")
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def fondo():
    """Calle con banqueta, edificios y líneas de carril (distractores)."""
    f = np.zeros((ALTO, ANCHO, 3), np.uint8)
    for y in range(ALTO):                       # cielo -> asfalto
        f[y] = (200 - y // 8, 190 - y // 8, 180 - y // 8) if y < 260 else (95, 95, 100)
    for x0 in range(0, ANCHO, 180):             # edificios con ventanas
        hb = int(rng.integers(80, 200))
        col = tuple(int(c) for c in rng.integers(90, 170, 3))
        cv2.rectangle(f, (x0, 260 - hb), (x0 + 160, 260), col, -1)
        for wy in range(260 - hb + 15, 250, 35):
            for wx in range(x0 + 15, x0 + 150, 40):
                cv2.rectangle(f, (wx, wy), (wx + 22, wy + 18), (220, 220, 200), -1)
    cv2.rectangle(f, (0, 260), (ANCHO, 300), (150, 150, 150), -1)  # banqueta
    for x0 in range(0, ANCHO, 160):
        cv2.rectangle(f, (x0, 640), (x0 + 90, 655), (230, 230, 230), -1)
    cv2.putText(f, "ALTO", (1080, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    return f


def dibujar_auto(frame, cx, cy, escala, placa):
    w, h = int(520 * escala), int(300 * escala)
    x0, y0 = cx - w // 2, cy - h // 2
    cv2.rectangle(frame, (x0, y0 + h // 4), (x0 + w, y0 + h), (40, 40, 140), -1)  # carrocería
    cv2.rectangle(frame, (x0 + w // 8, y0 - h // 6), (x0 + 7 * w // 8, y0 + h // 4), (35, 35, 120), -1)
    cv2.rectangle(frame, (x0 + w // 6, y0 - h // 10), (x0 + 5 * w // 6, y0 + h // 5), (90, 70, 60), -1)  # vidrio
    for lx in (x0 + w // 12, x0 + 9 * w // 12):  # calaveras
        cv2.rectangle(frame, (lx, y0 + h // 3), (lx + w // 6, y0 + h // 2), (30, 30, 220), -1)
    cv2.rectangle(frame, (x0 - 5, y0 + 4 * h // 5), (x0 + w + 5, y0 + h + 8), (30, 30, 30), -1)  # defensa
    # placa con ligera perspectiva
    pw, ph = int(150 * escala), int(75 * escala)
    px, py = cx - pw // 2, y0 + int(0.50 * h)
    sesgo = int(6 * escala)
    src = np.float32([[0, 0], [599, 0], [599, 299], [0, 299]])
    dst = np.float32([[px, py + sesgo], [px + pw, py], [px + pw, py + ph], [px, py + ph + sesgo]])
    H = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(placa, H, (ANCHO, ALTO))
    mask = cv2.warpPerspective(np.full(placa.shape[:2], 255, np.uint8), H, (ANCHO, ALTO))
    frame[mask > 0] = warp[mask > 0]


def degradar(frame, tramo, t):
    f = frame.astype(np.float32)
    if tramo == 1:                                  # baja iluminación + viñeta
        yy, xx = np.mgrid[0:ALTO, 0:ANCHO]
        vig = 1 - 0.6 * (((xx - ANCHO / 2) / ANCHO) ** 2 + ((yy - ALTO / 2) / ALTO) ** 2) * 2
        f = f * 0.30 * vig[..., None]
    f += rng.normal(0, 6 if tramo != 1 else 9, f.shape)   # ruido del sensor
    f = np.clip(f, 0, 255).astype(np.uint8)
    k = 5                                           # desenfoque por movimiento horizontal
    kern = np.zeros((k, k), np.float32); kern[k // 2, :] = 1 / k
    f = cv2.filter2D(f, -1, kern)
    if tramo == 2:                                  # lluvia: gotas diagonales + bruma
        capa = np.zeros_like(f)
        for _ in range(450):
            x, y = int(rng.integers(0, ANCHO)), int(rng.integers(0, ALTO))
            cv2.line(capa, (x, y), (x + 6, y + 22), (200, 200, 210), 1)
        capa = cv2.GaussianBlur(capa, (3, 3), 0)
        f = cv2.addWeighted(f, 0.80, np.full_like(f, 150), 0.20, 0)
        f = cv2.add(f, (capa * 0.6).astype(np.uint8))
    return f


def main(salida="video_prueba.mp4"):
    placa = dibujar_placa()
    base = fondo()
    vw = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ANCHO, ALTO))
    n = FPS * SEG
    for tramo in range(3):
        for i in range(n):
            t = i / (n - 1)
            frame = base.copy()
            escala = 0.75 + 0.75 * t                # el auto se acerca
            cx = int(380 + 520 * t)                 # y se desplaza a la derecha
            cy = int(420 + 70 * t)
            dibujar_auto(frame, cx, cy, escala, placa)
            vw.write(degradar(frame, tramo, t))
    vw.release()
    print("Video generado:", salida, f"({3 * n} cuadros, {FPS} fps)")


if __name__ == "__main__":
    main(*sys.argv[1:])
