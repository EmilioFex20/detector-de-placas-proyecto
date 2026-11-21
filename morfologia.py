"""
Morfología matemática implementada por el equipo (sin cv2.erode / cv2.dilate).

Todas las funciones trabajan con imágenes uint8 en escala de grises o binarias
(0 / 255). El elemento estructurante (EE) es un arreglo de 0/1 con el origen en
el centro.

Idea base:
  - Erosión  (A ⊖ B)(x) = min_{b ∈ B} A(x + b)
  - Dilatación (A ⊕ B)(x) = max_{b ∈ B} A(x - b)
Se implementan desplazando la imagen una vez por cada elemento "1" del EE y
quedándose con el mínimo (erosión) o el máximo (dilatación) acumulado. Esto es
equivalente a recorrer la ventana en cada píxel, pero vectorizado con NumPy.
"""
import numpy as np


# --------------------------------------------------------------------------
# Elementos estructurantes
# --------------------------------------------------------------------------
def ee_rect(ancho, alto):
    """EE rectangular de ancho x alto (todo unos)."""
    return np.ones((alto, ancho), dtype=np.uint8)


def ee_cruz(tam=3):
    """EE en cruz (vecindad 4 si tam=3)."""
    ee = np.zeros((tam, tam), dtype=np.uint8)
    ee[tam // 2, :] = 1
    ee[:, tam // 2] = 1
    return ee


def ee_disco(radio):
    """EE en forma de disco."""
    y, x = np.ogrid[-radio:radio + 1, -radio:radio + 1]
    return (x * x + y * y <= radio * radio).astype(np.uint8)


# --------------------------------------------------------------------------
# Operaciones básicas
# --------------------------------------------------------------------------
def _desplazamientos(ee):
    alto, ancho = ee.shape
    cy, cx = alto // 2, ancho // 2
    ys, xs = np.nonzero(ee)
    return ys - cy, xs - cx, cy, cx, alto, ancho


def erosion(img, ee, iteraciones=1):
    """Erosión: mínimo de la vecindad definida por el EE.
    El borde se rellena con 255 para no "comerse" objetos que tocan el borde
    (mismo criterio que OpenCV)."""
    out = img
    dys, dxs, cy, cx, alto, ancho = _desplazamientos(ee)
    for _ in range(iteraciones):
        pad = np.pad(out, ((cy, alto - cy - 1), (cx, ancho - cx - 1)),
                     mode="constant", constant_values=255)
        h, w = out.shape
        acc = np.full_like(out, 255)
        for dy, dx in zip(dys, dxs):
            np.minimum(acc, pad[cy + dy:cy + dy + h, cx + dx:cx + dx + w], out=acc)
        out = acc
    return out


def dilatacion(img, ee, iteraciones=1):
    """Dilatación: máximo de la vecindad reflejada del EE.
    El borde se rellena con 0."""
    out = img
    ee_ref = ee[::-1, ::-1]           # reflexión del EE (B^)
    dys, dxs, cy, cx, alto, ancho = _desplazamientos(ee_ref)
    for _ in range(iteraciones):
        pad = np.pad(out, ((cy, alto - cy - 1), (cx, ancho - cx - 1)),
                     mode="constant", constant_values=0)
        h, w = out.shape
        acc = np.zeros_like(out)
        for dy, dx in zip(dys, dxs):
            np.maximum(acc, pad[cy + dy:cy + dy + h, cx + dx:cx + dx + w], out=acc)
        out = acc
    return out


# --------------------------------------------------------------------------
# Operaciones compuestas
# --------------------------------------------------------------------------
def apertura(img, ee, iteraciones=1):
    """Apertura = erosión seguida de dilatación. Quita objetos pequeños y
    puntas delgadas sin cambiar mucho el tamaño de lo demás."""
    return dilatacion(erosion(img, ee, iteraciones), ee, iteraciones)


def cierre(img, ee, iteraciones=1):
    """Cierre = dilatación seguida de erosión. Rellena huecos y une regiones
    cercanas."""
    return erosion(dilatacion(img, ee, iteraciones), ee, iteraciones)


def top_hat(img, ee):
    """Top-hat (sombrero blanco) = A - apertura(A). Resalta detalles claros
    más pequeños que el EE."""
    return (img.astype(np.int16) - apertura(img, ee)).clip(0, 255).astype(np.uint8)


def black_hat(img, ee):
    """Black-hat (sombrero negro) = cierre(A) - A. Resalta detalles oscuros
    más pequeños que el EE, por ejemplo letras negras sobre la placa blanca."""
    return (cierre(img, ee).astype(np.int16) - img).clip(0, 255).astype(np.uint8)


def gradiente_morfologico(img, ee):
    """Dilatación - erosión: realza los bordes."""
    return (dilatacion(img, ee).astype(np.int16) - erosion(img, ee)).clip(0, 255).astype(np.uint8)


def esqueleto(binaria, ee=None):
    """Esqueleto morfológico (fórmula de Lantuéjoul):

        S(A) = U_k  [ (A ⊖ kB)  -  (A ⊖ kB) ∘ B ]

    Se erosiona k veces; en cada paso se guarda lo que la apertura elimina.
    Termina cuando la erosión deja la imagen vacía.
    Entrada binaria 0/255, salida binaria 0/255."""
    if ee is None:
        ee = ee_cruz(3)
    # Se agrega un marco de fondo (0) de 1 px. La erosión rellena el borde con
    # 255, así que un objeto que llena todo el recorte nunca se vaciaría y el
    # ciclo no terminaría; con el marco, lo de afuera cuenta como fondo.
    a = np.pad((binaria > 0).astype(np.uint8) * 255, 1)
    s = np.zeros_like(a)
    for _ in range(max(a.shape)):          # cota de seguridad
        if not a.any():
            break
        ab = apertura(a, ee)
        s = np.maximum(s, ((a > 0) & (ab == 0)).astype(np.uint8) * 255)
        a = erosion(a, ee)
    return s[1:-1, 1:-1]


def limpiar_borde(binaria):
    """Elimina componentes que tocan el borde (reconstrucción morfológica por
    dilatación geodésica a partir del marco de la imagen)."""
    mask = (binaria > 0).astype(np.uint8) * 255
    marcador = np.zeros_like(mask)
    marcador[0, :], marcador[-1, :] = mask[0, :], mask[-1, :]
    marcador[:, 0], marcador[:, -1] = mask[:, 0], mask[:, -1]
    ee = ee_rect(3, 3)
    while True:
        nuevo = np.minimum(dilatacion(marcador, ee), mask)
        if np.array_equal(nuevo, marcador):
            break
        marcador = nuevo
    return np.where(marcador > 0, 0, mask).astype(np.uint8)
