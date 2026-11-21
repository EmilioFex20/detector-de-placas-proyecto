"""
Detección y lectura de placas vehiculares en video, cuadro por cuadro.
Proyecto final - Procesamiento de Imágenes y Video (CETYS).

Pipeline por cuadro
-------------------
 1. Conversión BGR -> gris                         (modelo de color)
 2. Filtro gaussiano (propio)                       (filtrado espacial, quita ruido)
 3. CLAHE                                           (mejora de contraste local)
 4. Localización de la ROI:
      black-hat  -> Sobel X -> suavizado -> cierre -> Otsu -> apertura -> contornos
 5. Rectificación de perspectiva de cada candidata  (transformación geométrica)
 6. Binarización local de Sauvola en la placa       (binarización)
 7. Apertura + limpieza de borde                    (morfología)
 8. Componentes conexas + esqueleto                 (segmentación y validación de caracteres)
 9. OCR (Tesseract) sobre la imagen limpia          (reconocimiento de texto, NO es PDI)
10. Seguimiento simple + votación entre cuadros     (seguimiento, NO es PDI)

Filtro gaussiano, morfología, Otsu y Sauvola se usan con las funciones PROPIAS
(filtros.py, morfologia.py y binarizacion.py). OpenCV sólo se usa para
leer/escribir video, convertir a gris, CLAHE, Sobel, contornos, warp y
componentes conexas.

Uso
---
  python alpr_video.py --fuente video_prueba.mp4 --salida resultado.mp4
  python alpr_video.py --fuente 0            # cámara web en tiempo real
  python alpr_video.py --fuente foto.jpg       # una foto (o una carpeta de fotos)
  Opciones: --capturas DIR  (guarda las etapas de algunos cuadros)
            --cada N        (procesa 1 de cada N cuadros, útil en tiempo real)
            --mostrar       (abre ventana con original | procesado)
"""
import argparse
import csv
import os
import re
import time
from collections import Counter

import cv2
import numpy as np

import morfologia as M
import binarizacion as B
import filtros as F

try:
    import pytesseract
except ImportError:  # el pipeline de PDI funciona igual sin OCR
    pytesseract = None

ANCHO_TRABAJO = 960            # se redimensiona el cuadro para que el tiempo sea estable
PLACA_ALTO = 120               # alto de la placa rectificada (px)
PATRON_PLACA = re.compile(r"^[A-Z0-9]{5,8}$")
ESCALAS = (1.0, 0.5, 1 / 3, 0.25)   # escalas de la localización (ver procesar_cuadro)
CONF_MIN = 60                  # confianza mínima de Tesseract para aceptar una lectura


def es_placa(texto, conf):
    """Una placa mexicana mezcla letras y números (p. ej. ABC-123-D)."""
    letras = sum(ch.isalpha() for ch in texto)
    digitos = sum(ch.isdigit() for ch in texto)
    return bool(PATRON_PLACA.match(texto)) and letras >= 2 and digitos >= 2 and conf >= CONF_MIN


# ==========================================================================
# 1-3. Preprocesamiento
# ==========================================================================
def preprocesar(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtrada = F.gaussiano(gris, tam=5)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    realzada = clahe.apply(filtrada)
    return gris, filtrada, realzada


# ==========================================================================
# 4. Localización de la región de interés (ROI)
# ==========================================================================
def localizar_candidatas(realzada, etapas=None):
    """Devuelve una lista de rectángulos rotados (cv2.minAreaRect) que podrían
    ser placas: regiones claras con mucho contraste vertical (letras)."""
    h, w = realzada.shape
    # Los EE tienen tamaño FIJO en píxeles (pensados para una placa de ~100 px
    # de ancho). Para buscar placas más grandes no se agrandan los EE: se
    # reduce la imagen (ver ESCALAS en procesar_cuadro).
    k = lambda v: max(1, int(round(v)))

    # Black-hat: realza las letras oscuras sobre el fondo claro de la placa
    bh = M.black_hat(realzada, M.ee_rect(k(15), k(5)))
    # Gradiente horizontal: las letras producen muchos bordes verticales
    gx = np.abs(cv2.Sobel(bh, cv2.CV_32F, 1, 0, ksize=3))
    gx = (255 * gx / (gx.max() + 1e-6)).astype(np.uint8)
    gx = F.gaussiano(gx, tam=5)
    # Cierre con EE ancho: une las letras en un solo bloque
    bloque = M.cierre(gx, M.ee_rect(k(23), k(5)))
    mask, _ = B.otsu(bloque)
    # Apertura: quita bloques delgados/ruido; dilatación para recuperar margen
    mask = M.apertura(mask, M.ee_rect(k(5), k(5)), iteraciones=2)
    mask = M.dilatacion(mask, M.ee_rect(k(7), k(3)))
    # Sólo regiones claras (la placa es blanca): Otsu sobre el gris
    claro, _ = B.otsu(F.gaussiano(realzada, tam=5))
    claro = M.cierre(claro, M.ee_rect(k(7), k(7)))
    mask_final = np.minimum(mask, M.dilatacion(claro, M.ee_rect(k(9), k(9))))

    if etapas is not None:
        etapas.update(blackhat=bh, sobel=gx, cierre=bloque, mascara=mask_final)

    contornos, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatas = []
    for c in contornos:
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw < rh:
            rw, rh, ang = rh, rw, ang + 90
        if rh < 1:
            continue
        ar = rw / rh
        area = rw * rh
        if 1.8 <= ar <= 7.0 and (0.0004 * w * h) <= area <= (0.08 * w * h) and abs(((ang + 90) % 180) - 90) < 25:
            candidatas.append(((cx, cy), (rw, rh), ang))
    candidatas.sort(key=lambda r: -r[1][0] * r[1][1])
    return candidatas[:12]


# ==========================================================================
# 5. Rectificación
# ==========================================================================
def rectificar(gris, rect, escala=1.0, margen_x=0.15, margen_y=0.70):
    """Recorta y endereza la candidata. `escala` permite recortar del cuadro a
    resolución completa aunque la localización se hizo en el reducido. El
    margen vertical es amplio porque la máscara suele cubrir sólo el renglón
    de texto y los caracteres no deben tocar el borde del recorte."""
    (cx, cy), (rw, rh), ang = rect
    cx, cy, rw, rh = cx * escala, cy * escala, rw * escala, rh * escala
    rw, rh = rw * (1 + margen_x), rh * (1 + margen_y)
    caja = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)
    # ordenar esquinas: sup-izq, sup-der, inf-der, inf-izq
    s, d = caja.sum(1), np.diff(caja, axis=1).ravel()
    src = np.float32([caja[np.argmin(s)], caja[np.argmin(d)], caja[np.argmax(s)], caja[np.argmax(d)]])
    W = int(PLACA_ALTO * rw / rh)
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, PLACA_ALTO - 1], [0, PLACA_ALTO - 1]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gris, Hm, (W, PLACA_ALTO), flags=cv2.INTER_CUBIC), src


# ==========================================================================
# 6-8. Binarización, morfología y segmentación de caracteres
# ==========================================================================
def ancho_trazo(comp):
    """Ancho de trazo estimado = 2 x promedio de la transformada de distancia
    evaluada sobre el esqueleto (propio) del componente."""
    esq = M.esqueleto(comp)
    if not esq.any():
        return 0.0, esq
    # borde de ceros: si el componente toca todos los bordes no habría fondo
    # y distanceTransform devolvería ~FLT_MAX (overflow al promediar)
    bin_ = np.pad((comp > 0).astype(np.uint8), 1)
    dist = cv2.distanceTransform(bin_, cv2.DIST_L2, 3)[1:-1, 1:-1]
    return 2.0 * float(dist[esq > 0].mean()), esq


def cortar_por_proyeccion(binaria, frac=0.20):
    """Proyección vertical: cuenta píxeles blancos por columna dentro del
    renglón de texto. Las columnas con muy pocos píxeles (espacios y guiones)
    se ponen en cero, con lo que se separan caracteres que quedaron unidos."""
    filas = (binaria > 0).sum(1)
    if filas.max() == 0:
        return binaria
    # renglón principal = tramo contiguo más largo de filas con mucho texto
    # (así no se mezcla con el nombre del estado arriba ni "MEXICO" abajo)
    activas = filas >= 0.25 * filas.max()
    mejor, ini = (0, 0, 0), None
    for y, a in enumerate(np.append(activas, False)):
        if a and ini is None:
            ini = y
        elif not a and ini is not None:
            if y - ini > mejor[0]:
                mejor = (y - ini, ini, y)
            ini = None
    _, y0, y1 = mejor
    cols = (binaria[y0:y1] > 0).sum(0)
    out = binaria.copy()
    out[:, cols < frac * (y1 - y0)] = 0
    return out


def dividir_unidos(comp, h):
    """Si un componente es demasiado ancho para ser un solo caracter (dos
    letras que se tocaron por el desenfoque), se corta en la columna con
    menos píxeles de su zona central (mínimo de la proyección vertical)."""
    w = comp.shape[1]
    if w / h <= 1.1 or w / h > 2.6:
        return [(0, comp)]
    cols = (comp > 0).sum(0)
    a, b = int(0.3 * w), int(0.7 * w)
    corte = a + int(np.argmin(cols[a:b]))
    izq, der = comp[:, :corte], comp[:, corte + 1:]
    piezas = []
    for xs, p in ((0, izq), (corte + 1, der)):
        xx = np.nonzero(p.any(0))[0]
        if len(xx):
            piezas.append((xs + xx.min(), p[:, xx.min():xx.max() + 1]))
    return piezas


def seleccionar_renglon(cands):
    """Agrupa los candidatos por renglón (centro vertical parecido) y alto
    parecido (±25 %), y se queda con el grupo de caracteres MÁS ALTOS que tenga
    al menos 4 elementos. En una placa mexicana el número es el texto más
    grande; el nombre del estado y "MEXICO" son más chicos, así que se
    descartan aunque tengan más letras (antes se usaba la mediana del alto, y
    con 15 letras de "BAJA CALIFORNIA" la mediana caía en el renglón equivocado).
    Dentro del grupo se valida el ancho de trazo medido con el esqueleto."""
    mejor = []
    for c in cands:
        yc = c["y"] + c["h"] / 2
        g = [d for d in cands if 0.75 * c["h"] <= d["h"] <= 1.25 * c["h"]
             and abs(d["y"] + d["h"] / 2 - yc) <= 0.4 * c["h"]]
        if len(g) < 4:
            continue
        if not mejor or c["h"] > mejor[0] or (c["h"] == mejor[0] and len(g) > len(mejor[1])):
            mejor = (c["h"], g)
    if not mejor:
        return []
    g = mejor[1]
    med_sw = np.median([c["sw"] for c in g])
    return [c for c in g if 0.5 * med_sw <= c["sw"] <= 1.8 * med_sw]


def segmentar_caracteres(placa_gris, etapas=None):
    hp, wp = placa_gris.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 4))
    p = clahe.apply(F.gaussiano(placa_gris, tam=3))
    binaria = B.sauvola(p, ventana=(hp // 3) | 1, k=0.18)             # texto oscuro -> blanco
    binaria = M.apertura(binaria, M.ee_rect(2, 2))                   # quita puntos de ruido
    binaria = M.limpiar_borde(binaria)                               # quita el marco de la placa
    binaria = cortar_por_proyeccion(binaria)                         # separa caracteres pegados por guiones

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.15 * hp <= h <= 0.85 * hp):       # alto posible de un caracter
            continue
        comp = (etiquetas[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        for xs, pieza in dividir_unidos(comp, h):
            ys = np.nonzero(pieza.any(1))[0]
            pieza = pieza[ys.min():ys.max() + 1]
            ph_, pw_ = pieza.shape
            if not (0.08 <= pw_ / ph_ <= 1.1) or (pieza > 0).sum() < 0.15 * pw_ * ph_:
                continue
            sw, esq = ancho_trazo(pieza)
            cands.append(dict(x=x + xs, y=y + ys.min(), w=pw_, h=ph_, comp=pieza, esq=esq, sw=sw))

    cands = seleccionar_renglon(cands)
    cands.sort(key=lambda c: c["x"])

    limpia = np.zeros_like(binaria)
    esq_img = np.zeros_like(binaria)
    for c in cands:
        roi = limpia[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        np.maximum(roi, c["comp"], out=roi)
        roi = esq_img[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
        np.maximum(roi, c["esq"], out=roi)

    if etapas is not None:
        etapas.update(placa=placa_gris, placa_clahe=p, placa_bin=binaria,
                      placa_chars=limpia, placa_esq=esq_img)
    return cands, limpia


# ==========================================================================
# 9. OCR (fuera del alcance de PDI, se usa Tesseract)
# ==========================================================================
def leer_texto(limpia, cands):
    if pytesseract is None or len(cands) < 4:
        return "", 0.0
    x0 = max(min(c["x"] for c in cands) - 10, 0)
    x1 = max(c["x"] + c["w"] for c in cands) + 10
    y0 = max(min(c["y"] for c in cands) - 10, 0)
    y1 = max(c["y"] + c["h"] for c in cands) + 10
    img = 255 - limpia[y0:y1, x0:x1]                                   # negro sobre blanco
    img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    # Nota: en Tesseract 5 la lista blanca (tessedit_char_whitelist) hace que la
    # confianza salga en 0, así que se lee sin ella y se filtra después.
    cfg = "--psm 7 --oem 1"
    d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    palabras = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    texto = "".join(t for t, _ in palabras).upper()
    texto = re.sub(r"[^A-Z0-9]", "", texto)
    conf = float(np.mean([c for _, c in palabras])) if palabras else 0.0
    return texto, conf


def procesar_cuadro(frame, etapas=None):
    """Procesa un cuadro y devuelve la mejor lectura encontrada."""
    esc = ANCHO_TRABAJO / frame.shape[1]
    peq = cv2.resize(frame, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    gris, filtrada, realzada = preprocesar(peq)
    gris_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if etapas is not None:
        etapas.update(original=peq, gris=gris, filtrada=filtrada, clahe=realzada)
    # Localización MULTIESCALA: los elementos estructurantes tienen tamaño fijo
    # (pensado para una placa de ~100 px de ancho en el cuadro de 960 px). Si la
    # placa sale más grande (auto cerca o foto de cerca), el black-hat y el
    # cierre ya no abarcan las letras y el número no forma un bloque. En lugar
    # de cambiar los EE, se reduce la imagen a 1/2 y 1/3: ahí la placa grande
    # vuelve a tener el tamaño para el que están pensados los EE.
    candidatas, et_escala = [], {}
    for f in ESCALAS:
        img = realzada if f == 1 else cv2.resize(realzada, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        et_f = {} if etapas is not None else None
        for (cx, cy), (rw, rh), ang in localizar_candidatas(img, et_f):
            rect = ((cx / f, cy / f), (rw / f, rh / f), ang)
            # evitar repetir la misma región encontrada en dos escalas
            # (mismo centro Y mismo tamaño; una región del mismo centro pero de
            # otro tamaño es otra hipótesis, p. ej. un renglón contra la placa entera)
            def repetida(o):
                (ox, oy), (ow, oh), _ = o[0]
                return (np.hypot(rect[0][0] - ox, rect[0][1] - oy) < 0.3 * rect[1][0]
                        and 0.7 < rect[1][0] / ow < 1.4 and 0.7 < rect[1][1] / oh < 1.4)
            if not any(repetida(o) for o in candidatas):
                candidatas.append((rect, f))
        et_escala[f] = et_f
    mejor = None
    for rect, f in candidatas:
        placa, esquinas = rectificar(gris_full, rect, escala=1 / esc)
        sub = {} if etapas is not None else None
        cands, limpia = segmentar_caracteres(placa, sub)
        if len(cands) < 4:
            continue
        texto, conf = leer_texto(limpia, cands)
        valido = es_placa(texto, conf)
        puntaje = len(cands) + (10 if valido else 0) + conf / 100
        if mejor is None or puntaje > mejor["puntaje"]:
            mejor = dict(esquinas=esquinas, texto=texto if valido else "",
                         conf=conf, n_chars=len(cands), puntaje=puntaje, escala=f)
            if etapas is not None:
                etapas.update(sub)
                for k, v in et_escala[f].items():   # etapas de la escala que encontró la placa
                    etapas[k] = cv2.resize(v, (realzada.shape[1], realzada.shape[0]), interpolation=cv2.INTER_NEAREST)
                etapas["escala"] = f
    if etapas is not None and "blackhat" not in etapas:
        etapas.update({k: cv2.resize(v, (realzada.shape[1], realzada.shape[0]), interpolation=cv2.INTER_NEAREST)
                       for k, v in et_escala[1].items()})
    return mejor


# ==========================================================================
# 10. Seguimiento y votación entre cuadros (fuera del alcance de PDI)
# ==========================================================================
class Seguidor:
    """Asocia detecciones entre cuadros por cercanía del centro y vota la
    lectura más frecuente para que el texto mostrado sea estable."""

    def __init__(self, dist_max=150, olvidar=15):
        self.pistas, self.sig_id = [], 1
        self.dist_max, self.olvidar = dist_max, olvidar

    def actualizar(self, det, n_cuadro):
        """Devuelve (pistas activas, pista asociada a la detección o None)."""
        pista = None
        if det is not None:
            c = det["esquinas"].mean(0)
            pista = min(self.pistas, key=lambda p: np.linalg.norm(p["c"] - c), default=None)
            if pista is None or np.linalg.norm(pista["c"] - c) > self.dist_max:
                pista = dict(id=self.sig_id, votos=Counter(), lecturas=Counter()); self.sig_id += 1
                self.pistas.append(pista)
            pista.update(c=c, esquinas=det["esquinas"], ultimo=n_cuadro)
            if det["texto"]:
                pista["votos"][det["texto"]] += det["conf"]   # voto ponderado por la confianza del OCR
                pista["lecturas"][det["texto"]] += 1
        self.pistas = [p for p in self.pistas if n_cuadro - p["ultimo"] <= self.olvidar]
        return self.pistas, pista


MIN_LECTURAS = 3


def lectura_estable(pista):
    """Lectura que se muestra: la de mayor suma de confianzas, pero sólo si se
    leyó igual en al menos MIN_LECTURAS cuadros (evita mostrar la primera
    lectura equivocada mientras hay pocos votos)."""
    if not pista["votos"]:
        return ""
    texto = pista["votos"].most_common(1)[0][0]
    return texto if pista["lecturas"][texto] >= MIN_LECTURAS else ""


def dibujar(frame, pistas, n_cuadro):
    out = frame.copy()
    for p in pistas:
        texto = lectura_estable(p)
        if not texto:                          # todavía no hay una lectura confiable
            continue
        color = (0, 220, 0) if p["ultimo"] == n_cuadro else (0, 160, 255)
        cv2.polylines(out, [p["esquinas"].astype(np.int32)], True, color, 3)
        x, y = p["esquinas"][:, 0].min(), p["esquinas"][:, 1].min()
        cv2.rectangle(out, (int(x), int(y) - 38), (int(x) + 22 * len(texto) + 20, int(y) - 4), (0, 0, 0), -1)
        cv2.putText(out, texto, (int(x) + 8, int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def panel_etapas(et):
    """Arma una figura con las etapas intermedias para el reporte."""
    def lbl(img, t, w=480):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_NEAREST)
        barra = np.zeros((26, w, 3), np.uint8)        # el título va arriba, sin tapar la imagen
        cv2.putText(barra, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return np.vstack([barra, img])
    def fila(ims):
        h = max(i.shape[0] for i in ims)
        return np.hstack([cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in ims])
    # black-hat, Sobel y cierre tienen valores bajos (casi negros a la vista):
    # sólo para mostrarlos se estiran a 0-255; el algoritmo usa los valores reales
    vis = lambda im: cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
    et = dict(et, blackhat=vis(et["blackhat"]), sobel=vis(et["sobel"]), cierre=vis(et["cierre"]))
    a = fila([lbl(et["original"], "1. Original"), lbl(et["clahe"], "2-3. Gris + gaussiano + CLAHE"),
              lbl(et["blackhat"], "4a. Black-hat (propio)")])
    sufijo = f" [escala {et.get('escala', 1):.2f}]"
    b = fila([lbl(et["sobel"], "4b. Sobel X"), lbl(et["cierre"], "4c. Cierre (propio)"),
              lbl(et["mascara"], "4d. Otsu + apertura" + sufijo)])
    out = [a, b]
    if "placa" in et:
        out.append(fila([lbl(et["placa"], "5. Placa rectificada", 360), lbl(et["placa_bin"], "6-7. Sauvola+apertura+borde", 360),
                         lbl(et["placa_chars"], "8. Caracteres validados", 360), lbl(et["placa_esq"], "8b. Esqueleto (propio)", 360)]))
    ancho = max(o.shape[1] for o in out)
    return np.vstack([cv2.copyMakeBorder(o, 0, 0, 0, ancho - o.shape[1], cv2.BORDER_CONSTANT) for o in out])


EXT_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def procesar_imagenes(ruta, carpeta):
    """Modo fotos: procesa una imagen o todas las imágenes de una carpeta y
    guarda, para cada una, el resultado anotado y el panel de etapas."""
    os.makedirs(carpeta, exist_ok=True)
    rutas = ([os.path.join(ruta, f) for f in sorted(os.listdir(ruta)) if f.lower().endswith(EXT_IMAGEN)]
             if os.path.isdir(ruta) else [ruta])
    for r in rutas:
        img = cv2.imread(r)
        if img is None:
            print("No se pudo leer", r); continue
        et = {}
        det = procesar_cuadro(img, et)
        base = os.path.splitext(os.path.basename(r))[0]
        out = img.copy()
        if det:
            cv2.polylines(out, [det["esquinas"].astype(np.int32)], True, (0, 220, 0), 3)
            x, y = det["esquinas"][:, 0].min(), det["esquinas"][:, 1].min()
            cv2.putText(out, det["texto"] or "?", (int(x), max(int(y) - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
        cv2.imwrite(os.path.join(carpeta, f"{base}_resultado.png"), out)
        cv2.imwrite(os.path.join(carpeta, f"{base}_etapas.png"), panel_etapas(et))
        print(f"{base}: {det['texto'] if det and det['texto'] else 'sin lectura válida'}"
              + (f" (confianza {det['conf']:.0f}, {det['n_chars']} caracteres, escala {det['escala']:.2f})" if det else ""))


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="video_prueba.mp4", help="ruta de video o índice de cámara (0)")
    ap.add_argument("--salida", default="resultado.mp4")
    ap.add_argument("--registro", default="registro.csv")
    ap.add_argument("--capturas", default="", help="carpeta para guardar etapas de algunos cuadros")
    ap.add_argument("--cuadros_captura", default="", help="lista de cuadros a capturar, p. ej. 40,120,220")
    ap.add_argument("--cada", type=int, default=1)
    ap.add_argument("--mostrar", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(a.fuente) or a.fuente.lower().endswith(EXT_IMAGEN):
        procesar_imagenes(a.fuente, a.capturas or "capturas_fotos")
        return
    fuente = int(a.fuente) if a.fuente.isdigit() else a.fuente
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la fuente {a.fuente}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(a.salida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    capturar = {int(x) for x in a.cuadros_captura.split(",") if x}
    if a.capturas:
        os.makedirs(a.capturas, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 o negativo en cámara
    t0 = time.time()
    seg = Seguidor()
    filas, n = [], 0
    pistas = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % a.cada == 0:
            et = {} if (a.capturas and n in capturar) else None
            det = procesar_cuadro(frame, et)
            pistas, actual = seg.actualizar(det, n)
            filas.append([n, round(n / fps, 2), det["texto"] if det else "",
                          round(det["conf"], 1) if det else "", det["n_chars"] if det else 0,
                          lectura_estable(actual) if actual else ""])
            if et:
                cv2.imwrite(os.path.join(a.capturas, f"etapas_{n:04d}.png"), panel_etapas(et))
        anotado = dibujar(frame, pistas, n)
        if a.capturas and n in capturar:
            cv2.imwrite(os.path.join(a.capturas, f"resultado_{n:04d}.png"), anotado)
        vw.write(anotado)
        if a.mostrar:
            cv2.imshow("ALPR  (q para salir)", cv2.resize(anotado, None, fx=0.6, fy=0.6))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
        if n % 30 == 0:                              # progreso cada 30 cuadros
            vel = n / (time.time() - t0)
            if total > 0:
                print(f"\r{n}/{total} cuadros ({100 * n / total:.0f}%)  {vel:.1f} c/s  "
                      f"faltan ~{(total - n) / vel:.0f}s   ", end="", flush=True)
            else:
                print(f"\r{n} cuadros  {vel:.1f} c/s   ", end="", flush=True)
    print()
    cap.release(); vw.release(); cv2.destroyAllWindows()

    with open(a.registro, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["cuadro", "tiempo_s", "lectura_cuadro", "confianza", "n_caracteres", "lectura_votada"])
        wr.writerows(filas)
    print(f"Listo: {n} cuadros. Video: {a.salida}  Registro: {a.registro}")


if __name__ == "__main__":
    main()
