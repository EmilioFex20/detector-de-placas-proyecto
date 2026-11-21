"""
Binarización implementada por el equipo: Otsu (global) y Sauvola (local).
OpenALPR documenta Sauvola y Wolf-Jolion para la placa; aquí se programan
desde cero con NumPy.
"""
import numpy as np


def otsu_umbral(gris):
    """Calcula el umbral de Otsu: el que maximiza la varianza entre clases
    sigma_b^2(t) = w0(t) * w1(t) * (mu0(t) - mu1(t))^2."""
    hist = np.bincount(gris.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w0 = np.cumsum(p)                       # probabilidad de la clase 0
    mu = np.cumsum(p * np.arange(256))      # media acumulada
    mu_t = mu[-1]
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    sigma_b = np.nan_to_num(sigma_b)
    return int(np.argmax(sigma_b))


def otsu(gris, invertir=False):
    """Binariza con Otsu. invertir=True deja en blanco lo oscuro."""
    t = otsu_umbral(gris)
    b = gris > t
    if invertir:
        b = ~b
    return b.astype(np.uint8) * 255, t


def _suma_ventana(integral, r, h, w):
    """Suma en una ventana (2r+1)x(2r+1) usando la imagen integral."""
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    s = (integral[y1][:, x1] - integral[y0][:, x1]
         - integral[y1][:, x0] + integral[y0][:, x0])
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return s, area


def sauvola(gris, ventana=25, k=0.2, R=128.0):
    """Umbral local de Sauvola:  T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1)),
    donde m y s son la media y desviación estándar en la ventana.
    Devuelve una máscara con el TEXTO OSCURO en blanco (255)."""
    g = gris.astype(np.float64)
    h, w = g.shape
    r = ventana // 2
    I1 = np.zeros((h + 1, w + 1)); I1[1:, 1:] = g.cumsum(0).cumsum(1)
    I2 = np.zeros((h + 1, w + 1)); I2[1:, 1:] = (g * g).cumsum(0).cumsum(1)
    s1, area = _suma_ventana(I1, r, h, w)
    s2, _ = _suma_ventana(I2, r, h, w)
    m = s1 / area
    var = np.maximum(s2 / area - m * m, 0)
    T = m * (1 + k * (np.sqrt(var) / R - 1))
    return (g <= T).astype(np.uint8) * 255
