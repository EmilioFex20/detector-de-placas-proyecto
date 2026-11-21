"""
Filtro gaussiano implementado por el equipo (sin cv2.GaussianBlur).

El kernel gaussiano 2D es separable: G(x, y) = g(x) * g(y). Por eso, en lugar
de convolucionar con un kernel k x k (k^2 multiplicaciones por píxel), se
convoluciona primero por filas y luego por columnas con el kernel 1D
(2k multiplicaciones por píxel). El resultado es el mismo.
"""
import numpy as np


def kernel_gaussiano_1d(tam, sigma=0.0):
    """Kernel 1D normalizado. Si sigma <= 0 se calcula con la misma regla que
    OpenCV: sigma = 0.3 * ((tam - 1) * 0.5 - 1) + 0.8."""
    if tam % 2 == 0:
        raise ValueError("el tamaño del kernel debe ser impar")
    if sigma <= 0:
        sigma = 0.3 * ((tam - 1) * 0.5 - 1) + 0.8
    x = np.arange(tam) - tam // 2
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def _convolucion_1d(img, k, eje):
    """Convolución 1D a lo largo de un eje, con borde reflejado (tipo
    BORDER_REFLECT_101 de OpenCV: ...c b | a b c d | c b...)."""
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[eje] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    n = img.shape[eje]
    for i, w in enumerate(k):          # el kernel es simétrico: no hace falta voltearlo
        if eje == 0:
            out += w * p[i:i + n, :]
        else:
            out += w * p[:, i:i + n]
    return out


def gaussiano(img, tam=5, sigma=0.0):
    """Suavizado gaussiano separable de una imagen en escala de grises uint8."""
    k = kernel_gaussiano_1d(tam, sigma)
    f = _convolucion_1d(img.astype(np.float64), k, eje=1)   # filas
    f = _convolucion_1d(f, k, eje=0)                        # columnas
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)
