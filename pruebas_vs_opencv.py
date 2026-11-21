"""Compara las funciones propias contra OpenCV / scikit-image para validar
que la implementación es correcta.  Uso:  python pruebas_vs_opencv.py"""
import numpy as np
import cv2
from skimage.filters import threshold_sauvola

import morfologia as M
import binarizacion as B
import filtros as F

rng = np.random.default_rng(0)
img = (rng.random((120, 200)) * 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (7, 7), 0)
bin_ = (img > 128).astype(np.uint8) * 255

casos = [("rect 13x5", M.ee_rect(13, 5)), ("cruz 3", M.ee_cruz(3)), ("disco 3", M.ee_disco(3)),
         ("rect 4x2 (par)", M.ee_rect(4, 2))]
for nombre, ee in casos:
    ok_e = np.array_equal(M.erosion(img, ee), cv2.erode(img, ee))
    ok_d = np.array_equal(M.dilatacion(img, ee), cv2.dilate(img, ee))
    ok_o = np.array_equal(M.apertura(bin_, ee), cv2.morphologyEx(bin_, cv2.MORPH_OPEN, ee))
    ok_c = np.array_equal(M.cierre(bin_, ee), cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, ee))
    ok_b = np.array_equal(M.black_hat(img, ee), cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ee))
    print(f"{nombre:15s} erosión={ok_e} dilatación={ok_d} apertura={ok_o} cierre={ok_c} blackhat={ok_b}")

t_prop = B.otsu_umbral(img)
t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu propio={t_prop}  OpenCV={int(t_cv)}")

s_prop = B.sauvola(img, 25, 0.2) > 0
s_sk = img <= threshold_sauvola(img, window_size=25, k=0.2, r=128)
print(f"Sauvola: coincidencia con scikit-image = {100 * np.mean(s_prop == s_sk):.2f}% de los píxeles")

for tam in (3, 5, 7):
    dif = np.abs(F.gaussiano(img, tam).astype(int) - cv2.GaussianBlur(img, (tam, tam), 0).astype(int))
    print(f"Gaussiano {tam}x{tam}: diferencia máxima con OpenCV = {dif.max()} nivel(es) de gris, "
          f"{100 * np.mean(dif == 0):.2f}% de píxeles idénticos")
