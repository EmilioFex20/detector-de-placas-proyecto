"""Resume registro.csv por tramo.  Uso: python resumen.py [registro.csv] [TEXTO_REAL] [cuadros_por_tramo]"""
import csv, sys
from collections import Counter
ruta = sys.argv[1] if len(sys.argv) > 1 else "registro.csv"
real = sys.argv[2] if len(sys.argv) > 2 else "BCX482A"
paso = int(sys.argv[3]) if len(sys.argv) > 3 else 90
r = list(csv.DictReader(open(ruta, encoding="utf-8")))
for i in range(0, len(r), paso):
    sub = r[i:i + paso]
    det = sum(int(x["n_caracteres"]) > 0 for x in sub)
    ok = sum(x["lectura_cuadro"] == real for x in sub)
    otras = Counter(x["lectura_cuadro"] for x in sub if x["lectura_cuadro"] not in (real, ""))
    print(f"cuadros {i}-{i + len(sub) - 1}: candidata con >=4 caracteres {det}/{len(sub)}, lectura correcta {ok}/{len(sub)}, "
          f"votada final={sub[-1]['lectura_votada']}, errores={dict(otras.most_common(4))}")
