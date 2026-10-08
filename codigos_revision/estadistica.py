# -*- coding: utf-8 -*-
"""estadistica.py — Funciones estadísticas del manuscrito (Ecuaciones 6, 7 y 8) y TOST.
Ejecutar "python estadistica.py" comprueba que reproducen los valores publicados."""
import math
import numpy as np
from scipy.stats import binom, norm

# ---------------- Estadística (Ecuaciones 6-8 del manuscrito) ----------------
def wilson(k, n, z=1.959963984540054):
    """IC de Wilson para una proporción k/n (Ecuación 6). Devuelve (p, li, ls) en %."""
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return 100 * p, 100 * (c - h), 100 * (c + h)

def mcnemar_exact(b, c):
    """p bilateral exacto de McNemar (Ecuación 7). b, c = conteos discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))

def holm(pvals):
    """Ajuste de Holm (Ecuación 8). Devuelve p ajustados en el orden original."""
    p = np.asarray(pvals, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def paired_counts(ca, cb):
    """ca, cb: vectores 0/1 de acierto por imagen (mismo orden). Devuelve (b, c):
    b = solo A acierta, c = solo B acierta."""
    ca = np.asarray(ca, int); cb = np.asarray(cb, int)
    return int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())

def tost_paired(b, c, n, margin_pp=2.0, alpha=0.05):
    """TOST para la diferencia pareada de exactitud A - B (en puntos porcentuales),
    con error estándar asintótico de proporciones pareadas. Devuelve dict."""
    d = (b - c) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 1e-12)) / n
    m = margin_pp / 100
    p_low = 1 - norm.cdf((d + m) / se)    # H0: d <= -m
    p_up = norm.cdf((d - m) / se)         # H0: d >= +m
    z90 = norm.ppf(1 - alpha); z95 = norm.ppf(0.975)
    return dict(diff_pp=float(100 * d), ci90_lo=float(100 * (d - z90 * se)), ci90_hi=float(100 * (d + z90 * se)),
                ci95_lo=float(100 * (d - z95 * se)), ci95_hi=float(100 * (d + z95 * se)),
                margin_pp=margin_pp, p_tost=float(max(p_low, p_up)),
                equivalent=bool(max(p_low, p_up) < alpha))

if __name__ == "__main__":
    # Autoverificación con los valores del manuscrito
    print("Wilson Xception 553/562:", [round(v, 2) for v in wilson(553, 562)])   # 98.40, 96.98, 99.16
    print("McNemar 2 vs 17:", round(mcnemar_exact(2, 17), 6))                    # 0.000729
    print("McNemar 4 vs 20:", round(mcnemar_exact(4, 20), 6))                    # 0.001544
    print("Holm:", holm([0.000728607, 0.001543880, 1.0]).round(6))               # 0.002186 0.003088 1
    print("TOST Xception vs ResNet (4 vs 3):", {k: (round(v, 3) if isinstance(v, float) else v)
          for k, v in tost_paired(4, 3, 562).items()})
