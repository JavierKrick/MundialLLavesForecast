import math
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import defaultdict
from scipy.optimize import minimize
from scipy.special import gammaln

from datos import (
    datos_potencia_torneos_2026, 
    datos_torneos_completos, 
    partidos_filtrados, 
    teams_base,
    metricas_extra
)

# ══════════════════════════════════════════════════════════════════════════════
# §1 CONFIGURACIÓN GLOBAL Y MAPEOS
# ══════════════════════════════════════════════════════════════════════════════

BASE_ELO   = 1500   
BASE_POT   = 1850   
PRIOR_MEAN = 1.3    
PRIOR_PJ   = 3.0    
EPS_LOG    = 1e-6

ANFITRIONES_2026 = {"México", "Mexico", "Canadá", "Canada", "Estados Unidos", "USA", "EEUU"}

MAPA_EQUIPOS = {
    "Canada": "Canadá", "Morocco": "Marruecos", "France": "Francia", "Brazil": "Brasil",
    "Norway": "Noruega", "Mexico": "México", "England": "Inglaterra", "Spain": "España",
    "USMNT": "Estados Unidos", "United States": "Estados Unidos", "USA": "Estados Unidos",
    "Belgium": "Bélgica", "Switzerland": "Suiza", "Egypt": "Egipto", "South Africa": "Sudáfrica",
    "Sweden": "Suecia", "Japan": "Japón", "Netherlands": "Países Bajos", "Croatia": "Croacia",
    "Germany": "Alemania", "Algeria": "Argelia", "DR Congo": "RD Congo",
    "República Democrática del Congo": "RD Congo", "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Costa de Marfil", "South Korea": "Corea del Sur", "New Zealand": "Nueva Zelanda",
    "Saudi Arabia": "Arabia Saudita", "Czechia": "República Checa", "Turkey": "Turquía",
    "Scotland": "Escocia", "Ireland": "Irlanda", "Northern Ireland": "Irlanda del Norte",
    "Wales": "Gales", "Poland": "Polonia", "Denmark": "Dinamarca", "Russia": "Rusia",
    "Ukraine": "Ucrania", "Greece": "Grecia", "North Macedonia": "Macedonia del Norte",
    "Iceland": "Islandia", "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Jordan": "Jordania",
    "Iraq": "Irak", "Iran": "Irán", "Uzbekistan": "Uzbekistán", "Tunisia": "Túnez",
    "Cameroon": "Camerún", "Nigeria": "Nigeria"
}

MAPA_TORNEOS = {"World Cup": "World Cups"}

# Normalización inicial de datos importados
teams_base = {MAPA_EQUIPOS.get(k, k): v for k, v in teams_base.items()}
metricas_extra = {MAPA_EQUIPOS.get(k, k): v for k, v in metricas_extra.items()}


print("\n🏠 AJUSTE DE ELO POR LOCALÍA (ANFITRIONES 2026)")
for eq in ANFITRIONES_2026:
    if eq in teams_base:
        elo_actual = teams_base[eq][0]
        teams_base[eq] = [elo_actual + 50, teams_base[eq][1]]
        print(f"  ✅ {eq}: ELO {elo_actual} → {elo_actual + 50} (+50)")
print("─" * 50)

dict_potencias = {
    MAPA_TORNEOS.get(d["torneo"], d["torneo"]): d["potencia_torneo"] 
    for d in datos_potencia_torneos_2026
}

# ══════════════════════════════════════════════════════════════════════════════
# §2 PREPROCESAMIENTO: SEPARACIÓN 90' vs 120'
# ══════════════════════════════════════════════════════════════════════════════

def registrar_prorrogas(partidos, lista_prorrogas):
    """
    Separa explícitamente los goles a 90' de los goles a 120'.
    El modelo principal solo verá los goles a 90'.
    """
    partidos_copia = [p.copy() for p in partidos]
    for eq1, eq2, g90_1, g90_2, g120_1, g120_2 in lista_prorrogas:
        for p in partidos_copia:
            if {p["equipo1"], p["equipo2"]} == {eq1, eq2}:
                # Guardamos el resultado a 120'
                p["goles_120_equipo1"] = g120_1 if p["equipo1"] == eq1 else g120_2
                p["goles_120_equipo2"] = g120_2 if p["equipo1"] == eq1 else g120_1
                
                # Sobrescribimos el resultado a 90' (el modelo principal usará esto)
                p["goles_equipo1"] = g90_1 if p["equipo1"] == eq1 else g90_2
                p["goles_equipo2"] = g90_2 if p["equipo1"] == eq1 else g90_1
                break
    return partidos_copia

# Registramos los partidos que fueron a prórroga
partidos_con_90_separados = registrar_prorrogas(partidos_filtrados, [
    ("Argentina", "Cape Verde", 2, 2, 3, 2),
    ("Colombia", "Switzerland", 0, 0, 0, 0),
    ("Australia", "Egypt", 1, 1, 1, 1),
    ("Germany", "Paraguay", 1, 1, 1, 1), 
    ("Netherlands", "Morocco", 1, 1, 1, 1)
])

# ✅ NORMALIZACIÓN Y FILTRADO (TODO EN UN SOLO PASO)
partidos_normalizados = [
    {**p, "equipo1": MAPA_EQUIPOS.get(p["equipo1"], p["equipo1"]), 
          "equipo2": MAPA_EQUIPOS.get(p["equipo2"], p["equipo2"])}
    for p in partidos_con_90_separados
]

datos_normalizados = [
    {**r, "equipo": MAPA_EQUIPOS.get(r["equipo"], r["equipo"]), 
         "torneo": MAPA_TORNEOS.get(r["torneo"], r["torneo"])}
    for r in datos_torneos_completos
]

# ✅ FILTROS PARA EQUIPOS EN teams_base
partidos_con_ambos_en_teams = [
    p for p in partidos_normalizados
    if p["equipo1"] in teams_base and p["equipo2"] in teams_base
]

datos_con_equipos_en_teams = [
    r for r in datos_normalizados
    if r["equipo"] in teams_base
]

print(f"📊 Partidos totales: {len(partidos_normalizados)} → Filtrados: {len(partidos_con_ambos_en_teams)}")
print(f"📊 Equipos en datos: {len(set(r['equipo'] for r in datos_normalizados))} → Filtrados: {len(set(r['equipo'] for r in datos_con_equipos_en_teams))}")

# ══════════════════════════════════════════════════════════════════════════════
# §3 FASE 2: MOTOR EM (STATS LATENTES Y DIFICULTAD DE TORNEO)
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
def torneo_confianza(pot): return sigmoid((pot - BASE_POT) / 150.0)
def peso_observacion(n_partidos, pot): return n_partidos * torneo_confianza(pot)

def _e_step(datos, dict_potencias, params, lambda_elo=5.0):
    num = {"atk": defaultdict(float), "def": defaultdict(float), "gf": defaultdict(float), "ga": defaultdict(float)}
    den = {"atk": defaultdict(float), "def": defaultdict(float)}

    for r in datos:
        eq = r["equipo"]
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x_j = (pot - BASE_POT) / 200.0

        g_atk = math.exp(params["atk"][0] + params["atk"][1] * x_j)
        g_def = math.exp(params["def"][0] + params["def"][1] * x_j)

        w = peso_observacion(r["partidos_jugados"], pot)

        num["atk"][eq] += w * r["xg_90"] / g_atk
        num["gf"][eq] += w * r.get("goles_favor_90", r["xg_90"]) / g_atk
        den["atk"][eq] += w

        num["def"][eq] += w * r["xga_90"] / g_def
        num["ga"][eq] += w * r.get("goles_contra_90", r["xga_90"]) / g_def
        den["def"][eq] += w

    latente = {}
    for eq in den["atk"]:
        if den["atk"][eq] > 0.0:
            latente[eq] = {
                "atk": (num["atk"][eq] + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "gf": (num["gf"][eq] + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "def": (num["def"][eq] + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
                "ga": (num["ga"][eq] + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
            }
    return latente

def _m_step(datos, latente, dict_potencias):
    xs_atk, ys_atk, xs_def, ys_def = [], [], [], []
    PESO_XG, PESO_GF = 1.0, 0.3

    for r in datos:
        eq = r["equipo"]
        if eq not in latente: 
            continue
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x = (pot - BASE_POT) / 200.0

        if latente[eq]["atk"] > 1e-9:
            ratio_xg = r["xg_90"] / latente[eq]["atk"]
            xs_atk.extend([x] * int(PESO_XG * 10))
            ys_atk.extend([math.log(max(ratio_xg, EPS_LOG))] * int(PESO_XG * 10))

        if latente[eq]["gf"] > 1e-9:
            ratio_gf = r.get("goles_favor_90", r["xg_90"]) / latente[eq]["gf"]
            xs_atk.extend([x] * int(PESO_GF * 10))
            ys_atk.extend([math.log(max(ratio_gf, EPS_LOG))] * int(PESO_GF * 10))

        if latente[eq]["def"] > 1e-9:
            ratio_xga = r["xga_90"] / latente[eq]["def"]
            xs_def.extend([x] * int(PESO_XG * 10))
            ys_def.extend([math.log(max(ratio_xga, EPS_LOG))] * int(PESO_XG * 10))

        if latente[eq]["ga"] > 1e-9:
            ratio_ga = r.get("goles_contra_90", r["xga_90"]) / latente[eq]["ga"]
            xs_def.extend([x] * int(PESO_GF * 10))
            ys_def.extend([math.log(max(ratio_ga, EPS_LOG))] * int(PESO_GF * 10))

    def fit_ols(xs, ys):
        if len(xs) < 2: 
            return 0.0, 0.0
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - mx) ** 2 for x in xs)
        if var > 1e-12:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
        else:
            b = 0.0
        return my - b * mx, b

    a_atk, b_atk = fit_ols(xs_atk, ys_atk)
    a_def, b_def = fit_ols(xs_def, ys_def)
    
    return {"atk": (a_atk, b_atk), "def": (a_def, b_def)}

def estimar_modelo_latente(datos, dict_potencias, max_iter=20, tol=1e-5):
    params = {"atk": (0.0, 0.0), "def": (0.0, 0.0)}
    for i in range(max_iter):
        latente = _e_step(datos, dict_potencias, params)
        params_new = _m_step(datos, latente, dict_potencias)
        
        print(f"Iteración {i+1}: b_atk={params_new['atk'][1]:.4f}, b_def={params_new['def'][1]:.4f}")
        
        if abs(params_new["atk"][1] - params["atk"][1]) < tol and \
           abs(params_new["def"][1] - params["def"][1]) < tol:
            print(f"✅ Convergió en {i+1} iteraciones")
            break
        params = params_new
    return latente, params_new

# ✅ USANDO DATOS FILTRADOS
print("⏳ Entrenando modelo latente...")
_, PARAMS_TORNEO = estimar_modelo_latente(datos_con_equipos_en_teams, dict_potencias)

print(f"\n📊 PARÁMETROS FINALES:")
print(f"  Ataque:  a={PARAMS_TORNEO['atk'][0]:.4f}, b={PARAMS_TORNEO['atk'][1]:.4f}")
print(f"  Defensa: a={PARAMS_TORNEO['def'][0]:.4f}, b={PARAMS_TORNEO['def'][1]:.4f}")

if PARAMS_TORNEO['atk'][1] > 0:
    print("⚠️  WARNING: b_atk es POSITIVO (debería ser negativo)")
if PARAMS_TORNEO['def'][1] < 0:
    print("⚠️  WARNING: b_def es NEGATIVO (debería ser positivo)")

# ══════════════════════════════════════════════════════════════════════════════
# §4 ACUMULACIÓN DE STATS Y MÉTRICAS EXTRA
# ══════════════════════════════════════════════════════════════════════════════

def proyectar_a_mundial(q, pot, b, tipo="atk"):
    x = (pot - BASE_POT) / 200.0
    if tipo == "def":
        return q * math.exp(b * x)
    else:
        return q * math.exp(-b * x)

a_atk, b_atk = PARAMS_TORNEO["atk"]
a_def, b_def = PARAMS_TORNEO["def"]
FACTOR_RECENCIA_MUNDIAL = 3

stats_acumuladas = defaultdict(lambda: {"xg": 0.0, "xga": 0.0, "gf": 0.0, "ga": 0.0, "w": 0.0})

# ✅ USANDO DATOS FILTRADOS
for r in datos_con_equipos_en_teams:
    eq = r["equipo"]
    pot = dict_potencias.get(r["torneo"], BASE_POT)
    
    w = peso_observacion(r["partidos_jugados"], pot)
    
    if r["torneo"] in ["World Cup", "World Cups"] and r["partidos_jugados"] <= 5:
        w *= FACTOR_RECENCIA_MUNDIAL
    
    s = stats_acumuladas[eq]
    
    s["xg"] += proyectar_a_mundial(r["xg_90"], pot, b_atk, "atk") * w
    s["gf"] += proyectar_a_mundial(r.get("goles_favor_90", r["xg_90"]), pot, b_atk, "atk") * w
    s["xga"] += proyectar_a_mundial(r["xga_90"], pot, b_def, "def") * w
    s["ga"] += proyectar_a_mundial(r.get("goles_contra_90", r["xga_90"]), pot, b_def, "def") * w
    s["w"] += w

print("\n📊 STATS ACUMULADAS (equipos relevantes):")
equipos_clave = ["Argentina", "Brasil", "Francia", "Inglaterra", "España", "Alemania"]
for eq in equipos_clave:
    if eq in stats_acumuladas:
        s = stats_acumuladas[eq]
        print(f"  {eq:12s}: xG={s['xg']/s['w']:.3f}, xGA={s['xga']/s['w']:.3f}, w={s['w']:.1f}")

def get_elo_prior(eq, tipo="atk"):
    elo = teams_base.get(eq, [BASE_ELO])[0]
    signo = 1.0 if tipo == "atk" else -1.0
    return PRIOR_MEAN * math.exp(signo * (elo - BASE_ELO) / 600.0)

W_XG_ATK = 0.78
W_XG_DEF = 0.92

equipos_stats = {}
peso_prior = PRIOR_PJ * torneo_confianza(BASE_POT)          

for eq, s in stats_acumuladas.items():
    denom = s["w"] + peso_prior
    prior_atk, prior_def = get_elo_prior(eq, "atk"), get_elo_prior(eq, "def")
    
    xg_base  = (s["xg"]  + peso_prior * prior_atk) / denom
    gf_base  = (s["gf"]  + peso_prior * prior_atk) / denom
    xga_base = (s["xga"] + peso_prior * prior_def) / denom
    ga_base  = (s["ga"]  + peso_prior * prior_def) / denom
    
    equipos_stats[eq] = {
        "ataque_mix": (xg_base * W_XG_ATK) + (gf_base * (1.0 - W_XG_ATK)),
        "defensa_mix": (xga_base * W_XG_DEF) + (ga_base * (1.0 - W_XG_DEF))
    }

MAX_RANK = max((d["performance_rank"] for d in metricas_extra.values() if "performance_rank" in d), default=32)

def obtener_elo_efectivo(equipo):
    elo_base = teams_base.get(equipo, [BASE_ELO])[0]
    extra = metricas_extra.get(equipo, {})
    
    rank_actual = extra.get("performance_rank", MAX_RANK)
    boost_rank = abs(rank_actual - MAX_RANK) * 0.3
    
    elo_torneo = extra.get("diferencia_elo_torneo", 0)
    elo_pre_torneo = extra.get("diferencia_elo_año", 0) - elo_torneo
    boost_forma = 0.5 * elo_torneo + 0.15 * elo_pre_torneo
    
    boost_loc = 50.0 if equipo in ANFITRIONES_2026 else 0.0
    
    return elo_base + boost_rank + boost_forma + boost_loc






# ══════════════════════════════════════════════════════════════════════════════
# IMPRESIÓN DE LA TABLA (Ahora usa la misma función que el modelo)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 90)
print(f" DESGLOSE DE ELO EFECTIVO Y BOOSTS (MAX_RANK = {MAX_RANK})")
print("═" * 90)
print(f"{'EQUIPO':<16} | {'ELO BASE':<9} | {'BST RANK':<9} | {'BST FORMA':<10} | {'LOCALÍA':<8} | {'ELO REAL (TOTAL)'}")
print("─" * 90)

# Ordenamos usando la función oficial
equipos_ordenados = sorted(metricas_extra.keys(), key=obtener_elo_efectivo, reverse=True)

for eq in equipos_ordenados:
    elo_base = teams_base.get(eq, [BASE_ELO])[0]
    extra = metricas_extra.get(eq, {})
    
    # Recalculamos solo para mostrar en las columnas (usando los mismos pesos)
    rank_actual = extra.get("performance_rank", MAX_RANK)
    boost_rank = abs(rank_actual - MAX_RANK) * 0.3  # <-- Igual que arriba
    
    elo_torneo = extra.get("diferencia_elo_torneo", 0)
    elo_pre_torneo = extra.get("diferencia_elo_año", 0) - elo_torneo
    boost_forma = 0.5 * elo_torneo + 0.15 * elo_pre_torneo
    
    boost_loc = 50.0 if eq in ANFITRIONES_2026 else 0.0
    
    elo_total = obtener_elo_efectivo(eq) # Llamamos a la función real
    boost_acum = boost_rank + boost_forma + boost_loc
    
    print(f"{eq:<16} | {elo_base:<9.1f} | {boost_rank:+9.1f} | {boost_forma:+10.1f} | {boost_loc:+8.1f} | {elo_total:<8.1f} ({boost_acum:+.1f})")


def obtener_riesgo_partido(t1, t2):
    r1 = metricas_extra.get(t1, {}).get("prediction_risk", 25)
    r2 = metricas_extra.get(t2, {}).get("prediction_risk", 25)
    return min(1.0, ((r1 + r2) / 2.0) / 150.0)

# ══════════════════════════════════════════════════════════════════════════════
# §6 FASE 3: MODELO COMBINADO (90') Y FACTOR DE PRÓRROGA
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelo_90_minutos(partidos, equipos_stats):
    """Entrena el GLM Poisson y optimiza Alpha mediante Máxima Verosimilitud."""
    X_90, Y_90 = [], []
    for p in partidos:
        elo1, elo2 = obtener_elo_efectivo(p["equipo1"]), obtener_elo_efectivo(p["equipo2"])
        
        # Ahora usamos ataque_mix y defensa_mix
        s1 = equipos_stats.get(p["equipo1"], {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
        s2 = equipos_stats.get(p["equipo2"], {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})

        g90_1 = p.get("goles_90_equipo1", p["goles_equipo1"])
        g90_2 = p.get("goles_90_equipo2", p["goles_equipo2"])

        delta_elo = (elo1 - elo2) / 100.0

        X_90.extend([
            {
                "delta_elo": delta_elo,
                "ataque_mix": s1["ataque_mix"],
                "defensa_mix": s2["defensa_mix"]
            },
            {
                "delta_elo": -delta_elo,
                "ataque_mix": s2["ataque_mix"],
                "defensa_mix": s1["defensa_mix"] # Corregido para cruzar defensa
            }
        ])
        Y_90.extend([g90_1, g90_2])

    df = pd.DataFrame(X_90)
    
    # Entrenamos los Betas con Poisson y Ridge
    modelo = sm.GLM(
        Y_90,
        df[["delta_elo", "ataque_mix", "defensa_mix"]],
        family=sm.families.Poisson()
    )
    res = modelo.fit_regularized(L1_wt=0.0, alpha=0.1)
    
    # --- OPTIMIZACIÓN DE ALPHA (Maximum Likelihood) ---
    mu = res.predict()
    Y_np = np.array(Y_90)
    
    def nb2_log_likelihood(params):
        a = params[0]
        # Ecuación vectorizada de Log-Verosimilitud Negativa para NB2
        ll = np.sum(gammaln(Y_np + 1.0/a) - gammaln(Y_np + 1.0) - gammaln(1.0/a) + 
                    (1.0/a) * np.log(1.0 / (1.0 + a * mu)) + 
                    Y_np * np.log((a * mu) / (1.0 + a * mu)))
        return -ll # Retornamos negativo porque 'minimize' busca el mínimo

    # Buscamos el alpha óptimo entre 0.0001 y 1.0
    opt = minimize(nb2_log_likelihood, x0=[0.05], bounds=[(0.0001, 1.0)])
    alpha_base_optimizado = opt.x[0]
    
    return res.params.to_dict(), alpha_base_optimizado

def calcular_factor_prorroga(partidos):
    """Aprende empíricamente el ritmo de goles en prórroga vs 90'."""
    goles_90_total, goles_et_total = 0, 0
    for p in partidos:
        if "goles_120_equipo1" in p:
            g90_1 = p.get("goles_90_equipo1", p["goles_equipo1"])
            g90_2 = p.get("goles_90_equipo2", p["goles_equipo2"])
            g120_1, g120_2 = p["goles_120_equipo1"], p["goles_120_equipo2"]
            
            goles_90_total += (g90_1 + g90_2)
            goles_et_total += (g120_1 - g90_1) + (g120_2 - g90_2)
            
    if goles_90_total > 0:
        factor_bruto = goles_et_total / goles_90_total
        return max(0.05, min(0.25, factor_bruto)) # Acotado entre 5% y 25%
    return 0.18

BETAS, ALPHA_BASE = entrenar_modelo_90_minutos(partidos_con_ambos_en_teams, equipos_stats)
FACTOR_ET_MODELO = calcular_factor_prorroga(partidos_con_ambos_en_teams)


print(f"--- PARÁMETROS APRENDIDOS ---")
print(f"Alpha Base (NB2) : {ALPHA_BASE:.5f}")
print(f"Factor ET        : {FACTOR_ET_MODELO:.3f} (Ritmo de goles en prórroga)")

# ══════════════════════════════════════════════════════════════════════════════
# §7 FASE 4: MOTOR DE SIMULACIÓN (NB2 + CÓPULA)
# ══════════════════════════════════════════════════════════════════════════════

def prob_nb2(mu, alpha, k):
    if alpha < 1e-9:
        return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if mu > 1e-12 else (1.0 if k == 0 else 0.0)
    r, p = 1.0 / alpha, (1.0 / alpha) / ((1.0 / alpha) + mu)
    return math.exp(math.lgamma(k + r) - math.lgamma(k + 1) - math.lgamma(r) + r * math.log(p) + k * math.log1p(-p))

def cdf_nb2(mu, alpha, k):
    return sum(prob_nb2(mu, alpha, x) for x in range(k + 1)) if k >= 0 else 0.0

def frank_copula(u, v, theta):
    if abs(theta) < 1e-5: return u * v 
    num = (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
    den = math.exp(-theta) - 1.0
    adentro_log = 1.0 + num / den
    return - (1.0 / theta) * math.log(adentro_log) if adentro_log > 0 else 0.0

def normalizar_dist(dist, etiqueta=""):
    s = sum(dist.values())
    if s <= 1e-12: return dist
    if (1.0 - s) > 0.01: warnings.warn(f"Truncación no despreciable en {etiqueta}: masa perdida = {1.0 - s:.4%}")
    return {k: v / s for k, v in dist.items()}

def generar_distribucion(l1, l2, alpha, min_goles=0, max_goles=10, theta_copula=0.38):
    dist = {}
    for i in range(min_goles, max_goles):
        for j in range(min_goles, max_goles):
            u1, v1 = cdf_nb2(l1, alpha, i), cdf_nb2(l2, alpha, j)
            u0, v0 = cdf_nb2(l1, alpha, i - 1), cdf_nb2(l2, alpha, j - 1)
            C11 = frank_copula(u1, v1, theta_copula)
            C01 = frank_copula(u0, v1, theta_copula)
            C10 = frank_copula(u1, v0, theta_copula)
            C00 = frank_copula(u0, v0, theta_copula)
            dist[(i, j)] = max(C11 - C01 - C10 + C00, 0.0)
    return dist

def calcular_probabilidades(t1, t2, eliminatoria=True):
    elo1, elo2 = obtener_elo_efectivo(t1), obtener_elo_efectivo(t2)

    # 1. Usar las nuevas claves por defecto
    s1 = equipos_stats.get(t1, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    
    delta_elo = (elo1 - elo2) / 100.0  # Escalado
    factor_riesgo = obtener_riesgo_partido(t1, t2)
    
    # 2. Usar los nuevos nombres del diccionario BETAS y de las stats
    lam1_90 = math.exp(
        BETAS["delta_elo"] * delta_elo +
        BETAS["ataque_mix"] * s1["ataque_mix"] +
        BETAS["defensa_mix"] * s2["defensa_mix"]
    )

    lam2_90 = math.exp(
        BETAS["delta_elo"] * (-delta_elo) +
        BETAS["ataque_mix"] * s2["ataque_mix"] +
        BETAS["defensa_mix"] * s1["defensa_mix"]
    )
    
    # 3. Alpha dinámico 
    alpha_90 = ALPHA_BASE + 0.002 * abs(delta_elo) + (0.01 * factor_riesgo)
    prob_90 = normalizar_dist(generar_distribucion(lam1_90, lam2_90, alpha_90), f"90' ({t1} vs {t2})")




    top_90 = sorted(prob_90.items(), key=lambda x: x[1], reverse=True)[:5]
    exp_goals_90 = sum((i + j) * p for (i, j), p in prob_90.items())
    prob_draw_90 = sum(p for (i, j), p in prob_90.items() if i == j)

    if not eliminatoria:
        return {"top_90": top_90, "exp_goals_90": exp_goals_90, "prob_draw_90": prob_draw_90}

    # 2. MODELO DE PRÓRROGA (SOLO SI HAY EMPATE A 90')
    lam1_et = lam1_90 * FACTOR_ET_MODELO
    lam2_et = lam2_90 * FACTOR_ET_MODELO
    alpha_et = alpha_90 * FACTOR_ET_MODELO 
    
    prob_et = normalizar_dist(generar_distribucion(lam1_et, lam2_et, alpha_et, max_goles=5), f"ET ({t1} vs {t2})")
    
    # Convolución para obtener resultado a 120'
    prob_120 = {}
    for (i, j), p90 in prob_90.items():
        if i != j:
            prob_120[(i, j)] = prob_120.get((i, j), 0.0) + p90
        else:
            for (ea, eb), pet in prob_et.items():
                prob_120[(i + ea, j + eb)] = prob_120.get((i + ea, j + eb), 0.0) + (p90 * pet)
                
    prob_120 = {k: v / sum(prob_120.values()) for k, v in prob_120.items()}
    
    p_win1 = sum(p for (i, j), p in prob_120.items() if i > j)
    p_draw = sum(p for (i, j), p in prob_120.items() if i == j) 
    p_win2 = sum(p for (i, j), p in prob_120.items() if i < j)
    
    # 3. MODELO DE PENALES
    pen_edge = 0.08 * (1.0 - 0.2 * factor_riesgo)  # ±8% vs ±4.5%
    p_pen1 = max(0.46, min(0.54, 0.5 + pen_edge * math.tanh(delta_elo / 300.0)))

    return {
        "top_90": top_90, "exp_goals_90": exp_goals_90, "prob_draw_90": prob_draw_90,
        "top_120": sorted(prob_120.items(), key=lambda x: x[1], reverse=True)[:5],
        "prob_local": p_win1 + p_draw * p_pen1,   
        "prob_visita": p_win2 + p_draw * (1.0 - p_pen1),  
        "prob_local_120": p_win1, "prob_draw_120": p_draw, "prob_visita_120": p_win2
    }

# ══════════════════════════════════════════════════════════════════════════════
# §8 EJECUCIÓN Y AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════

codes = {
    "Sudáfrica": "AFS", "Canadá": "CAN", "Brasil": "BRA", "Japón": "JPN", "Alemania": "ALE",
    "Paraguay": "PAR", "Países Bajos": "NED", "Marruecos": "MAR", "Costa de Marfil": "CIV",
    "Noruega": "NOR", "Francia": "FRA", "Suecia": "SUE", "México": "MEX", "Ecuador": "ECU",
    "Inglaterra": "ENG", "RD Congo": "RDC", "Bélgica": "BEL", "Senegal": "SEN", "Estados Unidos": "USA",
    "Bosnia y Herzegovina": "BIH", "España": "ESP", "Austria": "AUT", "Portugal": "POR",
    "Croacia": "CRO", "Suiza": "SUI", "Argelia": "ALG", "Australia": "AUS", "Egipto": "EGY",
    "Argentina": "ARG", "Cabo Verde": "CPV", "Colombia": "COL", "Ghana": "GHA"
}

matches = [
    ("Francia", "Marruecos"), ("España", "Bélgica"), 
    ("Noruega", "Inglaterra"), ("Argentina", "Suiza"),("Argentina","Cabo Verde"),("Francia","Noruega"), ("Francia","Canadá"),("Paraguay","Francia"),("Francia","Paraguay"),("México","Inglaterra"),("Alemania","Paraguay")
]

w_partido, w_top, w_probs = 35, 35, 25
def _fmt(lista): return ", ".join(f"{i}-{j} ({p:.1%})" for (i, j), p in lista)

print(f"{'PARTIDO (Pasa de ronda)':<{w_partido}} | {'TOP 90 MINS':<{w_top}} | {'TOP 120 MINS':<{w_top}} | {'PROBS 120'}")
print("─" * (w_partido + w_top + w_top + w_probs + 10))

sum_goals_90, sum_draw_90, n = 0.0, 0.0, len(matches)
for t1, t2 in matches:
    try:
        res = calcular_probabilidades(t1, t2, eliminatoria=True)
        t1_c, t2_c = codes.get(t1, t1[:3].upper()), codes.get(t2, t2[:3].upper())
        fav = t1 if res['prob_local'] > res['prob_visita'] else t2
        fav_c = codes.get(fav, fav[:3].upper())
        prob_fav = max(res['prob_local'], res['prob_visita'])

        print(f"{t1_c} vs {t2_c} ({fav_c} {prob_fav:.0%}):<{w_partido-10} | {_fmt(res['top_90'][:3]):<{w_top}} | {_fmt(res['top_120'][:3]):<{w_top}} | L:{res['prob_local_120']:.0%} E:{res['prob_draw_120']:.0%} V:{res['prob_visita_120']:.0%}")
        sum_goals_90 += res["exp_goals_90"]
        sum_draw_90  += res["prob_draw_90"]
    except Exception as e:
        print(f"Error en {t1} vs {t2}: {e}")

print("-" * (w_partido + w_top + w_top + w_probs + 10))
print(f"Goles esperados promedio: {sum_goals_90 / n:.2f}")
print(f"Empates esperados promedio: {sum_draw_90 / n:.2%}")

# Auditoría de Parámetros
print("\n" + "═" * 90)
print(" AUDITORÍA DE PARÁMETROS DEL MODELO")
print("═" * 90)

print(f"► MOTOR EM: Atk(a,b)=({PARAMS_TORNEO['atk'][0]:.4f}, {PARAMS_TORNEO['atk'][1]:.4f}) | Def(a,b)=({PARAMS_TORNEO['def'][0]:.4f}, {PARAMS_TORNEO['def'][1]:.4f})")
print(f"► GLM 90': Betas={BETAS} | Alpha={ALPHA_BASE:.5f} | Factor ET={FACTOR_ET_MODELO:.3f}")
print("═" * 90 + "\n")

# Después de PARAMS_TORNEO (línea 156)
# Verificar que los signos son correctos para la proyección
print(f"\n📊 PARÁMETROS FINALES:")
print(f"  Ataque:  a={PARAMS_TORNEO['atk'][0]:.4f}, b={PARAMS_TORNEO['atk'][1]:.4f}")
print(f"  Defensa: a={PARAMS_TORNEO['def'][0]:.4f}, b={PARAMS_TORNEO['def'][1]:.4f}")

# ✅ CORRECCIÓN: b_def POSITIVO es lo que queremos
if PARAMS_TORNEO['atk'][1] > 0:
    print("⚠️  WARNING: b_atk es POSITIVO (esperado NEGATIVO para que ataque aumente)")
# ✅ b_def positivo es CORRECTO (aumenta el xGA)
if PARAMS_TORNEO['def'][1] < 0:
    print("⚠️  WARNING: b_def es NEGATIVO (esperado POSITIVO para que defensa empeore)")


# ══════════════════════════════════════════════════════════════════════════════
# §9 EVALUACIÓN IN-SAMPLE Y CALIBRACIÓN (90 MINUTOS)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 90)
print(" EVALUACIÓN Y CALIBRACIÓN DEL MODELO (Datos de Entrenamiento)")
print("═" * 90)

y_true = [] # 0: Local, 1: Empate, 2: Visita
probs_matrix = []
xg_model_total = 0.0
goles_reales_total = 0

# Estructura para agrupar predicciones (Buckets de 10%)
calibracion = {"count": [0]*10, "pred_sum": [0.0]*10, "hits_real": [0]*10}

for p in partidos_con_ambos_en_teams:
    t1, t2 = p["equipo1"], p["equipo2"]
    g1 = p.get("goles_90_equipo1", p["goles_equipo1"])
    g2 = p.get("goles_90_equipo2", p["goles_equipo2"])
    
    goles_reales_total += (g1 + g2)
    
    if g1 > g2: outcome = 0
    elif g1 == g2: outcome = 1
    else: outcome = 2
    y_true.append(outcome)
    
    # Recalcular las probabilidades puras a 90' para cada partido
    elo1, elo2 = obtener_elo_efectivo(t1), obtener_elo_efectivo(t2)
    s1 = equipos_stats.get(t1, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    delta_elo = (elo1 - elo2) / 100.0
    factor_riesgo = obtener_riesgo_partido(t1, t2)
    
    lam1 = math.exp(BETAS["delta_elo"] * delta_elo + BETAS["ataque_mix"] * s1["ataque_mix"] + BETAS["defensa_mix"] * s2["defensa_mix"])
    lam2 = math.exp(BETAS["delta_elo"] * (-delta_elo) + BETAS["ataque_mix"] * s2["ataque_mix"] + BETAS["defensa_mix"] * s1["defensa_mix"])
    alpha = ALPHA_BASE + 0.002 * abs(delta_elo) + (0.01 * factor_riesgo)
    
    dist = normalizar_dist(generar_distribucion(lam1, lam2, alpha))
    
    p_l = sum(prob for (i, j), prob in dist.items() if i > j)
    p_e = sum(prob for (i, j), prob in dist.items() if i == j)
    p_v = sum(prob for (i, j), prob in dist.items() if i < j)
    
    probs_matrix.append([p_l, p_e, p_v])
    xg_model_total += sum((i+j)*prob for (i,j), prob in dist.items())
    
    # Evaluar calibración (Solo para el equipo Local/Favorito para simplificar)
    idx_l = min(9, int(p_l * 10))
    calibracion["count"][idx_l] += 1
    calibracion["pred_sum"][idx_l] += p_l
    if outcome == 0: 
        calibracion["hits_real"][idx_l] += 1

# Cálculo de Brier Score y Precisión
n_partidos = len(y_true)
brier_score = 0
aciertos = 0

for i in range(n_partidos):
    probs = probs_matrix[i]
    real = y_true[i]
    
    if probs.index(max(probs)) == real:
        aciertos += 1
        
    for c in range(3):
        is_real_outcome = 1 if c == real else 0
        brier_score += (probs[c] - is_real_outcome)**2
        
brier_score /= n_partidos

# Impresión de Métricas Generales
print(f"Partidos evaluados   : {n_partidos}")
print(f"Precisión (Hit Rate) : {aciertos/n_partidos:.2%} (Ideal > 50% en fútbol)")
print(f"Brier Score Multiclase: {brier_score:.4f} (0.0=Perfecto, ~0.66=Azar)")
print("-" * 50)
print(f"Goles Reales Totales : {goles_reales_total}")
print(f"Goles Modelo (Suma)  : {xg_model_total:.1f}")

# Impresión de Tabla de Calibración
print("\nTABLA DE CALIBRACIÓN (Predicciones de Victoria Local)")
print("Rango Prob  | Promedio Pred | Frecuencia Real | Muestra (N)")
print("─────────────────────────────────────────────────────────────")
for i in range(10):
    c = calibracion["count"][i]
    if c > 0:
        avg_pred = calibracion["pred_sum"][i] / c
        freq_real = calibracion["hits_real"][i] / c
        print(f"{i*10:02d}% - {i*10+10:02d}% |     {avg_pred:.1%}     |      {freq_real:.1%}      | {c}")
print("══════════════════════════════════════════════════════════════════════════════════════════")

# ══════════════════════════════════════════════════════════════════════════════
# §10 EVALUACIÓN DE MARCADORES EXACTOS (SIMÉTRICA: GANADOR - PERDEDOR)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 90)
print(" CALIBRACIÓN DE RESULTADOS SIMÉTRICOS (Ganador - Perdedor)")
print("═" * 90)

from collections import defaultdict

frecuencia_real_simetrica = defaultdict(int)
prob_acumulada_modelo_simetrica = defaultdict(float)

for p in partidos_con_ambos_en_teams:
    t1, t2 = p["equipo1"], p["equipo2"]
    g1 = p.get("goles_90_equipo1", p["goles_equipo1"])
    g2 = p.get("goles_90_equipo2", p["goles_equipo2"])
    
    # 1. Registrar lo que pasó en la realidad (Ordenamos: Mayor - Menor)
    marcador_real = tuple(sorted([g1, g2], reverse=True))
    frecuencia_real_simetrica[marcador_real] += 1
    
    # 2. Recrear la matriz del modelo para este partido
    elo1, elo2 = obtener_elo_efectivo(t1), obtener_elo_efectivo(t2)
    s1 = equipos_stats.get(t1, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    delta_elo = (elo1 - elo2) / 100.0
    factor_riesgo = obtener_riesgo_partido(t1, t2)
    
    lam1 = math.exp(BETAS["delta_elo"] * delta_elo + BETAS["ataque_mix"] * s1["ataque_mix"] + BETAS["defensa_mix"] * s2["defensa_mix"])
    lam2 = math.exp(BETAS["delta_elo"] * (-delta_elo) + BETAS["ataque_mix"] * s2["ataque_mix"] + BETAS["defensa_mix"] * s1["defensa_mix"])
    alpha = ALPHA_BASE + 0.002 * abs(delta_elo) + (0.01 * factor_riesgo)
    
    dist = normalizar_dist(generar_distribucion(lam1, lam2, alpha))
    
    # Acumulamos la probabilidad sumando ambas combinaciones (ej: 1-0 y 0-1)
    for (i, j), prob in dist.items():
        marcador_modelo = tuple(sorted([i, j], reverse=True))
        prob_acumulada_modelo_simetrica[marcador_modelo] += prob

# Ordenar los resultados reales por cantidad de apariciones
top_resultados_simetricos = sorted(frecuencia_real_simetrica.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"{'Ganador-Perdedor':<18} | {'Apariciones Reales':<20} | {'% Real':<10} | {'% Promedio Modelo':<18} | {'Diferencia'}")
print("─" * 85)

for marcador, apariciones in top_resultados_simetricos:
    pct_real = apariciones / n_partidos
    pct_modelo = prob_acumulada_modelo_simetrica[marcador] / n_partidos
    diff = pct_modelo - pct_real
    
    str_marcador = f"{marcador[0]} - {marcador[1]}"
    if marcador[0] == marcador[1]:
        str_marcador += " (Empate)"
        
    print(f"{str_marcador:<18} | {apariciones:<20} | {pct_real:<10.2%} | {pct_modelo:<18.2%} | {diff:+.2%}")
print("══════════════════════════════════════════════════════════════════════════════════════════")

