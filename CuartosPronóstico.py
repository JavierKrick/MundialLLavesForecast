import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import defaultdict

# Asumimos que estos datos provienen de tu módulo local
from datos import (
    datos_potencia_torneos_2026, 
    datos_torneos_completos, 
    partidos_filtrados, 
    teams_base,
    metricas_extra
)

# ══════════════════════════════════════════════════════════════════════════════
# §1 CONSTANTES GLOBALES Y MAPEOS UNIFICADOS
# ══════════════════════════════════════════════════════════════════════════════

BASE_ELO   = 1500   
BASE_POT   = 1850   
PRIOR_MEAN = 1.3    
PRIOR_PJ   = 3.0    
PESO_ELO_MAX = 0.5
PESO_ELO_MIN = 0.45
PESO_STATS   = 1 - PESO_ELO_MAX

ANFITRIONES_2026 = {"México", "Mexico", "Canadá", "Canada", "Estados Unidos", "USA", "EEUU"}

MAPA_EQUIPOS = {
    # Octavos de Final / Copa América / Gold Cup
    "Canada": "Canadá", "Morocco": "Marruecos", "France": "Francia",
    "Brazil": "Brasil", "Norway": "Noruega", "Mexico": "México",
    "England": "Inglaterra", "Spain": "España", "USMNT": "Estados Unidos",
    "United States": "Estados Unidos", "USA": "Estados Unidos",
    "Belgium": "Bélgica", "Switzerland": "Suiza", "Egypt": "Egipto",
    "South Africa": "Sudáfrica", "Sweden": "Suecia", "Japan": "Japón",
    "Netherlands": "Países Bajos", "Croatia": "Croacia", "Germany": "Alemania",
    # Historial Previo
    "Algeria": "Argelia", "DR Congo": "RD Congo",
    "República Democrática del Congo": "RD Congo", "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Costa de Marfil", "South Korea": "Corea del Sur",
    "New Zealand": "Nueva Zelanda", "Saudi Arabia": "Arabia Saudita",
    "Czechia": "República Checa", "Turkey": "Turquía", "Scotland": "Escocia",
    "Ireland": "Irlanda", "Northern Ireland": "Irlanda del Norte",
    "Wales": "Gales", "Poland": "Polonia", "Denmark": "Dinamarca",
    "Russia": "Rusia", "Ukraine": "Ucrania", "Greece": "Grecia",
    "North Macedonia": "Macedonia del Norte", "Iceland": "Islandia",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina", "Jordan": "Jordania",
    "Iraq": "Irak", "Iran": "Irán", "Uzbekistan": "Uzbekistán",
    "Tunisia": "Túnez", "Cameroon": "Camerún", "Nigeria": "Nigeria"
}


teams_base = {MAPA_EQUIPOS.get(k, k): v for k, v in teams_base.items()}
metricas_extra = {MAPA_EQUIPOS.get(k, k): v for k, v in metricas_extra.items()}

MAPA_TORNEOS = {"World Cup": "World Cups"}

datos_normalizados = [
    {**r,
     "equipo": MAPA_EQUIPOS.get(r["equipo"], r["equipo"]),
     "torneo": MAPA_TORNEOS.get(r["torneo"], r["torneo"])}
    for r in datos_torneos_completos
]

partidos_normalizados = [
    {**p,
     "equipo1": MAPA_EQUIPOS.get(p["equipo1"], p["equipo1"]),
     "equipo2": MAPA_EQUIPOS.get(p["equipo2"], p["equipo2"])}
    for p in partidos_filtrados
]

dict_potencias = {
    MAPA_TORNEOS.get(d["torneo"], d["torneo"]): d["potencia_torneo"] 
    for d in datos_potencia_torneos_2026
}
# ══════════════════════════════════════════════════════════════════════════════
# §2 MODELO 1: REGRESIÓN ELO (Histórico)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelo_elo(partidos):
    X, Y = [], []
    for p in partidos:
        d = p["elo_equipo1"] - p["elo_equipo2"]
        X.extend([d, -d])
        Y.extend([p["goles_equipo1"], p["goles_equipo2"]])
    
    df = pd.DataFrame({"goles": Y, "delta_elo": X, "intercepto": 1.0})
    modelo = sm.GLM(df["goles"], df[["intercepto", "delta_elo"]], family=sm.families.Poisson())
    res = modelo.fit()
    return res.params["intercepto"], res.params["delta_elo"]

B0_PROPIO, B1_PROPIO = entrenar_modelo_elo(partidos_normalizados)
B0_ELO_Paper = 0.25
B1_ELO_Paper = 0.0023

B0_ELO = B0_PROPIO * 0.5 + B0_ELO_Paper * 0.5 
B1_ELO = B1_PROPIO * 0.5 + B1_ELO_Paper * 0.5 

# ══════════════════════════════════════════════════════════════════════════════
# §3 MOTOR EM: CALIBRACIÓN DE LA DIFICULTAD DEL TORNEO
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def torneo_confianza(potencia):
    return sigmoid((potencia - BASE_POT) / 150.0)

def peso_observacion(n_partidos, potencia):
    return n_partidos * torneo_confianza(potencia)

def _e_step(datos, dict_potencias, params, lambda_elo=5.0):
    num = {"atk": defaultdict(float), "def": defaultdict(float), "gf": defaultdict(float), "ga": defaultdict(float)}
    den = {"atk": defaultdict(float), "def": defaultdict(float)}

    for r in datos:
        eq  = r["equipo"]
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x_j = (pot - BASE_POT) / 200.0

        # exp(·) > 0 para cualquier argumento finito: la rama "g <= 0" ya no existe.
        g_atk = math.exp(params["atk"][0] + params["atk"][1] * x_j)
        g_def = math.exp(params["def"][0] + params["def"][1] * x_j)

        w = peso_observacion(r["partidos_jugados"], pot)

        num["atk"][eq] += w * r["xg_90"] / g_atk
        num["gf"][eq]  += w * r.get("goles_favor_90", r["xg_90"]) / g_atk
        den["atk"][eq] += w

        num["def"][eq] += w * r["xga_90"] / g_def
        num["ga"][eq]  += w * r.get("goles_contra_90", r["xga_90"]) / g_def
        den["def"][eq] += w

    latente = {}
    for eq in den["atk"]:
        if den["atk"][eq] > 0.0:
            latente[eq] = {
                "atk": (num["atk"][eq] + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "gf":  (num["gf"][eq]  + lambda_elo * PRIOR_MEAN) / (den["atk"][eq] + lambda_elo),
                "def": (num["def"][eq] + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
                "ga":  (num["ga"][eq]  + lambda_elo * PRIOR_MEAN) / (den["def"][eq] + lambda_elo),
            }
    return latente
EPS_LOG = 1e-6

def _m_step(datos, latente, dict_potencias):
    xs_atk, ys_atk, xs_def, ys_def = [], [], [], []
    PESO_XG, PESO_GF = 1.0, 0.3

    for r in datos:
        eq = r["equipo"]
        if eq not in latente: continue
        pot = dict_potencias.get(r["torneo"], BASE_POT)
        x = (pot - BASE_POT) / 200.0

        if latente[eq]["atk"] > 1e-9:
            ratio_xg = r["xg_90"] / latente[eq]["atk"]
            xs_atk.extend([x] * int(PESO_XG * 10))
            ys_atk.extend([math.log(max(ratio_xg, EPS_LOG))] * int(PESO_XG * 10))

        if latente[eq]["gf"] > 1e-9:   # antes se testeaba "atk" acá por error
            ratio_gf = r.get("goles_favor_90", r["xg_90"]) / latente[eq]["gf"]
            xs_atk.extend([x] * int(PESO_GF * 10))
            ys_atk.extend([math.log(max(ratio_gf, EPS_LOG))] * int(PESO_GF * 10))

        if latente[eq]["def"] > 1e-9:
            ratio_xga = r["xga_90"] / latente[eq]["def"]
            xs_def.extend([x] * int(PESO_XG * 10))
            ys_def.extend([math.log(max(ratio_xga, EPS_LOG))] * int(PESO_XG * 10))

        if latente[eq]["ga"] > 1e-9:   # ídem, testeaba "def"
            ratio_ga = r.get("goles_contra_90", r["xga_90"]) / latente[eq]["ga"]
            xs_def.extend([x] * int(PESO_GF * 10))
            ys_def.extend([math.log(max(ratio_ga, EPS_LOG))] * int(PESO_GF * 10))

    def fit_ols(xs, ys):
        if len(xs) < 2: return 0.0, 0.0   # a=0 → g(0)=exp(0)=1 (neutro; OJO: no 1.0 como antes)
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        var = sum((x - mx)**2 for x in xs)
        b = sum((x - mx)*(y - my) for x, y in zip(xs, ys)) / var if var > 1e-12 else 0.0
        return my - b * mx, b

    a_atk, b_atk = fit_ols(xs_atk, ys_atk)
    a_def, b_def = fit_ols(xs_def, ys_def)
    return {"atk": (a_atk, b_atk), "def": (a_def, b_def)}

def estimar_modelo_latente(datos, dict_potencias, max_iter=20, tol=1e-5):
    params = {"atk": (0.0, 0.0), "def": (0.0, 0.0)}  # g(x)=exp(a+bx): neutro en x=0 exige a=0
    for _ in range(max_iter):
        latente = _e_step(datos, dict_potencias, params)
        params_new = _m_step(datos, latente, dict_potencias)
        if abs(params_new["atk"][1] - params["atk"][1]) < tol:
            break
        params = params_new
    return latente, params_new

_, PARAMS_TORNEO = estimar_modelo_latente(datos_normalizados, dict_potencias)

# ══════════════════════════════════════════════════════════════════════════════
# §4 ACUMULACIÓN Y REGULARIZACIÓN ESTRUCTURAL
# ══════════════════════════════════════════════════════════════════════════════

def proyectar_a_mundial(q, pot, b):
    """g(0)/g(x) = exp(-bx): el intercepto 'a' se cancela, así que la
    proyección no depende del punto donde convergió (débilmente anclado).
    exp(-bx) > 0 siempre, así que tampoco hace falta el max(0.0, ...)
    ni el 'else q' de la versión con g = a+bx."""
    x = (pot - BASE_POT) / 200.0
    return q * math.exp(-b * x)

a_atk, b_atk = PARAMS_TORNEO["atk"]
a_def, b_def = PARAMS_TORNEO["def"]

stats_acumuladas = defaultdict(lambda: {"xg": 0.0, "xga": 0.0, "gf": 0.0, "ga": 0.0, "w": 0.0})

FACTOR_RECENCIA_MUNDIAL = 3.5

for r in datos_normalizados:
    eq, pot = r["equipo"], dict_potencias.get(r["torneo"], BASE_POT)
    w = peso_observacion(r["partidos_jugados"], pot)
    
    if r["torneo"] in ["World Cup", "World Cups"] and r["partidos_jugados"] <= 5:
        w *= FACTOR_RECENCIA_MUNDIAL
    
    s = stats_acumuladas[eq]


    s["xg"]  += proyectar_a_mundial(r["xg_90"], pot, b_atk) * w
    s["gf"]  += proyectar_a_mundial(r.get("goles_favor_90", r["xg_90"]), pot, b_atk) * w
    s["xga"] += proyectar_a_mundial(r["xga_90"], pot, b_def) * w
    s["ga"]  += proyectar_a_mundial(r.get("goles_contra_90", r["xga_90"]), pot, b_def) * w
    s["w"]   += w

def get_elo_prior(eq, tipo="atk"):
    elo = teams_base.get(eq, [BASE_ELO])[0]
    signo = 1.0 if tipo == "atk" else -1.0
    return PRIOR_MEAN * math.exp(signo * (elo - BASE_ELO) / 600.0)

equipos_stats = {}
peso_prior = PRIOR_PJ * torneo_confianza(BASE_POT)

for eq, s in stats_acumuladas.items():
    denom = s["w"] + peso_prior
    prior_atk = get_elo_prior(eq, "atk")
    prior_def = get_elo_prior(eq, "def")
    
    xg_base  = (s["xg"]  + peso_prior * prior_atk) / denom
    gf_base  = (s["gf"]  + peso_prior * prior_atk) / denom
    xga_base = (s["xga"] + peso_prior * prior_def) / denom
    ga_base  = (s["ga"]  + peso_prior * prior_def) / denom
    
    finishing = (gf_base / xg_base) if xg_base > 1e-9 else 1.0
    finishing_reg = 1.0 + (finishing - 1.0) * 0.2 
    finishing_reg = max(0.8, min(1.2, finishing_reg))

    equipos_stats[eq] = {
        "xg":  xg_base * finishing_reg,
        "xga": xga_base * 0.9 + ga_base * 0.1 
    }

global_atk = sum(s["xg"] for s in equipos_stats.values()) / len(equipos_stats) if equipos_stats else PRIOR_MEAN
global_def = sum(s["xga"] for s in equipos_stats.values()) / len(equipos_stats) if equipos_stats else PRIOR_MEAN

# ══════════════════════════════════════════════════════════════════════════════
# §4.1 PROCESAMIENTO DE MÉTRICAS EXTRA (MODIFICADORES DINÁMICOS)
# ══════════════════════════════════════════════════════════════════════════════

MAX_RANK = max(
    datos["performance_rank"] 
    for datos in metricas_extra.values() 
    if "performance_rank" in datos
) if metricas_extra else 32

def obtener_elo_efectivo(equipo):
    elo_base = teams_base.get(equipo, [BASE_ELO])[0]
    extra = metricas_extra.get(equipo)
    
    if not extra:
        return elo_base
        
    boost_rank = abs(extra["performance_rank"] - MAX_RANK) * 1.1
    elo_torneo = extra["diferencia_elo_torneo"]
    elo_pre_torneo = extra["diferencia_elo_año"] - elo_torneo
    
    boost_forma = 0.35 * elo_torneo + 0.15 * elo_pre_torneo
    return elo_base + boost_rank + boost_forma

def obtener_riesgo_partido(t1, t2):
    r1 = metricas_extra.get(t1, {}).get("prediction_risk", 25)
    r2 = metricas_extra.get(t2, {}).get("prediction_risk", 25)
    return min(1.0, ((r1 + r2) / 2.0) / 150.0)

# ══════════════════════════════════════════════════════════════════════════════
# §5 SIMULACIÓN MODIFICADA (Dixon & Coles + NB2 + Cópula de Frank)
# ══════════════════════════════════════════════════════════════════════════════

def prob_nb2(mu, alpha, k):
    if alpha < 1e-9:
        if mu < 1e-12: return 1.0 if k == 0 else 0.0
        return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))
    r = 1.0 / alpha
    p = r / (r + mu)
    return math.exp(
        math.lgamma(k + r) - math.lgamma(k + 1) - math.lgamma(r)
        + r * math.log(p) + k * math.log1p(-p)
    )

def cdf_nb2(mu, alpha, k):
    if k < 0: return 0.0
    return sum(prob_nb2(mu, alpha, x) for x in range(k + 1))

def frank_copula(u, v, theta):
    if abs(theta) < 1e-5: return u * v 
    num = (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
    den = math.exp(-theta) - 1.0
    adentro_log = 1.0 + num / den
    if adentro_log <= 0: return 0.0
    return - (1.0 / theta) * math.log(adentro_log)

TOL_TRUNCACION = 1

def normalizar_dist(dist, etiqueta=""):
    """Renormaliza una pmf discreta truncada para que sume 1 (condicional
    al soporte finito). Advierte si la pérdida de masa no es despreciable."""
    s = sum(dist.values())
    if s <= 1e-12:
        return dist
    if (1.0 - s) > TOL_TRUNCACION:
        import warnings
        warnings.warn(f"Truncación no despreciable en {etiqueta}: masa perdida = {1.0 - s:.4%}")
    return {k: v / s for k, v in dist.items()}

def _k_max_nb2(mu, alpha, tol=1e-4, k_min=5, hard_cap=40):
    """Menor K tal que 1 - cdf_nb2(mu, alpha, K-1) < tol, acotado por hard_cap
    (evita loop indefinido si mu se dispara por algún bug aguas arriba)."""
    k = k_min
    while cdf_nb2(mu, alpha, k) < 1.0 - tol and k < hard_cap:
        k += 1
    return k + 1  # +1: generar_distribucion usa range(min_goles, max_goles), exclusivo en el tope

def calcular_probabilidades(t1, t2, eliminatoria=True):
    elo1 = obtener_elo_efectivo(t1)
    elo2 = obtener_elo_efectivo(t2)
    
    if t1 in ANFITRIONES_2026: elo1 += 40
    if t2 in ANFITRIONES_2026: elo2 += 40
    
    # Mantenemos multiplicadores neutros (1.0) para no sobrestimar al anfitrión
    loc_atk1, loc_def1 = (1.05, 0.95) if t1 in ANFITRIONES_2026 else (1.0, 1.0)
    loc_atk2, loc_def2 = (1.05, 0.95) if t2 in ANFITRIONES_2026 else (1.0, 1.0)

    s1 = equipos_stats.get(t1, {"xg": PRIOR_MEAN, "xga": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"xg": PRIOR_MEAN, "xga": PRIOR_MEAN})
    
    lam1_stats = global_atk * (s1["xg"] * loc_atk1 / global_atk) * (s2["xga"] * loc_def2 / global_def)
    lam2_stats = global_atk * (s2["xg"] * loc_atk2 / global_atk) * (s1["xga"] * loc_def1 / global_def)
    
    delta_elo = elo1 - elo2
    factor_riesgo = obtener_riesgo_partido(t1, t2) 

    def generar_distribucion(l1, l2, alpha, min_goles=0, max_goles=10):
        dist = {}
        theta_copula = 0.38 

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

    lam1_elo = math.exp(B0_ELO + B1_ELO * delta_elo)
    lam2_elo = math.exp(B0_ELO - B1_ELO * delta_elo)
    
    alpha_90 = 0.002 + 0.0002 * abs(delta_elo) + (0.005 * factor_riesgo)
    
    dist_stats = normalizar_dist(
        generar_distribucion(lam1_stats, lam2_stats, alpha_90, min_goles=0, max_goles=10),
        etiqueta=f"stats 90' ({t1} vs {t2})")
    dist_elo = normalizar_dist(
        generar_distribucion(lam1_elo, lam2_elo, alpha_90, min_goles=0, max_goles=10),
        etiqueta=f"elo 90' ({t1} vs {t2})")
    # prob_90 = (1-w_elo)*dist_stats + w_elo*dist_elo ya suma 1 exactamente, sin tocar nada más aquí

    d = abs(delta_elo)
    factor_distancia = float(np.minimum(1.0, d / 400.0))
    w_elo = PESO_ELO_MAX - (PESO_ELO_MAX - PESO_ELO_MIN) * factor_distancia
    w_elo = max(PESO_ELO_MIN, min(PESO_ELO_MAX, w_elo))

    prob_90 = {k: (1 - w_elo) * dist_stats[k] + w_elo * dist_elo[k] for k in dist_stats.keys()}

    top_90 = sorted(prob_90.items(), key=lambda x: x[1], reverse=True)[:5]

   

    top_90 = sorted(prob_90.items(), key=lambda x: x[1], reverse=True)[:5]
    exp_goals_90 = sum((i + j) * p for (i, j), p in prob_90.items())
    prob_draw_90 = sum(p for (i, j), p in prob_90.items() if i == j)

    if not eliminatoria:
        return {
            "top_90": top_90,
            "exp_goals_90": exp_goals_90,
            "prob_draw_90": prob_draw_90
        }

    FACTOR_ET = 0.18
    lam1_et = ((1 - w_elo) * lam1_stats + w_elo * lam1_elo) * FACTOR_ET
    lam2_et = ((1 - w_elo) * lam2_stats + w_elo * lam2_elo) * FACTOR_ET
    
    alpha_et = 0.001 + 0.0002 * abs(delta_elo) + (0.002 * factor_riesgo)
    
    # Prórroga con cópula bivariada (de 0 a 5 goles)
    prob_et = normalizar_dist(
        generar_distribucion(lam1_et, lam2_et, alpha_et, min_goles=0, max_goles=5),
        etiqueta=f"ET ({t1} vs {t2})")
    
    prob_120 = {}
    for (i, j), p90 in prob_90.items():
        if i != j:
            prob_120[(i, j)] = prob_120.get((i, j), 0.0) + p90
        else:
            for (ea, eb), pet in prob_et.items():
                prob_120[(i + ea, j + eb)] = prob_120.get((i + ea, j + eb), 0.0) + (p90 * pet)
                
    suma_120 = sum(prob_120.values())
    prob_120 = {k: v / suma_120 for k, v in prob_120.items()}
    
    p_win1 = sum(p for (i, j), p in prob_120.items() if i > j)
    p_draw = sum(p for (i, j), p in prob_120.items() if i == j) 
    p_win2 = sum(p for (i, j), p in prob_120.items() if i < j)
    
    pen_edge = 0.045 * (1.0 - 0.3 * factor_riesgo)
    p_pen1 = 0.5 + pen_edge * math.tanh(delta_elo / 300.0)
    p_pen1 = max(0.46, min(0.54, p_pen1))

    p_final1 = p_win1 + p_draw * p_pen1
    p_final2 = p_win2 + p_draw * (1.0 - p_pen1)
        
    return {
        "top_90": top_90,
        "exp_goals_90": exp_goals_90,
        "prob_draw_90": prob_draw_90,
        "top_120": sorted(prob_120.items(), key=lambda x: x[1], reverse=True)[:5],
        "prob_local": p_final1,   
        "prob_visita": p_final2,  
        "prob_local_120": p_win1, 
        "prob_draw_120": p_draw,
        "prob_visita_120": p_win2
    }

# ══════════════════════════════════════════════════════════════════════════════
# §6 EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

codes = {
    "Sudáfrica": "AFS", "Canadá": "CAN", "Brasil": "BRA", "Japón": "JPN",
    "Alemania": "ALE", "Paraguay": "PAR", "Países Bajos": "NED", "Marruecos": "MAR",
    "Costa de Marfil": "CIV", "Noruega": "NOR", "Francia": "FRA", "Suecia": "SUE",
    "México": "MEX", "Ecuador": "ECU", "Inglaterra": "ENG", "RD Congo": "RDC",
    "Bélgica": "BEL", "Senegal": "SEN", "Estados Unidos": "USA", "Bosnia y Herzegovina": "BIH",
    "España": "ESP", "Austria": "AUT", "Portugal": "POR", "Croacia": "CRO",
    "Suiza": "SUI", "Argelia": "ALG", "Australia": "AUS", "Egipto": "EGY",
    "Argentina": "ARG", "Cabo Verde": "CPV", "Colombia": "COL", "Ghana": "GHA"
}


#PartidosCuartos
matches = [
    ("Francia", "Marruecos"), ("España", "Bélgica"), 
    ("Noruega", "Inglaterra"), ("Argentina", "Suiza")
]


w_partido = 35 
w_top = 35
w_probs = 25

def _fmt(lista): 
    return ", ".join(f"{i}-{j} ({p:.1%})" for (i, j), p in lista)

print(f"{'PARTIDO (Pasa de ronda)':<{w_partido}} | {'TOP 90 MINS':<{w_top}} | {'TOP 120 MINS':<{w_top}} | {'PROBS 120'}")
print("─" * (w_partido + w_top + w_top + w_probs + 10))

sum_goals_90 = 0.0
sum_draw_90  = 0.0
n = len(matches)

for t1, t2 in matches:
    try:
        res = calcular_probabilidades(t1, t2, eliminatoria=True)
        t1_c = codes.get(t1, t1[:3].upper())
        t2_c = codes.get(t2, t2[:3].upper())
        
        fav = t1 if res['prob_local'] > res['prob_visita'] else t2
        fav_c = codes.get(fav, fav[:3].upper())
        prob_fav = max(res['prob_local'], res['prob_visita'])

        partido_str = f"{t1_c} vs {t2_c} ({fav_c} {prob_fav:.0%})"
        top_90_3  = _fmt(res['top_90'][:3])
        top_120_3 = _fmt(res['top_120'][:3])
        
        p_l = res['prob_local_120']
        p_e = res['prob_draw_120']
        p_v = res['prob_visita_120']
        probs_120_str = f"L:{p_l:.0%} E:{p_e:.0%} V:{p_v:.0%}"

        print(f"{partido_str:<{w_partido}} | {top_90_3:<{w_top}} | {top_120_3:<{w_top}} | {probs_120_str}")

        sum_goals_90 += res["exp_goals_90"]
        sum_draw_90  += res["prob_draw_90"]

    except Exception as e:
        print(f"Error en {t1} vs {t2}: {e}")

print("-" * (w_partido + w_top + w_top + w_probs + 10))
print(f"Goles esperados promedio: {sum_goals_90 / n:.2f}")
print(f"Empates esperados promedio: {sum_draw_90 / n:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# §7 AUDITORÍA Y DESGLOSE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * (w_partido + w_top + w_top + w_probs + 10))
print(" AUDITORÍA DE PENDIENTES (SLOPES) Y PARÁMETROS DEL MODELO")
print("═" * (w_partido + w_top + w_top + w_probs + 10))
print("► MODELO 1 (Regresión GLM Poisson sobre Elo):")
print(f"  Intercepto (B0) : {B0_ELO:.4f}  (Base de goles en duelo igualado)")
print(f"  Pendiente  (B1) : {B1_ELO:.6f}  (Impacto por cada punto de Elo)") 
print("\n► MODELO 2 (Motor EM de Stats Latentes):")
print(f"  Ataque  -> Intercepto: {PARAMS_TORNEO['atk'][0]:.4f} | Pendiente: {PARAMS_TORNEO['atk'][1]:.4f}")
print(f"  Defensa -> Intercepto: {PARAMS_TORNEO['def'][0]:.4f} | Pendiente: {PARAMS_TORNEO['def'][1]:.4f}")
print("\n► MODELO 3 (Métricas Extra & Ajustes Dinámicos):")
print(f"  Rango Máximo Detectado (MAX_RANK) : {MAX_RANK}")
print(f"  Equipos con Ajuste Extra          : {len(metricas_extra)}")
print("═" * (w_partido + w_top + w_top + w_probs + 10) + "\n")

print("\n" + "═" * 90)
print(f" DESGLOSE DE ELO EFECTIVO Y BOOSTS (MAX_RANK = {MAX_RANK})")
print("═" * 90)
print(f"{'EQUIPO':<16} | {'ELO BASE':<9} | {'BST RANK':<9} | {'BST FORMA':<10} | {'LOCALÍA':<8} | {'ELO REAL (TOTAL)'}")
print("─" * 90)

def obtener_elo_total_cancha(eq):
    elo_base = teams_base.get(eq, [BASE_ELO])[0]
    extra = metricas_extra.get(eq, {})
    rank_actual = extra.get("performance_rank", MAX_RANK)
    boost_rank = abs(rank_actual - MAX_RANK) * 1.1
    boost_forma = 0.35 * extra.get("diferencia_elo_torneo", 0) + 0.15 * (extra.get("diferencia_elo_año", 0) - extra.get("diferencia_elo_torneo", 0))
    boost_loc = 40.0 if eq in ANFITRIONES_2026 else 0.0
    return elo_base + boost_rank + boost_forma + boost_loc

equipos_ordenados = sorted(metricas_extra.keys(), key=obtener_elo_total_cancha, reverse=True)

for eq in equipos_ordenados:
    elo_base = teams_base.get(eq, [BASE_ELO])[0]
    extra = metricas_extra.get(eq, {})
    rank_actual = extra.get("performance_rank", MAX_RANK)
    boost_rank = abs(rank_actual - MAX_RANK) * 1.1
    boost_forma = 0.35 * extra.get("diferencia_elo_torneo", 0) + 0.15 * (extra.get("diferencia_elo_año", 0) - extra.get("diferencia_elo_torneo", 0))
    boost_loc = 40.0 if eq in ANFITRIONES_2026 else 0.0
    elo_total = elo_base + boost_rank + boost_forma + boost_loc
    boost_acum = boost_rank + boost_forma + boost_loc
    print(f"{eq:<16} | {elo_base:<9.1f} | {boost_rank:+9.1f} | {boost_forma:+10.1f} | {boost_loc:+8.1f} | {elo_total:<8.1f} ({boost_acum:+.1f})")


    

print("─" * 90)
