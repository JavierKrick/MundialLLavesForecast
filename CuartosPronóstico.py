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
    # --- América (Norte y Centro) ---
    "Canada": "Canadá",
    "Canadá": "Canadá",
    "Canada National Team": "Canadá",
    "Mexico": "México",
    "México": "México",
    "Mexico National Team": "México",
    "United States": "Estados Unidos",
    "USA": "Estados Unidos",
    "USMNT": "Estados Unidos",
    "Estados Unidos": "Estados Unidos",
    "Panama": "Panamá",
    "Panamá": "Panamá",
    "Costa Rica": "Costa Rica",
    "Haiti": "Haití",
    "Haití": "Haití",
    "Curacao": "Curazao",
    "Curaçao": "Curazao",
    "Curazao": "Curazao",
    "Guatemala": "Guatemala",
    "Honduras": "Honduras",
    "Jamaica": "Jamaica",
    
    # --- América (Sur) ---
    "Brazil": "Brasil",
    "Brasil": "Brasil",
    "Argentina": "Argentina",
    "Colombia": "Colombia",
    "Chile": "Chile",
    "Bolivia": "Bolivia",
    "Ecuador": "Ecuador",
    "Uruguay": "Uruguay",
    "Venezuela": "Venezuela",
    "Paraguay": "Paraguay",
    "Peru": "Perú",
    "Perú": "Perú",
    
    # --- Europa ---
    "England": "Inglaterra",
    "Inglaterra": "Inglaterra",
    "Spain": "España",
    "España": "España",
    "France": "Francia",
    "Francia": "Francia",
    "Belgium": "Bélgica",
    "Bélgica": "Bélgica",
    "Switzerland": "Suiza",
    "Suiza": "Suiza",
    "Sweden": "Suecia",
    "Suecia": "Suecia",
    "Netherlands": "Países Bajos",
    "Países Bajos": "Países Bajos",
    "Croatia": "Croacia",
    "Croacia": "Croacia",
    "Germany": "Alemania",
    "Alemania": "Alemania",
    "Czech Republic": "Chequia",
    "Czechia": "Chequia",
    "República Checa": "Chequia",
    "Chequia": "Chequia",
    "Turkey": "Turquía",
    "Turquía": "Turquía",
    "Scotland": "Escocia",
    "Escocia": "Escocia",
    "Ireland": "Irlanda",
    "Irlanda": "Irlanda",
    "Northern Ireland": "Irlanda del Norte",
    "Irlanda del Norte": "Irlanda del Norte",
    "Wales": "Gales",
    "Gales": "Gales",
    "Poland": "Polonia",
    "Polonia": "Polonia",
    "Denmark": "Dinamarca",
    "Dinamarca": "Dinamarca",
    "Russia": "Rusia",
    "Rusia": "Rusia",
    "Ukraine": "Ucrania",
    "Ucrania": "Ucrania",
    "Greece": "Grecia",
    "Grecia": "Grecia",
    "North Macedonia": "Macedonia del Norte",
    "Macedonia del Norte": "Macedonia del Norte",
    "Iceland": "Islandia",
    "Islandia": "Islandia",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Bosnia/Herzeg": "Bosnia y Herzegovina",
    "Bosnia y Herzegovina": "Bosnia y Herzegovina",
    "Norway": "Noruega",
    "Noruega": "Noruega",
    "Albania": "Albania",
    "Belarus": "Bielorrusia",
    "Bielorrusia": "Bielorrusia",
    "Finland": "Finlandia",
    "Finlandia": "Finlandia",
    "Georgia": "Georgia",
    "Hungary": "Hungría",
    "Hungría": "Hungría",
    "Israel": "Israel",
    "Italy": "Italia",
    "Italia": "Italia",
    "Kosovo": "Kosovo",
    "Romania": "Rumania",
    "Rumania": "Rumania",
    "Serbia": "Serbia",
    "Slovakia": "Eslovaquia",
    "Eslovaquia": "Eslovaquia",
    "Slovenia": "Eslovenia",
    "Eslovenia": "Eslovenia",
    "Portugal": "Portugal",
    "Austria": "Austria",
    
    # --- África ---
    "Morocco": "Marruecos",
    "Marruecos": "Marruecos",
    "Egypt": "Egipto",
    "Egipto": "Egipto",
    "South Africa": "Sudáfrica",
    "Sudáfrica": "Sudáfrica",
    "Algeria": "Argelia",
    "Argelia": "Argelia",
    "Tunisia": "Túnez",
    "Túnez": "Túnez",
    "Tunisie": "Túnez",
    "Cape Verde": "Cabo Verde",
    "Cabo Verde": "Cabo Verde",
    "Ivory Coast": "Costa de Marfil",
    "Costa de Marfil": "Costa de Marfil",
    "Cameroon": "Camerún",
    "Camerún": "Camerún",
    "Nigeria": "Nigeria",
    "DR Congo": "RD Congo",
    "RD Congo": "RD Congo",
    "Congo DR": "RD Congo",
    "República Democrática del Congo": "RD Congo",
    "Congo (DR)": "RD Congo",
    "Congo": "RD Congo",
    "República Democrática del Congo": "RD Congo",
    "Burkina Faso": "Burkina Faso",
    "Mali": "Mali",
    "Senegal": "Senegal",
    "Ghana": "Ghana",
    
    # --- Asia y Oceanía ---
    "Japan": "Japón",
    "Japón": "Japón",
    "South Korea": "Corea del Sur",
    "Corea del Sur": "Corea del Sur",
    "Saudi Arabia": "Arabia Saudita",
    "Arabia Saudita": "Arabia Saudita",
    "Jordan": "Jordania",
    "Jordania": "Jordania",
    "Iraq": "Irak",
    "Irak": "Irak",
    "Iran": "Irán",
    "Irán": "Irán",
    "Uzbekistan": "Uzbekistán",
    "Uzbekistán": "Uzbekistán",
    "Qatar": "Catar",
    "Catar": "Catar",
    "New Zealand": "Nueva Zelanda",
    "Nueva Zelanda": "Nueva Zelanda",
    "Australia": "Australia",
}
MAPA_TORNEOS = {"World Cup": "World Cups"}

# Normalización inicial de datos importados
teams_base = {MAPA_EQUIPOS.get(k, k): v for k, v in teams_base.items()}
metricas_extra = {MAPA_EQUIPOS.get(k, k): v for k, v in metricas_extra.items()}


def mapear_partidos(lista_partidos, mapa_equipos):
    """
    Mapea los nombres de equipos en una lista de partidos
    
    Args:
        lista_partidos: Lista de diccionarios con 'equipo1' y 'equipo2'
        mapa_equipos: Diccionario con el mapeo de nombres
    
    Returns:
        Lista con los partidos mapeados
    """
    partidos_mapeados = []
    for partido in lista_partidos:
        partido_mapeado = partido.copy()  # Copiar para no modificar el original
        partido_mapeado["equipo1"] = mapa_equipos.get(partido["equipo1"], partido["equipo1"])
        partido_mapeado["equipo2"] = mapa_equipos.get(partido["equipo2"], partido["equipo2"])
        partidos_mapeados.append(partido_mapeado)
    return partidos_mapeados

# Usar la función
partidos_mapeados = mapear_partidos(partidos_filtrados, MAPA_EQUIPOS)



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

# ✅ PRIMERO: Mapeamos los partidos
partidos_mapeados = mapear_partidos(partidos_filtrados, MAPA_EQUIPOS)

# ✅ SEGUNDO: Registramos los partidos que fueron a prórroga (usando los nombres ya mapeados)
partidos_con_90_separados = registrar_prorrogas(partidos_mapeados, [
    ("Argentina", "Cabo Verde", 2, 2, 3, 2),      # <-- Usa "Cabo Verde" (ya mapeado)
    ("Colombia", "Suiza", 0, 0, 0, 0),            # <-- Usa "Suiza" (ya mapeado)
    ("Australia", "Egipto", 1, 1, 1, 1),          # <-- Usa "Egipto" (ya mapeado)
    ("Alemania", "Paraguay", 1, 1, 1, 1),         # <-- Usa "Alemania" (ya mapeado)
    ("Países Bajos", "Marruecos", 1, 1, 1, 1)     # <-- Usa "Países Bajos" (ya mapeado)
])

# ✅ TERCERO: Normalización final (por si acaso)
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
W_XG_DEF = 0.86

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

def bivariado_log_likelihood(params, X1_norm, X2_norm, Y1_np, Y2_np, lambda_l2=0.01):
    """Función objetivo global para optimizar los marcadores exactos cruzados."""
    # params: [b0, beta_delta_elo, beta_atk, beta_def, alpha, theta, L_limit]
    b0, b_elo, b_atk, b_def, alpha, theta, L_limit = params
    
    # Aseguramos que L_limit no sea cero para evitar división por cero
    L_limit = max(0.1, abs(L_limit)) 

    # Función ELU vectorizada: suaviza la desventaja estandarizada
    def aplicar_elu(x_matrix):
        x_mod = np.copy(x_matrix)
        d_elo = x_mod[:, 0]
        # Si es >= 0, queda igual. Si es < 0, se curva hacia -L_limit
        x_mod[:, 0] = np.where(d_elo >= 0, d_elo, L_limit * (np.exp(d_elo / L_limit) - 1.0))
        return x_mod

    # Aplicamos la transformación a las matrices de features
    X1_mod = aplicar_elu(X1_norm)
    X2_mod = aplicar_elu(X2_norm)
    
    # Calculamos los lambdas con las variables ya modificadas
    lam1 = np.exp(b0 + b_elo * X1_mod[:, 0] + b_atk * X1_mod[:, 1] + b_def * X1_mod[:, 2])
    lam2 = np.exp(b0 + b_elo * X2_mod[:, 0] + b_atk * X2_mod[:, 1] + b_def * X2_mod[:, 2])
    
    ll_total = 0.0
    
    for i in range(len(Y1_np)):
        y1, y2 = Y1_np[i], Y2_np[i]
        l1, l2 = lam1[i], lam2[i]
        
        u1 = cdf_nb2(l1, alpha, y1)
        v1 = cdf_nb2(l2, alpha, y2)
        u0 = cdf_nb2(l1, alpha, y1 - 1) if y1 > 0 else 0.0
        v0 = cdf_nb2(l2, alpha, y2 - 1) if y2 > 0 else 0.0
        
        C11 = frank_copula(u1, v1, theta)
        C01 = frank_copula(u0, v1, theta)
        C10 = frank_copula(u1, v0, theta)
        C00 = frank_copula(u0, v0, theta)
        
        prob_conjunta = max(C11 - C01 - C10 + C00, 1e-12)
        ll_total += math.log(prob_conjunta)
        
    penalizacion_l2 = lambda_l2 * (b_elo**2 + b_atk**2 + b_def**2)
        
    return -ll_total + penalizacion_l2

def entrenar_modelo_90_minutos(partidos, equipos_stats):
    X_90_1, X_90_2, Y_90_1, Y_90_2 = [], [], [], []
    
    for p in partidos:
        elo1, elo2 = obtener_elo_efectivo(p["equipo1"]), obtener_elo_efectivo(p["equipo2"])
        s1 = equipos_stats.get(p["equipo1"], {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
        s2 = equipos_stats.get(p["equipo2"], {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})

        g90_1 = p.get("goles_90_equipo1", p["goles_equipo1"])
        g90_2 = p.get("goles_90_equipo2", p["goles_equipo2"])
        delta_elo = (elo1 - elo2) / 100.0

        # OBSERVACIÓN ORIGINAL
        X_90_1.append([delta_elo, s1["ataque_mix"], s2["defensa_mix"]])
        X_90_2.append([-delta_elo, s2["ataque_mix"], s1["defensa_mix"]])
        Y_90_1.append(g90_1)
        Y_90_2.append(g90_2)
        
        # OBSERVACIÓN ESPEJO (Intercambiamos los equipos 1 y 2)
        # Esto destruye cualquier sesgo oculto de "el equipo 1 es siempre mejor"
        X_90_1.append([-delta_elo, s2["ataque_mix"], s1["defensa_mix"]])
        X_90_2.append([delta_elo, s1["ataque_mix"], s2["defensa_mix"]])
        Y_90_1.append(g90_2)  # Los goles de equipo2 ahora van al vector 1
        Y_90_2.append(g90_1)  # Los goles de equipo1 ahora van al vector 2


    df1 = pd.DataFrame(X_90_1, columns=["delta_elo", "ataque_mix", "defensa_mix"])
    df2 = pd.DataFrame(X_90_2, columns=["delta_elo", "ataque_mix", "defensa_mix"])
    
    df_combined = pd.concat([df1, df2], ignore_index=True)
    media_features = df_combined.mean()
    std_features = df_combined.std()

    # Normalización estricta
    X1_norm = ((df1 - media_features) / std_features).values
    X2_norm = ((df2 - media_features) / std_features).values
    Y1_np = np.array(Y_90_1)
    Y2_np = np.array(Y_90_2)

  

    # (Dentro de entrenar_modelo_90_minutos, justo antes de x0 = ...)
    print("\n⏳ Optimizando parámetros bivariados (Intercepto + Betas + Alpha + Theta + L_limit)...")
    
    # x0: [b0, beta_elo, beta_atk, beta_def, alpha, theta, L_limit]
    # Empezamos asumiendo un límite de 1.5 desvíos estándar
    x0 = [0.2, 0.1, 0.1, 0.1, 0.05, 0.4, 1.5] 
    
    # Agregamos el bound para L_limit (entre 0.1 y 5.0)
    bounds = [(None, None), (None, None), (None, None), (None, None), (0.0001, 1.5), (0.0001, 8.0), (0.1, 5.0)]
    
    opt = minimize(
        bivariado_log_likelihood, 
        x0=x0, 
        bounds=bounds, 
        args=(X1_norm, X2_norm, Y1_np, Y2_np, 0.01),
        method='L-BFGS-B'
    )
    
    if opt.success:
        print(f"✅ Optimización bivariada exitosa. Log-Likelihood: {-opt.fun:.2f}")
    else:
        warnings.warn("La optimización bivariada no convergió del todo. Revisa los bounds.")

    betas = {
        "b0": opt.x[0],
        "delta_elo": opt.x[1],
        "ataque_mix": opt.x[2],
        "defensa_mix": opt.x[3]
    }
    alpha_optimizado = opt.x[4]
    theta_optimizado = opt.x[5]
    L_limit_optimizado = opt.x[6] # <--- Guardamos el nuevo parámetro
    
    # DEVOLVEMOS L_limit_optimizado TAMBIÉN
    return betas, alpha_optimizado, theta_optimizado, L_limit_optimizado, media_features, std_features  

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
        return max(0.05, min(0.25, factor_bruto))
    return 0.18

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

# Modificar la llamada de entrenamiento:
BETAS, ALPHA_BASE, THETA_BASE, L_LIMIT_BASE, MEDIA_X, STD_X = entrenar_modelo_90_minutos(
    partidos_con_ambos_en_teams,
    equipos_stats
)

print(f"--- PARÁMETROS APRENDIDOS ---")
print(f"Alpha Base (NB2) : {ALPHA_BASE:.5f}")
print(f"Theta (Cópula)   : {THETA_BASE:.5f}")
print(f"L_Limit (ELU)    : {L_LIMIT_BASE:.5f}") # ¡El modelo aprendió dónde está el piso!


FACTOR_ET_MODELO = calcular_factor_prorroga(partidos_con_ambos_en_teams)


print(f"--- PARÁMETROS APRENDIDOS ---")
print(f"Alpha Base (NB2) : {ALPHA_BASE:.5f}")
print(f"Factor ET        : {FACTOR_ET_MODELO:.3f} (Ritmo de goles en prórroga)")

# ══════════════════════════════════════════════════════════════════════════════
# §7 FASE 4: MOTOR DE SIMULACIÓN (NB2 + CÓPULA)
# ══════════════════════════════════════════════════════════════════════════════






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

def asimetria_elo(val, L):
    """Aplica la asimetría suave aprendida por el modelo."""
    return val if val >= 0 else L * (math.exp(val / L) - 1.0)

def aplicar_tension_perdedor(dist, factor_traspaso=0.15):
    """
    Si un equipo va ganando por exactamente 1 o 2 goles (ej: 1-0, 2-0, 2-1),
    el equipo perdedor presiona para empatar/descontar.
    Le sacamos un % de probabilidad a ese resultado y lo pasamos al perdedor.
    Si la diferencia es mayor (ej: 3-0, 4-0), el partido está liquidado.
    """
    dist_ajustada = dist.copy()
    
    # Guardamos los traspasos en un dict temporal para no generar 
    # un "efecto dominó" irreal mientras iteramos la matriz.
    traspasos = {}
    
    for (i, j), p in dist.items():
        dif = i - j
        
        # Local gana por exactamente 1 o 2 goles (ej: 1-0, 2-0, 2-1, 3-1, 3-2)
        if dif in [1, 2]:
            movimiento = p * factor_traspaso
            dist_ajustada[(i, j)] -= movimiento
            traspasos[(i, j + 1)] = traspasos.get((i, j + 1), 0.0) + movimiento
            
        # Visita gana por exactamente 1 o 2 goles (ej: 0-1, 0-2, 1-2)
        elif dif in [-1, -2]:
            movimiento = p * factor_traspaso
            dist_ajustada[(i, j)] -= movimiento
            traspasos[(i + 1, j)] = traspasos.get((i + 1, j), 0.0) + movimiento

    # Aplicamos los traspasos calculados
    for (i, j), valor in traspasos.items():
        dist_ajustada[(i, j)] = dist_ajustada.get((i, j), 0.0) + valor

    # Normalizamos para evitar errores de redondeo de Python
    s = sum(dist_ajustada.values())
    return {k: v / s for k, v in dist_ajustada.items()}

import numpy as np

def entrenar_factor_tension(partidos, factor_max=0.40, pasos=41):
    print("\n" + "═" * 60)
    print(" ENTRENANDO FACTOR DE TENSIÓN (GAME STATE)")
    print("═" * 60)
    
    # 1. Guardamos las distribuciones BASE (sin tensión) para no recalcular todo
    distribuciones_base = []
    y_true_1x2 = []
    y_true_exacto = []
    
    for p in partidos:
        # Replicamos el cálculo base de lambda y alpha para cada partido
        t1, t2 = p["equipo1"], p["equipo2"]
        elo1, elo2 = obtener_elo_efectivo(t1), obtener_elo_efectivo(t2)
        delta_elo = (elo1 - elo2) / 100.0
        
        s1 = equipos_stats.get(t1, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
        s2 = equipos_stats.get(t2, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
        
        x1 = pd.Series({"delta_elo": delta_elo, "ataque_mix": s1["ataque_mix"], "defensa_mix": s2["defensa_mix"]})
        x1 = (x1 - MEDIA_X) / STD_X
        x2 = pd.Series({"delta_elo": -delta_elo, "ataque_mix": s2["ataque_mix"], "defensa_mix": s1["defensa_mix"]})
        x2 = (x2 - MEDIA_X) / STD_X

        x1["delta_elo"] = asimetria_elo(x1["delta_elo"], L_LIMIT_BASE)
        x2["delta_elo"] = asimetria_elo(x2["delta_elo"], L_LIMIT_BASE)

        lam1 = math.exp(BETAS["b0"] + BETAS["delta_elo"]*x1["delta_elo"] + BETAS["ataque_mix"]*x1["ataque_mix"] + BETAS["defensa_mix"]*x1["defensa_mix"])
        lam2 = math.exp(BETAS["b0"] + BETAS["delta_elo"]*x2["delta_elo"] + BETAS["ataque_mix"]*x2["ataque_mix"] + BETAS["defensa_mix"]*x2["defensa_mix"])
        
        alpha = ALPHA_BASE + 0.002 * abs(delta_elo) + (0.01 * obtener_riesgo_partido(t1, t2))
        dist_base = normalizar_dist(generar_distribucion(lam1, lam2, alpha, theta_copula=THETA_BASE))
        
        distribuciones_base.append(dist_base)
        
        g1 = p.get("goles_90_equipo1", p["goles_equipo1"])
        g2 = p.get("goles_90_equipo2", p["goles_equipo2"])
        y_true_exacto.append((g1, g2))
        
        if g1 > g2: y_true_1x2.append(0)
        elif g1 == g2: y_true_1x2.append(1)
        else: y_true_1x2.append(2)

    total_p = len(partidos)
    
    # Frecuencias reales exactas para el loss del 60%
    frecuencias_reales = defaultdict(float)
    for res in y_true_exacto:
        frecuencias_reales[res] += 1.0 / total_p
        
    # Resultados clave a evaluar
    resultados_clave = [(2,1), (1,1), (1,0), (0,0), (3,1), (2,0), (3,0), (4,1)]

    # Variables para guardar el ganador
    mejor_factor = 0.0
    menor_costo = float('inf')
    
    # 2. BARRIDO (Grid Search)
    factores = np.linspace(0.0, factor_max, pasos)
    
    for f in factores:
        loss_1x2 = 0.0
        preds_exactas = defaultdict(float)
        preds_empate = []
        
        # Evaluamos el factor actual en todos los partidos
        for i in range(total_p):
            dist_ajustada = aplicar_tension_perdedor(distribuciones_base[i], factor_traspaso=f)
            
            p_loc = sum(p for (g1, g2), p in dist_ajustada.items() if g1 > g2)
            p_emp = sum(p for (g1, g2), p in dist_ajustada.items() if g1 == g2)
            p_vis = sum(p for (g1, g2), p in dist_ajustada.items() if g1 < g2)
            
            # Recolectamos para la métrica del 20% (Calibración empate)
            preds_empate.append((p_emp, y_true_1x2[i] == 1))
            
            # Recolectamos para la métrica del 60% (Resultados Exactos)
            for res in resultados_clave:
                preds_exactas[res] += dist_ajustada.get(res, 0.0) / total_p
                
            # Métrica del 30% (Brier Score 1X2)
            real_arr = [1.0 if y_true_1x2[i] == 0 else 0.0, 
                        1.0 if y_true_1x2[i] == 1 else 0.0, 
                        1.0 if y_true_1x2[i] == 2 else 0.0]
            loss_1x2 += ((p_loc - real_arr[0])**2 + (p_emp - real_arr[1])**2 + (p_vis - real_arr[2])**2) / total_p

        # Calcular Loss 2: Resultados Exactos (MSE)
        loss_exacto = sum((preds_exactas[res] - frecuencias_reales[res])**2 for res in resultados_clave)
        
        # Calcular Loss 3: Calibración Empate (MSE por bins)
        bins = [0, 0.1, 0.2, 0.3, 0.4]
        loss_calib = 0.0
        for b_idx in range(len(bins)-1):
            en_bin = [x for x in preds_empate if bins[b_idx] <= x[0] < bins[b_idx+1]]
            if len(en_bin) > 0:
                pred_media = sum(x[0] for x in en_bin) / len(en_bin)
                real_media = sum(1.0 for x in en_bin if x[1]) / len(en_bin)
                loss_calib += (pred_media - real_media)**2 * (len(en_bin) / total_p) # Ponderado por volumen

        # NORMALIZACIÓN DE PESOS (30, 60, 20 sobre 110)
        costo_total = (30 * loss_1x2) + (60 * loss_exacto * 10) + (20 * loss_calib) 
        # Multiplico exacto * 10 para equilibrar escalas numéricas de los errores
        
        if costo_total < menor_costo:
            menor_costo = costo_total
            mejor_factor = f

    print(f"✅ Entrenamiento finalizado.")
    print(f"🏆 FACTOR ÓPTIMO ENCONTRADO: {mejor_factor:.3f} ({mejor_factor*100:.1f}%)")
    return mejor_factor

# ══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DEL ENTRENAMIENTO (Llamalo una sola vez antes de tus simulaciones)
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_TENSION_OPTIMO = entrenar_factor_tension(partidos_con_ambos_en_teams)

# A partir de acá, podés pasarle FACTOR_TENSION_OPTIMO a tu función 
# aplicar_tension_perdedor en vez de usar 0.15 hardcodeado.
def calcular_probabilidades(t1, t2, eliminatoria=True):
    elo1, elo2 = obtener_elo_efectivo(t1), obtener_elo_efectivo(t2)
    delta_elo = (elo1 - elo2) / 100.0

    s1 = equipos_stats.get(t1, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    s2 = equipos_stats.get(t2, {"ataque_mix": PRIOR_MEAN, "defensa_mix": PRIOR_MEAN})
    
    factor_riesgo = obtener_riesgo_partido(t1, t2)
    
    x1 = pd.Series({
        "delta_elo": delta_elo,
        "ataque_mix": s1["ataque_mix"],
        "defensa_mix": s2["defensa_mix"]
    })
    x1 = (x1 - MEDIA_X) / STD_X
    
    x2 = pd.Series({
        "delta_elo": -delta_elo,
        "ataque_mix": s2["ataque_mix"],
        "defensa_mix": s1["defensa_mix"]
    })
    x2 = (x2 - MEDIA_X) / STD_X

    # 1. Aplicamos el Límite ELU (Asimetría) aprendido por el optimizador
    x1["delta_elo"] = asimetria_elo(x1["delta_elo"], L_LIMIT_BASE)
    x2["delta_elo"] = asimetria_elo(x2["delta_elo"], L_LIMIT_BASE)

    lam1_90 = math.exp(
        BETAS["b0"] +
        BETAS["delta_elo"] * x1["delta_elo"] +
        BETAS["ataque_mix"] * x1["ataque_mix"] +
        BETAS["defensa_mix"] * x1["defensa_mix"]
    )

    lam2_90 = math.exp(
        BETAS["b0"] +
        BETAS["delta_elo"] * x2["delta_elo"] +
        BETAS["ataque_mix"] * x2["ataque_mix"] +
        BETAS["defensa_mix"] * x2["defensa_mix"]
    )
    
    # 2. Generamos la distribución base (Cópula de Frank)
    alpha_90 = ALPHA_BASE + 0.002 * abs(delta_elo) + (0.01 * factor_riesgo)
    
    prob_90 = normalizar_dist(
        generar_distribucion(lam1_90, lam2_90, alpha_90, theta_copula=THETA_BASE), 
        f"90' ({t1} vs {t2})"
    )

    # ------------------------------------------------------------------
    # 3. EFECTO GAME STATE (El descuento del perdedor)
    # Acá usamos el parámetro entrenado en vez del número fijo
    prob_90 = aplicar_tension_perdedor(prob_90, factor_traspaso=FACTOR_TENSION_OPTIMO)
    # ------------------------------------------------------------------

    top_90 = sorted(prob_90.items(), key=lambda x: x[1], reverse=True)[:5]
    exp_goals_90 = sum((i + j) * p for (i, j), p in prob_90.items())
    prob_draw_90 = sum(p for (i, j), p in prob_90.items() if i == j)

    if not eliminatoria:
        prob_local_90 = sum(p for (i,j),p in prob_90.items() if i > j)
        prob_visita_90 = sum(p for (i,j),p in prob_90.items() if i < j)

        return {
            "dist_90": prob_90,
            "top_90": top_90,
            "exp_goals_90": exp_goals_90,
            "prob_draw_90": prob_draw_90,
            "prob_local_90": prob_local_90,
            "prob_visita_90": prob_visita_90
        }

    # 4. MODELO DE PRÓRROGA (SOLO SI HAY EMPATE A 90')
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
    
    # 5. MODELO DE PENALES
    pen_edge = 0.08 * (1.0 - 0.2 * factor_riesgo)
    p_pen1 = max(0.46, min(0.54, 0.5 + pen_edge * math.tanh(delta_elo / 300.0)))

    return {
        "dist_90": prob_90,
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
    ("Noruega", "Inglaterra"), ("Argentina", "Suiza")
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
probs_matrix = []
y_true = []

for p in partidos_con_ambos_en_teams:

    res = calcular_probabilidades(
        p["equipo1"],
        p["equipo2"],
        eliminatoria=False
    )

    probs_matrix.append([
        res["prob_local_90"],
        res["prob_draw_90"],
        res["prob_visita_90"]
    ])

    if p["goles_equipo1"] > p["goles_equipo2"]:
        y_true.append(0)

    elif p["goles_equipo1"] == p["goles_equipo2"]:
        y_true.append(1)

    else:
        y_true.append(2)

# ══════════════════════════════════════════════════════════════════════════════
# CORRECCIÓN BUG 11: CALIBRACIÓN MULTICLASE
# ══════════════════════════════════════════════════════════════════════════════

calibracion = {
    "local": {
        "count": [0]*10,
        "pred_sum": [0.0]*10,
        "hits_real": [0]*10
    },
    "empate": {
        "count": [0]*10,
        "pred_sum": [0.0]*10,
        "hits_real": [0]*10
    },
    "visita": {
        "count": [0]*10,
        "pred_sum": [0.0]*10,
        "hits_real": [0]*10
    }
}


for i, probs in enumerate(probs_matrix):

    outcome = y_true[i]

    nombres = ["local", "empate", "visita"]

    for clase, nombre in enumerate(nombres):

        prob = probs[clase]

        bucket = min(9, int(prob * 10))

        calibracion[nombre]["count"][bucket] += 1
        calibracion[nombre]["pred_sum"][bucket] += prob

        if outcome == clase:
            calibracion[nombre]["hits_real"][bucket] += 1



for nombre in ["local", "empate", "visita"]:

    print("\n")
    print(f"CALIBRACIÓN {nombre.upper()}")
    print("Rango Prob | Predicción | Realidad | N")
    print("-"*45)

    for i in range(10):

        n = calibracion[nombre]["count"][i]

        if n > 0:

            pred = (
                calibracion[nombre]["pred_sum"][i] / n
            )

            real = (
                calibracion[nombre]["hits_real"][i] / n
            )

            print(
                f"{i*10:02d}-{i*10+10:02d}% | "
                f"{pred:.1%} | "
                f"{real:.1%} | "
                f"{n}"
            )

from collections import defaultdict

matriz_resultados = defaultdict(lambda: {
    "n": 0,
    "prob_promedio": 0,
    "aciertos": 0
})

for p in partidos_con_ambos_en_teams:

    res = calcular_probabilidades(
        p["equipo1"],
        p["equipo2"],
        eliminatoria=False
    )

    # resultado real
    real = (
        p["goles_equipo1"],
        p["goles_equipo2"]
    )

    # probabilidad que el modelo le daba exactamente a ese resultado
    prob_real = res["dist_90"].get(real, 0)

    matriz_resultados[real]["n"] += 1
    matriz_resultados[real]["prob_promedio"] += prob_real


total_partidos = len(partidos_con_ambos_en_teams)

print("\nRESULTADOS EXACTOS")
print("Resultado | N | Real | Modelo promedio")

for marcador, datos in sorted(
        matriz_resultados.items(),
        key=lambda x: x[1]["n"],
        reverse=True):

    if datos["n"] >= 5:
        real = datos["n"] / total_partidos
        modelo = datos["prob_promedio"] / datos["n"]

        print(
            f"{marcador[0]}-{marcador[1]} | "
            f"{datos['n']:3d} | "
            f"{real:6.2%} | "
            f"{modelo:6.2%}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN DE MARCADORES EXACTOS
# (bins de 2.5% + Intervalos de confianza de Wilson)
# ══════════════════════════════════════════════════════════════════════════════

from statsmodels.stats.proportion import proportion_confint

bins = np.arange(0.0, 0.275, 0.025)  # 0%, 2.5%, ..., 25%

cal = {
    i: {
        "n": 0,
        "hits": 0,
        "prob_sum": 0.0
    }
    for i in range(len(bins)-1)
}

for p in partidos_con_ambos_en_teams:

    res = calcular_probabilidades(
        p["equipo1"],
        p["equipo2"],
        eliminatoria=False
    )

    resultado_real = (
        p["goles_equipo1"],
        p["goles_equipo2"]
    )

    for marcador, prob in res["dist_90"].items():

        # ignorar probabilidades prácticamente nulas
        if prob < 1e-12:
            continue

        bucket = np.digitize(prob, bins) - 1

        # ignorar probabilidades mayores al último bin
        if bucket < 0 or bucket >= len(bins)-1:
            continue

        cal[bucket]["n"] += 1
        cal[bucket]["prob_sum"] += prob

        if marcador == resultado_real:
            cal[bucket]["hits"] += 1

print("\n" + "="*95)
print("CALIBRACIÓN DE MARCADORES EXACTOS")
print("="*95)
print("Rango        N pred.   Prob media    % ocurrió      IC95%")
print("-"*95)

for i in range(len(bins)-1):

    n = cal[i]["n"]

    if n == 0:
        continue

    hits = cal[i]["hits"]

    prob_media = cal[i]["prob_sum"] / n
    frecuencia = hits / n

    li, ls = proportion_confint(
        hits,
        n,
        alpha=0.05,
        method="wilson"
    )

    print(
        f"{bins[i]*100:4.1f}-{bins[i+1]*100:4.1f}%   "
        f"{n:7d}      "
        f"{prob_media:7.2%}      "
        f"{frecuencia:7.2%}   "
        f"[{li:6.2%}, {ls:6.2%}]"
    )



