import math

"""




Este modelo debe entenderse exclusivamente como un ejercicio exploratorio de modelización estadística aplicado a fútbol, con el objetivo de estudiar cómo diferentes capas de información (Poisson tipo Dixon & Coles, xG, Elo y ponderaciones estructurales) interactúan en un sistema unificado.

No tiene pretensiones de ser el mejor predictor de resultados reales ni de competir con modelos profesionales calibrados y validados empíricamente.

El proyecto fue desarrollado en un período breve de aproximadamente dos horas durante un domingo, con fines principalmente experimentales y de exploración metodológica.

IMPORTANTE: EL MODELO NO FUE VALIDADO CON UN CONJUNTO DE VALIDACIÓN




descripcion_modelo = 

El modelo está inspirado en el marco clásico de Dixon & Coles (1997) para la modelización de resultados en fútbol mediante procesos Poisson independientes, extendiendo dicho enfoque mediante la incorporación de variables modernas de rendimiento y factores estructurales.

1) Base Poisson tipo Dixon & Coles (1997):
Se asume que los goles de cada equipo siguen distribuciones Poisson independientes, donde los parámetros de intensidad dependen de la fuerza de ataque y defensa de los equipos.

2) Métricas modernas (xG / xGA):
A diferencia del enfoque original, se incorporan métricas de expected goals (xG) y expected goals against (xGA), combinadas con goles observados, para estimar de forma más robusta las capacidades ofensivas y defensivas.

3) Incorporación de Elo:
Se introduce una medida de fuerza relativa basada en Elo, utilizada como ajuste estructural en la determinación de los parámetros de intensidad de gol.

4) Ponderación por calidad del contexto competitivo:
Las observaciones históricas de xG y xGA se ponderan según la potencia del torneo, derivada del nivel promedio de Elo de los participantes, asignando mayor peso a competiciones de mayor nivel relativo.

5) Ponderación estructural (jerarquía competitiva):
Se introduce un esquema de ponderación adicional que refuerza la influencia de torneos de mayor importancia relativa (no corresponde a un decaimiento temporal estricto).

6) Ajuste por fuerza del rival:
Las métricas ofensivas y defensivas se ajustan en función de la calidad del oponente enfrentado, evitando sesgos derivados de calendarios heterogéneos.

7) Eficiencia de conversión (finishing):
Se incorpora un factor definido como la razón entre goles y xG, para distinguir capacidad de generación de ocasiones y eficiencia de finalización.

8) Regularización bayesiana (shrinkage):
Las estimaciones de parámetros individuales se contraen hacia la media del torneo, especialmente en muestras pequeñas, reduciendo sobreajuste.

9) Consistencia del rendimiento:
Se incluye una medida de varianza de xG y goles para capturar estabilidad o volatilidad del desempeño del equipo.

10) Efecto de localía:
Se incorpora un ajuste multiplicativo sobre la intensidad ofensiva del equipo local.

____________________________________________________________________________

fuentes_y_construccion_datos = 
Fuentes de datos y construcción de variables

1) Métricas de rendimiento (xG, xGA, goles):
Los datos de xG, xGA, goles a favor y goles en contra fueron compilados ""manualmente"" a partir de la página "footystats" y no provienen de una API estructurada. 
""manualmente"": Parte de estos valores fueron transcritos o generados con asistencia de modelos de lenguaje, por lo que pueden contener errores, inconsistencias o aproximaciones. 
En consecuencia, deben interpretarse como datos semi-sintéticos o no totalmente auditados. Ya que no es que fueron tomados por la API y filtrados con SQL.

2) Ratings Elo:
Los valores de Elo de equipos fueron obtenidos de eloratings.net, considerado como referencia estándar para ratings de fuerza relativa a nivel de selecciones.

3) Construcción del Elo de torneo:
El Elo representativo de cada torneo no es observado directamente, sino construido de forma heurística (y sin validar):

- Formatos tipo liga:
  Elo_torneo = (mediana(Elo equipos) + promedio(Elo equipos) + promedio(top 5 equipos)) / 3

- Formatos tipo copa:
  Elo_torneo = (mediana(Elo equipos) + promedio(Elo equipos)) / 2 + (promedio(top 3 equipos) / 2)

Esta construcción busca capturar diferencias estructurales entre formatos competitivos:
- En ligas, la interacción es más homogénea entre todos los equipos.
- En copas, los enfrentamientos tienden a concentrarse más entre equipos de mayor fuerza relativa, por lo que se pondera más fuertemente el rendimiento del top del torneo.



____________________________________________________________________________

limitaciones_y_uso = 

El modelo no debe interpretarse como una herramienta adecuada para apuestas deportivas, PRODE o entornos estratégicos de predicción competitiva.

1) Falta de validación estadística formal:
El modelo no ha sido sometido a un esquema riguroso de backtesting out-of-sample ni a tests sistemáticos de calibración (e.g. log-loss, Brier score), por lo que su capacidad predictiva no está cuantificada de manera robusta.

2) Dependencia del entorno estratégico:
En contextos como apuestas o PRODE, el problema no es únicamente estimar probabilidades reales, sino modelar la distribución de creencias del resto de agentes. Esto transforma el problema en uno de teoría de juegos más que de predicción estadística.

3) Consideración de valor esperado vs popularidad (analogía tipo poker):
En un PRODE o sistema de apuestas masivo, el valor no proviene necesariamente del evento más probable, sino del desajuste entre probabilidad real y probabilidad percibida por la masa de jugadores.

Por ejemplo, si un equipo A tiene 51% de probabilidad de ganar frente a un equipo B con 49%, la elección óptima en un entorno competitivo no es trivial:
- Si la mayoría selecciona A por ser favorito, la acción sobre A queda “concentrada” en la población.
- Esto implica que seleccionar B puede ser estratégicamente superior en términos de valor relativo en el espacio de resultados posibles.

En términos de teoría de apuestas:
no se maximiza únicamente la probabilidad de acierto, sino la expectativa relativa frente a la distribución de apuestas de otros agentes.

Es decir uno querría no solo elegir a los candidatos a ganar un partido, sino también por momento diferenciarse de otros.
Este efecto aumenta a medida que aumentan la cantidad de jugadores y se tiene más conocimiento de los prefiles de apostadores.

En consecuencia, el modelo está diseñado para predicción probabilística, no para optimización estratégica en entornos adversariales.






"""



# --- 1. BASE DE DATOS DE POTENCIA DE TORNEOS ---
datos_potencia_torneos_2026 = [
    {"torneo": "Mundial 2026 (Forma Previa)", "potencia_torneo": 1964.91},
    {"torneo": "WC Qualification South America", "potencia_torneo": 1925.07},
    {"torneo": "WC Qualification Europe", "potencia_torneo": 1727.33},
    {"torneo": "Copa America", "potencia_torneo": 1934.18},
    {"torneo": "UEFA Nations League", "potencia_torneo": 1721.82},
    {"torneo": "World Cups", "potencia_torneo": 1879.25},
    {"torneo": "AFC Asian Cup", "potencia_torneo": 1625.61},
    {"torneo": "Africa Cup of Nations", "potencia_torneo": 1644.07},
    {"torneo": "WC Qualification Africa", "potencia_torneo": 1495.98},
    {"torneo": "WC Qualification Asia", "potencia_torneo": 1361.29},
    {"torneo": "Gold Cup", "potencia_torneo": 1627.82},
    {"torneo": "CONCACAF Nations League", "potencia_torneo": 1290.41},
    {"torneo": "friendly", "potencia_torneo": 1290.41},
    
]

# --- 2. NUEVA BASE DE DATOS DE RENDIMIENTO ---
datos_torneos_completos = [
    {"torneo": "WC Qualification South America", "equipo": "Colombia", "partidos_jugados": 18, "xg_90": 1.45, "xga_90": 1.11, "goles_favor_90": 1.56, "goles_contra_90": 1.00},
    {"torneo": "WC Qualification South America", "equipo": "Argentina", "partidos_jugados": 18, "xg_90": 1.27, "xga_90": 0.71, "goles_favor_90": 1.72, "goles_contra_90": 0.56},
    {"torneo": "WC Qualification South America", "equipo": "Brasil", "partidos_jugados": 18, "xg_90": 1.26, "xga_90": 0.92, "goles_favor_90": 1.33, "goles_contra_90": 0.94},
    {"torneo": "WC Qualification South America", "equipo": "Paraguay", "partidos_jugados": 18, "xg_90": 1.11, "xga_90": 1.05, "goles_favor_90": 0.78, "goles_contra_90": 0.56},
    {"torneo": "WC Qualification South America", "equipo": "Ecuador", "partidos_jugados": 18, "xg_90": 1.09, "xga_90": 0.95, "goles_favor_90": 0.78, "goles_contra_90": 0.28},
    {"torneo": "Copa America", "equipo": "México", "partidos_jugados": 3, "xg_90": 1.58, "xga_90": 0.90, "goles_favor_90": 0.33, "goles_contra_90": 0.33},
    {"torneo": "Copa America", "equipo": "Paraguay", "partidos_jugados": 3, "xg_90": 1.42, "xga_90": 1.03, "goles_favor_90": 1.00, "goles_contra_90": 2.67},
    {"torneo": "Copa America", "equipo": "Argentina", "partidos_jugados": 6, "xg_90": 1.39, "xga_90": 0.87, "goles_favor_90": 1.50, "goles_contra_90": 0.17},
    {"torneo": "Copa America", "equipo": "Colombia", "partidos_jugados": 6, "xg_90": 1.27, "xga_90": 0.99, "goles_favor_90": 2.00, "goles_contra_90": 0.50},
    {"torneo": "Copa America", "equipo": "Brasil", "partidos_jugados": 4, "xg_90": 1.23, "xga_90": 1.00, "goles_favor_90": 1.25, "goles_contra_90": 0.50},
    {"torneo": "Copa America", "equipo": "Estados Unidos", "partidos_jugados": 3, "xg_90": 1.19, "xga_90": 1.05, "goles_favor_90": 1.00, "goles_contra_90": 1.00},
    {"torneo": "Copa America", "equipo": "Canadá", "partidos_jugados": 6, "xg_90": 1.12, "xga_90": 1.26, "goles_favor_90": 0.67, "goles_contra_90": 1.17},
    {"torneo": "Copa America", "equipo": "Ecuador", "partidos_jugados": 4, "xg_90": 0.97, "xga_90": 1.06, "goles_favor_90": 1.25, "goles_contra_90": 1.00},
    {"torneo": "WC Qualification Europe", "equipo": "Paraguay", "partidos_jugados": 3, "xg_90": 0.76, "xga_90": 2.27, "goles_favor_90": 0.67, "goles_contra_90": 1.33},
    {"torneo": "WC Qualification Europe", "equipo": "Portugal", "partidos_jugados": 6, "xg_90": 2.84, "xga_90": 0.89, "goles_favor_90": 3.33, "goles_contra_90": 1.17},
    {"torneo": "WC Qualification Europe", "equipo": "España", "partidos_jugados": 6, "xg_90": 2.74, "xga_90": 0.71, "goles_favor_90": 3.50, "goles_contra_90": 0.33},
    {"torneo": "WC Qualification Europe", "equipo": "Bélgica", "partidos_jugados": 8, "xg_90": 2.57, "xga_90": 0.70, "goles_favor_90": 3.63, "goles_contra_90": 0.88},
    {"torneo": "WC Qualification Europe", "equipo": "Croacia", "partidos_jugados": 8, "xg_90": 2.56, "xga_90": 0.78, "goles_favor_90": 3.25, "goles_contra_90": 0.50},
    {"torneo": "WC Qualification Europe", "equipo": "Francia", "partidos_jugados": 6, "xg_90": 2.54, "xga_90": 0.48, "goles_favor_90": 2.67, "goles_contra_90": 0.67},
    {"torneo": "WC Qualification Europe", "equipo": "Inglaterra", "partidos_jugados": 8, "xg_90": 2.38, "xga_90": 0.43, "goles_favor_90": 2.75, "goles_contra_90": 0.00},
    {"torneo": "WC Qualification Europe", "equipo": "Noruega", "partidos_jugados": 8, "xg_90": 2.21, "xga_90": 0.87, "goles_favor_90": 4.63, "goles_contra_90": 0.63},
    {"torneo": "WC Qualification Europe", "equipo": "Países Bajos", "partidos_jugados": 8, "xg_90": 2.07, "xga_90": 0.75, "goles_favor_90": 3.38, "goles_contra_90": 0.50},
    {"torneo": "WC Qualification Europe", "equipo": "Alemania", "partidos_jugados": 6, "xg_90": 2.01, "xga_90": 0.72, "goles_favor_90": 2.67, "goles_contra_90": 0.50},
    {"torneo": "WC Qualification Europe", "equipo": "Austria", "partidos_jugados": 8, "xg_90": 1.76, "xga_90": 0.93, "goles_favor_90": 2.75, "goles_contra_90": 0.50},
    {"torneo": "WC Qualification Europe", "equipo": "Bosnia y Herzegovina", "partidos_jugados": 10, "xg_90": 1.72, "xga_90": 1.19, "goles_favor_90": 1.90, "goles_contra_90": 0.90},
    {"torneo": "WC Qualification Europe", "equipo": "Suiza", "partidos_jugados": 6, "xg_90": 1.46, "xga_90": 0.76, "goles_favor_90": 2.33, "goles_contra_90": 0.33},
    {"torneo": "WC Qualification Europe", "equipo": "Sweden", "partidos_jugados": 8, "xg_90": 1.18, "xga_90": 1.41, "goles_favor_90": 1.25, "goles_contra_90": 1.88},
    {"torneo": "UEFA Nations League", "equipo": "Suecia", "partidos_jugados": 6, "xg_90": 2.45, "xga_90": 0.82, "goles_favor_90": 3.17, "goles_contra_90": 0.67},
    {"torneo": "UEFA Nations League", "equipo": "España", "partidos_jugados": 10, "xg_90": 1.94, "xga_90": 1.15, "goles_favor_90": 2.50, "goles_contra_90": 1.50},
    {"torneo": "UEFA Nations League", "equipo": "Austria", "partidos_jugados": 8, "xg_90": 1.91, "xga_90": 0.80, "goles_favor_90": 1.88, "goles_contra_90": 1.00},
    {"torneo": "UEFA Nations League", "equipo": "Francia", "partidos_jugados": 10, "xg_90": 1.90, "xga_90": 1.10, "goles_favor_90": 2.00, "goles_contra_90": 1.30},
    {"torneo": "UEFA Nations League", "equipo": "Inglaterra", "partidos_jugados": 6, "xg_90": 1.70, "xga_90": 0.64, "goles_favor_90": 2.67, "goles_contra_90": 0.50},
    {"torneo": "UEFA Nations League", "equipo": "Alemania", "partidos_jugados": 10, "xg_90": 1.68, "xga_90": 1.09, "goles_favor_90": 2.40, "goles_contra_90": 1.20},
    {"torneo": "UEFA Nations League", "equipo": "Portugal", "partidos_jugados": 10, "xg_90": 1.63, "xga_90": 1.33, "goles_favor_90": 2.20, "goles_contra_90": 1.10},
    {"torneo": "UEFA Nations League", "equipo": "Noruega", "partidos_jugados": 6, "xg_90": 1.60, "xga_90": 0.93, "goles_favor_90": 2.50, "goles_contra_90": 1.17},
    {"torneo": "UEFA Nations League", "equipo": "Bélgica", "partidos_jugados": 8, "xg_90": 1.57, "xga_90": 1.13, "goles_favor_90": 1.25, "goles_contra_90": 1.50},
    {"torneo": "UEFA Nations League", "equipo": "Países Bajos", "partidos_jugados": 8, "xg_90": 1.50, "xga_90": 1.23, "goles_favor_90": 2.25, "goles_contra_90": 1.50},
    {"torneo": "UEFA Nations League", "equipo": "Croacia", "partidos_jugados": 8, "xg_90": 1.38, "xga_90": 1.43, "goles_favor_90": 1.25, "goles_contra_90": 1.25},
    {"torneo": "UEFA Nations League", "equipo": "Suiza", "partidos_jugados": 6, "xg_90": 1.22, "xga_90": 1.30, "goles_favor_90": 1.00, "goles_contra_90": 2.33},
    {"torneo": "UEFA Nations League", "equipo": "Bosnia y Herzegovina", "partidos_jugados": 6, "xg_90": 0.73, "xga_90": 1.66, "goles_favor_90": 0.67, "goles_contra_90": 2.83},
    {"torneo": "WC Qualification Africa", "equipo": "Marruecos", "partidos_jugados": 8, "xg_90": 1.82, "xga_90": 0.58, "goles_favor_90": 2.75, "goles_contra_90": 0.25},
    {"torneo": "WC Qualification Africa", "equipo": "Costa de Marfil", "partidos_jugados": 10, "xg_90": 1.77, "xga_90": 0.57, "goles_favor_90": 2.50, "goles_contra_90": 0.00},
    {"torneo": "WC Qualification Africa", "equipo": "Senegal", "partidos_jugados": 10, "xg_90": 1.61, "xga_90": 0.93, "goles_favor_90": 2.20, "goles_contra_90": 0.30},
    {"torneo": "WC Qualification Africa", "equipo": "Egipto", "partidos_jugados": 10, "xg_90": 1.58, "xga_90": 0.70, "goles_favor_90": 2.00, "goles_contra_90": 0.20},
    {"torneo": "WC Qualification Africa", "equipo": "Ghana", "partidos_jugados": 10, "xg_90": 1.48, "xga_90": 0.93, "goles_favor_90": 2.30, "goles_contra_90": 0.60},
    {"torneo": "WC Qualification Africa", "equipo": "South Africa", "partidos_jugados": 10, "xg_90": 1.35, "xga_90": 0.71, "goles_favor_90": 1.70, "goles_contra_90": 0.60},
    {"torneo": "WC Qualification Africa", "equipo": "Argelia", "partidos_jugados": 10, "xg_90": 1.34, "xga_90": 0.86, "goles_favor_90": 2.40, "goles_contra_90": 0.80},
    {"torneo": "WC Qualification Africa", "equipo": "Cabo Verde", "partidos_jugados": 10, "xg_90": 1.31, "xga_90": 0.90, "goles_favor_90": 1.60, "goles_contra_90": 0.80},
    {"torneo": "WC Qualification Africa", "equipo": "RD Congo", "partidos_jugados": 12, "xg_90": 1.23, "xga_90": 0.99, "goles_favor_90": 1.42, "goles_contra_90": 0.58},
    {"torneo": "Africa Cup of Nations", "equipo": "Senegal", "partidos_jugados": 7, "xg_90": 1.93, "xga_90": 0.89, "goles_favor_90": 1.86, "goles_contra_90": 0.29},
    {"torneo": "Africa Cup of Nations", "equipo": "Marruecos", "partidos_jugados": 7, "xg_90": 1.75, "xga_90": 0.65, "goles_favor_90": 1.29, "goles_contra_90": 0.29},
    {"torneo": "Africa Cup of Nations", "equipo": "Costa de Marfil", "partidos_jugados": 5, "xg_90": 1.72, "xga_90": 0.86, "goles_favor_90": 2.00, "goles_contra_90": 1.20},
    {"torneo": "Africa Cup of Nations", "equipo": "Sudáfrica", "partidos_jugados": 4, "xg_90": 1.72, "xga_90": 1.04, "goles_favor_90": 1.50, "goles_contra_90": 1.50},
    {"torneo": "Africa Cup of Nations", "equipo": "RD Congo", "partidos_jugados": 4, "xg_90": 1.42, "xga_90": 1.16, "goles_favor_90": 1.25, "goles_contra_90": 0.50},
    {"torneo": "Africa Cup of Nations", "equipo": "Argelia", "partidos_jugados": 5, "xg_90": 1.31, "xga_90": 1.08, "goles_favor_90": 1.60, "goles_contra_90": 0.60},
    {"torneo": "Africa Cup of Nations", "equipo": "Egipto", "partidos_jugados": 7, "xg_90": 1.22, "xga_90": 1.37, "goles_favor_90": 1.29, "goles_contra_90": 0.71},
    {"torneo": "WC Qualification Asia", "equipo": "Japón", "partidos_jugados": 16, "xg_90": 1.52, "xga_90": 0.65, "goles_favor_90": 3.38, "goles_contra_90": 0.19},
    {"torneo": "WC Qualification Asia", "equipo": "Australia", "partidos_jugados": 16, "xg_90": 0.97, "xga_90": 1.13, "goles_favor_90": 2.38, "goles_contra_90": 0.44},
    {"torneo": "AFC Asian Cup", "equipo": "Japón", "partidos_jugados": 5, "xg_90": 1.70, "xga_90": 0.63, "goles_favor_90": 2.40, "goles_contra_90": 1.60},
    {"torneo": "AFC Asian Cup", "equipo": "Australia", "partidos_jugados": 5, "xg_90": 1.63, "xga_90": 0.62, "goles_favor_90": 1.80, "goles_contra_90": 0.60},
    {"torneo": "CONCACAF Nations League", "equipo": "México", "partidos_jugados": 4, "xg_90": 1.68, "xga_90": 1.05, "goles_favor_90": 2.00, "goles_contra_90": 0.75},
    {"torneo": "CONCACAF Nations League", "equipo": "Estados Unidos", "partidos_jugados": 4, "xg_90": 1.30, "xga_90": 0.95, "goles_favor_90": 1.50, "goles_contra_90": 1.25},
    {"torneo": "CONCACAF Nations League", "equipo": "Canadá", "partidos_jugados": 4, "xg_90": 1.25, "xga_90": 0.57, "goles_favor_90": 1.50, "goles_contra_90": 0.75},
    {"torneo": "Gold Cup", "equipo": "México", "partidos_jugados": 6, "xg_90": 1.61, "xga_90": 0.73, "goles_favor_90": 1.67, "goles_contra_90": 0.50},
    {"torneo": "Gold Cup", "equipo": "Estados Unidos", "partidos_jugados": 6, "xg_90": 1.51, "xga_90": 1.05, "goles_favor_90": 2.17, "goles_contra_90": 1.00},
    {"torneo": "Gold Cup", "equipo": "Canadá", "partidos_jugados": 4, "xg_90": 1.28, "xga_90": 0.70, "goles_favor_90": 2.25, "goles_contra_90": 0.25},
    {"torneo": "World Cup", "equipo": "Bélgica", "partidos_jugados": 3, "xg_90": 2.55, "xga_90": 0.98, "goles_favor_90": 2.00, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Canadá", "partidos_jugados": 3, "xg_90": 2.39, "xga_90": 0.62, "goles_favor_90": 2.67, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "España", "partidos_jugados": 3, "xg_90": 2.16, "xga_90": 0.55, "goles_favor_90": 1.67, "goles_contra_90": 0.00},
    {"torneo": "World Cup", "equipo": "Alemania", "partidos_jugados": 3, "xg_90": 2.14, "xga_90": 0.95, "goles_favor_90": 3.33, "goles_contra_90": 1.33},
    {"torneo": "World Cup", "equipo": "Inglaterra", "partidos_jugados": 3, "xg_90": 2.12, "xga_90": 0.88, "goles_favor_90": 2.00, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Colombia", "partidos_jugados": 3, "xg_90": 2.02, "xga_90": 0.92, "goles_favor_90": 1.33, "goles_contra_90": 0.33},
    {"torneo": "World Cup", "equipo": "Francia", "partidos_jugados": 3, "xg_90": 1.99, "xga_90": 0.80, "goles_favor_90": 3.33, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Senegal", "partidos_jugados": 3, "xg_90": 1.90, "xga_90": 1.20, "goles_favor_90": 2.67, "goles_contra_90": 2.00},
    {"torneo": "World Cup", "equipo": "Ecuador", "partidos_jugados": 3, "xg_90": 1.85, "xga_90": 1.32, "goles_favor_90": 0.67, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Suiza", "partidos_jugados": 3, "xg_90": 1.78, "xga_90": 1.03, "goles_favor_90": 2.33, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "Países Bajos", "partidos_jugados": 3, "xg_90": 1.78, "xga_90": 1.38, "goles_favor_90": 3.33, "goles_contra_90": 1.33},
    {"torneo": "World Cup", "equipo": "Estados Unidos", "partidos_jugados": 3, "xg_90": 1.75, "xga_90": 0.89, "goles_favor_90": 2.67, "goles_contra_90": 1.33},
    {"torneo": "World Cup", "equipo": "Marruecos", "partidos_jugados": 3, "xg_90": 1.75, "xga_90": 0.95, "goles_favor_90": 2.00, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "Suecia", "partidos_jugados": 3, "xg_90": 1.69, "xga_90": 1.09, "goles_favor_90": 2.33, "goles_contra_90": 2.33},
    {"torneo": "World Cup", "equipo": "Egipto", "partidos_jugados": 3, "xg_90": 1.63, "xga_90": 1.46, "goles_favor_90": 1.67, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "Brasil", "partidos_jugados": 3, "xg_90": 1.59, "xga_90": 1.28, "goles_favor_90": 2.33, "goles_contra_90": 0.33},
    {"torneo": "World Cup", "equipo": "Portugal", "partidos_jugados": 3, "xg_90": 1.48, "xga_90": 1.34, "goles_favor_90": 2.00, "goles_contra_90": 0.33},
    {"torneo": "World Cup", "equipo": "Argelia", "partidos_jugados": 3, "xg_90": 1.47, "xga_90": 1.11, "goles_favor_90": 1.67, "goles_contra_90": 2.33},
    {"torneo": "World Cup", "equipo": "Noruega", "partidos_jugados": 3, "xg_90": 1.38, "xga_90": 1.64, "goles_favor_90": 2.67, "goles_contra_90": 2.33},
    {"torneo": "World Cup", "equipo": "Argentina", "partidos_jugados": 3, "xg_90": 1.32, "xga_90": 0.74, "goles_favor_90": 2.67, "goles_contra_90": 0.33},
    {"torneo": "World Cup", "equipo": "Costa de Marfil", "partidos_jugados": 3, "xg_90": 1.27, "xga_90": 1.44, "goles_favor_90": 1.33, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "México", "partidos_jugados": 3, "xg_90": 1.23, "xga_90": 0.88, "goles_favor_90": 2.00, "goles_contra_90": 0.00},
    {"torneo": "World Cup", "equipo": "South Africa", "partidos_jugados": 3, "xg_90": 1.15, "xga_90": 1.40, "goles_favor_90": 0.67, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "Japón", "partidos_jugados": 3, "xg_90": 1.14, "xga_90": 1.11, "goles_favor_90": 2.33, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "RD Congo", "partidos_jugados": 3, "xg_90": 1.14, "xga_90": 1.26, "goles_favor_90": 1.33, "goles_contra_90": 1.00},
    {"torneo": "World Cup", "equipo": "Austria", "partidos_jugados": 3, "xg_90": 1.13, "xga_90": 1.36, "goles_favor_90": 2.00, "goles_contra_90": 2.00},
    {"torneo": "World Cup", "equipo": "Australia", "partidos_jugados": 3, "xg_90": 1.10, "xga_90": 1.72, "goles_favor_90": 0.67, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Croacia", "partidos_jugados": 3, "xg_90": 1.05, "xga_90": 1.35, "goles_favor_90": 1.67, "goles_contra_90": 1.67},
    {"torneo": "World Cup", "equipo": "Cabo Verde", "partidos_jugados": 3, "xg_90": 1.05, "xga_90": 1.98, "goles_favor_90": 0.67, "goles_contra_90": 0.67},
    {"torneo": "World Cup", "equipo": "Bosnia y Herzegovina", "partidos_jugados": 3, "xg_90": 1.04, "xga_90": 1.49, "goles_favor_90": 1.67, "goles_contra_90": 2.00},
    {"torneo": "World Cup", "equipo": "Paraguay", "partidos_jugados": 3, "xg_90": 0.76, "xga_90": 2.27, "goles_favor_90": 0.67, "goles_contra_90": 1.33},
    {"torneo": "World Cup", "equipo": "Ghana", "partidos_jugados": 3, "xg_90": 0.59, "xga_90": 1.48, "goles_favor_90": 0.67, "goles_contra_90": 0.67},
    {"torneo":"friendly","equipo":"Sudáfrica","partidos_jugados":4,"xg_90":1.42,"xga_90":0.23,"goles_favor_90":2.00,"goles_contra_90":0.50},
    {"torneo":"friendly","equipo":"Canadá","partidos_jugados":8,"xg_90":1.14,"xga_90":0.90,"goles_favor_90":1.25,"goles_contra_90":0.38},
    {"torneo":"friendly","equipo":"Brasil","partidos_jugados":4,"xg_90":1.62,"xga_90":1.04,"goles_favor_90":2.50,"goles_contra_90":1.00},
    {"torneo":"friendly","equipo":"Japón","partidos_jugados":6,"xg_90":1.42,"xga_90":1.10,"goles_favor_90":1.67,"goles_contra_90":1.00},
    {"torneo":"friendly","equipo":"Marruecos","partidos_jugados":5,"xg_90":1.92,"xga_90":0.11,"goles_favor_90":1.80,"goles_contra_90":0.00},
    {"torneo":"friendly","equipo":"Costa de Marfil","partidos_jugados":4,"xg_90":0.66,"xga_90":0.19,"goles_favor_90":0.50,"goles_contra_90":0.50},
    {"torneo":"friendly","equipo":"Noruega","partidos_jugados":2,"xg_90":1.46,"xga_90":0.31,"goles_favor_90":1.00,"goles_contra_90":0.50},
    {"torneo":"friendly","equipo":"Francia","partidos_jugados":10,"xg_90":1.90,"xga_90":1.10,"goles_favor_90":2.00,"goles_contra_90":1.30},
    {"torneo":"friendly","equipo":"Suecia","partidos_jugados":8,"xg_90":1.66,"xga_90":1.55,"goles_favor_90":2.75,"goles_contra_90":1.25},
    {"torneo":"friendly","equipo":"México","partidos_jugados":8,"xg_90":1.05,"xga_90":1.00,"goles_favor_90":0.88,"goles_contra_90":1.63},
    {"torneo":"friendly","equipo":"Ecuador","partidos_jugados":4,"xg_90":1.36,"xga_90":1.07,"goles_favor_90":1.00,"goles_contra_90":0.50},
    {"torneo":"friendly","equipo":"Inglaterra","partidos_jugados":6,"xg_90":1.44,"xga_90":0.53,"goles_favor_90":1.62,"goles_contra_90":1.00},
    {"torneo":"friendly","equipo":"República Democrática del Congo","partidos_jugados":5,"xg_90":0.36,"xga_90":0.48,"goles_favor_90":1.50,"goles_contra_90":0.16},
    {"torneo":"friendly","equipo":"Senegal","partidos_jugados":11,"xg_90":1.23,"xga_90":1.10,"goles_favor_90":1.88,"goles_contra_90":1.07},
    {"torneo":"friendly","equipo":"Portugal","partidos_jugados":14,"xg_90":1.62,"xga_90":1.14,"goles_favor_90":1.85,"goles_contra_90":0.80},
    {"torneo":"friendly","equipo":"Algeria","partidos_jugados":9,"xg_90":1.36,"xga_90":1.08,"goles_favor_90":2.70,"goles_contra_90":0.50},
    {"torneo":"friendly","equipo":"Argentina","partidos_jugados":7,"xg_90":1.79,"xga_90":0.53,"goles_favor_90":3.00,"goles_contra_90":0.12},
    {"torneo":"friendly","equipo":"Austria","partidos_jugados":3,"xg_90":0.77,"xga_90":0.64,"goles_favor_90":2.33,"goles_contra_90":0.33},
    {"torneo":"friendly","equipo":"Paraguay","partidos_jugados":4,"xg_90":0.76,"xga_90":0.99,"goles_favor_90":1.25,"goles_contra_90":1.75}
]

# --- 3. BASE DE DATOS BASE (ELO Y LOCALÍA) ---
teams_base = {
    "Sudáfrica": [1575, False], "Canadá": [1748, True], "Brasil": [2009, False],
    "Japón": [1910, False], "Alemania": [1916, False], "Paraguay": [1815, False],
    "Países Bajos": [1980, False], "Marruecos": [1877, False], "Costa de Marfil": [1743, False],
    "Noruega": [1918, False], "Francia": [2123, False], "Suecia": [1742, False],
    "México": [1912, True], "Ecuador": [1902, False], "Inglaterra": [2038, False],
    "RD Congo": [1712, False], "Bélgica": [1884, False], "Senegal": [1842, False],
    "Estados Unidos": [1781, True], "Bosnia y Herzegovina": [1622, False],
    "España": [2144, False], "Austria": [1836, False], "Portugal": [1990, False],
    "Croacia": [1905, False], "Suiza": [1914, False], "Argelia": [1785, False],
    "Australia": [1800, False], "Egipto": [1742, False], "Argentina": [2148, False],
    "Cabo Verde": [1622, False], "Colombia": [2004, False], "Ghana": [1575, False]
}

matches = [
    ("Sudáfrica", "Canadá"), ("Brasil", "Japón"), ("Alemania", "Paraguay"),
    ("Países Bajos", "Marruecos"), ("Costa de Marfil", "Noruega"), ("Francia", "Suecia"),
    ("México", "Ecuador"), ("Inglaterra", "RD Congo"), ("Bélgica", "Senegal"),
    ("Estados Unidos", "Bosnia y Herzegovina"), ("España", "Austria"), ("Portugal", "Croacia"),
    ("Suiza", "Argelia"), ("Australia", "Egipto"), ("Argentina", "Cabo Verde"), ("Colombia", "Ghana")
]

# --- 4. MAPEOS DE NORMALIZACIÓN ---
mapa_equipos = {"South Africa": "Sudáfrica", "Sweden": "Suecia"}
mapa_torneos = {"World Cup": "World Cups"}

# Mapeamos potencias a un diccionario simple
dict_potencias = {d["torneo"]: d["potencia_torneo"] for d in datos_potencia_torneos_2026}

# --- 5. PROCESAMIENTO Y AGREGACIÓN DE MÉTRICAS SOPESADAS ---
stats_acumuladas = {}

for reg in datos_torneos_completos:
    # Unificación de nombres
    equipo = mapa_equipos.get(reg["equipo"], reg["equipo"])
    torneo = mapa_torneos.get(reg["torneo"], reg["torneo"])
    
    potencia = dict_potencias.get(torneo, 1500.0)  # Valor neutro por si no coincide
    pj = reg["partidos_jugados"]
    
    # Peso = partidos_jugados * potencia_torneo
    peso = pj * potencia
    
    if equipo not in stats_acumuladas:
        stats_acumuladas[equipo] = {"sum_xg": 0, "sum_xga": 0, "sum_gf": 0, "sum_ga": 0, "total_peso": 0}
        
    stats_acumuladas[equipo]["sum_xg"] += reg["xg_90"] * peso
    stats_acumuladas[equipo]["sum_xga"] += reg["xga_90"] * peso
    stats_acumuladas[equipo]["sum_gf"] += reg["goles_favor_90"] * peso
    stats_acumuladas[equipo]["sum_ga"] += reg["goles_contra_90"] * peso
    stats_acumuladas[equipo]["total_peso"] += peso

# Calculamos los promedios ponderados finales
equipos_stats_nuevas = {}
for equipo, datos in stats_acumuladas.items():
    tp = datos["total_peso"]
    if tp > 0:
        equipos_stats_nuevas[equipo] = {
            "xg": datos["sum_xg"] / tp,
            "xga": datos["sum_xga"] / tp,
            "gf": datos["sum_gf"] / tp,
            "ga": datos["sum_ga"] / tp
        }

# --- 6. FUNCIONES DE CÁLCULO DE SIMULACIÓN ---
def get_stats(team):
    # Trae el Elo y la localía del diccionario base
    elo, es_local = teams_base.get(team, [1500, False])
    
    # Trae los nuevos promedios sopesados
    if team in equipos_stats_nuevas:
        ns = equipos_stats_nuevas[team]
        return elo, ns["xg"], ns["xga"], ns["gf"], ns["ga"], es_local
    else:
        # Fallback por si algún equipo no tuviera registros históricos en la lista
        return elo, 1.0, 1.0, 1.0, 1.0, es_local

def calculate_top_results(t1, t2):
    e1, xg1, xga1, gf1, ga1, loc1 = get_stats(t1)
    e2, xg2, xga2, gf2, ga2, loc2 = get_stats(t2)
    
    # Ajuste por localía
    adj1 = 1.15 if loc1 else 1.0
    adj2 = 1.15 if loc2 else 1.0
    
    # Cálculo ofensivo y defensivo combinando xG (70%) y Goles Reales (30%)
    atk1 = (xg1 * 0.7 + gf1 * 0.3) * adj1
    def1 = (xga1 * 0.7 + ga1 * 0.3)
    atk2 = (xg2 * 0.7 + gf2 * 0.3) * adj2
    def2 = (xga2 * 0.7 + ga2 * 0.3)
    
    elo_diff = (e1 - e2) / 400
    
    # Cálculo lambda para 120 minutos (factor 1.33)
    l1 = (1.4 * (atk1 / 1.7) * (def2 / 1.15) + elo_diff) * 1.33
    l2 = (1.4 * (atk2 / 1.7) * (def1 / 1.15) - elo_diff) * 1.33
    
    # Control de seguridad frente a lambdas negativas por diferencias drásticas de Elo
    l1 = max(0.01, l1)
    l2 = max(0.01, l2)
    
    results = []
    for i in range(6):
        for j in range(6):
            prob = (math.exp(-l1) * (l1**i) / math.factorial(i)) * \
                   (math.exp(-l2) * (l2**j) / math.factorial(j))
            results.append(((i, j), prob))
    
    return sorted(results, key=lambda x: x[1], reverse=True)[:3]

# --- 7. IMPRESIÓN DEL FIXTURE DE PARTIDOS ---
print(f"{'PARTIDO':<35} | {'TOP 3 RESULTADOS (Probabilidad)'}")
print("-" * 85)
for t1, t2 in matches:
    tops = calculate_top_results(t1, t2)
    res_str = ", ".join([f"{r[0][0]}-{r[0][1]} ({r[1]:.1%})" for r in tops])
    print(f"{t1 + ' vs ' + t2:<35} | {res_str}")