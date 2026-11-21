import numpy as np
import pandas as pd

# ==========================================
# 1. EXTRACCIÓN DE PHOIBLE Y GLOTTOLOG
# ==========================================
url_phoible = (
    "https://githubusercontent.com"
)
url_glottolog = "https://githubusercontent.com"

print("Descargando bases de datos...")
phoible = pd.read_csv(url_phoible, low_memory=False)
glotto = pd.read_csv(url_glottolog, low_memory=False)

# Filtrar vocales con marca de longitud 'ː' (puedes refinar si quieres excluir diptongos)
vocales_largas = phoible[
    (phoible["SegmentClass"] == "vowel")
    & (phoible["Phoneme"].str.contains("ː", na=False))
]

# Contar cuántas vocales largas reporta cada lengua y quedarnos con Glottocodes únicos
candidatas = (
    vocales_largas.groupby(["Glottocode", "ISO6393", "LanguageName"])["Phoneme"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"Phoneme": "Vocales_Largas"})
)

# Cruzar con Glottolog para obtener Familia, Macroárea y Coordenadas
glotto_clean = glotto[[
    "ID",
    "Name",
    "Family_ID",
    "Macroarea",
    "Latitude",
    "Longitude",
    "Classification",
]].rename(columns={"ID": "Glottocode"})
df_master = pd.merge(candidatas, glotto_clean, on="Glottocode", how="inner")

# Limpiar lenguas sin coordenadas o aisladas (a las aisladas les asignamos su propio Glottocode como familia)
df_master["Family_ID"] = df_master["Family_ID"].fillna(df_master["Glottocode"])
df_master = df_master.dropna(subset=["Latitude", "Longitude"]).reset_index(
    drop=True
)

print(
    f"Total de lenguas únicas con vocales largas y coordenadas: {len(df_master)}"
)


# ==========================================
# 2. DIVERSIDAD FILOGENÉTICA (Aprox. Rijkhoff & Bakker)
# ==========================================
# Calculamos el peso de diversidad de cada familia contando sus ramas únicas en la clasificación
def calcular_dv(grupo):
  # Cuenta cuántos nodos distintos existen en los niveles de clasificación de la familia
  niveles = set()
  for ruta in grupo["Classification"].dropna():
    partes = ruta.split("/")
    for i, nodo in enumerate(partes):
      niveles.add((i, nodo))
  # Si es lengua aislada (sin '/') le damos valor base 1
  return max(1, len(niveles))


dv_familias = (
    df_master.groupby("Family_ID")
    .apply(calcular_dv)
    .reset_index(name="DV_Score")
)
dv_familias["Proporcion"] = (
    dv_familias["DV_Score"] / dv_familias["DV_Score"].sum()
)

# Asignar cuotas para N = 50 usando el método de restos mayores (Largest Remainder Method)
N_MUESTRA = 50
dv_familias["Cuota_Exacta"] = dv_familias["Proporcion"] * N_MUESTRA
dv_familias["Cuota_Entera"] = np.floor(dv_familias["Cuota_Exacta"]).astype(int)
dv_familias["Residuo"] = dv_familias["Cuota_Exacta"] - dv_familias["Cuota_Entera"]

faltantes = N_MUESTRA - dv_familias["Cuota_Entera"].sum()
indices_top = dv_familias.nlargest(faltantes, "Residuo").index
dv_familias.loc[indices_top, "Cuota_Entera"] += 1

# Filtrar solo familias que obtuvieron al menos 1 cupo
cuotas_finales = dv_familias[dv_familias["Cuota_Entera"] > 0].sort_values(
    by="Cuota_Entera", ascending=False
)
print("\nDistribución de cupos por familia (Top 10):")
print(cuotas_finales[["Family_ID", "DV_Score", "Cuota_Entera"]].head(10))


# ==========================================
# 3. FILTRO ESPACIAL (HAVERSINE) PARA SELECCIONAR LAS 50
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
  R = 6371.0  # Radio de la Tierra en km
  phi1, phi2 = np.radians(lat1), np.radians(lat2)
  dphi = np.radians(lat2 - lat1)
  dlambda = np.radians(lon2 - lon1)
  a = (
      np.sin(dphi / 2.0) ** 2
      + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
  )
  return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


seleccionadas = []
np.random.seed(42)  # Para que tu selección sea 100% reproducible

for _, row in cuotas_finales.iterrows():
  fam = row["Family_ID"]
  cupos = row["Cuota_Entera"]
  candidatas_fam = df_master[df_master["Family_ID"] == fam].copy()

  # Si solo hay 1 lengua o pide todas las disponibles en esa familia
  if len(candidatas_fam) <= cupos:
    seleccionadas.append(candidatas_fam)
  else:
    # Elegir la que tenga más vocales largas documentadas como ancla inicial
    candidatas_fam["Num_Vocales"] = candidatas_fam["Vocales_Largas"].apply(
        lambda x: len(x.split(", "))
    )
    elegidas_fam = [
        candidatas_fam.sort_values(by="Num_Vocales", ascending=False).iloc[0]
    ]

    # Para cupos adicionales, maximizar distancia geográfica respecto a las ya seleccionadas
    while len(elegidas_fam) < cupos:
      restantes = candidatas_fam[
          ~candidatas_fam["Glottocode"].isin(
              [e["Glottocode"] for e in elegidas_fam]
          )
      ]
      mejor_cand = None
      max_dist_minima = -1

      for _, cand in restantes.iterrows():
        # Distancia mínima a cualquier lengua ya elegida en toda la muestra
        todas_previas = [
            e
            for sub in seleccionadas
            for _, e in (
                sub.iterrows()
                if isinstance(sub, pd.DataFrame)
                else enumerate([sub])
            )
        ] + elegidas_fam
        dists = [
            haversine(
                cand["Latitude"], cand["Longitude"], p["Latitude"], p["Longitude"]
            )
            for p in todas_previas
        ]
        dist_min = min(dists) if dists else 0

        if dist_min > max_dist_minima:
          max_dist_minima = dist_min
          mejor_cand = cand

      elegidas_fam.append(mejor_cand)
    seleccionadas.append(pd.DataFrame(elegidas_fam))

muestra_final = pd.concat(seleccionadas).reset_index(drop=True)
muestra_final.to_csv("muestra_50_lenguas_vocales_largas.csv", index=False)
print(
    f"\n¡Listo! Muestra generada con {len(muestra_final)} lenguas y guardada en"
    " CSV."
)
