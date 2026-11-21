import numpy as np
import pandas as pd

# 1. Descargar datos
url_phoible = "https://raw.githubusercontent.com/phoible/phoible/master/data/phoible.csv"
url_glottolog = "https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv"

phoible = pd.read_csv(url_phoible, low_memory=False)
glotto = pd.read_csv(url_glottolog, low_memory=False)

nombres_familias = dict(zip(glotto["ID"], glotto["Name"]))

# 2. Filtrar vocales largas en PHOIBLE
vocales_largas = phoible[
    (phoible["SegmentClass"] == "vowel")
    & (phoible["Phoneme"].str.contains("ː", na=False))
]

candidatas = (
    vocales_largas.groupby(["Glottocode", "ISO6393", "LanguageName"])["Phoneme"]
    .apply(lambda x: ", ".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"Phoneme": "Vocales_Largas"})
)

# 3. Cruzar con Glottolog y limpiar categorías no genealógicas
glotto_clean = glotto[
    ["ID", "Family_ID", "Macroarea", "Latitude", "Longitude"]
].rename(columns={"ID": "Glottocode"})
df_master = pd.merge(candidatas, glotto_clean, on="Glottocode", how="inner")

df_master["Family_ID"] = df_master["Family_ID"].fillna(df_master["Glottocode"])
df_master["Nombre_Familia"] = (
    df_master["Family_ID"].map(nombres_familias).fillna("Aislada")
)

# Excluir categorías administrativas de Glottolog (Bookkeeping, Unclassifiable, etc.)
familias_excluir = [
    "Bookkeeping",
    "Unclassifiable",
    "Artificial Language",
    "Speech Register",
    "Sign Language",
]
df_master = df_master[~df_master["Nombre_Familia"].isin(familias_excluir)]
df_master = df_master.dropna(subset=["Latitude", "Longitude"]).reset_index(
    drop=True
)

# 4. Cálculo de Diversidad Filogenética (Raíz cuadrada + Restos mayores)
tamano_real_glotto = glotto["Family_ID"].value_counts().to_dict()

diversidad = (
    df_master.groupby(["Family_ID", "Nombre_Familia"])
    .agg(
        Candidatas_Con_VL=("Glottocode", "count"),
        Macroareas=("Macroarea", lambda x: ", ".join(sorted(set(x.dropna())))),
    )
    .reset_index()
)

diversidad["Tamano_Glottolog"] = (
    diversidad["Family_ID"].map(tamano_real_glotto).fillna(1)
)
diversidad["Peso_Diversidad"] = np.sqrt(diversidad["Tamano_Glottolog"])

N_MUESTRA = 50
diversidad["Cuota_Exacta"] = (
    diversidad["Peso_Diversidad"] / diversidad["Peso_Diversidad"].sum()
) * N_MUESTRA
diversidad["Cuota_Entera"] = np.floor(diversidad["Cuota_Exacta"]).astype(int)
diversidad["Residuo"] = diversidad["Cuota_Exacta"] - diversidad["Cuota_Entera"]

faltantes = N_MUESTRA - diversidad["Cuota_Entera"].sum()
diversidad.loc[diversidad.nlargest(faltantes, "Residuo").index, "Cuota_Entera"] += 1

cuotas_finales = diversidad[diversidad["Cuota_Entera"] > 0].sort_values(
    by="Cuota_Entera", ascending=False
)


# 5. Selección espacial (Fórmula de Haversine para maximizar distancia entre lenguas)
def haversine(lat1, lon1, lat2, lon2):
  R = 6371.0
  phi1, phi2 = np.radians(lat1), np.radians(lat2)
  dphi = np.radians(lat2 - lat1)
  dlambda = np.radians(lon2 - lon1)
  a = (
      np.sin(dphi / 2.0) ** 2
      + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
  )
  return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


seleccionadas = []
np.random.seed(42)

for _, row in cuotas_finales.iterrows():
  fam = row["Family_ID"]
  cupos = row["Cuota_Entera"]
  candidatas_fam = df_master[df_master["Family_ID"] == fam].copy()

  # Priorizar las lenguas que tienen más vocales largas documentadas (mejor descripción fonética)
  candidatas_fam["Num_VL"] = candidatas_fam["Vocales_Largas"].apply(
      lambda x: len(x.split(", "))
  )
  candidatas_fam = candidatas_fam.sort_values(
      by=["Num_VL", "LanguageName"], ascending=[False, True]
  )

  if len(candidatas_fam) <= cupos:
    seleccionadas.append(candidatas_fam)
  else:
    # CORRECCIÓN: Usar .iloc[0] para obtener la primera fila
    elegidas_fam = [candidatas_fam.iloc[0]]
    while len(elegidas_fam) < cupos:
      restantes = candidatas_fam[
          ~candidatas_fam["Glottocode"].isin(
              [e["Glottocode"] for e in elegidas_fam]
          )
      ]
      mejor_cand = None
      max_dist_minima = -1

      todas_previas = [
          e
          for sub in seleccionadas
          for _, e in (
              sub.iterrows()
              if isinstance(sub, pd.DataFrame)
              else enumerate([sub])
          )
      ] + elegidas_fam

      for _, cand in restantes.iterrows():
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

muestra_final = pd.concat(seleccionadas).drop(columns=["Num_VL"]).reset_index(
    drop=True
)
muestra_final.to_csv("muestra_50_lenguas_limpia.csv", index=False)

print("Resumen por Macroárea en la muestra final:")
print(muestra_final["Macroarea"].value_counts(dropna=False))
print("\nPrimeras 15 lenguas seleccionadas:")
print(
    muestra_final[
        ["LanguageName", "ISO6393", "Nombre_Familia", "Macroarea", "Vocales_Largas"]
    ].head(15)
)
