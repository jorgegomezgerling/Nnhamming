"""
Limpieza y filtrado del dataset Mendeley
- Elimina clase "None"
- Elimina columnas vacías o con nombres inválidos
- Filtra clases con <10 casos
- Limpia nombres de columnas
- Elimina columnas duplicadas
"""

import pandas as pd
import numpy as np
import os
import re

print("="*70)
print("LIMPIEZA Y FILTRADO: Mendeley Dataset")
print("="*70)

# Cargar raw
df = pd.read_csv('../../datasets/mendeley_enfermedades/bronze/raw_data.csv')

print(f"\n[1/6] Dataset original:")
print(f"  Dimensiones: {df.shape}")
print(f"  Clases: {df['prognosis'].nunique()}")

# Limpiar caracteres no-ASCII
df = df.replace({r'[^\x00-\x7F]+': ''}, regex=True)
df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

# Limpiar nombres de columnas
def limpiar_nombre_columna(nombre):
    """
    Limpia nombres de columnas eliminando:
    - Paréntesis
    - Guiones bajos al inicio/final
    - Espacios múltiples
    """
    # Eliminar paréntesis
    nombre = re.sub(r'[()]', '', nombre)
    # Eliminar guiones bajos al inicio/final y espacios
    nombre = nombre.strip('_').strip()
    # Reemplazar múltiples espacios/guiones bajos por uno
    nombre = re.sub(r'[_\s]+', '_', nombre)
    # Eliminar guiones bajos finales de nuevo
    nombre = nombre.strip('_')
    return nombre

# Aplicar limpieza a todas las columnas excepto prognosis
nuevos_nombres = {}
for col in df.columns:
    if col != 'prognosis':
        nuevo = limpiar_nombre_columna(col)
        if nuevo and nuevo != col:
            nuevos_nombres[col] = nuevo

df = df.rename(columns=nuevos_nombres)

print(f"\n[2/6] Limpieza de nombres de columnas:")
print(f"  Columnas renombradas: {len(nuevos_nombres)}")

X = df.drop('prognosis', axis=1)
Y = df['prognosis'].str.strip().str.replace(r'[()]', '', regex=True)

# Convertir a numérico
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Eliminar columnas con nombres inválidos (vacíos, solo guiones bajos, muy cortos)
def es_nombre_valido(nombre):
    """
    Verifica si un nombre de columna es válido:
    - No vacío
    - No solo guiones bajos/espacios
    - Al menos 2 caracteres alfabéticos
    """
    nombre_limpio = nombre.strip('_').strip()
    
    # Vacío o muy corto
    if not nombre_limpio or len(nombre_limpio) < 2:
        return False
    
    # Solo guiones bajos/espacios/guiones
    if re.match(r'^[_\s\-]+$', nombre_limpio):
        return False
    
    # Debe tener al menos 2 letras
    letras = re.findall(r'[a-zA-Z]', nombre_limpio)
    if len(letras) < 2:
        return False
    
    return True

cols_validas = [col for col in X.columns if es_nombre_valido(col)]
cols_eliminadas = [col for col in X.columns if not es_nombre_valido(col)]

print(f"\n[3/6] Eliminación de columnas inválidas:")
print(f"  Features originales: {len(X.columns)}")
print(f"  Features válidas: {len(cols_validas)}")
print(f"  Eliminadas: {len(cols_eliminadas)}")
if len(cols_eliminadas) <= 10:
    print(f"  Columnas eliminadas: {cols_eliminadas}")

X = X[cols_validas]

# Eliminar columnas duplicadas
print(f"\n[4/6] Eliminación de columnas duplicadas:")
cols_antes = len(X.columns)
X = X.loc[:, ~X.columns.duplicated(keep='first')]
cols_despues = len(X.columns)
duplicadas = cols_antes - cols_despues
print(f"  Columnas duplicadas eliminadas: {duplicadas}")
if duplicadas > 0:
    # Encontrar cuáles eran duplicadas
    todas_cols = pd.Series(cols_validas)
    dups = todas_cols[todas_cols.duplicated()].unique()
    print(f"  Nombres duplicados: {dups.tolist()}")

# Eliminar clase "None"
mask_none = Y != 'None'
X_sin_none = X[mask_none]
Y_sin_none = Y[mask_none]

n_eliminados_none = (~mask_none).sum()

print(f"\n[5/6] Eliminación clase 'None':")
print(f"  Casos eliminados: {n_eliminados_none}")
print(f"  Muestras restantes: {len(Y_sin_none)}")

# Filtrar clases raras (<10 casos)
casos_por_enfermedad = Y_sin_none.value_counts()
enfermedades_validas = casos_por_enfermedad[casos_por_enfermedad >= 10].index

mask_validas = Y_sin_none.isin(enfermedades_validas)
X_filtrado = X_sin_none[mask_validas]
Y_filtrado = Y_sin_none[mask_validas]

n_clases_eliminadas = Y_sin_none.nunique() - Y_filtrado.nunique()
n_muestras_eliminadas = len(Y_sin_none) - len(Y_filtrado)

print(f"\n[6/6] Filtrado de clases raras (<10 casos):")
print(f"  Clases eliminadas: {n_clases_eliminadas}")
print(f"  Muestras eliminadas: {n_muestras_eliminadas}")
print(f"  Clases finales: {Y_filtrado.nunique()}")
print(f"  Muestras finales: {len(Y_filtrado)}")

# Recombinar y guardar
df_limpio = X_filtrado.copy()
df_limpio['prognosis'] = Y_filtrado.values

os.makedirs('../../datasets/mendeley_enfermedades/silver', exist_ok=True)
df_limpio.to_csv('../../datasets/mendeley_enfermedades/silver/00_clean_dataset.csv', index=False)

print(f"\n{'='*70}")
print("RESUMEN")
print(f"{'='*70}")
print(f"Dimensiones originales:  {df.shape}")
print(f"Dimensiones finales:     {df_limpio.shape}")
print(f"Reducción de muestras:   {len(df) - len(df_limpio)} ({(len(df) - len(df_limpio))/len(df)*100:.1f}%)")
print(f"Reducción de features:   {df.shape[1]-1} → {df_limpio.shape[1]-1}")
print(f"Reducción de clases:     {df['prognosis'].nunique()} → {Y_filtrado.nunique()}")
print(f"\nArchivo guardado: datasets/mendeley_enfermedades/silver/silver01/dataset.csv")
print(f"{'='*70}")

# Mostrar nombres finales
print(f"\nPrimeras 20 columnas finales:")
for i, col in enumerate(list(df_limpio.columns[:20]), 1):
    print(f"  {i:2d}. {col}")

# Verificar duplicados finales
if df_limpio.columns.duplicated().any():
    print("\n⚠️  ADVERTENCIA: Aún hay columnas duplicadas:")
    print(df_limpio.columns[df_limpio.columns.duplicated()].tolist())
else:
    print("\n✅ No hay columnas duplicadas")

# Informe
os.makedirs('../../resultados/mendeley_enfermedades/informes', exist_ok=True)

with open('../../resultados/mendeley_enfermedades/informes/01_limpieza_y_filtrado.txt', 'w') as f:
    f.write("LIMPIEZA Y FILTRADO: Mendeley Dataset\n\n")
    
    f.write("DATASET ORIGINAL\n")
    f.write(f"  Dimensiones: {df.shape}\n")
    f.write(f"  Clases: {df['prognosis'].nunique()}\n\n")
    
    f.write("LIMPIEZA DE NOMBRES\n")
    f.write(f"  Columnas renombradas: {len(nuevos_nombres)}\n\n")
    
    f.write("ELIMINACIÓN DE COLUMNAS INVÁLIDAS\n")
    f.write(f"  Features originales: {len(X.columns) + len(cols_eliminadas)}\n")
    f.write(f"  Columnas inválidas eliminadas: {len(cols_eliminadas)}\n\n")
    
    f.write("ELIMINACIÓN DE DUPLICADOS\n")
    f.write(f"  Columnas duplicadas eliminadas: {duplicadas}\n")
    if duplicadas > 0:
        f.write(f"  Nombres duplicados: {dups.tolist()}\n")
    f.write("\n")
    
    f.write("ELIMINACIÓN CLASE 'NONE'\n")
    f.write(f"  Casos eliminados: {n_eliminados_none}\n\n")
    
    f.write("FILTRADO CLASES RARAS (<10 casos)\n")
    f.write(f"  Clases eliminadas: {n_clases_eliminadas}\n")
    f.write(f"  Muestras eliminadas: {n_muestras_eliminadas}\n\n")
    
    f.write("RESULTADO FINAL\n")
    f.write(f"  Dimensiones: {df_limpio.shape}\n")
    f.write(f"  Features: {df_limpio.shape[1]-1}\n")
    f.write(f"  Clases: {Y_filtrado.nunique()}\n")
    f.write(f"  Muestras: {len(df_limpio)}\n")
    f.write(f"  Rango casos/clase: {casos_por_enfermedad[enfermedades_validas].min()}-{casos_por_enfermedad[enfermedades_validas].max()}\n")

print("\nInforme guardado: resultados/mendeley_enfermedades/informes/01_limpieza_y_filtrado.txt")