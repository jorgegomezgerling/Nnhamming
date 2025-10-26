import pandas as pd 

# Cargar datos discretizados
df = pd.read_csv('../dataset/silver/03_discretizado_10comp.csv')

print("="*60)
print("DISTRIBUCIÓN DE CLASES POR BIN")
print("="*60)

# Analizar las primeras 3 componentes
for col in df.drop('prognosis', axis=1).columns[:3]:
    print(f"\n{'='*60}")
    print(f"{col}")
    print(f"{'='*60}")
    
    # Para cada bin, ver qué clases caen ahí
    for bin_val in [0, 1, 2]:
        # Filtrar filas en este bin
        filas_en_bin = df[df[col] == bin_val]
        
        print(f"\nBin {int(bin_val)} ({len(filas_en_bin)} muestras):")
        
        # Contar clases en este bin
        clases_en_bin = filas_en_bin['prognosis'].value_counts()
        
        # Mostrar top 5 clases más frecuentes en este bin
        print("  Top 5 clases:")
        for clase, count in clases_en_bin.head(5).items():
            porcentaje = (count / len(filas_en_bin)) * 100
            print(f"    {clase}: {count} ({porcentaje:.1f}%)")
        
        # ¿Cuántas clases diferentes hay en este bin?
        print(f"  Total de clases diferentes en este bin: {len(clases_en_bin)}")
