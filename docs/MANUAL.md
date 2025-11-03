# Manual de Referencia - Red de Hamming

## Tabla de Contenidos

1. [Alcances y Limitaciones](#alcances-y-limitaciones)
2. [Proceso de Instalación](#proceso-de-instalación)
3. [Modo de Correr y Test Demo](#modo-de-correr-y-test-demo)
4. [FAQ - Preguntas Frecuentes](#faq---preguntas-frecuentes)

---

## Alcances y Limitaciones

### Alcances

La Red de Hamming implementada puede:

- Clasificar hasta 133 clases diferentes (probado con el dataset Kaggle)
- Trabajar con entradas binarias de cualquier longitud
- Devolver los K candidatos más cercanos (Top-K predicciones)
- Tolerar ruido de hasta 10-15% de bits invertidos
- Manejar datasets desde 358 hasta 2,564 muestras

**Escenarios donde funciona bien:**

Cuando el ratio clases/features es bajo (menor a 2.0), la red alcanza accuracy mayor a 90%. Por ejemplo, con Mendeley (24 clases, 20 features) se obtuvo 91.67% accuracy.

**Escenarios con limitaciones:**

Cuando el ratio clases/features es alto (mayor a 5.0), el accuracy baja considerablemente. Por ejemplo, con Kaggle (133 clases, 20 features) se obtuvo solo 24.37% accuracy. Sin embargo, usando K=10 (top 10 candidatos), el accuracy sube a 69.20%.

**Ventajas:**

- Interpretable: La distancia de Hamming es fácil de entender (cuenta bits diferentes)
- Rápida: Cálculo simple, predicciones en tiempo real
- Simple: No requiere ajuste complejo de parámetros
- Robusta: Degrada gradualmente con ruido, sin fallos abruptos

---

### Limitaciones

**1. Solo acepta datos binarios**

La red solo trabaja con valores 0 y 1. En caso de datos continuos (temperatura, presión, etc.), se deben procesar siguiendo el siguiente pipeline:

```
Datos originales → PCA → Mutual Information → Discretización → Binarización
```

**2. Dependencia del ratio clases/features**

Las conclusiones parciales que son posibles de extraer de nuestra red y su relación con el ratio clases/features son las siguientes:

| Ratio clases/features | Accuracy esperado | Recomendación                  |
| --------------------- | ----------------- | ------------------------------ |
| Menor a 1.0           | 80-99%            | Ideal                          |
| Mayor a 5.0           | Menor a 30%       | Usar K grande o cambiar método |

Con muchas clases y pocos bits, las enfermedades comparten patrones y se confunden. La solución puede ser (en parte) usar K mayor a 1 para presentar múltiples candidatos.

**3. Clases con pocos casos**

En el análisis del database de Mendeley se comprobó que: las clases con menos de 10 casos tienden a tener 0% accuracy.
Se recomienda filtrarlas durante el preprocesamiento para óptimo comportamiento de la red.

**4. Pérdida de información**

Al binarizar (convertir valores continuos a 0/1), se pierde granularidad. Por ejemplo:

```
37.5°C y 39.0°C → Ambos se codifican como "fiebre alta" [0,1]
```

---

## Proceso de Instalación

### Requisitos

- Python 3.8 o superior
- Sistema operativo: Windows, macOS, o Linux

### Paso 1: Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Nnhamming
```

### Paso 2: Crear entorno virtual (opcional pero recomendado)

En Linux/macOS:

```bash
python3 -m venv {nombre_entorno_virtual}
source {nombre_entorno_virtual}/bin/activate
```

En Windows:

```bash
python -m venv {nombre_entorno_virtual}
{nombre_entorno_virtual}\Scripts\activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

El archivo requirements.txt contiene:

```
contourpy==1.3.3
cycler==0.12.1
fonttools==4.60.1
joblib==1.5.2
kiwisolver==1.4.9
kneed==0.8.5
matplotlib==3.10.6
numpy==2.3.2
packaging==25.0
pandas==2.3.2
pillow==11.3.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.7.2
scipy==1.16.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2025.2
```

### Paso 4: Verificar instalación

```bash
python3 -c "from src.Nnhamming import Nnhamming; print('Instalación exitosa')"
```

Si ves "Instalación exitosa", la red está lista.

### Paso 5: Verificar datasets

Los datasets están en:

```
datasets/kaggle_enfermedades/gold/kaggle_dataset.csv
datasets/mendeley_enfermedades/gold/mendeley_dataset.csv
```

### Solución de problemas comunes

**Error: "ModuleNotFoundError: No module named 'sklearn'"**

Solución:

```bash
pip install scikit-learn
```

**Error: "FileNotFoundError: datasets/..."**

Solución: Verificar que estás en el directorio raíz del proyecto (Nnhamming/).

**Error: "ImportError: cannot import name 'Nnhamming'"**

Solución: Agrega al inicio del script:

```python
import sys
sys.path.append("src")
```

---

## Modo de Correr y Test Demo

### Configuración: Seleccionar dataset

Edita el archivo `evaluaciones/config.py`:

```python
# Opciones: "kaggle_enfermedades" o "mendeley_enfermedades"
DATASET_ACTIVO = "mendeley_enfermedades"
```

### Test Demo 1: Predicción simple

El proyecto incluye un script de demostración (`demo.py`) que muestra el uso básico de la Red de Hamming con el dataset Mendeley limpio.

### Ejecutar el Demo

Desde la raíz del proyecto:

```bash
python3 demo.py
```

# Cargar dataset

df = pd.read_csv('datasets/mendeley_enfermedades/gold/mendeley_dataset.csv')

# Entrenar red

red = Nnhamming()
red.fit_from_df(df)

# Preparar patrón de prueba

patron = df.drop('prognosis', axis=1).iloc[0].values.tolist()
enfermedad_real = df['prognosis'].iloc[0]

# Predecir

predicciones = red.predict(patron, k=3)

print("RESULTADOS")

print("Top 3 predicciones:")
for i, (enfermedad, confianza) in enumerate(predicciones, 1):
marca = "[CORRECTO]" if enfermedad == enfermedad_real else "[INCORRECTO]"
print(f"{i}. {marca} {enfermedad:30s} (confianza: {confianza:.2f})")

````

Ejecutar:

```bash
python3 test_demo_simple.py
````

Salida esperada:

```

TEST DEMO: PREDICCIÓN SIMPLE

RESULTADOS

Enfermedad real: Diabetes

Top 3 predicciones:
1. [CORRECTO] Diabetes                       (confianza: 1.00)
2. [INCORRECTO] Dengue                       (confianza: 0.60)
3. [INCORRECTO] Malaria                      (confianza: 0.55)

```

### Evaluaciones completas

Para ejecutar todas las evaluaciones sobre un dataset:

```bash
cd evaluaciones

# Hay que asegurarse de configurar el dataset en config.py primero

python3 00_analisis_gold_ds.py        # Análisis del dataset
python3 01_matriz_confusion.py        # Matriz de confusión
python3 02_optimizacion_k.py          # Optimización de K
python3 03_test_ruido.py              # Test de ruido
python3 04_sensibilidad_patrones.py   # Sensibilidad a datos
python3 05_sensibilidad_epsilon.py    # Sensibilidad a epsilon
```

Los resultados se guardan en `resultados/<dataset_id>/graficos/` y `resultados/<dataset_id>/metricas/`.

---

## FAQ - Preguntas Frecuentes

### 1. ¿Por qué el accuracy es tan diferente entre Kaggle y Mendeley?

**Respuesta:**

Consideramos que la diferencia principal radica en el ratio clases/features:

- Kaggle: 133 clases / 20 features = 6.65
- Mendeley: 24 clases / 20 features = 1.20

Con solo 20 bits es imposible representar de forma única 133 enfermedades. Muchas comparten el mismo patrón binario. En cambio, 20 bits son suficientes para distinguir 24 enfermedades.

**Solución para Kaggle:**

Usar K=10. Aunque la red no acierta en primera opción (24% accuracy), el diagnóstico correcto sí está entre los 10 candidatos más probables el 69% de las veces.

---

### 2. ¿Cómo uso la red con mis propios datos?

**Respuesta:**

Se deben seguir 5 pasos:

**Paso 1: Preparación de datos**

El dataset debe ser un CSV con columnas numéricas (features) y una columna target (clase). Ejemplo:

```
temperatura,presion,colesterol,edad,enfermedad
37.5,120,200,45,Gripe
38.2,130,180,52,Neumonía
...
```

**Paso 2: Aplicar preprocesamiento**

La red solo acepta datos binarios, por eso se deben transformar los datos. Usar los scripts en `scripts/` como referencia. El proceso general es:

# 1. PCA para reducir dimensionalidad

# 2. Mutual Information para seleccionar las mejores features

# 3. Discretizar a 3 bins

# 4. Binarizar (3 bins → 2 bits cada uno)

**Paso 3: Crear dataset GOLD**

Guarda el resultado en formato CSV:

**Paso 4: Configurar**

Editar `evaluaciones/config.py`:

```python
DATASETS = {
    "datos_nuevo_usuario": {
        "id": "datos_nuevo_usuario",
        "nombre": "Mi Dataset",
        "path": "../datasets/datos_nuevo_usuario/gold/dataset.csv",
        "target": "prognosis"
    }
}

DATASET_ACTIVO = "datos_nuevo_usuario"
```

**Paso 5: Evaluar**

```bash
cd evaluaciones
python3 01_matriz_confusion.py
python3 02_optimizacion_k.py
```

---

### 3. ¿Qué es el parámetro K y cómo elegirlo?

**Respuesta:**

K es el número de candidatos que la red devuelve como posibles diagnósticos.

**K=1 (una sola predicción):**

```python
predicciones = red.predict(patron, k=1)
# Devuelve: [('Diabetes', 0.95)]
```

La red da un solo diagnóstico con su nivel de confianza (0-1). Cuando el accuracy es elevado utilizar k=1 es una excelente opción.

**K=3 (top 3 candidatos):**

```python
predicciones = red.predict(patron, k=3)
# Devuelve: [('Diabetes', 0.95), ('Dengue', 0.78), ('Malaria', 0.65)]
```

La red da las 3 enfermedades más probables ordenadas por confianza. Útil para analizar diferentes opciones.

**K=10 (top 10 candidatos):**

```python
predicciones = red.predict(patron, k=10)
```

La red da las 10 enfermedades más probables. Útil para casos de ratio clases/features alto (como nuestros datos de Kaggle).

**¿Cómo elegir K?**

Ejecuta la optimización:

```bash
python3 evaluaciones/02_optimizacion_k.py
```

El script prueba K=1, 2, 3, 4, 5, 7, 10 y te dice cuál da mejor accuracy.

**Ejemplo (Kaggle):**

```
K=1  → 24.37% accuracy
K=3  → 45.61% accuracy
K=5  → 55.95% accuracy
K=10 → 69.20% accuracy (mejor)
```

**Aplicación práctica:**

En vez de dar un diagnóstico único que puede estar mal:

```
"Diagnóstico: Diabetes"
```

Es mejor dar opciones ordenadas por confianza:

```
"Top 3 diagnósticos posibles:"
  1. Diabetes (confianza: 0.95)
  2. Prediabetes (confianza: 0.78)
  3. Síndrome Metabólico (confianza: 0.65)
```

Esto aumenta la utilidad del sistema.

---

**Última actualización:** Noviembre 2025
