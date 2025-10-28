# RED DE HAMMING - Diagnóstico de Enfermedades

_Trabajo académico: Implementación de una Red Neuronal de Hamming para clasificación multiclase aplicada al diagnóstico médico._

---

## Descripción

Implementación de una Red de Hamming competitiva para clasificar enfermedades basándose en síntomas binarios. El proyecto incluye preprocesamiento completo de datos, evaluaciones exhaustivas y comparación entre dos datasets de diferente complejidad.

### Características Principales

- **Clasificación basada en Distancia de Hamming**: Mide la similitud entre el vector de síntomas de entrada y los prototipos de enfermedades almacenados.
- **Representación Bipolar**: Transformación a representación bipolar (-1, 1) para cálculo eficiente de distancias.
- **Predicción Top-K**: Retorna los K candidatos más probables ordenados por distancia.
- **Capa de Inhibición Lateral**: Competencia entre neuronas para seleccionar el mejor candidato.
- **Preprocesamiento Completo**: PCA -> Mutual Information -> Discretización -> Binarización.
- **Evaluaciones Exhaustivas**: Matriz de confusión, optimización K, test de ruido, sensibilidad a datos y epsilon.

---

## Resultados Destacados

| Métrica             | Kaggle  | Mendeley   |
| ------------------- | ------- | ---------- |
| Muestras            | 2,564   | 358        |
| Clases              | 133     | 24         |
| Features            | 20 bits | 20 bits    |
| **Accuracy (K=1)**  | 24.37%  | **91.67%** |
| **Accuracy (K=10)** | 69.20%  | **98.61%** |

---

## Instalación

### Requisitos

- Python 3.8+
- pip

### Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/jorgegomezgerling/Nnhamming
cd IA_TP_FINAL

# Crear entorno virtual (recomendado)
python3 -m venv IA
source IA/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**

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

---

## Estructura del Proyecto

```
IA_TP_FINAL/
├── datasets/                      # Datos preprocesados (gold)
│   ├── kaggle_enfermedades/
│   │   └── gold/
│   │       └── kaggle_dataset.csv
│   └── mendeley_enfermedades/
│       └── gold/
│           └── mendeley_dataset.csv
├── docs/                          # Documentación
│   ├── MANUAL.md                  # Manual completo
│   └── INFORME_CORRIDAS.md        # Evaluación de resultados
├── evaluaciones/                  # Scripts de evaluación
│   ├── config.py                  # Configuración de datasets
│   ├── 00_analisis_gold_ds.py
│   ├── 01_matriz_confusion.py
│   ├── 02_optimizacion_k.py
│   ├── 03_test_ruido.py
│   ├── 04_sensibilidad_patrones.py
│   └── 05_sensibilidad_epsilon.py
├── resultados/                    # Gráficos y métricas generadas
│   ├── kaggle_enfermedades/
│   └── mendeley_enfermedades/
├── scripts/                       # Preprocesamiento de datos
│   ├── kaggle_enfermedades/
│   └── mendeley_enfermedades/
├── src/                           # Código fuente
│   ├── Nnhamming.py               # Implementación de la red
│   └── main.py
├── IA/                            # Entorno virtual
├── README.md
└── requirements.txt
```

---

## Uso Rápido

### 1. Configurar Dataset

Editar `evaluaciones/config.py`:

```python
# Opciones: "kaggle_enfermedades" o "mendeley_enfermedades"
DATASET_ACTIVO = "mendeley_enfermedades"
```

### 2. Ejecutar Evaluaciones

```bash
cd evaluaciones

# Análisis del dataset
python3 00_analisis_gold_ds.py

# Evaluaciones de la red
python3 01_matriz_confusion.py      # Matriz de confusión
python3 02_optimizacion_k.py        # Optimización de K
python3 03_test_ruido.py            # Robustez al ruido
python3 04_sensibilidad_patrones.py # Sensibilidad a datos
python3 05_sensibilidad_epsilon.py  # Sensibilidad a epsilon
```

### 3. Ver Resultados

Los gráficos e informes se generan en:

```
resultados/<dataset_id>/
├── graficos/      # Visualizaciones PNG
└── metricas/      # Informes TXT y CSV
```

---

## Test

Ejemplo básico de uso:

```python
from src.Nnhamming import Nnhamming
import pandas as pd

# Cargar dataset
df = pd.read_csv('datasets/mendeley_enfermedades/gold/mendeley_dataset.csv')

# Entrenar red
red = Nnhamming()
red.fit_from_df(df)

# Predecir con un patrón
patron = [1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,0]
predicciones = red.predict(patron, k=3)

# Mostrar resultados
print("Top 3 predicciones:")
for enfermedad, distancia in predicciones:
    print(f"  {enfermedad}: distancia Hamming = {distancia}")
```

**Ver más ejemplos en:** [`docs/MANUAL.md`](docs/MANUAL.md)

---

## Documentación

### Manual Completo

**[`docs/MANUAL.md`](docs/MANUAL.md)** - Incluye:

- Alcances y limitaciones detalladas
- Proceso de instalación paso a paso
- Tests demo completos (código ejecutable)
- FAQ con 3+ preguntas frecuentes

### Informe de Evaluaciones

**[`docs/INFORME_CORRIDAS.md`](docs/INFORME_CORRIDAS.md)** - Incluye:

- Evaluación exhaustiva en ambos datasets
- Análisis comparativo de resultados
- Gráficos y métricas detalladas
- Conclusiones y recomendaciones

---

## Integrantes

- **Galarza, Francisco**
- **Gómez, Francisco**
- **Gómez, Jorge**

**Materia:** Inteligencia Artificial  
**Institución:** UNIVERSIDAD AUTONOMA DE ENTRE RIOS FACULTAD DE CIENCIA Y TECNOLOGIA (UADER FCYT).
**Año:** 2025

---

## Licencia

Este proyecto es de uso académico.
