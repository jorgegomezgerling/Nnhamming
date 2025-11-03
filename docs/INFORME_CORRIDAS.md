# Informe de Evaluación - Red de Hamming

## Resumen Ejecutivo

Mediante este informe presentamos los resultados de la evaluación de una Red de Hamming aplicada al diagnóstico de enfermedades en dos datasets de diferente complejidad: Kaggle (133 clases) y Mendeley (24 clases).
Realizamos las pruebas exhaustivas solicitadas y correspondientes, las cuales incluyen matriz de confusión, optimización del parámetro K, test de ruido, sensibilidad a cantidad de patrones y sensibilidad al parámetro epsilon.

---

## 1. Introducción

### 1.1 Objetivos

- Evaluar el desempeño de la Red de Hamming en problemas de clasificación de clases (diagnósticos/enfermedades).
- Comparar resultados entre datasets con diferentes ratios clases/features.
- Identificar fortalezas y limitaciones del método.
- Determinar parámetros óptimos de configuración.

### 1.2 Metodología

Todas las evaluaciones se realizaron bajo las siguientes condiciones:

- Train/Test split: 80/20 con estratificación (para evitar desbalanceo de clases).
- Random state: 42 (para reproducibilidad)
- Epsilon: Valor por defecto (1.0)
- Rango de K evaluado: 1, 2, 3, 4, 5, 7, 10

---

## 2. Descripción de los Datasets

### 2.1 Dataset Kaggle

**Origen:** Kaggle Competition - Disease Prediction Dataset

**Características originales:**

- Muestras totales: 2,564
- Features originales: 400 síntomas binarios
- Clases: 133 enfermedades únicas
- Balance: **Desbalanceado** (rango: 3-43 casos por clase, promedio: 19.3)
  - 8 clases con <10 casos (ruido muestral)
  - 31 clases con <15 casos
  - Clase más rara: 'decubitus ulcer' (3 casos)
  - Clase más frecuente: 43 casos

**Pipeline de preprocesamiento:**

```
400 features -> PCA (100 componentes, 70.44% varianza)
            -> Mutual Info (10 componentes)
            -> Discretización (3 bins)
            -> Binarización (20 bits)
```

**Dataset final (GOLD):**

- Muestras: 2,564
- Features: 20 bits binarios
- Clases: 133
- Train: 2,051 muestras
- Test: 513 muestras
- Ratio clases/features: 6.65

---

### 2.2 Dataset Mendeley

**Origen:** Mendeley Data Repository - Medical Diagnosis Dataset

**Características originales:**

- Muestras totales: 757
- Features originales: 172 síntomas binarios
- Clases: 84 enfermedades únicas
- Balance: Desbalanceado (algunas clases con menos de 10 casos)

**Limpieza y filtrado:**

- Eliminación de columnas inválidas: 97
- Eliminación de duplicados: 3
- Eliminación de clase "None": 19 muestras
- Filtrado de clases raras (menos de 10 casos): 59 clases, 380 muestras

**Pipeline de preprocesamiento:**

```
74 features → PCA (40 componentes, 96.10% varianza)
           → Mutual Info (10 componentes)
           → Discretización (3 bins)
           → Binarización (20 bits)
```

**Dataset final (GOLD):**

- Muestras: 358
- Features: 20 bits binarios
- Clases: 24
- Train: 286 muestras
- Test: 72 muestras
- Ratio clases/features: 1.20

---

### 2.3 Comparación de Datasets

| Característica        | Kaggle | Mendeley |
| --------------------- | ------ | -------- |
| Muestras              | 2,564  | 358      |
| Clases                | 133    | 24       |
| Features finales      | 20     | 20       |
| Ratio clases/features | 6.65   | 1.20     |
| Varianza PCA retenida | 70.44% | 96.10%   |
| Mín. casos/clase      | 3      | 10       |
| Máx. casos/clase      | 29     | 41       |

**Observación clave:** Mendeley tiene un ratio clases/features 5.5 veces menor que Kaggle, lo que sugiere mejor separabilidad.

---

## 3. Resultados Kaggle

### 3.1 Caracterización del Problema

**Complejidad:**

- Bits necesarios (teórico): log₂(133) = 7.06 ≈ 7 bits
- Bits disponibles: 20 bits
- Ratio clases/features: 6.65

**Distribución de casos:**

- Mínimo: 3 casos
- Máximo: 43 casos
- Promedio: 19.3 casos

---

### 3.2 Matriz de Confusión (K=1)

**Resultados:**

- Accuracy: 24.37%
- Correctas: 125/513
- Incorrectas: 388/513

**Análisis por accuracy:**

- Enfermedades con 0% accuracy: 58/133 (43.6%)
- Muestras afectadas: 218/513 (42.5%)
- Accuracy excluyendo 0%: 42.4%
- Ratio clases/features: 6.65

**Top 10 mejor accuracy:**

```
1. Dengue                                :  100.0% (4/4)
2. Jaundice                              :  100.0% (4/4)
3. Malaria                               :  100.0% (4/4)
4. Typhoid                               :  100.0% (4/4)
5. Varicose veins                        :  100.0% (4/4)
6. Hepatitis E                           :   75.0% (3/4)
7. Chicken pox                           :   66.7% (2/3)
8. Gastroenteritis                       :   66.7% (2/3)
9. Tuberculosis                          :   66.7% (2/3)
10. Bronchial Asthma                     :   60.0% (3/5)
```

**Top 10 peor accuracy:**

```
124. Acne                                :    0.0% (0/4)
125. AIDS                                :    0.0% (0/4)
126. Alcoholic hepatitis                 :    0.0% (0/3)
127. Allergy                             :    0.0% (0/3)
128. Arthritis                           :    0.0% (0/3)
129. Cervical spondylosis                :    0.0% (0/4)
130. Chronic cholestasis                 :    0.0% (0/4)
131. Common Cold                         :    0.0% (0/4)
132. Drug Reaction                       :    0.0% (0/3)
133. Fungal infection                    :    0.0% (0/4)
```

**Interpretación posible:**
El alto número de clases con 0% accuracy (43.6%) indica que con solo 20 bits es muy difícil disntiguir de forma única 133 enfermedades. Muchas comparten patrones similares.

---

### 3.3 Optimización del Parámetro K

**Resultados:**

| K   | Aciertos | Accuracy   | Mejora vs K=1 |
| --- | -------- | ---------- | ------------- |
| 1   | 125/513  | 24.37%     | +0.00%        |
| 2   | 192/513  | 37.43%     | +13.06%       |
| 3   | 234/513  | 45.61%     | +21.25%       |
| 4   | 267/513  | 52.05%     | +27.68%       |
| 5   | 287/513  | 55.95%     | +31.58%       |
| 7   | 321/513  | 62.57%     | +38.21%       |
| 10  | 355/513  | **69.20%** | **+44.83%**   |

**Mejor K:** 10

**Interpretación:**

La mejora de +44.83% al usar K=10 demuestra que aunque la red no puede dar el diagnóstico exacto como primera opción, el diagnóstico correcto sí está dentro de los 10 candidatos más cercanos en el 69.2% de los casos.

---

### 3.4 Test de Ruido

**Configuración:**

- Features: 20 bits
- Niveles de ruido: 0%, 5%, 10%, 15%, 20%, 25%, 30%

**Resultados:**

| Ruido | Bits Invertidos | Accuracy | Pérdida |
| ----- | --------------- | -------- | ------- |
| 0%    | 0               | 24.37%   | -       |
| 5%    | 1               | 22.61%   | -1.76%  |
| 10%   | 2               | 21.05%   | -3.32%  |
| 15%   | 3               | 19.49%   | -4.88%  |
| 20%   | 4               | 17.54%   | -6.83%  |
| 25%   | 5               | 15.79%   | -8.58%  |
| 30%   | 6               | 14.23%   | -10.14% |

**Degradación total (0% → 30%):** -10.14%

**Interpretación:**

La red muestra degradación gradual y controlada ante ruido. Con 30% de bits invertidos (6 de 20), el accuracy solo baja 10 puntos porcentuales.

---

### 3.5 Sensibilidad a Cantidad de Patrones

**Configuración:**

- Train completo: 2,051 muestras
- Test (fijo): 513 muestras
- Porcentajes evaluados: 25%, 50%, 75%, 100%

**Resultados:**

| Train | Muestras | Prototipos | Accuracy |
| ----- | -------- | ---------- | -------- |
| 25%   | 512      | 512        | 23.78%   |
| 50%   | 1,025    | 1,025      | 24.17%   |
| 75%   | 1,538    | 1,538      | 24.27%   |
| 100%  | 2,051    | 2,051      | 24.37%   |

**Diferencia (25% → 100%):** +0.59%

**Interpretación:**

La cantidad de datos tiene impacto mínimo en el accuracy (solo +0.59%). Esto indica que el problema está limitado por la separabilidad de las clases (ratio alto), no por falta de datos de entrenamiento.

---

### 3.6 Sensibilidad al Parámetro Epsilon

**Configuración:**

- Prototipos: 2,051
- Epsilon factors evaluados: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
- Fórmula: ε = factor / (M + 1)

**Resultados:**

| ε factor | ε real   | Accuracy | Iter. promedio |
| -------- | -------- | -------- | -------------- |
| 0.1      | 0.000049 | 24.37%   | 2.1            |
| 0.5      | 0.000244 | 24.37%   | 2.1            |
| 1.0      | 0.000488 | 24.37%   | 2.1            |
| 2.0      | 0.000975 | 1.95%    | 20.0           |
| 5.0      | 0.002439 | 1.56%    | 20.0           |
| 10.0     | 0.004878 | 1.56%    | 20.0           |

**Mejor ε factor:** 1.0

**Interpretación:**

Existe un punto de quiebre crítico entre ε=1.0 y ε=2.0. Con valores mayores a 1.0, la inhibición es excesiva y la red colapsa (accuracy menor a 2%). El valor por defecto (1.0) es óptimo.

---

## 4. Resultados Mendeley

### 4.1 Caracterización del Problema

**Complejidad:**

- Bits necesarios (teórico): log₂(24) = 4.58 ≈ 5 bits
- Bits disponibles: 20 bits
- Ratio clases/features: 1.20

**Distribución de casos:**

- Mínimo: 10 casos
- Máximo: 41 casos
- Promedio: 14.9 casos
- Mediana: 12 casos
- Clases con menos de 10 casos: 0

---

### 4.2 Matriz de Confusión (K=1)

**Resultados:**

- Accuracy: 91.67%
- Correctas: 66/72
- Incorrectas: 6/72

**Análisis por accuracy:**

- Enfermedades con 0% accuracy: 0/24 (0.0%)
- Muestras afectadas: 0/72 (0.0%)
- Accuracy excluyendo 0%: 91.67%
- Ratio clases/features: 1.20

**Top 10 mejor accuracy:**

```
1. Dengue                                :  100.0% (3/3)
2. Diabetes                              :  100.0% (3/3)
3. Dysentery                             :  100.0% (4/4)
4. Gastroenteritis                       :  100.0% (3/3)
5. Hepatitis                             :  100.0% (3/3)
6. Jaundice                              :  100.0% (2/2)
7. Malaria                               :  100.0% (3/3)
8. Peptic Ulcer                          :  100.0% (2/2)
9. Pneumonia                             :  100.0% (3/3)
10. Tuberculosis                         :  100.0% (2/2)
```

**Top 10 peor accuracy (todas las demás tienen 66.7%-100%):**

```
15. Arthritis                            :   66.7% (2/3)
16. Bronchial Asthma                     :   66.7% (2/3)
17. Common Cold                          :   66.7% (2/3)
18. Drug Reaction                        :   66.7% (2/3)
19. Fungal infection                     :   66.7% (2/3)
20. Heart attack                         :   66.7% (2/3)
```

**Interpretación:**

Ninguna enfermedad tiene 0% accuracy. Todas las clases son clasificables con al menos 66.7% de precisión, demostrando buena separabilidad con ratio 1.20.

---

### 4.3 Optimización del Parámetro K

**Resultados:**

| K   | Aciertos | Accuracy   | Mejora vs K=1 |
| --- | -------- | ---------- | ------------- |
| 1   | 66/72    | 91.67%     | +0.00%        |
| 2   | 67/72    | 93.06%     | +1.39%        |
| 3   | 69/72    | 95.83%     | +4.17%        |
| 4   | 69/72    | 95.83%     | +4.17%        |
| 5   | 69/72    | 95.83%     | +4.17%        |
| 7   | 70/72    | 97.22%     | +5.56%        |
| 10  | 71/72    | **98.61%** | **+6.94%**    |

**Mejor K:** 10

**Interpretación:**

Aunque el accuracy con K=1 ya es excelente (91.67%), usar K=10 permite alcanzar 98.61%.

---

### 4.4 Test de Ruido

**Configuración:**

- Features: 20 bits
- Niveles de ruido: 0%, 5%, 10%, 15%, 20%, 25%, 30%

**Resultados:**

| Ruido | Bits Invertidos | Accuracy | Pérdida |
| ----- | --------------- | -------- | ------- |
| 0%    | 0               | 91.67%   | -       |
| 5%    | 1               | 90.28%   | -1.39%  |
| 10%   | 2               | 87.50%   | -4.17%  |
| 15%   | 3               | 84.72%   | -6.95%  |
| 20%   | 4               | 81.94%   | -9.73%  |
| 25%   | 5               | 79.17%   | -12.50% |
| 30%   | 6               | 76.39%   | -15.28% |

**Degradación total (0% → 30%):** -15.28%

**Interpretación:**
Incluso con 30% de ruido (6 bits invertidos), el accuracy se mantiene en 76.39%, demostrando alta robustez.

---

### 4.5 Sensibilidad a Cantidad de Patrones

**Configuración:**

- Train completo: 286 muestras
- Test (fijo): 72 muestras
- Porcentajes evaluados: 25%, 50%, 75%, 100%

**Resultados:**

| Train | Muestras | Prototipos | Accuracy |
| ----- | -------- | ---------- | -------- |
| 25%   | 71       | 71         | 88.89%   |
| 50%   | 143      | 143        | 90.28%   |
| 75%   | 214      | 214        | 91.67%   |
| 100%  | 286      | 286        | 91.67%   |

**Diferencia (25% → 100%):** +2.78%

**Interpretación:**
Con solo 25% del train (71 muestras) ya se alcanza 88.89% accuracy. La mejora es mayor que en Kaggle (+2.78% vs +0.59%), sugiriendo que más datos sí aportan cuando el ratio es favorable.

---

### 4.6 Sensibilidad al Parámetro Epsilon

**Configuración:**

- Prototipos: 286
- Epsilon factors evaluados: 0.1, 0.5, 1.0, 2.0, 5.0, 10.0
- Fórmula: ε = factor / (M + 1)

**Resultados:**

| ε factor | ε real   | Accuracy | Iter. promedio |
| -------- | -------- | -------- | -------------- |
| 0.1      | 0.000349 | 91.67%   | 2.3            |
| 0.5      | 0.001742 | 91.67%   | 2.3            |
| 1.0      | 0.003484 | 91.67%   | 2.3            |
| 2.0      | 0.006969 | 4.17%    | 20.0           |
| 5.0      | 0.017422 | 4.17%    | 20.0           |
| 10.0     | 0.034843 | 4.17%    | 20.0           |

**Mejor ε factor:** 1.0

**Interpretación:**

Similar a Kaggle, existe un punto de quiebre entre ε=1.0 y ε=2.0. El comportamiento es consistente entre ambos datasets.

---

## 5. Análisis Comparativo

### 5.1 Resumen de Métricas Principales

| Métrica                        | Kaggle         | Mendeley    | Ganador           |
| ------------------------------ | -------------- | ----------- | ----------------- |
| **Accuracy (K=1)**             | 24.37%         | 91.67%      | Mendeley (+67.3%) |
| **Accuracy (K=10)**            | 69.20%         | 98.61%      | Mendeley (+29.4%) |
| **Mejora con K**               | +44.83%        | +6.94%      | Kaggle            |
| **Clases con 0% accuracy**     | 58/133 (43.6%) | 0/24 (0.0%) | Mendeley          |
| **Robustez (30% ruido)**       | -10.14%        | -15.28%     | Kaggle            |
| **Sensibilidad a datos**       | +0.59%         | +2.78%      | Mendeley          |
| **Convergencia (iteraciones)** | 2.1            | 2.3         | Similar           |

---

### 5.2 Impacto del Ratio Clases/Features

**Hallazgo principal:** El ratio clases/features es el factor determinante del accuracy.

Kaggle: Ratio 6.65 → 24.37% accuracy (K=1)
Mendeley: Ratio 1.20 → 91.67% accuracy (K=1)

---

### 5.3 Efectividad del Parámetro K

**Observación:** Ambos datasets mejoran con K mayor a 1, pero de forma diferente.

**Kaggle (ratio alto):**

- Mejora significativa: +44.83% (de 24.37% a 69.20%)
- El diagnóstico correcto rara vez es la primera opción
- K=10 es esencial para utilidad clínica

**Mendeley (ratio bajo):**

- Mejora moderada: +6.94% (de 91.67% a 98.61%)
- El diagnóstico correcto suele ser la primera opción
- K=3 ya alcanza 95.83% accuracy

---

### 5.4 Robustez al Ruido

**Comparación a 30% de ruido:**

| Dataset  | Accuracy base | Accuracy con ruido | Pérdida |
| -------- | ------------- | ------------------ | ------- |
| Kaggle   | 24.37%        | 14.23%             | -10.14% |
| Mendeley | 91.67%        | 76.39%             | -15.28% |

**Pérdida relativa:**

- Kaggle: 41.6% del accuracy original
- Mendeley: 16.7% del accuracy original

---

### 5.5 Sensibilidad a Cantidad de Datos

**Mejora al usar 100% vs 25% del train:**

| Dataset  | 25% train | 100% train | Mejora |
| -------- | --------- | ---------- | ------ |
| Kaggle   | 23.78%    | 24.37%     | +0.59% |
| Mendeley | 88.89%    | 91.67%     | +2.78% |

**Interpretación:**

- Kaggle: Más datos no ayudan (problema de separabilidad)
- Mendeley: Más datos sí ayudan (problema de generalización)

---

## 6. Conclusiones Iniciales

### 6.1 Consideraciones:

1. **El ratio clases/features determina el accuracy:**

   - Ratio menor a 2.0: Accuracy mayor a 80%
   - Ratio mayor a 5.0: Accuracy menor a 30%

2. **La Red de Hamming es efectiva para datasets pequeños:**

   - Mendeley con solo 358 muestras logra 91.67% accuracy

3. **El parámetro K amplía la utilidad clínica:**

   - En Kaggle, K=10 mejora accuracy de 24% a 69%
   - Permite presentar múltiples diagnósticos probables

4. **Robustez moderada al ruido:**

   - Tolera hasta 15-20% de bits invertidos
   - Degradación gradual, no colapso súbito

5. **El parámetro epsilon tiene valor óptimo claro:**
   - ε=1.0 funciona bien en ambos datasets
   - Valores mayores causan colapso de la red

---

### 6.2 Fortalezas del Método

1. **Interpretabilidad:** La distancia de Hamming es intuitiva y explicable
2. **Velocidad:** Predicción en tiempo real (O(n) por consulta)
3. **Simplicidad:** No requiere ajuste complejo.
4. **Efectividad en ratios bajos:** Accuracy comparable a métodos modernos
5. **Pocos datos requeridos:** Funciona con cientos de muestras

---

### 6.3 Limitaciones Identificadas

1. **Dependencia del ratio:** Accuracy muestra gran diferencia dependiendo el ratio clases/features.
2. **Requiere preprocesamiento:** Pipeline PCA + Discretización + Binarización obligatorio
3. **Pérdida de información:** La binarización elimina granularidad
4. **Sensibilidad al desbalanceo:** Clases con menos de 10 casos tienen 0% accuracy
5. **No supera a métodos modernos en ratios altos:** Random Forest o Deep Learning son superiores para problemas complejos

---

### 6.4 Recomendaciones de Uso

**Usar Red de Hamming cuando:**

- Dataset pequeño (menos de 1,000 muestras)
- Ratio clases/features menor a 2.0
- Se requiere interpretabilidad
- Se necesita predicción rápida
- Aplicación médica con presentación de top-K candidatos

**No usar Red de Hamming cuando:**

- Ratio clases/features mayor a 5.0
- Se dispone de miles de muestras

---

## 7. Referencias

- Dataset Kaggle: Disease Prediction from Symptoms (Kaggle Competition)
- Dataset Mendeley: Medical Diagnosis Dataset (Mendeley Data Repository)

---

## Anexos

### Anexo A: Configuración de Experimentos

**Software:**

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

**Reproducibilidad:**
Todos los experimentos utilizaron random_state=42 para garantizar reproducibilidad.

---

### Anexo B: Ubicación de Resultados

Todos los gráficos e informes generados se encuentran en:

```
resultados/
├── kaggle_enfermedades/
│   ├── graficos/
│   │   ├── 00_analisis_dataset_gold.png
│   │   ├── 01_analisis_pca_varianza_barras.png
│   │   ├── 02_mutual_information_analisis.png
│   │   ├── 03_discretizacion_analisis.png
│   │   ├── 04_binarizacion_analisis.png
│   │   ├── 05_matriz_confusion_top20.png
│   │   ├── 06_distribucion_accuracy.png
│   │   ├── 07_optimizacion_k.png
│   │   ├── 08_test_ruido.png
│   │   ├── 09_sensibilidad_patrones.png
│   │   └── 10_sensibilidad_epsilon.png
│   ├── informes/
│   │   ├── 01_informe_para_pca.txt
│   │   ├── 02_informe_mutual_information.txt
│   │   ├── 03_informe_discretizacion.txt
│   │   └── 04_informe_binarizacion.txt
└── └── metricas/
│       ├── 00_caracterizacion_problema.txt
│       ├── 01_matriz_confusion_completa.csv
│       ├── 02_metricas_confusion.txt
│       ├── 03_metricas_por_enfermedad.csv
│       ├── 04_optimizacion_k.txt
│       ├── 05_optimizacion_k_detalle.csv
│       ├── 06_test_ruido.txt
│       ├── 07_test_ruido_detalle.csv
│       ├── 08_sensibilidad_patrones.txt
│       ├── 09_sensibilidad_patrones_detalle.csv
│       ├── 09_sensibilidad_epsilon.txt
│       └── 10_sensibilidad_epsilon_detalle.csv
└── mendeley_enfermedades/
    └── (misma estructura)
```

---

**Fecha de elaboración:** Noviembre 2025  
**Versión:** 1.0
