# 🤖 TelecomX — Predicción de Evasión de Clientes (Machine Learning)

<div align="center">

**Parte 2: Modelado Predictivo y Sistema de Alerta Temprana**

**Oracle Next Education (ONE) · Alura Latam — Challenge 2 · Data Science**

*Autora: Daniela Andrea Puebla Mosca*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
</div>

---

## 📋 Descripción

Este proyecto constituye la **segunda fase** del análisis de evasión de clientes de **Telecom X**, enfocándose en la construcción de **modelos predictivos** de Machine Learning capaces de anticipar qué clientes tienen mayor probabilidad de cancelar sus servicios.

Partiendo del dataset limpio generado en la [Parte 1](../TelecomX_LATAM.ipynb), implementamos un **pipeline completo de ML** que incluye:
- Preprocesamiento especializado para algoritmos de clasificación
- Entrenamiento y evaluación de múltiples modelos
- Análisis de interpretabilidad con SHAP values
- Generación de recomendaciones estratégicas accionables

**Objetivo final:** Reducir la tasa de evasión del **26.5%** mediante un sistema de scoring predictivo que permita intervenciones proactivas.

---

## 🗂️ Estructura del Proyecto

```
TelecomX-ML-Prediction/
│
├── TelecomX_Parte2_DPuebla.ipynb    # Notebook principal (ML Pipeline)
├── telecomx_limpio.csv              # Dataset procesado (generado en Parte 1)
├── README.md                        # Este archivo
└── papeline_machine_leraning.png   # Diagrama del pipeline ML
```

---

## 🔄 Pipeline de Machine Learning

El proyecto sigue un flujo estructurado de 6 etapas:

```
┌──────────────────────────────────────────────────────────────────┐
│  1️⃣  PREPARACIÓN       Codificación + Normalización              │
│  2️⃣  CORRELACIÓN       Análisis + Selección de variables         │
│  3️⃣  ENTRENAMIENTO     Random Forest + XGBoost + Reg. Logística │
│  4️⃣  EVALUACIÓN        Accuracy, Precision, Recall, F1, ROC-AUC │
│  5️⃣  INTERPRETACIÓN    Feature Importance + SHAP values         │
│  6️⃣  CONCLUSIÓN        Recomendaciones estratégicas             │
└──────────────────────────────────────────────────────────────────┘
```

### Etapa 1 — Preparación de Datos

| Proceso | Técnica aplicada | Objetivo |
|---------|------------------|----------|
| **Codificación binaria** | Label Encoding: `{'Sí': 1, 'No': 0}` | Variables Sí/No → numérico |
| **Codificación categórica** | One-Hot Encoding con `drop_first=True` | Tipo_Contrato, Metodo_Pago, etc. |
| **Normalización** | StandardScaler (μ=0, σ=1) | Cargos_Mensuales, Cargos_Totales, Meses_Contrato |
| **División de datos** | Train (80%) / Test (20%) con `stratify` | Mantener proporción de Evasión |

### Etapa 2 — Análisis de Correlación

- Matriz de correlación completa con variable objetivo `Evasion_Binaria`
- Detección de multicolinealidad (threshold > 0.85)
- Identificación de top 15 variables más correlacionadas con evasión
- Visualización con heatmaps y gráficos de barras

### 🔧 Etapa 3 — Entrenamiento de Modelos

Se entrenan **tres modelos** de clasificación:

| Modelo | Tipo | Hiperparámetros clave | Ventaja principal |
|--------|------|----------------------|-------------------|
| **Regresión Logística** | Lineal | `max_iter=1000` | Interpretabilidad, baseline rápido |
| **Random Forest** | Ensemble (Bagging) | `n_estimators=100`, `max_depth=10` | Robusto, maneja no-linealidad |
| **XGBoost** | Ensemble (Boosting) | `n_estimators=100`, `learning_rate=0.1` | Mejor performance, optimizado |

### 📊 Etapa 4 — Evaluación de Modelos

**Métricas calculadas para cada modelo:**

```
✅ Accuracy   → % de predicciones correctas
✅ Precision  → De los predichos como evasión, ¿cuántos realmente lo son?
✅ Recall     → De todos los que evaden, ¿cuántos detectamos?
✅ F1-Score   → Balance armónico entre Precision y Recall
✅ ROC-AUC    → Capacidad de discriminación entre clases
```

**Herramientas de evaluación:**
- Matriz de confusión (True/False Positives/Negatives)
- Curvas ROC con AUC
- Comparación visual de métricas entre modelos

### Etapa 5 — Interpretabilidad

**Feature Importance (Random Forest & XGBoost)**
- Ranking de las 15 variables más importantes
- Visualización comparativa entre modelos

**SHAP Values (SHapley Additive exPlanations)**
- Contribución individual de cada variable
- Dirección del impacto (aumenta/disminuye probabilidad de evasión)
- Summary plots y dependence plots

### 💻Etapa 6 — Recomendaciones Estratégicas

Cinco estrategias concretas basadas en los hallazgos del modelo:
1. Sistema de alerta temprana (scoring mensual)
2. Programa de retención focalizado por factor de riesgo
3. Mejora continua de satisfacción (NPS, CSAT, CES)
4. Optimización de calidad de servicio técnico
5. Personalización masiva con ML

---

## 📊 Resultados Principales

### Rendimiento de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Regresión Logística** | **0.7935** | 0.6352 | **0.5214** | **0.5727** | **0.8431** |
| Random Forest | 0.7921 | **0.6431** | 0.4866 | 0.5540 | 0.8399 |
| XGBoost | 0.7842 | 0.6182 | 0.4893 | 0.5463 | 0.8395 |

**🏆 Mejores modelos por métrica:**
- **Accuracy:** Regresión Logística (79.35%)
- **Precision:** Random Forest (64.31%)
- **Recall:** Regresión Logística (52.14%)
- **F1-Score:** Regresión Logística (57.27%)
- **ROC-AUC:** Regresión Logística (84.31%)

> 🏆 **Modelo recomendado:** **Regresión Logística** por liderar en 4 de 5 métricas clave, incluyendo el mejor ROC-AUC (0.8431)

### Análisis de Resultados

**Hallazgo clave:** Aunque XGBoost suele ser superior en competiciones, en este caso la **Regresión Logística** demostró ser el mejor modelo por:

1. **Mayor ROC-AUC (0.8431):** Mejor capacidad de discriminación entre clientes que evaden vs no evaden
2. **Mejor Recall (0.5214):** Detecta el 52.14% de los clientes que realmente evaden (minimiza falsos negativos)
3. **Mejor F1-Score (0.5727):** Mejor balance entre precisión y exhaustividad
4. **Mayor Accuracy (0.7935):** 79.35% de predicciones correctas

**¿Por qué Regresión Logística superó a modelos complejos?**
- Dataset con relaciones predominantemente lineales
- Menor riesgo de overfitting
- Más interpretable para el negocio
- Entrenamiento y predicción más rápidos

**Ventaja de Random Forest:**
- Mejor Precision (0.6431): De los que predice como evasión, 64.31% realmente lo son
- Útil si el costo de falsos positivos es alto (invertir en retención innecesariamente)

1. **Meses_Contrato** — Clientes nuevos (< 12 meses) tienen 3x más riesgo
2. **Tipo_Contrato_Mensual** — Contratos mes a mes vs anuales/bianuales
3. **Cargos_Totales** — Acumulación de cargos históricos
4. **Cargos_Mensuales** — Facturación mensual promedio
5. **Metodo_Pago_Cheque** — Cheque electrónico vs débito/tarjeta
6. **Tipo_Internet_Fibra** — Fibra óptica vs DSL
7. **Factura_Digital** — Clientes con factura electrónica
8. **Multiples_Lineas** — Cantidad de líneas telefónicas
9. **Servicio_Telefono** — Contratación de telefonía
10. **Cuentas_Diarias** — Costo diario calculado (feature engineered)

### Perfil de Alto Riesgo Validado por el Modelo

```

┌─────────────────────────────────────────────────────┐
│  🔴 CLIENTE DE ALTO RIESGO (Prob. Evasión > 70%)   │
│                                                     │
│  • Antigüedad: < 6 meses                            │
│  • Contrato: Mensual                                │
│  • Internet: Fibra óptica                           │
│  • Pago: Cheque electrónico                         │
│  • Cargos mensuales: > $70 USD                      │
│  • Servicios adicionales: < 2                       │
└─────────────────────────────────────────────────────┘

```


---

## 🚀 Impacto Esperado

### Capacidad del Modelo

Con un **ROC-AUC de 0.8431**, el modelo de Regresión Logística tiene:
- **Excelente capacidad discriminativa** (>0.80 se considera muy bueno)
- **Recall del 52.14%:** Detecta a la mitad de los clientes que realmente evaden
- **Precision del 63.52%:** 2 de cada 3 alertas son correctas

### Reducción de Evasión Proyectada

Asumiendo intervención efectiva en el 60% de los clientes detectados:

```
Escenario actual (sin modelo):    26.5% tasa de evasión
Meta 6 meses (con modelo):        ~23.8% (-10% relativo)  
Meta 12 meses (optimización):     ~21.2% (-20% relativo)
```

### ROI del Proyecto

**Supuestos conservadores:**
- Clientes totales: 7,267
- Clientes que evaden mensualmente: ~1,927 (26.5%)
- Modelo detecta correctamente: ~1,005 (52.14% recall)
- Tasa de retención exitosa: 40%
- CLV promedio: $1,500 USD

**Impacto mensual:**  
1,005 detectados × 40% retenidos × $1,500 CLV = **$603,000 USD**

**Impacto anual:** **$7.24M USD** en ingresos retenidos

---

## 🛠️ Tecnologías Utilizadas

### Librerías Core

| Librería | Versión | Uso |
|----------|---------|-----|
| `pandas` | ≥ 1.5 | Manipulación de datos |
| `numpy` | ≥ 1.23 | Operaciones numéricas |
| `scikit-learn` | ≥ 1.2 | Preprocesamiento, modelos, métricas |
| `xgboost` | ≥ 1.7 | Gradient Boosting optimizado |

### Visualización

| Librería | Uso |
|----------|-----|
| `matplotlib` | Gráficos base, curvas ROC |
| `seaborn` | Heatmaps, matrices de confusión |

### Interpretabilidad

| Librería | Uso |
|----------|-----|
| `shap` | SHAP values, summary plots |

---

## ▶️ Cómo Ejecutar

### Requisitos Previos

1. **Dataset limpio:** Debes haber ejecutado primero [TelecomX_LATAM.ipynb](../TelecomX_LATAM.ipynb) (Parte 1)
2. **Archivo requerido:** `telecomx_limpio.csv` en el mismo directorio

### Opción 1 — Google Colab *(recomendado)*

```python
# 1. Subir a Colab:
#    - TelecomX_Parte2_DPuebla.ipynb
#    - telecomx_limpio.csv

# 2. Modificar ruta de lectura en Celda 5:
df = pd.read_csv('/content/telecomx_limpio.csv')

# 3. Ejecutar todas las celdas:
#    Runtime → Run all (Ctrl + F9)
```

### Opción 2 — Local con Jupyter

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/telecomx-ml-prediction.git
cd telecomx-ml-prediction

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap jupyter

# 4. Verificar que tengas telecomx_limpio.csv
ls telecomx_limpio.csv

# 5. Abrir el notebook
jupyter notebook TelecomX_Parte2_DPuebla.ipynb
```

> ⚠️ **Importante:** El notebook asume que `telecomx_limpio.csv` está en el mismo directorio. Ajusta la ruta en la Celda 5 si es necesario.

---

## 🔗 Dependencias con Parte 1

Este notebook es la **continuación directa** de:

📂 [TelecomX_LATAM.ipynb](../TelecomX_LATAM.ipynb) — Parte 1: ETL y Análisis Exploratorio

**Conexión:**
- Utiliza el dataset `telecomx_limpio.csv` generado en Parte 1
- Las columnas están en español según el diccionario de renombrado de Parte 1
- La variable objetivo es `Evasion_Binaria` (0 = No evasión, 1 = Sí evasión)

**Estructura de columnas esperada (23 columnas):**
```
ID_Cliente, Evasion, Genero, Adulto_Mayor, Tiene_Pareja, 
Tiene_Dependientes, Meses_Contrato, Servicio_Telefono, 
Multiples_Lineas, Tipo_Internet, Seguridad_Online, 
Backup_Online, Proteccion_Dispositivo, Soporte_Tecnico, 
Streaming_TV, Streaming_Peliculas, Tipo_Contrato, 
Factura_Digital, Metodo_Pago, Cargos_Mensuales, 
Cargos_Totales, Cuentas_Diarias, Evasion_Binaria
```

---

## ⚠️ Solución de Problemas

| Problema | Causa probable | Solución |
|----------|---------------|----------|
| `FileNotFoundError: telecomx_limpio.csv` | Archivo no encontrado | Ejecutar primero la Parte 1 o ajustar ruta en Celda 5 |
| `KeyError: 'Churn'` | Columna renombrada a español | Usar `Evasion_Binaria` en lugar de `Churn` |
| `ModuleNotFoundError: xgboost` | Librería no instalada | `pip install xgboost` |
| `ModuleNotFoundError: shap` | Librería no instalada | `pip install shap` |
| SHAP tarda mucho | Dataset muy grande | Reducir `sample_size` en Celda 43 (ej: 300) |
| Gráficos no se ven | Celdas ejecutadas fuera de orden | `Runtime → Restart and run all` |
| `ValueError: could not convert string to float` | Codificación no aplicada | Verificar que Celda 16 se ejecutó correctamente |

### Verificación de Datos

Ejecuta esto antes de empezar para verificar que tu dataset es correcto:

```python
# Verificación rápida
df = pd.read_csv('telecomx_limpio.csv')

print("✅ Verificaciones:")
print(f"  - Filas: {df.shape[0]} (esperado: ~7,267)")
print(f"  - Columnas: {df.shape[1]} (esperado: 23)")
print(f"  - Tiene 'Evasion_Binaria': {'Evasion_Binaria' in df.columns}")
print(f"  - Valores Evasion_Binaria: {df['Evasion_Binaria'].unique()}")

assert 'Evasion_Binaria' in df.columns, "❌ Falta columna Evasion_Binaria"
assert set(df['Evasion_Binaria'].unique()) == {0.0, 1.0}, "❌ Valores incorrectos"
print("\n🎉 Dataset correcto, listo para ML!")
```

---

## 📈 Próximos Pasos

Este proyecto establece las bases para:

1. **Productización del modelo**
   - Despliegue en API REST (FastAPI/Flask)
   - Integración con CRM de TelecomX
   - Sistema de scoring batch mensual

2. **Optimización avanzada**
   - Hyperparameter tuning con GridSearchCV/RandomizedSearchCV
   - Técnicas de balanceo (SMOTE, undersampling)
   - Ensemble de modelos (stacking)

3. **Dashboard ejecutivo**
   - Visualización en tiempo real (Streamlit/Dash)
   - KPIs de evasión por segmento
   - Alertas automáticas para clientes de alto riesgo

4. **A/B Testing**
   - Piloto con 1,000 clientes de alto riesgo
   - Comparación grupo control vs grupo con intervención
   - Medición de impacto real en reducción de evasión

---

## 📚 Recursos Adicionales

### Documentación

- [Scikit-learn: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP: Interpretable ML](https://shap.readthedocs.io/)

### Papers de Referencia

- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP)
- Chen & Guestrin (2016): "XGBoost: A Scalable Tree Boosting System"

---

## 📄 Licencia

Proyecto desarrollado con fines educativos como parte del programa
**Oracle Next Education (ONE) + Alura Latam — Challenge 2 · Data Science**.

Los datos son ficticios y proporcionados por Alura Latam exclusivamente para entrenamiento.

---

## 🙏 Agradecimientos

- **Alura Latam** por el dataset y el desafío técnico
- **Oracle Next Education (ONE)** por el programa de formación
- **Comunidad de Data Science** por las mejores prácticas en ML

---

<div align="center">
  <sub>Desarrollado con 💜 por <strong>Daniela Andrea Puebla Mosca</strong></sub><br>
  <sub>Python · Scikit-learn · XGBoost · SHAP · Machine Learning</sub><br><br>
  
  [![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/daniela-pueblam31)
  [![GitHub](https://img.shields.io/badge/GitHub-Seguir-181717?style=flat&logo=github)](https://github.com/Danny343)
  
  <br>
  <sub>⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub</sub>
</div>
