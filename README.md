# Detección de Fraude en Transacciones con Tarjeta de Crédito

## Dashboard interactivo
🔗 [Ver dashboard en vivo](https://fraude-tarjeta-credito.streamlit.app/)

## Contexto del problema
Las compañías de tarjetas de crédito necesitan identificar transacciones 
fraudulentas para evitar que sus clientes sean cobrados por compras que 
no realizaron. Este proyecto aplica análisis estadístico y aprendizaje 
automático para detectar fraude en un dataset severamente desbalanceado.

## Dataset
- **Fuente:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transacciones:** 284,807 registros de titulares de tarjeta europeos (septiembre 2013)
- **Fraudes:** 492 (0.172% del total)
- **Variables:** Time, Amount, V1-V28 (componentes principales por PCA) y Class (variable objetivo: 0=legítima, 1=fraude)

## Herramientas utilizadas
- **Python:** pandas, numpy, scikit-learn, scipy, seaborn, matplotlib
- **SQL:** SQLite
- **Dashboard:** Streamlit + Plotly
- **Modelo:** Regresión Logística

## Estructura del análisis
1. Limpieza y preparación de datos
2. Análisis exploratorio y estadística
3. Modelo predictivo
4. Análisis SQL
5. Dashboard interactivo

## Resultados del modelo
| Métrica | Valor |
|---|---|
| Recall (fraude) | 0.87 |
| Precisión (fraude) | 0.12 |
| F1 (fraude) | 0.21 |
| AUPRC | 0.6720 |
| Fraudes detectados | 83 de 95 |
| Falsas alarmas | 629 |
| Umbral de decisión | 0.7 |
| Monto interceptado | €51,190 |
| Monto no recuperado | €7,401 |
| Reducción de pérdida | 87.4% |

## Hallazgos principales

1. **Dataset severamente desbalanceado:** solo el 0.167% de las 
   transacciones son fraude (473 de 283,726 tras limpieza). 
   Accuracy no es una métrica válida en este contexto.

2. **El monto distingue parcialmente entre clases:** el test t confirma 
   que la diferencia entre el monto promedio de fraude (€123.87) y 
   legítima (€88.41) es estadísticamente significativa (p=0.0021). 
   Sin embargo, la diferencia práctica es pequeña — el monto por sí 
   solo no es suficiente para detectar fraude.

3. **Datos atípicos y fraude:** las transacciones con monto atípico 
   (fuera del rango ±3σ) tienen una tasa de fraude de 0.27%, 
   1.6 veces mayor que el promedio del dataset (0.167%).

4. **Variables más discriminantes:** V14, V3, V17 y V12 muestran 
   las mayores diferencias entre clases en el EDA, confirmado 
   por los coeficientes del modelo.

5. **Distribución horaria del fraude:** los fraudes se distribuyen 
   de forma relativamente uniforme durante el día, con un leve pico 
   en la tarde (12h–18h) con 133 casos. Esto sugiere que los patrones 
   de fraude dependen más del comportamiento transaccional que del 
   momento del día.

6. **La tasa de fraude aumenta con el monto:** las transacciones 
   de monto muy alto tienen una tasa de fraude 2.6 veces mayor 
   que las de monto medio (0.398% vs 0.153%).

7. **Impacto del modelo:** con umbral 0.7 el modelo intercepta el 87% 
   de los fraudes reales, protegiendo €51,190 de los €58,591 en riesgo.

## Decisiones técnicas

1. **Algoritmo — Regresión Logística:** se eligió por su base estadística 
   clara e interpretabilidad. Sus coeficientes permiten explicar 
   directamente qué variables influyen en la detección de fraude.

2. **Balanceo — class_weight='balanced':** el desbalance severo 
   (0.167% fraude) hace que un modelo sin balanceo aprenda a predecir 
   siempre "legítima" y detecte 0% de fraudes. Este parámetro le asigna 
   mayor peso a la clase minoritaria durante el entrenamiento sin 
   necesidad de generar datos sintéticos.

3. **Métrica principal — Recall y AUPRC:** en detección de fraude 
   el costo de no detectar un fraude real supera el de generar una 
   falsa alarma. Por eso priorizamos recall sobre precisión.

4. **Umbral de decisión — 0.7:** se evaluaron cuatro umbrales (0.3, 0.5, 
   0.7, 0.9). El umbral 0.7 mantiene el recall en 0.87 y reduce las 
   falsas alarmas de 1,396 a 629 sin sacrificar detección de fraudes.

5. **Estandarización — StandardScaler:** Amount y Time no pasaron por 
   PCA y se encontraban en la escala original. Se estandarizaron para 
   igualarlas al rango de V1-V28. Se eligió StandardScaler sobre 
   MinMaxScaler porque este último es sensible a datos atípicos, 
   frecuentes en datasets de fraude.

## Cómo ejecutar el dashboard localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

> El dashboard descarga los datos automáticamente desde Kaggle 
> al iniciar. Se requiere conexión a internet en el primer uso.

## Trabajo futuro
- Explorar técnicas de balanceo como SMOTE
- Evaluar modelos como Random Forest o XGBoost
- Ajuste de hiperparámetros con validación cruzada
