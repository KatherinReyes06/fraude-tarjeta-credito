# Detección de Fraude en Transacciones con Tarjeta de Crédito

## Contexto del problema
Las compañías de tarjetas de crédito necesitan identificar transacciones 
fraudulentas para evitar que sus clientes sean cobrados por compras que 
no realizaron. Este proyecto aplica análisis estadístico y aprendizaje 
automático para detectar fraude en un dataset severamente desbalanceado.

## Dataset
- **Fuente:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transacciones:** 284,807 registros de  europeos (septiembre 2013)
- **Fraudes:** 492 (0.172% del total)
- **Variables:** Time, Amount, V1-V28 (componentes principales por PCA) y Class (variable objetivo: 0=legítima, 1=fraude)

## Herramientas utilizadas
- **Python:** pandas, numpy, scikit-learn, scipy, seaborn, matplotlib
- **SQL:** SQLite
- **Modelo:** Regresión Logística

## Estructura del análisis
1. Limpieza y preparación de datos
2. Análisis exploratorio y estadística
3. Modelo predictivo
4. Análisis SQL

## Hallazgos principales

1. **Dataset severamente desbalanceado:** solo el 0.167% de las 
   transacciones son fraude (473 de 283,726 tras limpieza). 
   Accuracy no es una métrica válida en este contexto.

2. **El monto distingue parcialmente entre clases:** el test t confirma 
   que la diferencia entre el monto promedio de fraude (0.1414) y 
   legítima (-0.0002) es estadísticamente significativa (p=0.0021). 
   Sin embargo, la diferencia práctica es pequeña — el monto por sí 
   solo no es suficiente para detectar fraude.

3. **Datos atípicos y fraude:** las transacciones con monto atípico 
   (fuera del rango ±3σ) tienen una tasa de fraude de 0.27%, 
   1.6 veces mayor que el promedio del dataset (0.167%).

4. **Variables más discriminantes:** V14, V3, V17 y V12 muestran 
   las mayores diferencias entre clases en el EDA, confirmado 
   por los coeficientes del modelo.

5. **Los fraudes ocurren principalmente en madrugada y mañana:** 
   el 64% de los fraudes ocurren cuando los titulares están 
   menos atentos a sus transacciones.

6. **La tasa de fraude aumenta con el monto:** las transacciones 
   de monto muy alto tienen una tasa de fraude 2.6 veces mayor 
   que las de monto medio.

7. **Modelo:** con umbral 0.7 el modelo detecta el 87% de los fraudes 
   reales (83 de 95) reduciendo las falsas alarmas de 1,396 a 629, 
   con AUPRC de 0.6720.

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
   PCA y se encontraban en la escala original. Se estandarizaron para igualarlas 
   al rango de V1-V28. Se eligió StandardScaler sobre MinMaxScaler 
   porque este último es sensible a datos atípicos, y se encontró una cantidad importante de ellos     en el dataset.

## Trabajo futuro
- Explorar técnicas de balanceo como SMOTE
- Evaluar modelos como Random Forest o XGBoost
- Ajuste de hiperparámetros con validación cruzada
