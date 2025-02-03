# shadow_model_inference_5g

# **Ataque de Modelos Sombra en Aprendizaje Federado - Experimento**

## **📌 Resumen**
Este repositorio contiene un experimento diseñado para probar **ataques de inferencia utilizando modelos sombra** en un sistema de **Aprendizaje Federado (FL)**. El objetivo principal es analizar qué tan bien un modelo sombra puede inferir si una propiedad específica ("Slice1") está presente en una ronda federada, basándose en las actualizaciones de pesos intercambiadas entre los clientes y el modelo global.

Además, el experimento evalúa diferentes medidas de seguridad (cambio de etiquetas y ruido) para mitigar estos ataques y medir su impacto en la efectividad de la inferencia.

## **🛠️ Funcionamiento**
1. **Configuración de Aprendizaje Federado:**
   - El modelo de FL se entrena colaborativamente entre **10 clientes**, cada uno con su propio conjunto de datos privado.
   - Los clientes se dividen en dos grupos:
     - **5 clientes** poseen datos con la propiedad "Slice1".
     - **5 clientes** tienen datos sin esta propiedad.
   - El modelo global agrega actualizaciones de los clientes seleccionados en cada ronda.

2. **Ataque con Modelos Sombra:**
   - Los modelos sombra intentan **inferir la probabilidad** de que "Slice1" esté presente en una ronda dada.
   - Existen dos modos de funcionamiento:
     - **ENTRENO_ANTES = True**: Los modelos sombra se entrenan previamente utilizando una fracción del conjunto de datos (`SHADOW_DATA_FRACTION`).
     - **ENTRENO_ANTES = False**: Los modelos sombra infieren dinámicamente en base a las actualizaciones de pesos globales.

3. **Adaptación Dinámica del Umbral:**
   - Inicialmente, se establece un umbral (`PROPERTY_THRESHOLD`) para determinar si "Slice1" está presente.
   - A medida que avanza el experimento, el umbral se ajusta dinámicamente (`dynamic_threshold`) basado en variaciones observadas en la probabilidad estimada.

4. **Métricas de Evaluación:**
   - **ROC AUC:** Evalúa qué tan bien el ataque distingue rondas con/sin "Slice1".
   - **Precisión y Recall:** Miden la exactitud de las predicciones del ataque.
   - **Precisión Personalizada:** Determina si las **transiciones de probabilidad** coinciden con el comportamiento esperado basado en la presencia de "Slice1".

## **⚙️ Parámetros y Configuración**

| Parámetro | Descripción | Valores Posibles |
|-----------|-------------|----------------|
| `NUM_ROUNDS` | Número de rondas de aprendizaje federado | Predeterminado: `100` |
| `NUM_CLIENTS` | Número de clientes en FL | Predeterminado: `10` |
| `FRACTION` | Fracción de datos utilizados para el entrenamiento | Predeterminado: `0.2` |
| `ENTRENO_ANTES` | Indica si los modelos sombra se entrenan antes de la inferencia | `True` / `False` |
| `SHADOW_DATA_FRACTION` | Fracción de datos utilizada para entrenar los modelos sombra (si `ENTRENO_ANTES = True`) | Predeterminado: `0.01` |
| `PROPERTY_THRESHOLD` | Umbral inicial para detectar "Slice1" | Predeterminado: `0.5` |
| `PROB_RANGE` | Rango para clasificar cambios de probabilidad como significativos | Predeterminado: `0.1` |

## **🛡️ Medidas de Seguridad Implementadas**
Para mitigar los ataques de inferencia con modelos sombra, se incorporan **dos mecanismos de seguridad**:

### **1️⃣ Cambio de Etiquetas (Label Flipping)**
   - Modifica aleatoriamente las etiquetas (por ejemplo, intercambiando "Slice1" ↔ "Slice0") para confundir los modelos sombra.
   - Configuraciones:
     - `LABEL_FLIPPING = True` (Activar/Desactivar)
     - `FLIPPING_ANTES = True` (Antes del entrenamiento) o `False` (Después del entrenamiento)
     - `PROB_FLIP_0` (Probabilidad de cambiar 0 → 1)
     - `PROB_FLIP_1` (Probabilidad de cambiar 1 → 0)

### **2️⃣ Inyección de Ruido Diferencial**
   - Se añade **ruido gaussiano o laplaciano** a las actualizaciones de los pesos para ocultar patrones y dificultar la inferencia.
   - Configuraciones:
     - `APLICAR_RUIDO = True` (Activar/Desactivar)
     - `RUIDO_ANTES = True` (Antes del entrenamiento) o `False` (Después del entrenamiento)
     - `RUIDO_PER` (Porcentaje de datos afectados por ruido)
     - `NOISE_STD` (Desviación estándar del ruido aplicado)
     - `EPSILON` & `DELTA` (Parámetros de privacidad diferencial)

   - **Elección del tipo de ruido:**
     - Si `DELTA` es diferente de `None`, se usa **ruido gaussiano**:
       ```python
       noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
       noise = np.random.normal(0, noise_scale, size=(num_samples, num_features))
       ```
     - Si `DELTA` es `None`, se usa **ruido laplaciano**:
       ```python
       noise_scale = sensitivity / epsilon
       noise = np.random.laplace(0, noise_scale, size=(num_samples, num_features))
       ```

## **📊 Resultados y Análisis**


## **📌 Cómo Ejecutar el Experimento**


## **📎 Mejoras Futuras**

