# Predictive Maintenance Platform

Plataforma web de mantenimiento predictivo para el análisis de motores turbofán mediante técnicas de Machine Learning y Deep Learning.

La aplicación permite monitorizar la evolución temporal de los motores, detectar comportamientos anómalos mediante **Isolation Forest**, estimar su **Remaining Useful Life (RUL)** mediante redes neuronales **LSTM**, incorporar nuevos historiales mediante archivos CSV y actualizar posteriormente los motores externos ya registrados.

El sistema utiliza como conjuntos de referencia los subconjuntos **FD001 y FD002 de NASA C-MAPSS**, que permiten trabajar con dos perfiles predictivos diferentes:

- **Condición única**, asociado a FD001.
- **Multirrégimen**, asociado a FD002.

Cada perfil dispone de su propio preprocesamiento, detector de anomalías y modelo de prognosis.

---

# Funcionalidades principales

La plataforma está organizada en cuatro áreas principales:

## Visión general de la flota

Permite consultar el estado global de todos los motores disponibles en la aplicación.

Incluye:

- Número total de motores monitorizados.
- Número de alertas activas.
- RUL media estimada.
- Motor con menor RUL estimada.
- Tabla con el estado individual de cada motor.
- Identificación del origen de los datos.
- Perfil operativo.
- Estado de alerta actual.
- Existencia de alertas anteriores.
- RUL estimada.
- Búsqueda de motores.
- Filtro por origen.
- Filtro por estado de alerta.
- Ordenación por RUL, último ciclo o identificador.

Cada motor puede seleccionarse directamente desde la tabla para acceder a su monitorización individual.

Los valores numéricos de anomaly score no se utilizan para ordenar globalmente la flota porque FD001 y FD002 emplean modelos y preprocesamientos diferentes y, por tanto, sus puntuaciones no deben compararse directamente.

---

## Monitorización individual

Permite analizar la evolución de un motor a lo largo de sus ciclos de funcionamiento.

La información se divide en tres vistas.

### Resumen

Muestra:

- Motor seleccionado.
- Perfil operativo.
- Régimen operativo actual, cuando corresponde.
- Ciclo consultado.
- Estado de alerta.
- RUL estimada.

### Sensores

Representa gráficamente la evolución temporal de las variables utilizadas por los modelos.

Los 14 sensores seleccionados son:

```text
sensor_2
sensor_3
sensor_4
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

### Análisis de IA

Incluye:

- Evolución del anomaly score.
- Umbral de detección.
- Estado de alerta.
- Alertas confirmadas.
- Evolución de la RUL estimada.
- Comparación con la RUL real en modo evaluación para los motores NASA de referencia.

La plataforma dispone además de un simulador temporal que permite recorrer el historial de un motor ciclo a ciclo de forma manual o automática.

El análisis realizado en cada instante utiliza únicamente los datos disponibles hasta el ciclo seleccionado.

---

# Dataset

## NASA C-MAPSS

El proyecto utiliza el dataset **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** desarrollado por NASA.

C-MAPSS contiene series temporales multivariantes que representan la evolución y degradación de motores turbofán mediante variables operativas y señales procedentes de sensores.

La plataforma utiliza dos subconjuntos:

```text
FD001
FD002
```

En conjunto se utilizan **359 motores de referencia del conjunto de test**:

```text
FD001: 100 motores
FD002: 259 motores
Total: 359 motores
```

Los motores externos incorporados mediante CSV se añaden a esta flota de referencia, por lo que el número total mostrado por la aplicación puede ser superior.

---

## FD001 · Condición única

FD001 representa motores sometidos a:

- Una única condición operativa.
- Un modo de degradación asociado al compresor de alta presión, HPC.
- 100 motores en entrenamiento.
- 100 motores en test.
- 21 variables originales de sensores.

Tras el análisis exploratorio y el preprocesamiento se seleccionaron 14 sensores para los modelos predictivos.

En la plataforma este perfil se muestra como:

```text
Condición única
```

---

## FD002 · Multirrégimen

FD002 representa motores sometidos a:

- Seis condiciones operativas diferentes.
- Un modo de degradación asociado al compresor de alta presión, HPC.
- 260 motores en entrenamiento.
- 259 motores en test.
- 21 variables originales de sensores.

Cada motor puede pasar por distintos regímenes durante su trayectoria.

Debido a que los valores de los sensores pueden variar significativamente por efecto de la condición operativa, los datos se normalizan mediante estadísticas específicas de cada régimen antes de ejecutar los modelos.

En la plataforma este perfil se muestra como:

```text
Multirrégimen
```

---

# Perfiles predictivos

Los motores NASA de referencia disponen de un perfil conocido por su procedencia:

```text
FD001 → Condición única
FD002 → Multirrégimen
```

En los motores externos el perfil se detecta automáticamente a partir de las tres variables de condición operativa:

```text
op_setting_1
op_setting_2
op_setting_3
```

Cada observación se compara con los seis regímenes conocidos durante el desarrollo de FD002.

La lógica utilizada es:

```text
Solo régimen 1
→ FD001
→ Condición única

Aparece algún régimen 2-6
→ FD002
→ Multirrégimen

Condición operativa desconocida
→ Perfil predictivo no disponible
→ Importación rechazada
```

No se aproxima una condición desconocida al régimen más cercano.

Para motores externos, la clasificación como FD001 es provisional mientras únicamente se observe el régimen 1.

Si posteriormente aparecen otros regímenes, el motor evoluciona automáticamente:

```text
FD001 → FD002
```

Esta transición es irreversible. Una vez identificado como multirrégimen, el motor permanece asociado al pipeline FD002.

---

# Preprocesamiento

Los dos perfiles utilizan los mismos 14 sensores raw como variables predictivas, pero aplican transformaciones diferentes.

## FD001

### Detección de anomalías

Los 14 sensores se transforman mediante el `StandardScaler` aprendido durante el entrenamiento antes de introducirlos en Isolation Forest.

### Prognosis

Las ventanas temporales de 30 ciclos se transforman mediante el `StandardScaler` utilizado durante el entrenamiento del modelo LSTM FD001.

---

## FD002

FD002 necesita tener en cuenta el régimen operativo.

Antes de ejecutar los modelos:

1. Se determina el régimen de cada observación a partir de las tres variables operativas.
2. Cada sensor se normaliza mediante una transformación z-score utilizando la media y desviación estándar aprendidas para ese régimen.
3. Las variables normalizadas se introducen en los modelos FD002.

Este procedimiento permite reducir el efecto de los cambios operativos y separar mejor la variación debida al régimen de la asociada a la degradación.

---

# Detección de anomalías

La detección de comportamientos anómalos se realiza mediante:

```text
Isolation Forest
```

La plataforma dispone de un detector independiente para cada perfil predictivo.

El anomaly score se calcula a partir de:

```python
-score_samples(X)
```

Por tanto:

```text
Mayor anomaly score
→ comportamiento más anómalo
```

El anomaly score:

- No representa una probabilidad de fallo.
- No representa un porcentaje de degradación.
- No constituye por sí mismo un diagnóstico de avería.

Además, los valores de anomaly score de FD001 y FD002 **no deben compararse directamente entre sí**, ya que proceden de modelos y transformaciones diferentes.

---

## Umbral FD001

El umbral utilizado por el detector correspondiente a FD001 es aproximadamente:

```text
0.5199
```

Se obtuvo utilizando el percentil 97,5 de las puntuaciones de referencia.

---

## Umbral FD002

El umbral utilizado por el detector correspondiente a FD002 es aproximadamente:

```text
0.5187
```

También se obtuvo utilizando el percentil 97,5 de las puntuaciones de referencia.

---

## Confirmación de alertas

Una observación aislada por encima del umbral no genera una alerta confirmada.

La plataforma exige:

```text
3 ciclos consecutivos por encima del umbral
```

para activar una alerta.

Este criterio permite reducir la sensibilidad frente a observaciones anómalas puntuales.

La interfaz diferencia entre:

```text
Alerta activa
```

y:

```text
Alertas anteriores
```

Una alerta anterior indica que el motor ha presentado al menos un episodio confirmado en algún punto de su historial.

---

# Estimación de la vida útil restante

La prognosis se realiza mediante redes neuronales:

```text
Long Short-Term Memory (LSTM)
```

Existe un modelo independiente para cada perfil predictivo.

La **Remaining Useful Life (RUL)** representa el número estimado de ciclos de funcionamiento restantes antes del final de la vida útil del motor.

---

## Entrada de las redes

Los dos modelos utilizan:

```text
30 ciclos temporales
×
14 sensores
```

Por tanto, la primera estimación puede obtenerse a partir del ciclo:

```text
30
```

Los motores con menos de 30 observaciones pueden disponer de detección de anomalías, pero no de una estimación de RUL.

En el conjunto oficial FD002 existen seis motores con menos de 30 ciclos observados. Estos motores se mantienen dentro de la flota, aunque su RUL se muestra como:

```text
No disponible
```

La cobertura de prognosis sobre FD002 es por ello:

```text
253 / 259 motores
97.68 %
```

---

# Objetivo de entrenamiento de RUL

Durante el entrenamiento de los modelos se utilizó una RUL objetivo limitada:

```text
RUL_max = 120 ciclos
```

La definición empleada es:

```text
RUL_objetivo = min(RUL_real, 120)
```

Este tratamiento reduce la importancia de distinguir valores muy elevados durante las primeras fases de funcionamiento y concentra el aprendizaje en la región de degradación más relevante.

Las predicciones generadas por las redes **no se recortan artificialmente después de la inferencia**, por lo que pueden superar ligeramente los 120 ciclos.

---

# Rendimiento de los modelos de prognosis

## FD001 · Condición única

Resultados sobre los 100 motores del conjunto de test FD001:

| Métrica | Resultado |
|---|---:|
| MAE | 8.71 ciclos |
| RMSE | 12.18 ciclos |
| R² | 0.9041 |
| Cobertura | 100 % |

---

## FD002 · Multirrégimen

Resultados sobre los motores evaluables del conjunto oficial de test FD002:

| Métrica | Resultado |
|---|---:|
| MAE | 10.44 ciclos |
| RMSE | 14.41 ciclos |
| R² | 0.8791 |
| Cobertura | 97.68 % |

La evaluación se realiza sobre 253 de los 259 motores porque seis trayectorias contienen menos de los 30 ciclos necesarios para construir una ventana completa de entrada a la LSTM.

---

# Modo evaluación

Los motores NASA de referencia disponen de valores reales de RUL proporcionados en:

```text
RUL_FD001.txt
RUL_FD002.txt
```

Estos datos permiten comparar las estimaciones de la LSTM con la referencia oficial.

La aplicación muestra:

- RUL real.
- RUL real limitada.
- RUL estimada.
- Error absoluto.

Para ciclos anteriores al último ciclo registrado, la RUL real se reconstruye mediante:

```text
RUL_real(ciclo)
=
RUL_final
+
(último_ciclo - ciclo)
```

Para comparar la predicción con el objetivo utilizado durante el entrenamiento se calcula:

```text
RUL_real_limitada
=
min(RUL_real, 120)
```

El error absoluto se obtiene mediante:

```text
Error_absoluto
=
|RUL_estimada - RUL_real_limitada|
```

Los valores reales se utilizan exclusivamente para evaluación y **nunca forman parte de las variables de entrada de los modelos**.

---

## Evaluación de motores externos

Los motores externos incorporados mediante CSV no disponen de una RUL real conocida.

Por tanto:

```text
Motores NASA
→ Inferencia disponible
→ Modo evaluación disponible

Motores externos
→ Inferencia disponible
→ Modo evaluación no disponible
```

Los motores externos pueden utilizar normalmente:

- Detección de anomalías.
- Alertas.
- Visualización de sensores.
- Estimación de RUL.
- Evolución temporal.
- Actualización del historial.

---

# Importación de nuevos motores

La plataforma permite incorporar nuevos historiales mediante archivos CSV.

Antes de guardar una importación, el archivo se valida y se determina automáticamente qué pipeline predictivo corresponde a cada motor.

El usuario debe proporcionar:

- Un nombre para identificar el origen.
- Un archivo CSV compatible.

El origen puede utilizarse, por ejemplo, para identificar:

```text
Banco de ensayo A
Campaña experimental B
Conjunto de validación
```

---

# Formato del CSV

Cada fila representa un ciclo de funcionamiento.

El esquema mínimo contiene **19 columnas obligatorias**:

```text
unit
cycle

op_setting_1
op_setting_2
op_setting_3

sensor_2
sensor_3
sensor_4
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

También pueden incluirse otros sensores originales de C-MAPSS:

```text
sensor_1
...
sensor_21
```

siempre que no se añadan columnas desconocidas por la plataforma.

Las variables internas utilizadas por la aplicación no deben incluirse en el CSV de entrada.

---

# Validación del CSV

Antes de importar los datos se comprueba que:

- El archivo no esté vacío.
- Contenga todas las columnas obligatorias.
- No incluya columnas desconocidas.
- Las variables sean numéricas.
- No existan valores nulos.
- Los identificadores de motor sean enteros positivos.
- Los ciclos sean enteros positivos.
- El historial comience en el ciclo 1.
- No existan ciclos duplicados.
- Los ciclos sean consecutivos.
- Cada motor disponga de al menos 30 observaciones.
- Las condiciones operativas sean compatibles con alguno de los perfiles disponibles.
- Pueda determinarse automáticamente el perfil predictivo.

La validación muestra:

- Motores detectados.
- Número de observaciones.
- Sensores disponibles.
- Número de columnas.
- Perfiles detectados.

La importación solo puede confirmarse después de superar correctamente la validación.

---

# Identificación de motores externos

Los identificadores originales incluidos en un CSV pueden coincidir con motores existentes en la plataforma.

Para evitar colisiones, el backend asigna automáticamente un **identificador interno único**.

Los motores NASA utilizan los identificadores internos:

```text
FD001 → 1-100
FD002 → 101-359
```

Los nuevos motores externos reciben identificadores a partir del siguiente ID disponible.

Por ejemplo:

```text
ID original del CSV    ID interno
-------------------    ----------
9001                   360
```

El identificador original se conserva mediante:

```text
source_unit
```

Cada observación persistida dispone de metadatos internos como:

```text
unit
source_unit
origin
is_reference
profile
```

donde:

```text
unit
→ identificador interno de la plataforma

source_unit
→ identificador original del CSV

origin
→ procedencia indicada por el usuario

is_reference
→ indica si pertenece a los conjuntos NASA de referencia

profile
→ pipeline predictivo asignado
```

---

# Actualización de motores externos

Los motores externos pueden recibir nuevas observaciones después de haber sido incorporados.

Para actualizar un motor debe proporcionarse nuevamente su **historial completo desde el ciclo 1**, incluyendo las nuevas observaciones.

La actualización comprueba que:

- El motor sea externo.
- El CSV contenga un único motor.
- El identificador original coincida con el motor almacenado.
- El historial incluya todos los ciclos existentes.
- Los ciclos anteriores no hayan sido modificados.
- Exista al menos un ciclo nuevo.
- Las nuevas condiciones operativas sean compatibles.

El motor conserva:

```text
ID interno
ID original
Origen
Historial previo
```

Si un motor previamente clasificado como FD001 comienza a presentar condiciones correspondientes a otros regímenes, se produce la transición:

```text
Condición única
→ Multirrégimen
```

Al producirse el cambio, la trayectoria completa se reprocesa utilizando el pipeline FD002.

Una vez realizada esta transición, el motor no vuelve automáticamente a FD001.

---

# Persistencia de las importaciones

Los motores importados se almacenan en:

```text
data/imported/
```

Cada importación genera un archivo:

```text
import_<id>.csv
```

El archivo:

```text
registry.json
```

mantiene la información necesaria para reconstruir las importaciones al reiniciar la aplicación.

Entre otros datos, conserva:

```text
Identificador de importación
Origen
Archivo persistido
Correspondencia de IDs
Perfiles predictivos
Número de observaciones
```

Por tanto, detener FastAPI no elimina los motores externos ya registrados.

---

# Datos de demostración

La carpeta:

```text
data/demo/
```

puede contener historiales preparados para demostrar las funcionalidades de importación y actualización:

```text
motor_demo_historial_40.csv
motor_demo_actualizacion_45.csv
```

El primer archivo representa el historial completo del motor de demostración hasta el ciclo 40.

El segundo contiene el mismo historial ampliado hasta el ciclo 45 y permite demostrar la actualización del motor desde la propia interfaz.

Estos archivos se utilizan exclusivamente para demostrar el funcionamiento de la aplicación y no forman parte de los conjuntos NASA utilizados como referencia o entrenamiento.

---

# Arquitectura de la aplicación

La plataforma sigue una arquitectura basada en un backend FastAPI y un frontend web.

```text
        NASA C-MAPSS
        FD001 + FD002
              │
              │
        CSV externos
              │
              ▼
       Validación de datos
              │
              ▼
     Detección de perfil
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
     FD001          FD002
       │             │
       ▼             ▼
 Preprocesamiento específico
       │             │
       ├──────┬──────┤
       │      │      │
       ▼      │      ▼
 Isolation    │     LSTM
  Forest      │
       │      │
       └──┬───┘
          ▼
        FastAPI
          │
          ▼
    HTML + CSS + JS
          │
          ▼
       Dashboard
```

---

# Estructura del proyecto

```text
Dashboard/
│
├── README.md
├── requirements.txt
│
├── backend/
│   └── main.py
│
├── data/
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── RUL_FD001.txt
│   ├── RUL_FD002.txt
│   │
│   ├── imported/
│   │   ├── registry.json
│   │   └── import_<id>.csv
│   │
│   └── demo/
│       ├── motor_demo_historial_40.csv
│       └── motor_demo_actualizacion_45.csv
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── models/
    │
    ├── anomaly_detection/
    │   ├── anomaly_detection_FD001.pkl
    │   └── anomaly_detection_FD002.pkl
    │
    ├── preprocessing/
    │   └── preprocessing_FD002.pkl
    │
    └── prognosis/
        ├── config_RUL_FD001.json
        ├── config_RUL_FD002.json
        ├── lstm_RUL_FD001.keras
        ├── lstm_RUL_FD002.keras
        └── scaler_RUL_FD001.pkl
```

---

# Backend

El backend está desarrollado con:

```text
FastAPI
```

Sus principales responsabilidades son:

- Cargar los conjuntos NASA FD001 y FD002.
- Recuperar las importaciones persistidas.
- Validar archivos CSV externos.
- Detectar automáticamente el perfil predictivo.
- Asignar identificadores internos.
- Actualizar motores externos.
- Aplicar el preprocesamiento específico de cada perfil.
- Ejecutar Isolation Forest.
- Ejecutar las redes LSTM.
- Calcular la RUL.
- Gestionar las alertas.
- Reconstruir la RUL real para evaluación.
- Calcular el resumen global de la flota.
- Servir los datos mediante una API REST.
- Servir la aplicación web.

---

# Frontend

El frontend está desarrollado mediante:

```text
HTML
CSS
JavaScript
Chart.js
```

No se utiliza ningún framework frontend adicional.

La aplicación consume directamente los endpoints proporcionados por FastAPI.

Chart.js se utiliza para representar:

- Evolución de sensores.
- Evolución del anomaly score.
- Umbral de detección.
- Ciclos con alerta activa.
- Evolución de la RUL estimada.
- RUL real limitada en modo evaluación.

> **Nota:** Chart.js se carga actualmente desde CDN. Por este motivo, es necesaria conexión a Internet para cargar la biblioteca de gráficos. El backend y el resto de la interfaz se ejecutan localmente.

---

# API

## Estado de la aplicación

```http
GET /
```

Devuelve información básica sobre el estado del backend.

---

## Motores disponibles

```http
GET /api/motores
```

Devuelve los motores registrados y sus principales metadatos.

---

## Información de un motor

```http
GET /api/motores/{motor_id}
```

Devuelve información general sobre un motor concreto.

---

## Historial de sensores

```http
GET /api/motores/{motor_id}/historial
```

También permite seleccionar un ciclo:

```http
GET /api/motores/{motor_id}/historial?ciclo=100
```

---

## Detección de anomalías

```http
GET /api/motores/{motor_id}/anomalias
```

Ejemplo:

```http
GET /api/motores/25/anomalias?ciclo=48
```

---

## Estimación de RUL

```http
GET /api/motores/{motor_id}/rul
```

---

## Historial de RUL

```http
GET /api/motores/{motor_id}/rul/historial
```

---

## Estado general del motor

```http
GET /api/motores/{motor_id}/estado
```

Combina información de detección de anomalías y prognosis.

---

## Evaluación de RUL

```http
GET /api/motores/{motor_id}/evaluacion
```

Solo está disponible para motores NASA de referencia.

---

## Resumen de la flota

```http
GET /api/flota/resumen
```

Calcula el estado del último ciclo disponible de todos los motores utilizando el pipeline correspondiente a cada perfil.

---

## Validar un CSV

```http
POST /api/importaciones/validar
```

Recibe:

```text
origen
archivo
```

Comprueba la compatibilidad del archivo y detecta los perfiles sin modificar los datos almacenados.

---

## Importar motores

```http
POST /api/importaciones/importar
```

Valida nuevamente el archivo, asigna identificadores internos, incorpora los metadatos necesarios y almacena la importación de forma persistente.

---

## Actualizar un motor externo

```http
POST /api/motores/{motor_id}/actualizar
```

Recibe un CSV con la trayectoria completa actualizada del motor.

No permite modificar los motores NASA de referencia.

---

# Requisitos

El proyecto utiliza:

```text
Python 3.11
```

Las dependencias Python se encuentran en `requirements.txt`:

```txt
fastapi==0.142.2
uvicorn[standard]==0.54.0

pandas==3.0.6
numpy==1.26.4

scikit-learn==1.6.1
joblib==1.6.0

tensorflow==2.16.2

python-multipart==0.0.32
```

---

# Instalación

## 1. Crear un entorno virtual

### macOS / Linux

```bash
python3 -m venv venv_tfm
source venv_tfm/bin/activate
```

### Windows

```bash
python -m venv venv_tfm
venv_tfm\Scripts\activate
```

---

## 2. Instalar las dependencias

Desde el directorio raíz del proyecto:

```bash
pip install -r requirements.txt
```

---

# Ejecución

Desde el directorio raíz:

```bash
python -m uvicorn backend.main:app --reload
```

El servidor estará disponible en:

```text
http://127.0.0.1:8000
```

---

# Acceso a la aplicación

## Dashboard

```text
http://127.0.0.1:8000/dashboard
```

## Documentación interactiva de FastAPI

```text
http://127.0.0.1:8000/docs
```

## Estado de la API

```text
http://127.0.0.1:8000/
```

---

# Uso de la plataforma

## Analizar un motor existente

1. Abrir **Visión general**.
2. Buscar o filtrar el motor deseado.
3. Pulsar sobre su identificador.
4. Acceder a **Monitorización**.
5. Seleccionar el ciclo que se desea analizar.
6. Consultar las vistas de Resumen, Sensores o Análisis de IA.
7. Activar el modo evaluación si se trata de un motor NASA de referencia.

---

## Importar un motor

1. Abrir **Importar motores**.
2. Indicar el origen del conjunto.
3. Seleccionar el archivo CSV.
4. Pulsar **Validar CSV**.
5. Revisar los motores y perfiles detectados.
6. Pulsar **Importar motores**.
7. Volver a **Visión general**.
8. Utilizar el filtro de origen para localizar los nuevos motores.
9. Seleccionar el motor para analizarlo.

---

## Actualizar un motor externo

1. Seleccionar el motor externo desde la flota.
2. Abrir la pestaña **Resumen**.
3. Localizar el apartado **Actualizar historial del motor**.
4. Seleccionar un CSV con la trayectoria completa actualizada.
5. Pulsar **Actualizar motor**.
6. La plataforma comprobará el historial anterior y añadirá los nuevos ciclos.
7. Si aparecen nuevas condiciones operativas, se actualizará automáticamente el perfil predictivo cuando corresponda.

---

# Interpretación de resultados

## Anomaly score

Indica cómo de atípico considera Isolation Forest el comportamiento observado dentro del perfil predictivo correspondiente.

```text
Mayor puntuación
→ comportamiento más anómalo
```

No debe interpretarse como:

- Probabilidad de fallo.
- Porcentaje de degradación.
- Diagnóstico automático de una avería.

Los valores de los dos perfiles deben interpretarse respecto a sus propios umbrales.

---

## Estado de alerta

Una alerta se confirma cuando el anomaly score supera el umbral correspondiente durante tres ciclos consecutivos.

Una alerta indica un comportamiento anómalo persistente, pero no determina automáticamente la causa concreta del mismo.

---

## RUL estimada

Representa el número de ciclos de funcionamiento restantes estimados por la LSTM correspondiente al perfil del motor.

Una RUL reducida puede utilizarse para priorizar la revisión de determinados motores.

Sin embargo, constituye una predicción y está sujeta a error.

---

## Régimen operativo

En los motores multirrégimen indica la condición operativa asociada al ciclo consultado.

El régimen se utiliza además para seleccionar las estadísticas de normalización que se aplican antes de ejecutar los modelos FD002.

---

# Alcance y limitaciones

La plataforma ha sido desarrollada y evaluada utilizando los subconjuntos **FD001 y FD002 de NASA C-MAPSS**.

Los modelos han aprendido los patrones presentes en estos datos y utilizan las mismas variables y transformaciones empleadas durante su entrenamiento.

Por tanto, sus resultados no pueden generalizarse automáticamente a cualquier tipo de motor o sistema industrial.

Los motores externos deben presentar una estructura y un significado de las variables compatibles con los utilizados durante el desarrollo.

La posibilidad de importar y actualizar trayectorias demuestra la capacidad de la arquitectura para incorporar nuevos motores compatibles, pero no implica que los modelos puedan aplicarse sin validación a cualquier conjunto de datos.

Además:

- El sistema trabaja actualmente con datos históricos.
- No existe adquisición directa de sensores en tiempo real.
- Solo se contemplan los perfiles predictivos aprendidos a partir de FD001 y FD002.
- Las condiciones operativas desconocidas son rechazadas.
- La detección de anomalías no determina la causa concreta del posible fallo.
- La RUL es una estimación y presenta incertidumbre.
- Los motores externos sin referencia real no pueden utilizar el modo evaluación.
- Los anomaly scores de FD001 y FD002 no deben compararse directamente.
- Un despliegue en un entorno real requeriría validación adicional con datos representativos del sistema objetivo.
- También sería necesario supervisar el rendimiento de los modelos frente a posibles cambios en la distribución de los datos.

---

# Tecnologías utilizadas

## Machine Learning

```text
scikit-learn
Isolation Forest
StandardScaler
```

## Deep Learning

```text
TensorFlow
Keras
LSTM
```

## Procesamiento de datos

```text
Python
Pandas
NumPy
Joblib
```

## Backend

```text
FastAPI
Uvicorn
python-multipart
```

## Frontend

```text
HTML
CSS
JavaScript
Chart.js
```

---

# Objetivo de la herramienta

El objetivo de la plataforma es proporcionar un entorno integrado de mantenimiento predictivo que permita transformar series temporales de sensores en información interpretable para el seguimiento del estado de los motores.

```text
Datos de sensores
        ↓
Condiciones operativas
        ↓
Selección del perfil predictivo
        ↓
Preprocesamiento
        ↓
Análisis temporal
        ↓
Detección de anomalías
        ↓
Estimación de RUL
        ↓
Visualización e interpretación
        ↓
Apoyo al mantenimiento predictivo
```

La aplicación complementa el desarrollo de los modelos predictivos proporcionando una herramienta que permite ejecutar, visualizar e interpretar sus resultados sobre los motores NASA de referencia y sobre nuevas trayectorias compatibles incorporadas mediante archivos CSV.