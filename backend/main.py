# --------------------------------------------------
# LIBRERÍAS
# --------------------------------------------------
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pathlib import Path
from functools import lru_cache
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import pandas as pd
import joblib
import json
import numpy as np
import tensorflow as tf
import io
from uuid import uuid4
from datetime import datetime

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="Herramienta de mantenimiento predictivo basada en IA",
    version="1.0.0"
)

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset original NASA C-MAPSS
DATA_DIR = BASE_DIR / "data"

# Datos raw oficiales
TEST_FD001_RAW_PATH = DATA_DIR / "test_FD001.txt"
TEST_FD002_RAW_PATH = DATA_DIR / "test_FD002.txt"

# Columnas originales de los archivos C-MAPSS
CMAPSS_COLUMNS = (["unit", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)])

# --------------------------------------------------
# ESQUEMA COMÚN PARA MOTORES EXTERNOS
# --------------------------------------------------
SENSORES_MODELO_COMUNES = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21"
]

COLUMNAS_CSV_OBLIGATORIAS = ([
        "unit",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3"
    ] + SENSORES_MODELO_COMUNES
)

COLUMNAS_SENSOR_VALIDAS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

COLUMNAS_CSV_PERMITIDAS = ([
        "unit",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3"
    ] + COLUMNAS_SENSOR_VALIDAS
)

# Carpeta destinada a motores importados
IMPORTED_DATA_DIR = BASE_DIR/"data"/"imported"

# Registro de importaciones
IMPORT_REGISTRY_PATH = IMPORTED_DATA_DIR/"registry.json"

# Crear la carpeta si todavía no existe
IMPORTED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def cargar_registro_importaciones():
    """
    Recupera el listado de CSV importados previamente
    """
    # Crear un registro vacío si no existe
    if not IMPORT_REGISTRY_PATH.exists():
        with open(
            IMPORT_REGISTRY_PATH,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump([], f, indent=4)
        return []

    # Leer el registro existente
    with open(
        IMPORT_REGISTRY_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        registro = json.load(f)

    # Comprobar que el formato sea correcto
    if not isinstance(registro, list):
        raise ValueError(
            "El registro de importaciones "
            "debe contener una lista."
        )
    return registro

def guardar_registro_importaciones(registro):
    """
    Guarda en disco el registro de CSV importados
    """
    with open(
        IMPORT_REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            registro,
            f,
            indent=4,
            ensure_ascii=False
        )

# --------------------------------------------------
# CARGA DE DATASETS NASA RAW
# --------------------------------------------------
def cargar_test_cmapss(ruta: Path):
    """
    Carga un archivo test oficial de NASA C-MAPSS utilizando la estructura original de 26 columnas
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se ha encontrado el archivo: {ruta}")

    datos = pd.read_csv(ruta, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)

    # Comprobación estructural
    if list(datos.columns) != CMAPSS_COLUMNS:
        raise ValueError(f"Estructura inesperada en {ruta.name}.")

    if datos.empty:
        raise ValueError(f"El archivo {ruta.name} está vacío.")

    # Los identificadores y ciclos deben ser enteros
    datos["unit"] = datos["unit"].astype(int)
    datos["cycle"] = datos["cycle"].astype(int)
    return datos

def preparar_dataset_referencia(ruta: Path, profile: str, origin: str, primer_id_interno: int):
    """
    Prepara un dataset NASA de referencia añadiendo los metadatos internos utilizados por la plataforma
    El identificador original de NASA se conserva en source_unit y unit pasa a representar el ID interno
    """
    datos = cargar_test_cmapss(ruta).copy()

    # Conservar ID original NASA
    datos["source_unit"] = datos["unit"].astype(int)

    # Crear IDs internos consecutivos
    motores_originales = sorted(datos["source_unit"].unique().tolist())
    mapa_ids = {
        int(source_unit): (
            primer_id_interno + indice
        )
        for indice, source_unit
        in enumerate(motores_originales)
    }

    datos["unit"] = datos["source_unit"].map(mapa_ids).astype(int)

    # Metadatos conocidos por procedencia
    datos["origin"] = origin
    datos["is_reference"] = True
    datos["profile"] = profile
    return datos

def cargar_referencias_nasa():
    """
    Construye la flota oficial NASA de referencia
    FD001:
        IDs internos 1-100
        profile = FD001
    FD002:
        IDs internos 101-359
        profile = FD002
    """
    # --------------------------------------------------
    # FD001
    # --------------------------------------------------
    fd001 = preparar_dataset_referencia(
        ruta=TEST_FD001_RAW_PATH,
        profile="FD001",
        origin="NASA C-MAPSS FD001",
        primer_id_interno=1
    )

    # FD001 debe contener 100 motores
    total_fd001 = int(fd001["unit"].nunique())

    if total_fd001 != 100:
        raise ValueError(
            "FD001 debería contener 100 motores "
            f"y se han encontrado {total_fd001}."
        )

    # --------------------------------------------------
    # FD002
    # --------------------------------------------------
    fd002 = preparar_dataset_referencia(
        ruta=TEST_FD002_RAW_PATH,
        profile="FD002",
        origin="NASA C-MAPSS FD002",
        primer_id_interno=101
    )

    # FD002 debe contener 259 motores
    total_fd002 = int(fd002["unit"].nunique())

    if total_fd002 != 259:
        raise ValueError(
            "FD002 debería contener 259 motores "
            f"y se han encontrado {total_fd002}."
        )

    # --------------------------------------------------
    # UNIFICAR REFERENCIAS
    # --------------------------------------------------
    referencias = pd.concat([fd001, fd002], ignore_index=True)

    # Comprobar que los IDs internos no se repiten
    perfiles_por_motor = referencias.groupby("unit")["profile"].nunique()

    if (perfiles_por_motor != 1).any():
        raise ValueError(
            "Se han detectado IDs internos asociados "
            "a más de un perfil."
        )
    return referencias

# --------------------------------------------------
# FLOTA NASA RAW PREPARADA PARA LA NUEVA ARQUITECTURA
# --------------------------------------------------
df_referencias_nasa = cargar_referencias_nasa()

# --------------------------------------------------
# CARGA DE PIPELINES PREDICTIVOS
# --------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
ANOMALY_DIR = MODELS_DIR / "anomaly_detection"
PROGNOSIS_DIR = MODELS_DIR / "prognosis"
PREPROCESSING_DIR = MODELS_DIR / "preprocessing"

# --------------------------------------------------
# PIPELINE FD001 - Perfil de condición operativa única
# --------------------------------------------------
FD001_ANOMALY_PATH = ANOMALY_DIR / "anomaly_detection_FD001.pkl"

fd001_anomaly_artifacts = joblib.load(FD001_ANOMALY_PATH)
fd001_anomaly_model = fd001_anomaly_artifacts["isolation_forest"]
fd001_anomaly_scaler = fd001_anomaly_artifacts["scaler_development"]
fd001_anomaly_features = fd001_anomaly_artifacts["sensor_features"]
fd001_anomaly_threshold = fd001_anomaly_artifacts["thresholds"]["isolation_forest"]
fd001_anomaly_persistence = fd001_anomaly_artifacts["consecutive_cycles"]

# Modelo LSTM FD001
fd001_lstm_model = tf.keras.models.load_model(PROGNOSIS_DIR / "lstm_RUL_FD001.keras",compile=False)

# Escalador RUL FD001
fd001_rul_scaler = joblib.load(PROGNOSIS_DIR / "scaler_RUL_FD001.pkl")

# Configuración RUL FD001
with open(
    PROGNOSIS_DIR / "config_RUL_FD001.json",
    "r",
    encoding="utf-8"
) as f:
    fd001_rul_config = json.load(f)

fd001_rul_features = fd001_rul_config["sensor_features"]
fd001_rul_window_size = int(fd001_rul_config["window_size"])
fd001_rul_cap = float(fd001_rul_config["rul_cap"])

# Verificaciones FD001
assert len(fd001_anomaly_features) == 14
assert fd001_rul_features == fd001_anomaly_features
assert fd001_rul_scaler.n_features_in_ == len(fd001_rul_features)
assert fd001_lstm_model.input_shape[-2:] == (fd001_rul_window_size, len(fd001_rul_features))

print("Pipeline FD001 cargado correctamente.")

# --------------------------------------------------
# PIPELINE FD002 - Perfil operativo multirrégimen
# --------------------------------------------------
FD002_PREPROCESSING_PATH = PREPROCESSING_DIR / "preprocessing_FD002.pkl"
FD002_ANOMALY_PATH = ANOMALY_DIR / "anomaly_detection_FD002.pkl"

# Preprocesamiento FD002
fd002_preprocessing = joblib.load(FD002_PREPROCESSING_PATH)

# Detector de anomalías FD002
fd002_anomaly_artifacts = joblib.load(FD002_ANOMALY_PATH)
fd002_anomaly_model = fd002_anomaly_artifacts["model"]
fd002_anomaly_threshold = float(fd002_anomaly_artifacts["threshold"])
fd002_anomaly_persistence = int(fd002_anomaly_artifacts["persistence_cycles"])
fd002_anomaly_features = fd002_anomaly_artifacts["features_modelo"]

# Modelo LSTM FD002
fd002_lstm_model = tf.keras.models.load_model(PROGNOSIS_DIR / "lstm_RUL_FD002.keras",compile=False)

# Configuración RUL FD002
with open(
    PROGNOSIS_DIR / "config_RUL_FD002.json",
    "r",
    encoding="utf-8"
) as f:
    fd002_rul_config = json.load(f)

fd002_sensor_features = fd002_rul_config["sensores_modelo"]
fd002_rul_features = fd002_rul_config["features_modelo"]
fd002_rul_window_size = int(fd002_rul_config["window_size"])
fd002_rul_cap = float(fd002_rul_config["rul_max"])

# Información necesaria para asignar regímenes
fd002_regime_mapping = fd002_preprocessing["regime_mapping"]
fd002_regime_rounding = fd002_preprocessing["regime_rounding"]
fd002_regime_statistics = fd002_preprocessing["estadisticas_regimen"]

# Verificaciones FD002
assert fd002_preprocessing["sensores_modelo"] == fd002_sensor_features
assert fd002_preprocessing["features_modelo"] == fd002_rul_features
assert fd002_anomaly_features == fd002_rul_features
assert fd002_preprocessing["normalization"] == "zscore_by_regime"
assert fd002_anomaly_artifacts["score_definition"] == "-score_samples"
assert fd002_anomaly_artifacts["threshold_operator"] == ">"
assert len(fd002_sensor_features) == 14
assert len(fd002_regime_mapping) == 6
assert fd002_lstm_model.input_shape[-2:] == (fd002_rul_window_size, len(fd002_rul_features))

# Ambos perfiles deben utilizar los mismos 14 sensores raw de entrada
assert fd001_rul_features == SENSORES_MODELO_COMUNES
assert fd002_sensor_features == SENSORES_MODELO_COMUNES

print("Pipeline FD002 cargado correctamente.")

# --------------------------------------------------
# REGISTRO DE PIPELINES
# --------------------------------------------------
PIPELINES = {
    "FD001": {
        "profile_name": "Condición única",
        "anomaly": {
            "model": fd001_anomaly_model,
            "scaler": fd001_anomaly_scaler,
            "features": fd001_anomaly_features,
            "threshold": fd001_anomaly_threshold,
            "persistence": fd001_anomaly_persistence
        },
        "rul": {
            "model": fd001_lstm_model,
            "scaler": fd001_rul_scaler,
            "features": fd001_rul_features,
            "window_size": fd001_rul_window_size,
            "rul_cap": fd001_rul_cap
        }
    },
    "FD002": {
        "profile_name": "Multirrégimen",
        "preprocessing": {
            "sensors": fd002_sensor_features,
            "features": fd002_rul_features,
            "regime_mapping": fd002_regime_mapping,
            "regime_rounding": fd002_regime_rounding,
            "regime_statistics": fd002_regime_statistics
        },
        "anomaly": {
            "model": fd002_anomaly_model,
            "features": fd002_anomaly_features,
            "threshold": fd002_anomaly_threshold,
            "persistence": fd002_anomaly_persistence
        },
        "rul": {
            "model": fd002_lstm_model,
            "features": fd002_rul_features,
            "window_size": fd002_rul_window_size,
            "rul_cap": fd002_rul_cap
        }
    }
}

# --------------------------------------------------
# PREPROCESAMIENTO SEGÚN PERFIL
# --------------------------------------------------
def obtener_perfil_motor(motor_df: pd.DataFrame):
    """
    Recupera el perfil predictivo asociado a un motor
    Todos los registros de una misma trayectoria deben pertenecer al mismo perfil
    """
    if "profile" not in motor_df.columns:
        raise ValueError("El motor no tiene un perfil predictivo asignado.")

    perfiles = motor_df["profile"].dropna().unique().tolist()

    if len(perfiles) != 1:
        raise ValueError(
            "La trayectoria del motor contiene "
            "más de un perfil predictivo."
        )

    perfil = str(perfiles[0])

    if perfil not in PIPELINES:
        raise ValueError(f"Perfil predictivo desconocido: {perfil}.")

    return perfil

def asignar_regimenes_fd002(datos: pd.DataFrame):
    """
    Asigna a cada observación FD002 uno de los seis regímenes operativos definidos durante el desarrollo
    La asignación reproduce exactamente el redondeo utilizado al construir el modelo
    """
    resultado = datos.copy()
    columnas_operativas = ["op_setting_1", "op_setting_2", "op_setting_3"]

    # Comprobar que están disponibles
    columnas_faltantes = [columna for columna in columnas_operativas if columna not in resultado.columns]

    if columnas_faltantes:
        raise ValueError(
            "Faltan variables operativas necesarias "
            "para FD002: "
            + ", ".join(columnas_faltantes)
        )

    # Aplicar exactamente el redondeo almacenado
    valores_redondeados = {}
    for columna in columnas_operativas:
        decimales = int(fd002_regime_rounding[columna])
        valores_redondeados[columna] = resultado[columna].round(decimales)

    # Crear la clave operacional de cada observación
    claves_regimen = list(
        zip(
            valores_redondeados["op_setting_1"],
            valores_redondeados["op_setting_2"],
            valores_redondeados["op_setting_3"]
        )
    )
    resultado["operating_regime"] = [fd002_regime_mapping.get(tuple(clave))for clave in claves_regimen]

    # Detectar condiciones no reconocidas
    mascara_desconocida = (resultado["operating_regime"].isna())

    if mascara_desconocida.any():
        condiciones_desconocidas = (
            pd.DataFrame({
                "op_setting_1": valores_redondeados["op_setting_1"][mascara_desconocida],
                "op_setting_2": valores_redondeados["op_setting_2"][mascara_desconocida],
                "op_setting_3": valores_redondeados["op_setting_3"][mascara_desconocida]
            }).drop_duplicates().to_dict("records")
        )

        raise ValueError(
            "Se han encontrado condiciones operativas "
            "no reconocidas por FD002: "
            f"{condiciones_desconocidas}"
        )

    resultado["operating_regime"] = resultado["operating_regime"].astype(int)
    return resultado

def normalizar_fd002(datos: pd.DataFrame):
    """
    Aplica la normalización z-score específica de cada régimen utilizando exclusivamente las estadísticas aprendidas durante el entrenamiento de FD002
    """
    resultado = asignar_regimenes_fd002(datos)
    for sensor, feature_norm in zip(fd002_sensor_features,fd002_rul_features):
        medias = {int(regimen): float(fd002_regime_statistics.loc[regimen,(sensor, "mean")]) for regimen in fd002_regime_statistics.index}
        desviaciones = {int(regimen): float(fd002_regime_statistics.loc[regimen, (sensor, "std")]) for regimen in fd002_regime_statistics.index}
        media_fila = resultado["operating_regime"].map(medias)
        std_fila = resultado["operating_regime"].map(desviaciones)

        if (std_fila <= 0).any():
            raise ValueError(
                f"Desviación estándar no válida "
                f"para {sensor}."
            )

        resultado[feature_norm] = (resultado[sensor] - media_fila) / std_fila

    return resultado



# --------------------------------------------------
# DETECCIÓN AUTOMÁTICA DE PERFIL
# --------------------------------------------------
def detectar_perfil_motor_externo(motor_df: pd.DataFrame):
    """
    Determina automáticamente qué pipeline debe utilizar un motor externo a partir de sus condiciones operativas
    - Solo régimen 1 -> FD001
    - Algún régimen 2-6 -> FD002
    - Condición desconocida -> incompatible
    """

    try:
        datos_regimen = asignar_regimenes_fd002(motor_df)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Perfil predictivo no disponible "
                "para estas condiciones operativas."
            )
        )

    regimenes = set(datos_regimen["operating_regime"].astype(int).unique().tolist())

    if regimenes == {1}:
        return "FD001"

    if any(regimen in {2, 3, 4, 5, 6} for regimen in regimenes):
        return "FD002"

    raise HTTPException(
        status_code=400,
        detail=(
            "Perfil predictivo no disponible "
            "para estas condiciones operativas."
        )
    )



# --------------------------------------------------
# VALIDACIÓN DE CSV EXTERNOS
# --------------------------------------------------

def validar_csv_motores(datos_csv: pd.DataFrame):
    """
    Valida un CSV raw externo y determina automáticamente el perfil predictivo decada motor
    """
    # --------------------------------------------------
    # 1. ARCHIVO NO VACÍO
    # --------------------------------------------------
    if datos_csv.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo CSV no contiene datos."
            )
        )

    # --------------------------------------------------
    # 2. COLUMNAS
    # --------------------------------------------------
    columnas_recibidas = list(datos_csv.columns)
    columnas_faltantes = [columna for columna in COLUMNAS_CSV_OBLIGATORIAS if columna not in columnas_recibidas]

    if columnas_faltantes:
        raise HTTPException(
            status_code=400,
            detail={
                "mensaje": (
                    "El CSV no contiene todas "
                    "las columnas necesarias."
                ),
                "columnas_faltantes": columnas_faltantes
            }
        )

    columnas_adicionales = [columna for columna in columnas_recibidas if columna not in COLUMNAS_CSV_PERMITIDAS]

    if columnas_adicionales:
        raise HTTPException(
            status_code=400,
            detail={
                "mensaje": (
                    "El CSV contiene columnas "
                    "no reconocidas."
                ),
                "columnas_adicionales": columnas_adicionales
            }
        )

    # --------------------------------------------------
    # 3. ORDENAR COLUMNAS
    # --------------------------------------------------
    columnas_presentes = [columna for columna in COLUMNAS_CSV_PERMITIDAS if columna in datos_csv.columns]
    datos_csv = datos_csv[columnas_presentes].copy()

    # --------------------------------------------------
    # 4. VALORES NULOS
    # --------------------------------------------------
    if (datos_csv.isnull().any().any()):
        columnas_con_nulos = datos_csv.columns[datos_csv.isnull().any()].tolist()
        
        raise HTTPException(
            status_code=400,
            detail={
                "mensaje": (
                    "El CSV contiene "
                    "valores nulos."
                ),
                "columnas_con_nulos": columnas_con_nulos
            }
        )

    # --------------------------------------------------
    # 5. DATOS NUMÉRICOS
    # --------------------------------------------------
    for columna in (datos_csv.columns):
        try:
            datos_csv[columna] = pd.to_numeric(datos_csv[columna],errors="raise")
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"La columna '{columna}' "
                    "contiene valores "
                    "no numéricos."
                )
            )

    # --------------------------------------------------
    # 6. IDENTIFICADORES
    # --------------------------------------------------
    if ((datos_csv["unit"] <= 0).any() or (datos_csv["unit"] % 1 != 0).any()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los identificadores de motor "
                "deben ser números enteros "
                "positivos."
            )
        )

    datos_csv["unit"] = datos_csv["unit"].astype(int)

    # --------------------------------------------------
    # 7. CICLOS
    # --------------------------------------------------
    if ((datos_csv["cycle"] <= 0).any() or (datos_csv["cycle"] % 1 != 0).any()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los ciclos deben ser "
                "números enteros positivos."
            )
        )

    datos_csv["cycle"] = datos_csv["cycle"].astype(int)

    # --------------------------------------------------
    # 8. VALIDAR CADA MOTOR
    # --------------------------------------------------
    resumen_motores = []

    window_size_minimo = max(int(fd001_rul_window_size), int(fd002_rul_window_size))

    for (motor_id, motor_df) in datos_csv.groupby("unit"):
        motor_df = motor_df.sort_values("cycle")
        ciclos = motor_df["cycle"].tolist()

        # Ciclos duplicados
        if (len(ciclos) != len(set(ciclos))):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El motor {motor_id} "
                    "contiene ciclos "
                    "duplicados."
                )
            )

        # El historial debe comenzar en 1
        if ciclos[0] != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El motor {motor_id} "
                    "debe comenzar en "
                    "el ciclo 1."
                )
            )

        # Ciclos consecutivos
        ciclos_esperados = list(range(1, ciclos[-1] + 1))

        if (ciclos != ciclos_esperados):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El motor {motor_id} "
                    "contiene saltos en "
                    "la secuencia de ciclos."
                )
            )

        # Los motores externos deben poder utilizar también la LSTM
        if (len(motor_df) < window_size_minimo):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El motor {motor_id} "
                    f"tiene {len(motor_df)} "
                    "ciclos. Se requieren "
                    f"al menos "
                    f"{window_size_minimo}."
                )
            )

        # Detección automática
        perfil = detectar_perfil_motor_externo(motor_df)

        resumen_motores.append(
            {
                "motor_original": int(motor_id),
                "ciclos": int(len(motor_df)),
                "primer_ciclo": int(ciclos[0]),
                "ultimo_ciclo": int(ciclos[-1]),
                "profile": perfil,
                "profile_name": PIPELINES[perfil]["profile_name"]
            }
        )

    return (datos_csv, resumen_motores)

# --------------------------------------------------
# CARGA DE MOTORES EXTERNOS PERSISTIDOS
# --------------------------------------------------
def cargar_motores_importados():
    datasets = []
    registro = cargar_registro_importaciones()

    for importacion in registro:
        nombre_archivo = importacion.get("filename")

        if not nombre_archivo:
            continue

        ruta_csv = IMPORTED_DATA_DIR / nombre_archivo

        if not ruta_csv.exists():
            print(
                "Aviso: no se ha encontrado "
                f"el archivo importado "
                f"{nombre_archivo}."
            )
            continue

        datos = pd.read_csv(ruta_csv)
        columnas_necesarias = COLUMNAS_CSV_OBLIGATORIAS + ["source_unit", "origin", "is_reference", "profile"]
        columnas_faltantes = [columna for columna in columnas_necesarias if columna not in datos.columns]

        if columnas_faltantes:
            print(
                "Aviso: se omite una "
                "importación antigua "
                f"({nombre_archivo}) porque "
                "no utiliza el nuevo esquema."
            )
            continue

        # --------------------------------------------------
        # PERFIL PREDICTIVO PERSISTIDO
        # --------------------------------------------------
        perfiles_guardados = importacion.get("predictive_profiles", {})

        if perfiles_guardados:
            for (source_unit, perfil) in perfiles_guardados.items():
                if perfil not in PIPELINES:
                    raise ValueError(
                        "La importación contiene "
                        f"un perfil desconocido: {perfil}."
                    )

                mascara_motor = datos["source_unit"].astype(int) == int(source_unit)

                if not mascara_motor.any():
                    raise ValueError(
                        "El registro de importación "
                        "no coincide con el CSV "
                        f"guardado ({nombre_archivo})."
                    )

                datos.loc[mascara_motor, "profile"] = perfil
        
        datasets.append(datos)

    if not datasets:
        return None

    return pd.concat(datasets, ignore_index=True, sort=False)

def cargar_datos_operativos():
    datasets = [df_referencias_nasa.copy()]
    importados = cargar_motores_importados()

    if importados is not None:
        datasets.append(importados)

    datos = pd.concat(datasets, ignore_index=True, sort=False)

    # Un ID interno solo puede pertenecer a una trayectoria/perfil
    perfiles_por_motor = datos.groupby("unit")["profile"].nunique()

    if (perfiles_por_motor > 1).any():
        raise ValueError(
            "Hay identificadores internos "
            "asociados a más de un perfil."
        )

    return datos

# --------------------------------------------------
# INFERENCIA SEGÚN PERFIL PREDICTIVO
# --------------------------------------------------
def preparar_motor_para_inferencia(motor_df: pd.DataFrame):
    """
    Prepara una trayectoria completa para el pipeline predictivo que le corresponde
    """
    if motor_df.empty:
        raise ValueError("No se han recibido observaciones del motor.")

    motor_df = motor_df.sort_values("cycle").copy()
    perfil = obtener_perfil_motor(motor_df)

    if perfil == "FD001":
        datos_preparados = motor_df
    elif perfil == "FD002":
        datos_preparados = normalizar_fd002(motor_df)
    else:
        raise ValueError(f"Perfil no soportado: {perfil}.")

    return perfil, datos_preparados

def inferir_anomalias_motor(motor_df: pd.DataFrame):
    """
    Calcula el historial completo de anomaly score utilizando el pipeline correspondiente al motor
    """
    perfil, datos = preparar_motor_para_inferencia(motor_df)
    pipeline = PIPELINES[perfil]["anomaly"]
    features = pipeline["features"]
    X = datos[features]

    # FD001 necesita StandardScaler antes del IF
    # FD002 ya llega normalizado por régimen
    scaler = pipeline.get("scaler")
    
    if scaler is not None:
        X_modelo = pd.DataFrame(
            scaler.transform(X),
            columns=features,
            index=X.index
        )
    else:
        X_modelo = X.copy()

    modelo = pipeline["model"]
    umbral = float(pipeline["threshold"])
    persistencia = int(pipeline["persistence"])
    scores = (-modelo.score_samples(X_modelo))
    
    historial = []
    ciclos_confirmacion = []
    racha = 0

    for posicion, (ciclo_actual,score) in enumerate(zip(datos["cycle"],scores)):
        supera_umbral = bool(score > umbral)

        if supera_umbral:
            racha += 1
        else:
            racha = 0

        alerta_activa = bool(racha >= persistencia)

        if racha == persistencia:
            ciclos_confirmacion.append(int(ciclo_actual))

        registro = {
            "cycle": int(ciclo_actual),
            "anomaly_score": float(score),
            "supera_umbral": (supera_umbral),
            "alerta_activa": (alerta_activa)
        }

        # FD002 puede mostrar también el régimen
        if perfil == "FD002":
            registro["operating_regime"] = int(datos.iloc[posicion]["operating_regime"])
        
        historial.append(registro)

    estado_actual = historial[-1]

    return {
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "detector": "Isolation Forest",
        "anomaly_score": estado_actual["anomaly_score"],
        "umbral": umbral,
        "persistencia": persistencia,
        "alerta_activa": estado_actual["alerta_activa"],
        "ha_tenido_alertas": len(ciclos_confirmacion) > 0,
        "ciclos_confirmacion": ciclos_confirmacion,
        "historial": historial
    }

def inferir_historial_rul_motor(motor_df: pd.DataFrame):
    """
    Genera todas las predicciones RUL disponibles para una trayectoria utilizando su pipeline
    """
    perfil, datos = preparar_motor_para_inferencia(motor_df)
    pipeline = PIPELINES[perfil]["rul"]
    features = pipeline["features"]
    window_size = int(pipeline["window_size"])
    modelo = pipeline["model"]
    X_base = datos[features]

    # FD001 necesita su StandardScaler
    # FD002 utiliza directamente las features normalizadas por régimen
    scaler = pipeline.get("scaler")

    if scaler is not None:
        X_transformado = scaler.transform(X_base)
    else:
        X_transformado = X_base.to_numpy(dtype=float)

    ventanas = []
    ciclos = []
    regimenes = []

    for i in range(window_size, len(datos) + 1):
        ventana = X_transformado[i - window_size:i]
        ventanas.append(ventana)
        ciclos.append(int(datos.iloc[i - 1]["cycle"]))

        if perfil == "FD002":
            regimenes.append(int(datos.iloc[i - 1]["operating_regime"]))

    # Motor todavía no evaluable por la LSTM
    if not ventanas:
        return {
            "profile": perfil,
            "profile_name": PIPELINES[perfil]["profile_name"],
            "window_size": window_size,
            "historial": []
        }

    X_lstm = np.asarray(ventanas, dtype=np.float32)
    predicciones = modelo.predict(X_lstm, verbose=0).ravel()

    if not np.isfinite(predicciones).all():
        raise ValueError(
            "Se han obtenido predicciones "
            "RUL no válidas."
        )

    historial = []

    for posicion, (ciclo_actual, rul) in enumerate(zip(ciclos, predicciones)):
        registro = {
            "cycle": int(ciclo_actual),
            "rul_estimada": float(rul)
        }

        if perfil == "FD002":
            registro["operating_regime"] = int(regimenes[posicion])

        historial.append(registro)

    return {
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "window_size": window_size,
        "rul_cap": float(pipeline["rul_cap"]),
        "historial": historial
    }

def inferir_rul_motor(motor_df: pd.DataFrame):
    """
    Devuelve la estimación RUL correspondiente al último ciclo observado de una trayectoria
    La predicción se realiza sobre una única ventana para reproducir el mismo procedimiento utilizado por el endpoint individual de la aplicación
    """
    perfil, datos = preparar_motor_para_inferencia(motor_df)
    pipeline = PIPELINES[perfil]["rul"]
    features = pipeline["features"]
    window_size = int(pipeline["window_size"])
    modelo = pipeline["model"]

    # --------------------------------------------------
    # 1. COMPROBAR SI HAY SUFICIENTES CICLOS
    # --------------------------------------------------
    if len(datos) < window_size:
        return {
            "profile": perfil,
            "profile_name": PIPELINES[perfil]["profile_name"],
            "disponible": False,
            "rul_estimada": None,
            "ciclos_necesarios": window_size,
            "ciclos_disponibles": int(len(datos))
        }

    # --------------------------------------------------
    # 2. SELECCIONAR ÚLTIMA VENTANA
    # --------------------------------------------------
    ventana = datos[features].tail(window_size)

    # --------------------------------------------------
    # 3. APLICAR PREPROCESAMIENTO DEL PERFIL
    # --------------------------------------------------
    scaler = pipeline.get("scaler")

    if scaler is not None:
        # FD001
        ventana_transformada = scaler.transform(ventana)
    else:
        # FD002 ya está normalizado por régimen
        ventana_transformada = ventana.to_numpy(dtype=float)

    # --------------------------------------------------
    # 4. PREPARAR ENTRADA LSTM
    # --------------------------------------------------
    X = np.array(ventana_transformada,dtype=np.float32).reshape(1,window_size,len(features))

    # --------------------------------------------------
    # 5. PREDICCIÓN
    # --------------------------------------------------
    prediccion = modelo.predict(X, verbose=0)
    rul_estimada = float(prediccion[0, 0])

    if not np.isfinite(rul_estimada):
        raise ValueError(
            "La predicción RUL obtenida "
            "no es válida."
        )

    # --------------------------------------------------
    # 6. RESPUESTA
    # --------------------------------------------------
    resultado = {
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "disponible": True,
        "rul_estimada": rul_estimada,
        "rul_cap": float(pipeline["rul_cap"]),
        "window_size": window_size
    }

    if perfil == "FD002":
        resultado["operating_regime"] = int(datos.iloc[-1]["operating_regime"])

    return resultado

# --------------------------------------------------
# ENDPOINT INICIAL
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API",
        "status": "running",
        "dataset": "NASA C-MAPSS FD001 + FD002"
    }

# --------------------------------------------------
# ENDPOINT: LISTADO DE MOTORES
# --------------------------------------------------
@app.get("/api/motores")
def obtener_motores():
    motores = (
        df_motores
        .groupby("unit")
        .agg(
            ultimo_ciclo=("cycle","max"),
            observaciones=("cycle","count"),
            origen=("origin","first"),
            motor_original=("source_unit","first"),
            es_referencia=("is_reference","first"),
            profile=("profile","first")
        )
        .reset_index()
        .sort_values("unit")
    )

    lista_motores = []

    for _, motor in (motores.iterrows()):
        perfil = str(motor["profile"])

        lista_motores.append(
            {
                "id": int(motor["unit"]),
                "motor_original": int(motor["motor_original"]),
                "origen": str(motor["origen"]),
                "ultimo_ciclo": int(motor["ultimo_ciclo"]),
                "observaciones": int(motor["observaciones"]),
                "es_referencia": bool(motor["es_referencia"]),
                "profile": perfil,
                "profile_name": (PIPELINES[perfil]["profile_name"]
                )
            }
        )

    return {
        "total_motores": len(lista_motores),
        "motores": lista_motores
    }

# --------------------------------------------------
# ENDPOINT: INFORMACIÓN DE UN MOTOR
# --------------------------------------------------
@app.get("/api/motores/{motor_id}")
def obtener_motor(motor_id: int):
    motor_df = obtener_datos_motor(motor_id)
    perfil = obtener_perfil_motor(motor_df)
    sensores = list(fd001_rul_features)
    ultima_observacion = motor_df.iloc[-1]
    valores_sensores = {sensor: float(ultima_observacion[sensor]) for sensor in sensores}

    respuesta = {
        "motor_id": motor_id,
        "motor_original": int(motor_df["source_unit"].iloc[0]),
        "origen": str(motor_df["origin"].iloc[0]),
        "es_referencia": bool(motor_df["is_reference"].iloc[0]),
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "total_ciclos": int(len(motor_df)),
        "primer_ciclo": int(motor_df["cycle"].min()),
        "ultimo_ciclo": int(motor_df["cycle"].max()),
        "sensores": valores_sensores
    }

    if perfil == "FD002":
        datos_regimen = asignar_regimenes_fd002(motor_df.tail(1))
        respuesta["operating_regime"] = int(datos_regimen["operating_regime"].iloc[0])

    return respuesta

# --------------------------------------------------
# ENDPOINT: HISTORIAL DE SENSORES DE UN MOTOR
# --------------------------------------------------
@app.get("/api/motores/{motor_id}/historial")
def obtener_historial(motor_id: int, ciclo: int | None = None):
    # --------------------------------------------------
    # 1. RECUPERAR MOTOR
    # --------------------------------------------------
    motor_df = obtener_datos_motor(motor_id)
    perfil = obtener_perfil_motor(motor_df)
    ultimo_ciclo = int(motor_df["cycle"].max())

    if ciclo is None:
        ciclo = ultimo_ciclo

    if (ciclo < 1 or ciclo > ultimo_ciclo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"El ciclo debe estar entre 1 y "
                f"{ultimo_ciclo} para el motor "
                f"{motor_id}."
            )
        )

    # --------------------------------------------------
    # 2. LIMITAR HISTORIAL
    # --------------------------------------------------
    historial_df = motor_df[motor_df["cycle"] <= ciclo].copy()

    # --------------------------------------------------
    # 3. RÉGIMEN FD002
    # --------------------------------------------------
    if perfil == "FD002":
        historial_df = asignar_regimenes_fd002(historial_df)

    # --------------------------------------------------
    # 4. SENSORES MOSTRADOS
    # --------------------------------------------------
    sensores = list(fd001_rul_features)

    # --------------------------------------------------
    # 5. PREPARAR HISTORIAL
    # --------------------------------------------------
    historial = []

    for _, fila in (historial_df.iterrows()):
        valores_sensores = {sensor: float(fila[sensor]) for sensor in sensores}

        registro = {
            "cycle": int(fila["cycle"]),
            "sensores": valores_sensores
        }

        if perfil == "FD002":
            registro["operating_regime"] = int(fila["operating_regime"])

        historial.append(registro)

    # --------------------------------------------------
    # 6. RESPUESTA
    # --------------------------------------------------
    respuesta = {
        "motor_id": motor_id,
        "motor_original": int(motor_df["source_unit"].iloc[0]),
        "origen": str(motor_df["origin"].iloc[0]),
        "es_referencia": bool(motor_df["is_reference"].iloc[0]),
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "total_ciclos": int(len(motor_df)),
        "primer_ciclo": int(motor_df["cycle"].min()),
        "ultimo_ciclo": ultimo_ciclo,
        "ciclo_consultado": ciclo,
        "historial": historial
    }

    if perfil == "FD002":
        respuesta["operating_regime"] = int(historial_df["operating_regime"].iloc[-1])

    return respuesta


# --------------------------------------------------
# DATASET OPERATIVO
# --------------------------------------------------
# Incluye las referencias NASA y los motores externos importados compatibles con la arquitectura actual
df_motores = cargar_datos_operativos()

# --------------------------------------------------
# RECUPERACIÓN DE MOTORES
# --------------------------------------------------
def obtener_datos_motor(motor_id: int):
    """
    Recupera la trayectoria completa de un motor utilizando su identificador interno
    """
    motor_df = df_motores[df_motores["unit"] == motor_id].sort_values("cycle").copy()

    if motor_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"El motor {motor_id} "
                "no existe."
            )
        )

    return motor_df
    
@app.get("/api/motores/{motor_id}/anomalias")
def obtener_anomalias(motor_id: int, ciclo: int | None = None):
    """
    Ejecuta el detector de anomalías correspondiente al perfil predictivo del motor
    """
    # --------------------------------------------------
    # 1. RECUPERAR MOTOR
    # --------------------------------------------------
    motor_df = obtener_datos_motor(motor_id)

    ultimo_ciclo = int(motor_df["cycle"].max())

    if ciclo is None:
        ciclo = ultimo_ciclo

    if (ciclo < 1 or ciclo > ultimo_ciclo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"El ciclo debe estar entre 1 y "
                f"{ultimo_ciclo}."
            )
        )

    # --------------------------------------------------
    # 2. UTILIZAR SOLO EL HISTORIAL DISPONIBLE
    # --------------------------------------------------
    motor_hasta_ciclo = motor_df[motor_df["cycle"] <= ciclo].copy()

    # --------------------------------------------------
    # 3. INFERENCIA SEGÚN PERFIL
    # --------------------------------------------------
    try:
        resultado = inferir_anomalias_motor(motor_hasta_ciclo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # --------------------------------------------------
    # 4. RESPUESTA
    # --------------------------------------------------
    respuesta = {
        "motor_id": motor_id,
        "profile": resultado["profile"],
        "profile_name": resultado["profile_name"],
        "ciclo_consultado": ciclo,
        "ultimo_ciclo_disponible": ultimo_ciclo,
        "detector": resultado["detector"],
        "anomaly_score": resultado["anomaly_score"],
        "umbral": resultado["umbral"],
        "persistencia": resultado["persistencia"],
        "alerta_activa": resultado["alerta_activa"],
        "ha_tenido_alertas": resultado["ha_tenido_alertas"],
        "ciclos_confirmacion": resultado["ciclos_confirmacion"],
        "historial": resultado["historial"]
    }

    # FD002 dispone además del régimen operativo actual
    if (resultado["profile"] == "FD002"):
        respuesta["operating_regime"] = int(resultado["historial"][-1]["operating_regime"])

    return respuesta

# --------------------------------------------------
# ENDPOINT: ESTIMACIÓN DE RUL
# --------------------------------------------------
@app.get("/api/motores/{motor_id}/rul")
def obtener_rul(motor_id: int, ciclo: int | None = None):
    """
    Calcula la RUL con la LSTM correspondiente al perfil predictivo del motor
    """
    # --------------------------------------------------
    # 1. RECUPERAR MOTOR
    # --------------------------------------------------
    motor_df = obtener_datos_motor(motor_id)
    ultimo_ciclo = int(motor_df["cycle"].max())

    if ciclo is None:
        ciclo = ultimo_ciclo
    if (ciclo < 1 or ciclo > ultimo_ciclo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"El ciclo debe estar entre 1 y "
                f"{ultimo_ciclo}."
            )
        )

    # --------------------------------------------------
    # 2. HISTORIAL DISPONIBLE HASTA ESE CICLO
    # --------------------------------------------------
    motor_hasta_ciclo = motor_df[motor_df["cycle"] <= ciclo].copy()

    # --------------------------------------------------
    # 3. INFERENCIA
    # --------------------------------------------------
    try:
        resultado = inferir_rul_motor(motor_hasta_ciclo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    perfil = resultado["profile"]
    pipeline_rul = PIPELINES[perfil]["rul"]
    window_size = int(pipeline_rul["window_size"])
    rul_cap_motor = float(pipeline_rul["rul_cap"])

    # --------------------------------------------------
    # 4. RUL TODAVÍA NO DISPONIBLE
    # --------------------------------------------------
    if not resultado["disponible"]:
        return {
            "motor_id": motor_id,
            "profile": perfil,
            "profile_name": resultado["profile_name"],
            "ciclo_consultado": ciclo,
            "ultimo_ciclo_disponible": ultimo_ciclo,
            "disponible": False,
            "rul_estimada": None,
            "mensaje": (
                "Datos insuficientes "
                "para estimar RUL."
            ),
            "ciclos_necesarios": window_size,
            "ciclos_disponibles": int(len(motor_hasta_ciclo)),
            "rul_cap": rul_cap_motor,
            "unidad": "ciclos"
        }

    # --------------------------------------------------
    # 5. RESPUESTA CON PREDICCIÓN
    # --------------------------------------------------
    respuesta = {
        "motor_id": motor_id,
        "profile": perfil,
        "profile_name": resultado["profile_name"],
        "ciclo_consultado": ciclo,
        "ultimo_ciclo_disponible": ultimo_ciclo,
        "disponible": True,
        "modelo": "LSTM",
        "ventana_inicio": int(motor_hasta_ciclo.iloc[-window_size]["cycle"]),
        "ventana_fin": ciclo,
        "ventana_temporal": window_size,
        "rul_estimada": float(resultado["rul_estimada"]),
        "rul_cap": rul_cap_motor,
        "unidad": "ciclos"
    }

    if perfil == "FD002":
        respuesta["operating_regime"] = int(resultado["operating_regime"])

    return respuesta

# --------------------------------------------------
# ENDPOINT: ESTADO GENERAL DE UN MOTOR
# --------------------------------------------------
@app.get("/api/motores/{motor_id}/estado")
def obtener_estado_motor(motor_id: int, ciclo: int | None = None):
    """
    Resume el estado predictivo del motor utilizando automáticamente el pipeline correspondiente.
    """
    # --------------------------------------------------
    # 1. ANOMALÍAS
    # --------------------------------------------------
    resultado_anomalias = obtener_anomalias(motor_id=motor_id, ciclo=ciclo)

    # --------------------------------------------------
    # 2. RUL
    # --------------------------------------------------
    resultado_rul = obtener_rul(motor_id=motor_id, ciclo=ciclo)
    perfil = resultado_anomalias["profile"]
    pipeline_rul = PIPELINES[perfil]["rul"]

    # --------------------------------------------------
    # 3. RESPUESTA COMÚN
    # --------------------------------------------------
    estado_motor = {
        "motor_id": motor_id,
        "profile": perfil,
        "profile_name": resultado_anomalias["profile_name"],
        "ciclo_actual": resultado_anomalias["ciclo_consultado"],
        "ultimo_ciclo_disponible": resultado_anomalias["ultimo_ciclo_disponible"],
        "anomalias": {
            "modelo": "Isolation Forest",
            "anomaly_score": resultado_anomalias["anomaly_score"],
            "umbral": resultado_anomalias["umbral"],
            "alerta_activa": resultado_anomalias["alerta_activa"],
            "ha_tenido_alertas": resultado_anomalias["ha_tenido_alertas"],
            "ciclos_confirmacion": resultado_anomalias["ciclos_confirmacion"]
        },
        "prognosis": {
            "modelo": "LSTM",
            "disponible": resultado_rul["disponible"],
            "rul_estimada": resultado_rul["rul_estimada"],
            "rul_cap": float(pipeline_rul["rul_cap"]),
            "ventana_temporal": int(pipeline_rul["window_size"]),
            "unidad": "ciclos"
        }
    }

    # --------------------------------------------------
    # 4. INFORMACIÓN ESPECÍFICA FD002
    # --------------------------------------------------
    if perfil == "FD002":
        estado_motor["operating_regime"] = int(resultado_anomalias["operating_regime"])

    return estado_motor

# --------------------------------------------------
# RESUMEN DE FLOTA MULTIPIPELINE
# --------------------------------------------------
@lru_cache(maxsize=1)
def calcular_resumen_flota_multipipeline():
    """
    Calcula el estado de toda la flota utilizando independientemente los pipelines FD001 y FD002
    Los anomaly scores se calculan con el detector correspondiente a cada perfil y no se comparan directamente entre perfiles
    """
    datos_completos = df_motores.sort_values(["unit", "cycle"]).copy()
    resultados_perfiles = []

    # ==================================================
    # PROCESAR CADA PERFIL POR SEPARADO
    # ==================================================
    for perfil in ["FD001", "FD002"]:
        datos = datos_completos[datos_completos["profile"] == perfil].copy()

        if datos.empty:
            continue

        # --------------------------------------------------
        # 1. PREPROCESAMIENTO
        # --------------------------------------------------
        if perfil == "FD002":
            datos = normalizar_fd002(datos)

        # --------------------------------------------------
        # 2. ANOMALÍAS
        # --------------------------------------------------
        pipeline_anomalia = PIPELINES[perfil]["anomaly"]
        features_anomalia = pipeline_anomalia["features"]
        X_anomalia = datos[features_anomalia]
        scaler_anomalia = pipeline_anomalia.get("scaler")

        if scaler_anomalia is not None:
            X_anomalia_modelo = pd.DataFrame(scaler_anomalia.transform(X_anomalia),columns=features_anomalia,index=X_anomalia.index)
        else:
            X_anomalia_modelo = X_anomalia.copy()

        modelo_anomalia = pipeline_anomalia["model"]
        umbral = float(pipeline_anomalia["threshold"])
        persistencia = int(pipeline_anomalia["persistence"])
        
        datos["anomaly_score"] = (-modelo_anomalia.score_samples(X_anomalia_modelo))
        datos["supera_umbral"] = (datos["anomaly_score"] > umbral)

        # Confirmación tras N ciclos consecutivos
        datos["alerta_activa"] = (datos.groupby("unit")["supera_umbral"]
            .transform(
                lambda serie: (
                    serie
                    .rolling(
                        window=persistencia,
                        min_periods=persistencia
                    )
                    .sum()
                    .eq(persistencia)
                )
            )
        )

        # --------------------------------------------------
        # 3. RUL TERMINAL
        # --------------------------------------------------
        pipeline_rul = PIPELINES[perfil]["rul"]
        features_rul = pipeline_rul["features"]
        window_size = int(pipeline_rul["window_size"])
        scaler_rul_perfil = pipeline_rul.get("scaler")
        
        ventanas = []
        ids_con_rul = []

        for (motor_id, motor_df) in datos.groupby("unit"):
            motor_df = motor_df.sort_values("cycle")

            # Algunos motores oficiales FD002 tienen menos de 30 ciclos
            if (len(motor_df) < window_size):
                continue

            ventana = motor_df[features_rul].tail(window_size)

            if (scaler_rul_perfil is not None):
                ventana_modelo = scaler_rul_perfil.transform(ventana)
            else:
                ventana_modelo = ventana.to_numpy(dtype=float)

            ventanas.append(ventana_modelo)
            ids_con_rul.append(int(motor_id))

        rul_por_motor = {}

        if ventanas:
            X_lstm = np.asarray(ventanas,dtype=np.float32)
            predicciones = pipeline_rul["model"].predict(X_lstm, verbose=0).ravel()

            if not np.isfinite(predicciones).all():
                raise ValueError(
                    "Se han obtenido "
                    "predicciones RUL "
                    "no válidas."
                )

            rul_por_motor = {motor_id: float(rul_predicha) for (motor_id, rul_predicha) in zip(ids_con_rul, predicciones)}

        # --------------------------------------------------
        # 4. RESUMEN POR MOTOR
        # --------------------------------------------------
        ultimos = datos.groupby("unit").tail(1).set_index("unit")
        ha_tenido_alerta = datos.groupby("unit")["alerta_activa"].any()

        for motor_id in (sorted(datos["unit"].unique().tolist())):
            ultima = ultimos.loc[motor_id]
            resultado_motor = {
                "motor_id": int(motor_id),
                "motor_original": int(ultima["source_unit"]),
                "origen": str(ultima["origin"]),
                "es_referencia": bool(ultima["is_reference"]),
                "profile": perfil,
                "profile_name": PIPELINES[perfil]["profile_name"],
                "ultimo_ciclo": int(ultima["cycle"]),
                "anomaly_score": float(ultima["anomaly_score"]),
                "anomaly_threshold": umbral,
                "alerta_activa": bool(ultima["alerta_activa"]),
                "ha_tenido_alertas": bool(ha_tenido_alerta.loc[motor_id]),
                "rul_disponible": int(motor_id) in rul_por_motor,
                "rul_estimada": rul_por_motor.get(int(motor_id))
            }

            # Régimen actual únicamente para FD002
            if perfil == "FD002":
                resultado_motor["operating_regime"] = int(ultima["operating_regime"])
            
            resultados_perfiles.append(resultado_motor)

    # ==================================================
    # COMBINAR LOS DOS PERFILES
    # ==================================================
    lista_motores = resultados_perfiles

    # RUL disponibles únicamente
    rul_disponibles = [motor["rul_estimada"] for motor in lista_motores if motor["rul_estimada"] is not None]
    total_alertas = sum(motor["alerta_activa"]for motor in lista_motores)

    if rul_disponibles:
        rul_media = float(np.mean(rul_disponibles))
        motores_con_rul = [motor for motor in lista_motores if motor["rul_estimada"] is not None]
        motor_min_rul = min(motores_con_rul,key=lambda motor: (motor["rul_estimada"]))
        motor_menor_rul = {
            "motor_id": motor_min_rul["motor_id"],
            "rul_estimada": motor_min_rul["rul_estimada"]
        }
    else:
        rul_media = None
        motor_menor_rul = None

    # --------------------------------------------------
    # ORDENAR
    # --------------------------------------------------
    # Los motores sin RUL quedan al final
    lista_motores.sort(
        key=lambda motor: (
            motor["rul_estimada"] is None,
            motor["rul_estimada"] if motor["rul_estimada"] is not None else float("inf"),
            motor["motor_id"]
        )
    )

    perfiles = {perfil: sum(motor["profile"] == perfil for motor in lista_motores) for perfil in PIPELINES}
    return {
        "dataset": (
            "NASA C-MAPSS "
            "FD001 + FD002"
        ),
        "total_motores": len(lista_motores),
        "perfiles": perfiles,
        "motores_alerta_activa": int(total_alertas),
        "motores_rul_disponible": len(rul_disponibles),
        "motores_rul_no_disponible": len(lista_motores) - len(rul_disponibles),
        "rul_media_estimada": rul_media,
        "motor_menor_rul": motor_menor_rul,
        "motores": lista_motores
    }

@app.get("/api/flota/resumen")
def obtener_resumen_flota():
    return calcular_resumen_flota_multipipeline()

# --------------------------------------------------
# FRONTEND
# --------------------------------------------------
FRONTEND_DIR = BASE_DIR / "frontend"

# Servir archivos CSS y JavaScript
app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

# Mostrar la aplicación web
@app.get("/dashboard")
def mostrar_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")

# --------------------------------------------------
# HISTORIAL DE PREDICCIONES DE RUL
# --------------------------------------------------
@lru_cache(maxsize=512)
def calcular_historial_rul(motor_id: int):
    motor_df = obtener_datos_motor(motor_id)

    try:
        resultado = inferir_historial_rul_motor(motor_df)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
    
    return resultado["historial"]

@app.get("/api/motores/{motor_id}/rul/historial")
def obtener_historial_rul(motor_id: int,ciclo: int | None = None):
    motor_df = obtener_datos_motor(motor_id)
    perfil = obtener_perfil_motor(motor_df)
    ultimo_ciclo = int(motor_df["cycle"].max())

    if ciclo is None:
        ciclo = ultimo_ciclo

    if (ciclo < 1 or ciclo > ultimo_ciclo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"El ciclo debe estar entre "
                f"1 y {ultimo_ciclo}."
            )
        )

    historial_completo = calcular_historial_rul(motor_id)
    historial = [registro for registro in historial_completo if registro["cycle"] <= ciclo]
    window_size = int(PIPELINES[perfil]["rul"]["window_size"])
    respuesta = {
        "motor_id": motor_id,
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "ciclo_consultado": ciclo,
        "ultimo_ciclo_disponible": ultimo_ciclo,
        "modelo": "LSTM",
        "ventana_temporal": window_size,
        "total_predicciones": len(historial),
        "historial": historial
    }

    if (perfil == "FD002" and historial):
        respuesta["operating_regime"] = int(historial[-1]["operating_regime"])
        
    return respuesta

# --------------------------------------------------
# RUL REAL DE LOS DATASETS NASA DE REFERENCIA
# --------------------------------------------------
RUL_REAL_PATHS = {
    "FD001": DATA_DIR / "RUL_FD001.txt",
    "FD002": DATA_DIR / "RUL_FD002.txt"
}

rul_finales_referencia = {}

for perfil, ruta in (RUL_REAL_PATHS.items()):
    valores = pd.read_csv(ruta, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    rul_finales_referencia[perfil] = valores

# Comprobaciones estructurales
assert len(rul_finales_referencia["FD001"]) == 100
assert len(rul_finales_referencia["FD002"]) == 259

# --------------------------------------------------
# ENDPOINT: EVALUACIÓN DE PREDICCIONES RUL
# --------------------------------------------------
@app.get("/api/motores/{motor_id}/evaluacion")
def evaluar_motor(motor_id: int, ciclo: int | None = None):
    # --------------------------------------------------
    # 1. RECUPERAR MOTOR
    # --------------------------------------------------
    motor_df = obtener_datos_motor(motor_id)

    if not bool(motor_df["is_reference"].iloc[0]):
        raise HTTPException(
            status_code=400,
            detail=(
                "El modo evaluación no está "
                "disponible para este motor "
                "porque no existe un valor real "
                "de RUL de referencia."
            )
        )

    perfil = obtener_perfil_motor(motor_df)
    source_unit = int(motor_df["source_unit"].iloc[0])
    ultimo_ciclo = int(motor_df["cycle"].max())

    if ciclo is None:
        ciclo = ultimo_ciclo

    if (ciclo < 1 or ciclo > ultimo_ciclo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"El ciclo debe estar entre "
                f"1 y {ultimo_ciclo}."
            )
        )

    # --------------------------------------------------
    # 2. RECUPERAR RUL REAL TERMINAL
    # --------------------------------------------------
    valores_reales = rul_finales_referencia[perfil]

    if (source_unit < 1 or source_unit > len(valores_reales)):
        raise HTTPException(
            status_code=500,
            detail=(
                "No se ha podido localizar "
                "la RUL real del motor."
            )
        )

    rul_final_real = float(valores_reales[source_unit - 1])

    # --------------------------------------------------
    # 3. RECONSTRUIR RUL REAL DEL CICLO
    # --------------------------------------------------
    rul_real = rul_final_real + (ultimo_ciclo - ciclo)
    rul_cap_motor = float(PIPELINES[perfil]["rul"]["rul_cap"])
    rul_real_limitada = min(rul_real, rul_cap_motor)

    # --------------------------------------------------
    # 4. PREDICCIÓN
    # --------------------------------------------------
    resultado_rul = obtener_rul(motor_id=motor_id, ciclo=ciclo)
    rul_predicha = resultado_rul["rul_estimada"]

    # --------------------------------------------------
    # 5. ERROR
    # --------------------------------------------------
    if resultado_rul["disponible"]:
        error = rul_predicha - rul_real_limitada
        error_absoluto = abs(error)
    else:
        error = None
        error_absoluto = None

    # --------------------------------------------------
    # 6. RESPUESTA
    # --------------------------------------------------
    respuesta = {
        "motor_id": motor_id,
        "motor_original": source_unit,
        "profile": perfil,
        "profile_name": PIPELINES[perfil]["profile_name"],
        "ciclo_consultado": ciclo,
        "modo": "evaluacion",
        "prediccion_disponible": resultado_rul["disponible"],
        "rul_predicha": rul_predicha,
        "rul_real": float(rul_real),
        "rul_real_limitada": float(rul_real_limitada),
        "error": error,
        "error_absoluto": error_absoluto,
        "rul_cap": rul_cap_motor,
        "unidad": "ciclos"
    }

    if (perfil == "FD002" and "operating_regime" in resultado_rul):
        respuesta["operating_regime"] = int(resultado_rul["operating_regime"])

    return respuesta

# --------------------------------------------------
# ENDPOINT: VALIDAR CSV DE NUEVOS MOTORES
# --------------------------------------------------
@app.post("/api/importaciones/validar")
async def validar_importacion(origen: str = Form(...), archivo: UploadFile = File(...)):
    origen = origen.strip()

    if not origen:
        raise HTTPException(
            status_code=400,
            detail=(
                "Debes indicar un nombre "
                "para el origen de los motores."
            )
        )

    if len(origen) > 100:
        raise HTTPException(
            status_code=400,
            detail=(
                "El nombre del origen no puede "
                "superar los 100 caracteres."
            )
        )

    nombre_archivo = archivo.filename or ""

    if not (nombre_archivo.lower().endswith(".csv")):
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo seleccionado "
                "debe tener formato CSV."
            )
        )

    contenido = await archivo.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo está vacío."
            )
        )

    try:
        datos_csv = pd.read_csv(io.BytesIO(contenido))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "No se ha podido leer "
                "el archivo como un "
                "CSV válido."
            )
        )

    (datos_validados, motores) = validar_csv_motores(datos_csv)
    perfiles = {}

    for motor in motores:
        perfil = motor["profile"]
        perfiles[perfil] = (perfiles.get(perfil, 0) + 1)

    return {
        "valido": True,
        "origen": origen,
        "archivo": nombre_archivo,
        "motores_detectados": len(motores),
        "observaciones": int(len(datos_validados)),
        "columnas": int(len(datos_validados.columns)),
        "sensores_disponibles": len([columna for columna in datos_validados.columns if columna.startswith("sensor_")]),
        "perfiles_detectados": perfiles,
        "motores": motores
    }

# --------------------------------------------------
# ENDPOINT: IMPORTAR NUEVOS MOTORES
# --------------------------------------------------
@app.post("/api/importaciones/importar")
async def importar_motores(origen: str = Form(...), archivo: UploadFile = File(...)):
    global df_motores
    # --------------------------------------------------
    # 1. ORIGEN
    # --------------------------------------------------
    origen = origen.strip()

    if not origen:
        raise HTTPException(
            status_code=400,
            detail=(
                "Debes indicar un nombre "
                "para el origen de los motores."
            )
        )

    if len(origen) > 100:
        raise HTTPException(
            status_code=400,
            detail=(
                "El nombre del origen no puede "
                "superar los 100 caracteres."
            )
        )

    # --------------------------------------------------
    # 2. ARCHIVO
    # --------------------------------------------------
    nombre_archivo_original = archivo.filename or ""

    if not (nombre_archivo_original.lower().endswith(".csv")):
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo seleccionado "
                "debe tener formato CSV."
            )
        )

    contenido = await archivo.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo está vacío."
            )
        )

    try:
        datos_csv = pd.read_csv(io.BytesIO(contenido))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "No se ha podido leer "
                "el archivo como un "
                "CSV válido."
            )
        )

    # --------------------------------------------------
    # 3. VALIDAR Y DETECTAR PERFILES
    # --------------------------------------------------
    (datos_validados, motores) = validar_csv_motores(datos_csv)
    motores_originales = sorted(datos_validados["unit"].unique().tolist())
    perfiles_por_motor = {int(motor["motor_original"]): motor["profile"] for motor in motores}

    # --------------------------------------------------
    # 4. NUEVOS IDs INTERNOS
    # --------------------------------------------------
    if df_motores.empty:
        siguiente_id = 1
    else:
        siguiente_id = int(df_motores["unit"].max()) + 1

    mapa_ids = {}

    for motor_original in (motores_originales):
        mapa_ids[int(motor_original)] = siguiente_id
        siguiente_id += 1

    # --------------------------------------------------
    # 5. METADATOS
    # --------------------------------------------------
    datos_validados["source_unit"] = datos_validados["unit"].astype(int)
    datos_validados["profile"] = datos_validados["source_unit"].map(perfiles_por_motor)
    datos_validados["unit"] = datos_validados["source_unit"].map(mapa_ids).astype(int)
    datos_validados["origin"] = origen
    datos_validados["is_reference"] = False

    # --------------------------------------------------
    # 6. GUARDAR IMPORTACIÓN
    # --------------------------------------------------
    import_id = uuid4().hex
    nombre_archivo_guardado = f"import_{import_id}.csv"
    ruta_destino = IMPORTED_DATA_DIR / nombre_archivo_guardado
    
    datos_validados.to_csv( ruta_destino, index=False)

    # --------------------------------------------------
    # 7. REGISTRY
    # --------------------------------------------------
    registro = cargar_registro_importaciones()
    nueva_importacion = {
        "id": import_id,
        "origin": origen,
        "original_filename": nombre_archivo_original,
        "filename": nombre_archivo_guardado,
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "motors": int(len(motores_originales)),
        "observations": int(len(datos_validados)),
        "unit_mapping": {str(origen_id): int(nuevo_id) for (origen_id, nuevo_id) in mapa_ids.items()},
        "predictive_profiles": {str(origen_id): (perfiles_por_motor[int(origen_id)]) for origen_id in motores_originales}
    }

    registro.append(nueva_importacion)
    guardar_registro_importaciones(registro)

    # --------------------------------------------------
    # 8. RECARGAR FLOTA COMPLETA
    # --------------------------------------------------
    df_motores = cargar_datos_operativos()
    calcular_resumen_flota_multipipeline.cache_clear()
    calcular_historial_rul.cache_clear()

    # --------------------------------------------------
    # 9. RESPUESTA
    # --------------------------------------------------
    motores_importados = []

    for motor in motores:
        motor_original = int(motor["motor_original"])
        motores_importados.append(
            {
                "motor_original": motor_original,
                "motor_id": int(mapa_ids[motor_original]),
                "profile": motor["profile"],
                "profile_name": motor["profile_name"],
                "ciclos": int(motor["ciclos"])
            }
        )

    return {
        "importado": True,
        "importacion_id": import_id,
        "origen": origen,
        "archivo": nombre_archivo_original,
        "motores_importados": len(motores_importados),
        "observaciones": int(len(datos_validados)),
        "motores": motores_importados
    }

# --------------------------------------------------
# LOCALIZAR MOTOR EXTERNO PERSISTIDO
# --------------------------------------------------
def localizar_importacion_motor(motor_id: int):
    """
    Localiza la importación a la que pertenece un motor externo utilizando exclusivamente su ID interno
    source_unit se conserva como metadato, pero no se utiliza para localizar el motor
    """
    registro = cargar_registro_importaciones()
    coincidencias = []

    for indice, importacion in enumerate(registro):
        mapa_ids = importacion.get("unit_mapping", {})

        for (source_unit, id_interno) in mapa_ids.items():
            if int(id_interno) == int(motor_id):
                coincidencias.append((indice, importacion, int(source_unit)))

    if not coincidencias:
        raise HTTPException(
            status_code=404,
            detail=(
                "No se ha encontrado una "
                "importación asociada al "
                f"motor {motor_id}."
            )
        )

    if len(coincidencias) != 1:
        raise HTTPException(
            status_code=500,
            detail=(
                "El motor aparece asociado "
                "a más de una importación."
            )
        )

    (indice, importacion, source_unit) = coincidencias[0]

    return (registro, indice, importacion, source_unit)

# --------------------------------------------------
# ENDPOINT: ACTUALIZAR MOTOR EXTERNO
# --------------------------------------------------
@app.post("/api/motores/{motor_id}/actualizar")
async def actualizar_motor_externo(motor_id: int, archivo: UploadFile = File(...)):
    global df_motores
    # --------------------------------------------------
    # 1. COMPROBAR MOTOR ACTUAL
    # --------------------------------------------------
    motor_actual = obtener_datos_motor(motor_id)

    if bool(motor_actual["is_reference"].iloc[0]):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los motores NASA de referencia "
                "son de solo lectura y no pueden "
                "actualizarse."
            )
        )

    perfil_previo = str(motor_actual["profile"].iloc[0])
    source_unit_actual = int(motor_actual["source_unit"].iloc[0])
    origen_actual = str(motor_actual["origin"].iloc[0])
    ultimo_ciclo_actual = int(motor_actual["cycle"].max())

    # --------------------------------------------------
    # 2. LEER CSV
    # --------------------------------------------------
    nombre_archivo = archivo.filename or ""

    if not nombre_archivo.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo seleccionado "
                "debe tener formato CSV."
            )
        )

    contenido = await archivo.read()

    if not contenido:
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo está vacío."
            )
        )

    try:
        datos_csv = pd.read_csv(io.BytesIO(contenido))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "No se ha podido leer "
                "el archivo como un "
                "CSV válido."
            )
        )

    # --------------------------------------------------
    # 3. VALIDAR HISTORIAL COMPLETO
    # --------------------------------------------------
    (datos_validados, motores) = validar_csv_motores(datos_csv)

    if len(motores) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "La actualización debe contener "
                "exactamente un motor."
            )
        )

    source_unit_csv = int(datos_validados["unit"].iloc[0])

    if (source_unit_csv != source_unit_actual):
        raise HTTPException(
            status_code=400,
            detail=(
                "El identificador del motor "
                "contenido en el CSV no coincide "
                "con el motor que se está "
                "actualizando."
            )
        )

    ultimo_ciclo_nuevo = int(datos_validados["cycle"].max())

    if (ultimo_ciclo_nuevo <= ultimo_ciclo_actual):
        raise HTTPException(
            status_code=400,
            detail=(
                "La actualización debe incluir "
                "al menos un ciclo nuevo."
            )
        )

    # --------------------------------------------------
    # 4. COMPROBAR QUE EL HISTORIAL ANTERIOR NO HA SIDO MODIFICADO
    # --------------------------------------------------
    columnas_comparacion = [
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        *SENSORES_MODELO_COMUNES
    ]

    historial_actual = motor_actual[columnas_comparacion].sort_values("cycle").reset_index(drop=True)
    historial_nuevo_anterior = datos_validados[datos_validados["cycle"] <= ultimo_ciclo_actual][columnas_comparacion].sort_values("cycle").reset_index(drop=True)

    if (len(historial_actual) != len(historial_nuevo_anterior)):
        raise HTTPException(
            status_code=400,
            detail=(
                "El CSV debe contener el "
                "historial completo del motor."
            )
        )

    if not np.array_equal(historial_actual["cycle"].to_numpy(),historial_nuevo_anterior["cycle"].to_numpy()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los ciclos históricos del motor "
                "no coinciden con los ya "
                "almacenados."
            )
        )

    columnas_valores = [columna for columna in columnas_comparacion if columna != "cycle"]

    if not np.allclose(historial_actual[columnas_valores].to_numpy(dtype=float), historial_nuevo_anterior[columnas_valores].to_numpy(dtype=float), rtol=0.0, atol=1e-9):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los datos de los ciclos ya "
                "almacenados han sido modificados. "
                "La actualización solo puede "
                "añadir nuevos ciclos."
            )
        )

    # --------------------------------------------------
    # 5. RESOLVER PERFIL PREDICTIVO
    # --------------------------------------------------
    perfil_detectado = str(motores[0]["profile"])

    # Una vez FD002, nunca vuelve a FD001
    if perfil_previo == "FD002":
        perfil_final = "FD002"
    else:
        perfil_final = perfil_detectado

    transicion_perfil = bool(perfil_previo == "FD001"and perfil_final == "FD002")

    # --------------------------------------------------
    # 6. PREPARAR NUEVA TRAYECTORIA
    # --------------------------------------------------
    datos_validados["source_unit"] = source_unit_actual
    datos_validados["unit"] = int(motor_id)
    datos_validados["origin"] = origen_actual
    datos_validados["is_reference"] = False
    datos_validados["profile"] = perfil_final

    # --------------------------------------------------
    # 7. LOCALIZAR ARCHIVO PERSISTIDO
    # --------------------------------------------------
    (registro, indice_registro, importacion, source_unit_registro) = localizar_importacion_motor(motor_id)

    if (source_unit_registro != source_unit_actual):
        raise HTTPException(
            status_code=500,
            detail=(
                "El registro de importación "
                "no coincide con los datos "
                "del motor."
            )
        )

    nombre_archivo_guardado = importacion.get("filename")

    if not nombre_archivo_guardado:
        raise HTTPException(
            status_code=500,
            detail=(
                "La importación no contiene "
                "un archivo persistido válido."
            )
        )

    ruta_importacion = IMPORTED_DATA_DIR / nombre_archivo_guardado

    if not ruta_importacion.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "No se ha encontrado el archivo "
                "persistido del motor."
            )
        )

    datos_importacion = pd.read_csv(ruta_importacion)

    # --------------------------------------------------
    # 8. REEMPLAZAR ÚNICAMENTE ESTE MOTOR
    # --------------------------------------------------
    resto_importacion = datos_importacion[datos_importacion["unit"].astype(int) != int(motor_id)].copy()
    datos_actualizados = pd.concat([resto_importacion, datos_validados], ignore_index=True, sort=False).sort_values(["unit", "cycle"]).reset_index(drop=True)
    ruta_temporal = ruta_importacion.with_name(ruta_importacion.stem + "_tmp.csv")
    datos_actualizados.to_csv(ruta_temporal, index=False)
    ruta_temporal.replace(ruta_importacion)

    # --------------------------------------------------
    # 9. ACTUALIZAR REGISTRY.JSON
    # --------------------------------------------------
    perfiles_guardados = importacion.setdefault("predictive_profiles", {})
    perfiles_guardados[str(source_unit_actual)] = perfil_final
    importacion["updated_at"] = datetime.now().isoformat(timespec="seconds")
    importacion["observations"] = int(len(datos_actualizados))
    registro[indice_registro] = importacion
    guardar_registro_importaciones(registro)

    # --------------------------------------------------
    # 10. RECARGAR FLOTA
    # --------------------------------------------------
    df_motores = cargar_datos_operativos()
    calcular_resumen_flota_multipipeline.cache_clear()
    calcular_historial_rul.cache_clear()

    # --------------------------------------------------
    # 11. RESPUESTA
    # --------------------------------------------------
    respuesta = {
        "actualizado": True,
        "motor_id": int(motor_id),
        "motor_original": source_unit_actual,
        "origen": origen_actual,
        "ciclos_anteriores": ultimo_ciclo_actual,
        "ciclos_actuales": ultimo_ciclo_nuevo,
        "ciclos_añadidos": int(ultimo_ciclo_nuevo - ultimo_ciclo_actual),
        "profile_anterior": perfil_previo,
        "profile_detectado": perfil_detectado,
        "profile": perfil_final,
        "profile_name": PIPELINES[perfil_final]["profile_name"],
        "transicion_perfil": transicion_perfil
    }

    if perfil_final == "FD002":
        datos_regimen = asignar_regimenes_fd002(datos_validados)
        respuesta["operating_regime"] = int(datos_regimen["operating_regime"].iloc[-1])

    return respuesta