
/* =====================================================
   PREDICTIVE MAINTENANCE
   Vista general de la flota
===================================================== */
const API_URL = "/api/flota/resumen";

// Guardar los motores recibidos desde FastAPI
let motoresFlota = [];

// Estado de la monitorización
let motorSeleccionado = null;
let cicloActual = null;
let ultimoCicloMotor = null;
let motorSeleccionadoEsReferencia = true;

// Estado de la importación de CSV
let importacionValidada = false;

// Evitar que respuestas antiguas sustituyan a las recientes
let ultimaSolicitudEstado = 0;

// Estado de la reproducción automática
let reproduciendo = false;
let temporizadorReproduccion = null;
let sesionReproduccion = 0;

// Gráfico de Isolation Forest
let graficoAnomalias = null;

// Gráfico de sensores
let graficoSensores = null;

// Historial correspondiente al ciclo consultado
let datosHistorialActual = null;

// Historial de anomalías del ciclo consultado
let datosAnomaliasActual = null;

// Gráfico e historial de prognosis
let graficoRul = null;
let datosRulActual = null;

let ultimaSolicitudEvaluacion = 0;

// Datos de referencia para comparar la RUL
let datosEvaluacionActual = null;

/* Formatear números para la interfaz */
function formatearNumero(valor, decimales = 2) {
    return Number(valor).toLocaleString("es-ES", {
        minimumFractionDigits: decimales,
        maximumFractionDigits: decimales
    });
}

/* Crear una etiqueta de estado */
function crearEstado(texto, clase) {
    const etiqueta = document.createElement("span");
    etiqueta.className = `status-badge ${clase}`;
    etiqueta.textContent = texto;
    return etiqueta;
}

/* Actualizar los cuatro indicadores */
function actualizarIndicadores(datos) {
    document.getElementById("kpi-total").textContent = datos.total_motores;
    document.getElementById("kpi-alertas").textContent = datos.motores_alerta_activa;
    document.getElementById("kpi-rul-media").textContent = formatearNumero(datos.rul_media_estimada);
    document.getElementById("kpi-motor-critico").textContent = `Motor ${datos.motor_menor_rul.motor_id}`;
    document.getElementById("kpi-motor-critico-rul").textContent =`${formatearNumero(datos.motor_menor_rul.rul_estimada)} ciclos estimados`;
}

/* Crear una celda de texto */
function crearCelda(fila, contenido, clase = "") {
    const celda = document.createElement("td");
    celda.textContent = contenido;
    if (clase) {
        celda.className = clase;
    }
    fila.appendChild(celda);
    return celda;
}

/* Construir la tabla con los motores reales */
function actualizarTabla(motores) {
    const tbody = document.getElementById("tabla-motores");
    tbody.replaceChildren();

    // Mostrar mensaje si los filtros no encuentran motores
    if (motores.length === 0) {
        const fila = document.createElement("tr");
        const celda = document.createElement("td");
        celda.colSpan = 7;
        celda.textContent = "No se encontraron motores con los filtros seleccionados.";
        fila.appendChild(celda);
        tbody.appendChild(fila);
        return;
    }

    motores.forEach(motor => {
        const fila = document.createElement("tr");

        // Identificador: enlace a monitorización
        const celdaMotor = document.createElement("td");
        const enlaceMotor = document.createElement("a");

        enlaceMotor.href = "#monitorizacion";
        enlaceMotor.className = "motor-link";
        enlaceMotor.textContent = `Motor ${motor.motor_id}`;
        enlaceMotor.addEventListener("click", () => {seleccionarMotor(motor.motor_id);});
        celdaMotor.appendChild(enlaceMotor);
        fila.appendChild(celdaMotor);

        // Mostrar el identificador original cuando el backend haya reasignado el ID del motor
        if (!motor.es_referencia && motor.motor_original !== motor.motor_id) {
            const identificadorOriginal = document.createElement("small");
            identificadorOriginal.className = "motor-original-id";
            identificadorOriginal.textContent = `ID original: ${motor.motor_original}`;
            celdaMotor.appendChild(identificadorOriginal);
        }

        // Origen del motor
        crearCelda(fila, motor.origen, "origin-cell");

        // Perfil predictivo
        crearCelda(fila, motor.profile_name || motor.profile);

        // Último ciclo disponible
        crearCelda(fila, motor.ultimo_ciclo);

        // Alerta actual
        const celdaAlerta = document.createElement("td");
        celdaAlerta.appendChild(
            motor.alerta_activa
                ? crearEstado("Activa", "status-danger")
                : crearEstado("Sin alerta", "status-success")
        );
        fila.appendChild(celdaAlerta);

        // Alertas históricas
        const celdaHistorial = document.createElement("td");
        celdaHistorial.appendChild(
            motor.ha_tenido_alertas
                ? crearEstado("Registradas", "status-warning")
                : crearEstado("Ninguna", "status-success")
        );
        fila.appendChild(celdaHistorial);

        // RUL estimada
        const textoRul =
            motor.rul_disponible &&
            motor.rul_estimada !== null
                ? `${formatearNumero(
                    motor.rul_estimada
                )} ciclos` : "No disponible";

        crearCelda(fila, textoRul);
        tbody.appendChild(fila);
    });
}

function actualizarFiltroOrigen() {
    const selector = document.getElementById("filtro-origen");
    const valorAnterior = selector.value;
    const origenes = [
        ...new Set(
            motoresFlota
                .map(motor => motor.origen)
                .filter(Boolean)
        )
    ].sort((a, b) => a.localeCompare(b, "es"));
    selector.replaceChildren();

    const opcionTodos = document.createElement("option");
    opcionTodos.value = "todos";
    opcionTodos.textContent = "Todos los orígenes";
    selector.appendChild(opcionTodos);

    origenes.forEach(origen => {
        const opcion = document.createElement("option");
        opcion.value = origen;
        opcion.textContent = origen;
        selector.appendChild(opcion);
    });

    const valorSigueDisponible = [
        ...selector.options
    ].some(
        opcion => opcion.value === valorAnterior
    );

    selector.value = valorSigueDisponible ? valorAnterior : "todos";
}

/* =====================================================
   BUSQUEDA, FILTROS Y ORDENACION
===================================================== */
function aplicarBusqueda() {
    const busqueda = document.getElementById("buscar-motor").value.trim();
    const filtroAlertas = document.getElementById("filtro-alertas").value;
    const filtroOrigen = document.getElementById("filtro-origen").value;
    const orden = document.getElementById("orden-motores").value;

    // 1. Filtrar por identificador y alertas
    let motoresFiltrados = motoresFlota.filter(motor => {

        const coincideBusqueda = String(
            motor.motor_id
        ).includes(busqueda);

        const coincideOrigen =
            filtroOrigen === "todos" ||
            motor.origen === filtroOrigen;

        let coincideAlerta = true;

        if (filtroAlertas === "activas") {
            coincideAlerta = motor.alerta_activa;
        }

        else if (filtroAlertas === "sin-alerta") {
            coincideAlerta = !motor.alerta_activa;
        }

        else if (filtroAlertas === "historicas") {
            coincideAlerta = motor.ha_tenido_alertas;
        }

        return coincideBusqueda && coincideOrigen && coincideAlerta;
    });

    // 2. Ordenar resultados
    const criterios = {

        "rul-asc": (a, b) => {

            if (
                a.rul_estimada === null &&
                b.rul_estimada === null
            ) {
                return 0;
            }

            if (a.rul_estimada === null) {
                return 1;
            }

            if (b.rul_estimada === null) {
                return -1;
            }

            return (
                a.rul_estimada -
                b.rul_estimada
            );
        },

        "rul-desc": (a, b) => {

            if (
                a.rul_estimada === null &&
                b.rul_estimada === null
            ) {
                return 0;
            }

            if (a.rul_estimada === null) {
                return 1;
            }

            if (b.rul_estimada === null) {
                return -1;
            }

            return (
                b.rul_estimada -
                a.rul_estimada
            );
        },

        "ciclo-desc": (a, b) =>
            b.ultimo_ciclo -
            a.ultimo_ciclo,

        "ciclo-asc": (a, b) =>
            a.ultimo_ciclo -
            b.ultimo_ciclo,

        "motor-asc": (a, b) =>
            a.motor_id -
            b.motor_id
    };

    const comparador = criterios[orden];

    if (comparador) {
        motoresFiltrados.sort((a, b) =>
            comparador(a, b) || a.motor_id - b.motor_id
        );
    }

    // 3. Actualizar tabla
    actualizarTabla(motoresFiltrados);

    // 4. Actualizar contador
    const mensaje = document.getElementById(
        "mensaje-flota"
    );

    mensaje.textContent =
        `${motoresFiltrados.length} de ` +
        `${motoresFlota.length} motores mostrados.`;
}

/* Consultar el backend */
async function cargarFlota() {
    const mensaje = document.getElementById("mensaje-flota");

    mensaje.textContent = "Cargando información de la flota...";

    try {
        const respuesta = await fetch(API_URL);

        if (!respuesta.ok) {
            throw new Error(
                `Error HTTP ${respuesta.status}`
            );
        }

        const datos = await respuesta.json();

        if (
            !Array.isArray(datos.motores) ||
            datos.motores.length !== datos.total_motores
        ) {
            throw new Error(
                "La respuesta de la flota está incompleta."
            );
        }

        actualizarIndicadores(datos);
        motoresFlota = datos.motores;
        actualizarFiltroOrigen();
        aplicarBusqueda();

    } catch (error) {
        console.error(
            "Error al cargar la flota:",
            error
        );

        mensaje.textContent =
            "No se han podido cargar los datos. " +
            "Comprueba que FastAPI esté funcionando.";
    }
}

/* =====================================================
   MONITORIZACION POR CICLOS
===================================================== */
function limpiarActualizacionMotor() {
    const archivo = document.getElementById("update-file");
    const estado = document.getElementById("update-status");

    if (archivo) {
        archivo.value = "";
    }

    if (estado) {
        estado.textContent = "";
    }
}

function actualizarVisibilidadActualizacion() {
    const seccion = document.getElementById("update-section");

    if (!seccion) {
        return;
    }

    const panel = document.querySelector(".monitor-panel");
    const vistaResumen = panel.dataset.vista === "resumen";
    const motorExterno = motorSeleccionado !== null && !motorSeleccionadoEsReferencia;

    seccion.hidden = !(vistaResumen && motorExterno);
}

/* Seleccionar un motor desde la tabla */
function seleccionarMotor(motorId) {
    detenerReproduccion();

    document.getElementById(
        "monitor-context"
    ).hidden = true;

    // Desactivar evaluación al cambiar de motor
    ultimaSolicitudEvaluacion++;
    datosEvaluacionActual = null;

    document.getElementById(
        "evaluation-toggle"
    ).checked = false;

    document.getElementById(
        "evaluation-results"
    ).hidden = true;

    document.getElementById(
        "evaluation-section"
    ).hidden = true;

    document.getElementById(
        "evaluation-message"
    ).textContent = "";

    document.getElementById(
        "anomaly-chart-section"
    ).hidden = true;

    // Ocultar sensores del motor anterior
    document.getElementById(
        "sensor-chart-section"
    ).hidden = true;

    // Limpiar historial anterior
    datosHistorialActual = null;
    datosAnomaliasActual = null;
    datosRulActual = null;

    document.getElementById(
        "rul-chart-section"
    ).hidden = true;

    // Mostrar las pestañas
    document.getElementById(
        "monitor-tabs"
    ).hidden = false;

    // Cada motor comienza en Resumen
    cambiarVistaMonitorizacion("resumen");

    const motor = motoresFlota.find(
        item => item.motor_id === motorId
    );

    if (!motor) {
        console.error("Motor no encontrado:", motorId);
        return;
    }

    motorSeleccionadoEsReferencia =
        Boolean(
            motor.es_referencia
        );

    const interruptorEvaluacion =
        document.getElementById(
            "evaluation-toggle"
        );

    interruptorEvaluacion.disabled =
        !motorSeleccionadoEsReferencia;

    document.querySelector(
        ".evaluation-toggle-label"
    ).classList.toggle(
        "is-disabled",
        !motorSeleccionadoEsReferencia
    );

    motorSeleccionado = motorId;
    ultimoCicloMotor = motor.ultimo_ciclo;

    // Limpiar cualquier actualización del motor anterior
    limpiarActualizacionMotor();

    // Mostrar la actualización solo para motores externos
    actualizarVisibilidadActualizacion();

    // Comenzar en el último ciclo disponible
    cicloActual = ultimoCicloMotor;

    // Configurar barra deslizante
    const slider = document.getElementById(
        "cycle-slider"
    );

    slider.min = 1;
    slider.max = ultimoCicloMotor;
    slider.value = cicloActual;

    // Mostrar controles
    document.getElementById(
        "cycle-controls"
    ).hidden = false;

    actualizarControlesCiclo();

    // Consultar el estado inicial
    consultarEstadoCiclo();
}


/* Actualizar barra, indicador y botones */
function actualizarControlesCiclo() {

    if (motorSeleccionado === null) {
        return;
    }

    // Actualizar la posición del deslizador
    const slider = document.getElementById(
        "cycle-slider"
    );

    slider.value = cicloActual;

    // Porcentaje recorrido del historial disponible
    const progreso = ultimoCicloMotor > 1
        ? ((cicloActual - 1) / (ultimoCicloMotor - 1)) * 100
        : 100;

    // Actualizar el relleno visual de la barra
    slider.style.setProperty(
        "--cycle-progress",
        `${progreso}%`
    );

    // Mejorar la descripción accesible del control
    slider.setAttribute(
        "aria-valuetext",
        `Ciclo ${cicloActual} de ${ultimoCicloMotor}`
    );

    document.getElementById(
        "cycle-indicator"
    ).textContent =
        `Ciclo ${cicloActual} / ${ultimoCicloMotor}`;

    document.getElementById(
        "cycle-prev"
    ).disabled = cicloActual <= 1;

    document.getElementById(
        "cycle-next"
    ).disabled = cicloActual >= ultimoCicloMotor;

    document.getElementById(
        "cycle-reset"
    ).disabled = cicloActual <= 1;

    actualizarBotonReproduccion();
}


/* Consultar los modelos en el ciclo seleccionado */
async function consultarEstadoCiclo() {

    if (motorSeleccionado === null) {
        return;
    }

    const motorId = motorSeleccionado;
    const ciclo = cicloActual;

    const solicitud = ++ultimaSolicitudEstado;

    // Invalidar evaluaciones anteriores
    ultimaSolicitudEvaluacion++;
    datosEvaluacionActual = null;

    const mensaje = document.getElementById(
        "monitor-mensaje"
    );

    const resumen = document.getElementById(
        "monitor-resumen"
    );

    mensaje.textContent =
        `Analizando motor ${motorId}, ciclo ${ciclo}...`;

    try {
        // Consultar estado, anomalías, sensores y prognosis
        const [
            respuesta,
            respuestaAnomalias,
            respuestaHistorial,
            respuestaRul
        ] = await Promise.all([

            fetch(
                `/api/motores/${motorId}/estado?ciclo=${ciclo}`
            ),

            fetch(
                `/api/motores/${motorId}/anomalias?ciclo=${ciclo}`
            ),

            fetch(
                `/api/motores/${motorId}/historial?ciclo=${ciclo}`
            ),

            fetch(
                `/api/motores/${motorId}/rul/historial?ciclo=${ciclo}`
            )

        ]);

        // Comprobar las cuatro respuestas
        if (
            !respuesta.ok ||
            !respuestaAnomalias.ok ||
            !respuestaHistorial.ok ||
            !respuestaRul.ok
        ) {
            throw new Error(
                `Error al consultar el motor ${motorId}`
            );
        }

        // Convertir las respuestas a JSON
        const [
            datos,
            datosAnomalias,
            datosHistorial,
            datosRul
        ] = await Promise.all([

            respuesta.json(),
            respuestaAnomalias.json(),
            respuestaHistorial.json(),
            respuestaRul.json()

        ]);

        // Ignorar respuestas de consultas anteriores
        if (solicitud !== ultimaSolicitudEstado) {
            return;
        }

        // Motor seleccionado
        document.getElementById(
            "monitor-id"
        ).textContent = `Motor ${datos.motor_id}`;

        // Perfil predictivo
        document.getElementById(
            "monitor-profile"
        ).textContent =
            datos.profile_name ||
            datos.profile;


        // Régimen operativo actual
        document.getElementById(
            "monitor-regime"
        ).textContent =
            datos.profile === "FD002" &&
            datos.operating_regime !== undefined
                ? `Régimen ${datos.operating_regime}`
                : "No aplica";

        // Ciclo actual
        document.getElementById(
            "monitor-ciclo"
        ).textContent =
            `${datos.ciclo_actual} / ${datos.ultimo_ciclo_disponible}`;

        // Estado de Isolation Forest
        document.getElementById(
            "monitor-alerta"
        ).textContent =
            datos.anomalias.alerta_activa
                ? "Alerta activa"
                : "Sin alerta activa";

        // Predicción LSTM
        document.getElementById(
            "monitor-rul"
        ).textContent =
            datos.prognosis.disponible
                ? `${formatearNumero(
                    datos.prognosis.rul_estimada
                )} ciclos`
                : "Datos insuficientes";

        mensaje.textContent =
            `Estado calculado hasta el ciclo ${ciclo}.`;

        resumen.hidden = false;

        // Actualizar el contexto del motor
        actualizarContextoMotor(datos);

        // Actualizar indicadores de Isolation Forest
        actualizarIndicadoresIA(datos.anomalias);

        // Conservar los historiales para los gráficos
        datosAnomaliasActual = datosAnomalias;
        datosHistorialActual = datosHistorial;
        datosRulActual = datosRul;

        // Consultar qué pestaña está activa
        const vistaActiva = document.querySelector(
            ".monitor-panel"
        ).dataset.vista;

        // Actualizar los gráficos de la pestaña visible

        if (vistaActiva === "sensores") {
            actualizarGraficoSensores();
        }

        if (vistaActiva === "ia") {

            // Actualizar siempre el historial de anomalías
            actualizarGraficoAnomalias(
                datosAnomaliasActual
            );

            // Actualizar la RUL directamente si no
            // estamos utilizando el modo evaluación.
            if (
                !document.getElementById(
                    "evaluation-toggle"
                ).checked
            ) {
                actualizarGraficoRul(
                    datosRulActual
                );
            }

        }

        // --------------------------------------------------
        // MODO EVALUACIÓN
        // --------------------------------------------------

        const seccionEvaluacion =
            document.getElementById(
                "evaluation-section"
            );

        const interruptorEvaluacion =
            document.getElementById(
                "evaluation-toggle"
            );

        const mensajeEvaluacion =
            document.getElementById(
                "evaluation-message"
            );

        seccionEvaluacion.hidden = false;

        interruptorEvaluacion.disabled =
            !motorSeleccionadoEsReferencia;

        document.querySelector(
            ".evaluation-toggle-label"
        ).classList.toggle(
            "is-disabled",
            !motorSeleccionadoEsReferencia
        );

        if (!motorSeleccionadoEsReferencia) {

            interruptorEvaluacion.checked = false;

            document.getElementById(
                "evaluation-results"
            ).hidden = true;

            datosEvaluacionActual = null;

            mensajeEvaluacion.textContent =
                "El modo evaluación no está disponible " +
                "para este motor porque el CSV importado " +
                "no incluye una RUL real de referencia.";

        } else {

            if (!interruptorEvaluacion.checked) {
                mensajeEvaluacion.textContent = "";
            }

            if (interruptorEvaluacion.checked) {

                await consultarEvaluacionCiclo(
                    motorId,
                    ciclo,
                    solicitud
                );
            }
        }

        return true;

    } catch (error) {

        if (solicitud !== ultimaSolicitudEstado) {
            return;
        }

        console.error(
            "Error al consultar el ciclo:",
            error
        );

        mensaje.textContent =
            "No se ha podido calcular el estado del motor.";

        return false;
    }
}


/* Cambiar a un ciclo concreto */
function irACiclo(nuevoCiclo) {
    if (motorSeleccionado === null) {
        return;
    }

    detenerReproduccion();

    // Evitar ciclos fuera del historial
    const cicloValidado = Math.max(
        1,
        Math.min(ultimoCicloMotor, nuevoCiclo)
    );

    if (cicloValidado === cicloActual) {
        return;
    }

    cicloActual = cicloValidado;

    actualizarControlesCiclo();
    consultarEstadoCiclo();
}


/* =====================================================
   REPRODUCCIÓN AUTOMÁTICA
===================================================== */

/* Actualizar el botón Reproducir / Pausar */
function actualizarBotonReproduccion() {

    const boton = document.getElementById(
        "playback-toggle"
    );

    boton.textContent = reproduciendo
        ? "⏸ Pausar"
        : "▶ Reproducir";

    boton.disabled =
        motorSeleccionado === null ||
        (
            !reproduciendo &&
            cicloActual >= ultimoCicloMotor
        );
}


/* Detener la reproducción */
function detenerReproduccion() {

    reproduciendo = false;

    // Invalidar la reproducción anterior
    sesionReproduccion++;

    clearTimeout(temporizadorReproduccion);

    temporizadorReproduccion = null;

    actualizarBotonReproduccion();
}


/* Programar el siguiente ciclo */
function programarSiguienteCiclo(sesion) {

    // Intervalos base según la velocidad elegida
    const intervalos = {
        1: 1200,
        2: 600,
        5: 240
    };

    const velocidad = Number(
        document.getElementById("playback-speed").value
    );

    const intervalo = intervalos[velocidad] ?? 1200;

    temporizadorReproduccion = setTimeout(async () => {

        // Comprobar que seguimos reproduciendo
        if (
            !reproduciendo ||
            sesion !== sesionReproduccion
        ) {
            return;
        }

        // Detener al alcanzar el último ciclo
        if (cicloActual >= ultimoCicloMotor) {
            detenerReproduccion();
            return;
        }

        // Avanzar un ciclo
        cicloActual++;

        actualizarControlesCiclo();

        // Esperar a que FastAPI termine el cálculo
        const correcto = await consultarEstadoCiclo();

        // Comprobar que no hemos pausado entretanto
        if (
            !reproduciendo ||
            sesion !== sesionReproduccion
        ) {
            return;
        }

        // Detener si hubo un error o llegamos al final
        if (
            !correcto ||
            cicloActual >= ultimoCicloMotor
        ) {
            detenerReproduccion();
            return;
        }

        // Programar el siguiente avance
        programarSiguienteCiclo(sesion);

    }, intervalo);
}


/* Alternar entre reproducir y pausar */
function alternarReproduccion() {

    if (reproduciendo) {
        detenerReproduccion();
        return;
    }

    if (
        motorSeleccionado === null ||
        cicloActual >= ultimoCicloMotor
    ) {
        return;
    }

    reproduciendo = true;

    const sesion = ++sesionReproduccion;

    actualizarBotonReproduccion();

    programarSiguienteCiclo(sesion);
}


/* =====================================================
   GRÁFICO DE EVOLUCIÓN DE ANOMALÍAS
===================================================== */

function actualizarGraficoAnomalias(datos) {

    const historial = datos.historial;

    // Ciclos observados
    const ciclos = historial.map(
        registro => registro.cycle
    );

    // Puntuaciones de Isolation Forest
    const puntuaciones = historial.map(
        registro => registro.anomaly_score
    );

    // Umbral constante
    const umbrales = historial.map(
        () => datos.umbral
    );

    // Marcar únicamente ciclos con alerta activa
    const alertas = historial.map(registro =>
        registro.alerta_activa
            ? registro.anomaly_score
            : null
    );

    const datasets = [
        {
            label: "Anomaly score",
            data: puntuaciones,
            borderColor: "#2563eb",
            backgroundColor: "#2563eb",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0.15
        },
        {
            label: "Umbral",
            data: umbrales,
            borderColor: "#dc2626",
            borderWidth: 2,
            borderDash: [6, 5],
            pointRadius: 0,
            tension: 0
        },
        {
            label: "Alerta activa",
            data: alertas,
            borderColor: "#dc2626",
            backgroundColor: "#dc2626",
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: false
        }
    ];

    const seccion = document.getElementById(
        "anomaly-chart-section"
    );

    // Mostrar el contenedor antes de crear el gráfico
    seccion.hidden = false;

    // Primera creación
    if (graficoAnomalias === null) {

        const canvas = document.getElementById(
            "anomaly-chart"
        );

        graficoAnomalias = new Chart(canvas, {
            type: "line",

            data: {
                labels: ciclos,
                datasets: datasets
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,

                interaction: {
                    mode: "index",
                    intersect: false
                },

                plugins: {
                    legend: {
                        position: "bottom"
                    },

                    tooltip: {
                        callbacks: {
                            title: elementos =>
                                `Ciclo ${elementos[0].label}`
                        }
                    }
                },

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Ciclo"
                        },

                        ticks: {
                            maxTicksLimit: 12
                        }
                    },

                    y: {
                        title: {
                            display: true,
                            text: "Anomaly score"
                        }
                    }
                }
            }
        });

    } else {

        // Actualizar el gráfico existente
        graficoAnomalias.data.labels = ciclos;
        graficoAnomalias.data.datasets = datasets;

        // Actualizar sin animación para facilitar
        // la reproducción automática
        graficoAnomalias.update("none");
    }
}


/* =====================================================
   GRÁFICO DE EVOLUCIÓN DE SENSORES
===================================================== */

function actualizarGraficoSensores() {

    if (datosHistorialActual === null) {
        return;
    }

    // Sensor elegido
    const sensor = document.getElementById(
        "sensor-selector"
    ).value;

    // Historial recibido desde FastAPI
    const historial = datosHistorialActual.historial;

    // Ciclos registrados
    const ciclos = historial.map(
        registro => registro.cycle
    );

    // Valores reales del sensor seleccionado
    const valores = historial.map(
        registro => registro.sensores[sensor]
    );

    const seccion = document.getElementById(
        "sensor-chart-section"
    );

    seccion.hidden = false;

    const dataset = {
        label: sensor.replace("_", " ").toUpperCase(),
        data: valores,

        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.08)",

        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,

        tension: 0.15,
        fill: true
    };

    // Primera creación
    if (graficoSensores === null) {

        const canvas = document.getElementById(
            "sensor-chart"
        );

        graficoSensores = new Chart(canvas, {
            type: "line",

            data: {
                labels: ciclos,
                datasets: [dataset]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,

                interaction: {
                    mode: "index",
                    intersect: false
                },

                plugins: {
                    legend: {
                        display: false
                    },

                    tooltip: {
                        callbacks: {
                            title: elementos =>
                                `Ciclo ${elementos[0].label}`
                        }
                    }
                },

                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Ciclo"
                        },

                        ticks: {
                            maxTicksLimit: 12
                        }
                    },

                    y: {
                        title: {
                            display: true,
                            text: "Valor del sensor"
                        }
                    }
                }
            }
        });

    } else {

        // Actualizar el gráfico existente
        graficoSensores.data.labels = ciclos;

        graficoSensores.data.datasets = [dataset];

        graficoSensores.update("none");
    }
}

/* =====================================================
   NAVEGACIÓN ENTRE PESTAÑAS
===================================================== */

function cambiarVistaMonitorizacion(vista) {

    const panel = document.querySelector(
        ".monitor-panel"
    );

    // Establecer vista activa
    panel.dataset.vista = vista;

    // Actualizar aspecto de los botones
    document.querySelectorAll(
        ".monitor-tab"
    ).forEach(boton => {

        const activa = boton.dataset.vista === vista;

        boton.classList.toggle("active", activa);

        boton.setAttribute(
            "aria-pressed",
            String(activa)
        );
    });

    // Dibujar únicamente el gráfico visible
    if (
        vista === "sensores" &&
        datosHistorialActual !== null
    ) {
        actualizarGraficoSensores();
    }

    if (
        vista === "ia" &&
        datosAnomaliasActual !== null
    ) {
        actualizarGraficoAnomalias(
            datosAnomaliasActual
        );
    }

    if (
        vista === "ia" &&
        datosRulActual !== null
    ) {
        actualizarGraficoRul(
            datosRulActual
        );
    }
    
    actualizarVisibilidadActualizacion();
}



/* =====================================================
   INDICADORES DE ANÁLISIS DE IA
===================================================== */

function actualizarIndicadoresIA(anomalias) {

    // Puntuación actual de Isolation Forest
    document.getElementById(
        "ai-score"
    ).textContent = formatearNumero(
        anomalias.anomaly_score,
        4
    );

    // Umbral de detección
    document.getElementById(
        "ai-threshold"
    ).textContent = formatearNumero(
        anomalias.umbral,
        4
    );

    // Estado de la alerta en el ciclo seleccionado
    const estado = document.getElementById("ai-status");

    estado.textContent = anomalias.alerta_activa
        ? "Alerta activa"
        : "Sin alerta activa";

    estado.style.color = anomalias.alerta_activa
        ? "var(--danger)"
        : "var(--success)";

    // Número de episodios de alerta confirmados
    document.getElementById(
        "ai-alert-count"
    ).textContent = anomalias.ciclos_confirmacion.length;
}


/* =====================================================
   GRÁFICO DE EVOLUCIÓN DE RUL
===================================================== */

function actualizarGraficoRul(datos) {

    const seccion = document.getElementById(
        "rul-chart-section"
    );

    const mensaje = document.getElementById(
        "rul-chart-message"
    );

    const contenedor = document.getElementById(
        "rul-chart-container"
    );

    const historial = datos.historial;

    // Mostrar la sección de prognosis
    seccion.hidden = false;

    // Si no hay 30 ciclos, todavía no existe predicción
    if (historial.length === 0) {

        mensaje.hidden = false;
        contenedor.hidden = true;

        return;
    }

    // Mostrar el gráfico
    mensaje.hidden = true;
    contenedor.hidden = false;

    // Ciclos para los que existe una predicción
    const ciclos = historial.map(
        registro => registro.cycle
    );

    // Predicciones originales de nuestra LSTM
    const predicciones = historial.map(
        registro => registro.rul_estimada
    );

    const dataset = {
        label: "RUL estimada",

        data: predicciones,

        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.08)",

        borderWidth: 2,
        pointRadius: historial.length === 1 ? 5 : 2,
        pointHoverRadius: 6,

        tension: 0,
        fill: true
    };

    // Conjunto de curvas que vamos a representar
    const datasets = [dataset];

    // Añadir la referencia real únicamente en modo evaluación
    const evaluacionActiva = document.getElementById(
        "evaluation-toggle"
    ).checked;

    if (
        evaluacionActiva &&
        datosEvaluacionActual !== null &&
        datosEvaluacionActual.motor_id === datos.motor_id &&
        datosEvaluacionActual.ciclo_consultado === datos.ciclo_consultado
    ) {

        // Reconstruir la RUL real de cada ciclo representado
        const rulRealLimitada = historial.map(registro => {

            const rulReal = datosEvaluacionActual.rul_real +
                (
                    datosEvaluacionActual.ciclo_consultado -
                    registro.cycle
                );

            return Math.min(
                rulReal,
                datosEvaluacionActual.rul_cap
            );
        });

        datasets.push({
            label: "RUL real limitada (evaluación)",
            data: rulRealLimitada,
            borderColor: "#dc2626",
            backgroundColor: "#dc2626",
            borderWidth: 2,
            borderDash: [7, 5],
            pointRadius: 0,
            pointHoverRadius: 5,
            tension: 0,
            fill: false
        });
    }

    // Crear el gráfico por primera vez
    if (graficoRul === null) {

        const canvas = document.getElementById(
            "rul-chart"
        );

        graficoRul = new Chart(canvas, {

            type: "line",

            data: {
                labels: ciclos,
                datasets: datasets
            },

            options: {

                responsive: true,
                maintainAspectRatio: false,
                animation: false,

                interaction: {
                    mode: "index",
                    intersect: false
                },

                plugins: {

                    legend: {
                        position: "bottom"
                    },

                    tooltip: {
                        callbacks: {

                            title: elementos =>
                                `Ciclo ${elementos[0].label}`,

                            label: contexto =>
                                `${contexto.dataset.label}: ${
                                    formatearNumero(contexto.parsed.y)
                                } ciclos`
                        }
                    }
                },

                scales: {

                    x: {
                        title: {
                            display: true,
                            text: "Ciclo de funcionamiento"
                        },

                        ticks: {
                            maxTicksLimit: 12
                        }
                    },

                    y: {
                        title: {
                            display: true,
                            text: "RUL estimada (ciclos)"
                        },

                        beginAtZero: true
                    }
                }
            }
        });

    } else {

        // Actualizar el gráfico existente
        graficoRul.data.labels = ciclos;
        graficoRul.data.datasets = datasets;

        graficoRul.update("none");
    }
}



/* =====================================================
   MODO DE EVALUACIÓN
===================================================== */

async function consultarEvaluacionCiclo(
    motorId,
    ciclo,
    solicitudEstado
) {

    const solicitudEvaluacion =
        ++ultimaSolicitudEvaluacion;

    const resultados = document.getElementById(
        "evaluation-results"
    );

    const mensaje = document.getElementById(
        "evaluation-message"
    );

    const interruptor = document.getElementById(
        "evaluation-toggle"
    );

    // Conservar visibles los resultados anteriores mientras FastAPI calcula el nuevo ciclo
    if (resultados.hidden) {
        mensaje.textContent = "Calculando evaluación...";
    }

    try {

        const respuesta = await fetch(
            `/api/motores/${motorId}/evaluacion?ciclo=${ciclo}`
        );

        if (!respuesta.ok) {
            throw new Error(
                `Error HTTP ${respuesta.status}`
            );
        }

        const datos = await respuesta.json();

        // Evitar resultados de peticiones antiguas
        if (
            solicitudEstado !== ultimaSolicitudEstado ||
            solicitudEvaluacion !== ultimaSolicitudEvaluacion ||
            !interruptor.checked
        ) {
            return;
        }

        // Guardar los valores reales recibidos
        datosEvaluacionActual = datos;

        // Actualizar el gráfico si está visible
        const vistaActiva = document.querySelector(
            ".monitor-panel"
        ).dataset.vista;

        if (
            vistaActiva === "ia" &&
            datosRulActual !== null
        ) {
            actualizarGraficoRul(datosRulActual);
        }

        // RUL real sin limitar
        document.getElementById(
            "evaluation-real"
        ).textContent =
            `${formatearNumero(datos.rul_real)} ciclos`;

        // RUL real limitada a 120
        document.getElementById(
            "evaluation-real-capped"
        ).textContent =
            `${formatearNumero(datos.rul_real_limitada)} ciclos`;

        // RUL predicha
        document.getElementById(
            "evaluation-predicted"
        ).textContent =
            datos.prediccion_disponible
                ? `${formatearNumero(datos.rul_predicha)} ciclos`
                : "No disponible";

        // Error absoluto
        document.getElementById(
            "evaluation-error"
        ).textContent =
            datos.prediccion_disponible
                ? `${formatearNumero(datos.error_absoluto)} ciclos`
                : "No disponible";

        resultados.hidden = false;

        mensaje.textContent =
            datos.prediccion_disponible
                ? "Error calculado respecto a la RUL real limitada."
                : "La LSTM necesita al menos 30 ciclos para predecir.";

    } catch (error) {

        if (
            solicitudEstado !== ultimaSolicitudEstado ||
            solicitudEvaluacion !== ultimaSolicitudEvaluacion ||
            !interruptor.checked
        ) {
            return;
        }

        console.error(
            "Error en evaluación:",
            error
        );

        mensaje.textContent = "No se ha podido calcular la evaluación.";

        // Si falla la evaluación, conservar al menos la trayectoria estimada de la LSTM
        if (
            datosRulActual !== null &&
            document.querySelector(".monitor-panel").dataset.vista === "ia"
        ) {
            actualizarGraficoRul(datosRulActual);
        }
    }
}

/* =====================================================
   ACTUALIZACIÓN DE MOTORES EXTERNOS
===================================================== */
async function actualizarMotorExterno() {
    if (motorSeleccionado === null || motorSeleccionadoEsReferencia) {
        return;
    }

    const archivo = document
        .getElementById(
            "update-file"
        )
        .files[0];

    const estado = document.getElementById(
        "update-status"
    );

    const boton = document.getElementById(
        "update-confirm"
    );

    if (!archivo) {

        estado.textContent =
            "Selecciona un archivo CSV.";

        return;
    }

    if (
        !archivo.name
            .toLowerCase()
            .endsWith(".csv")
    ) {

        estado.textContent =
            "El archivo seleccionado debe ser CSV.";

        return;
    }

    boton.disabled = true;
    estado.textContent = `Actualizando motor ${motorSeleccionado}...`;

    try {
        const formulario = new FormData();
        formulario.append(
            "archivo",
            archivo
        );

        const respuesta = await fetch(
            `/api/motores/${motorSeleccionado}/actualizar`,
            {
                method: "POST",
                body: formulario
            }
        );

        if (!respuesta.ok) {
            throw new Error(
                await leerErrorImportacion(
                    respuesta
                )
            );
        }

        const datos = await respuesta.json();

        // ----------------------------------------------
        // ACTUALIZAR FLOTA
        // ----------------------------------------------
        await cargarFlota();

        const motorActualizado =
            motoresFlota.find(
                motor =>
                    motor.motor_id ===
                    motorSeleccionado
            );

        if (motorActualizado) {
            ultimoCicloMotor = motorActualizado.ultimo_ciclo;
        } else {
            ultimoCicloMotor = datos.ciclos_actuales;
        }

        // Ir directamente al último ciclo nuevo
        cicloActual = ultimoCicloMotor;

        const slider = document.getElementById(
            "cycle-slider"
        );

        slider.max = ultimoCicloMotor;
        slider.value = cicloActual;
        actualizarControlesCiclo();

        // ----------------------------------------------
        // MENSAJE DE RESULTADO
        // ----------------------------------------------
        let mensajeResultado =
            `Motor ${datos.motor_id} actualizado: ` +
            `${datos.ciclos_anteriores} → ` +
            `${datos.ciclos_actuales} ciclos. `;

        if (datos.transicion_perfil) {
            mensajeResultado +=
                `El perfil predictivo ha cambiado ` +
                `a ${datos.profile_name}.`;
        } else {
            mensajeResultado +=
                `Perfil actual: ` +
                `${datos.profile_name}.`;
        }

        estado.textContent = mensajeResultado;

        // Vaciar archivo después del éxito
        document.getElementById(
            "update-file"
        ).value = "";

        // Recalcular monitorización en el nuevo último ciclo
        await consultarEstadoCiclo();

        actualizarVisibilidadActualizacion();

    } catch (error) {
        console.error(
            "Error al actualizar el motor:",
            error
        );

        estado.textContent = error.message;

    } finally {
        boton.disabled = false;
    }
}

/* =====================================================
   IMPORTACIÓN DE MOTORES
===================================================== */
function mostrarEstadoImportacion(
    texto,
    tipo = ""
) {

    const estado = document.getElementById(
        "import-status"
    );

    estado.textContent = texto;

    estado.className =
        "import-status";

    if (tipo) {
        estado.classList.add(
            `is-${tipo}`
        );
    }
}


function invalidarValidacionImportacion() {

    importacionValidada = false;

    document.getElementById(
        "import-confirm"
    ).disabled = true;

    document.getElementById(
        "import-validation"
    ).hidden = true;

    document.getElementById(
        "import-success"
    ).hidden = true;

    mostrarEstadoImportacion("");
}


function obtenerDatosImportacion() {

    const origen = document
        .getElementById("import-origin")
        .value
        .trim();

    const archivo = document
        .getElementById("import-file")
        .files[0];

    if (!origen) {
        throw new Error(
            "Indica el origen de los motores."
        );
    }

    if (!archivo) {
        throw new Error(
            "Selecciona un archivo CSV."
        );
    }

    if (
        !archivo.name
            .toLowerCase()
            .endsWith(".csv")
    ) {
        throw new Error(
            "El archivo seleccionado debe ser CSV."
        );
    }

    return {
        origen,
        archivo
    };
}


function crearFormularioImportacion(
    origen,
    archivo
) {

    const formulario =
        new FormData();

    formulario.append(
        "origen",
        origen
    );

    formulario.append(
        "archivo",
        archivo
    );

    return formulario;
}


async function leerErrorImportacion(
    respuesta
) {

    let datos = null;

    try {
        datos = await respuesta.json();
    } catch {
        return `Error HTTP ${respuesta.status}`;
    }

    const detalle = datos.detail;

    if (typeof detalle === "string") {
        return detalle;
    }

    if (
        detalle &&
        typeof detalle === "object"
    ) {

        let mensaje =
            detalle.mensaje ||
            "El archivo no es válido.";

        if (
            Array.isArray(
                detalle.columnas_faltantes
            ) &&
            detalle.columnas_faltantes.length
        ) {
            mensaje +=
                " Columnas faltantes: " +
                detalle.columnas_faltantes.join(", ") +
                ".";
        }

        if (
            Array.isArray(
                detalle.columnas_adicionales
            ) &&
            detalle.columnas_adicionales.length
        ) {
            mensaje +=
                " Columnas no reconocidas: " +
                detalle.columnas_adicionales.join(", ") +
                ".";
        }

        if (
            Array.isArray(
                detalle.columnas_con_nulos
            ) &&
            detalle.columnas_con_nulos.length
        ) {
            mensaje +=
                " Columnas con valores nulos: " +
                detalle.columnas_con_nulos.join(", ") +
                ".";
        }

        return mensaje;
    }

    return `Error HTTP ${respuesta.status}`;
}


async function validarImportacion() {

    const botonValidar =
        document.getElementById(
            "import-validate"
        );

    const botonImportar =
        document.getElementById(
            "import-confirm"
        );

    try {

        const {
            origen,
            archivo
        } = obtenerDatosImportacion();

        importacionValidada = false;

        botonValidar.disabled = true;
        botonImportar.disabled = true;

        document.getElementById(
            "import-success"
        ).hidden = true;

        mostrarEstadoImportacion(
            "Validando estructura del archivo..."
        );

        const formulario =
            crearFormularioImportacion(
                origen,
                archivo
            );

        const respuesta = await fetch(
            "/api/importaciones/validar",
            {
                method: "POST",
                body: formulario
            }
        );

        if (!respuesta.ok) {

            throw new Error(
                await leerErrorImportacion(
                    respuesta
                )
            );
        }

        const datos =
            await respuesta.json();

        document.getElementById(
            "import-validation-file"
        ).textContent =
            `${datos.archivo} · ${datos.origen}`;

        document.getElementById(
            "import-summary-motors"
        ).textContent =
            datos.motores_detectados;

        document.getElementById(
            "import-summary-observations"
        ).textContent =
            datos.observaciones.toLocaleString(
                "es-ES"
            );

        document.getElementById(
            "import-summary-sensors"
        ).textContent =
            datos.sensores_disponibles;

        document.getElementById(
            "import-summary-columns"
        ).textContent =
            datos.columnas;

        const nombresPerfiles = {
            FD001: "Condición única",
            FD002: "Multirrégimen"
        };


        const textoPerfiles = Object
            .entries(
                datos.perfiles_detectados || {}
            )
            .map(
                ([perfil, cantidad]) =>
                    `${nombresPerfiles[perfil] || perfil}: ${cantidad}`
            )
            .join(" · ");


        document.getElementById(
            "import-summary-profiles"
        ).textContent =
            textoPerfiles || "—";

        document.getElementById(
            "import-validation"
        ).hidden = false;

        importacionValidada = true;

        botonImportar.disabled = false;

        mostrarEstadoImportacion(
            "El archivo es compatible y está listo para importarse.",
            "success"
        );

    } catch (error) {

        console.error(
            "Error al validar la importación:",
            error
        );

        importacionValidada = false;

        botonImportar.disabled = true;

        document.getElementById(
            "import-validation"
        ).hidden = true;

        mostrarEstadoImportacion(
            error.message,
            "error"
        );

    } finally {

        botonValidar.disabled = false;
    }
}


async function importarMotores() {

    if (!importacionValidada) {
        return;
    }

    const botonValidar =
        document.getElementById(
            "import-validate"
        );

    const botonImportar =
        document.getElementById(
            "import-confirm"
        );

    try {

        const {
            origen,
            archivo
        } = obtenerDatosImportacion();

        botonValidar.disabled = true;
        botonImportar.disabled = true;

        mostrarEstadoImportacion(
            "Importando motores y calculando su estado..."
        );

        const formulario =
            crearFormularioImportacion(
                origen,
                archivo
            );

        const respuesta = await fetch(
            "/api/importaciones/importar",
            {
                method: "POST",
                body: formulario
            }
        );

        if (!respuesta.ok) {

            throw new Error(
                await leerErrorImportacion(
                    respuesta
                )
            );
        }

        const datos =
            await respuesta.json();

        // Actualizar la flota completa con
        // los motores recién incorporados.
        await cargarFlota();

        const idsInternos =
            datos.motores.map(
                motor => motor.motor_id
            );

        let textoIds = "";

        if (idsInternos.length === 1) {

            textoIds =
                ` ID interno asignado: ` +
                `${idsInternos[0]}.`;

        } else if (idsInternos.length > 1) {

            const primerId =
                Math.min(...idsInternos);

            const ultimoId =
                Math.max(...idsInternos);

            textoIds =
                ` IDs internos asignados: ` +
                `${primerId}–${ultimoId}.`;
        }

        document.getElementById(
            "import-success-message"
        ).textContent =
            `${datos.motores_importados} motores ` +
            `del origen "${datos.origen}" ` +
            `se han añadido a la plataforma.` +
            textoIds;

        document.getElementById(
            "import-validation"
        ).hidden = true;

        document.getElementById(
            "import-success"
        ).hidden = false;

        importacionValidada = false;

        mostrarEstadoImportacion(
            "Importación finalizada correctamente.",
            "success"
        );

    } catch (error) {

        console.error(
            "Error al importar motores:",
            error
        );

        botonImportar.disabled = false;

        mostrarEstadoImportacion(
            error.message,
            "error"
        );

    } finally {

        botonValidar.disabled = false;
    }
}


function resetearFormularioImportacion() {
    document.getElementById(
        "import-form"
    ).reset();

    importacionValidada = false;

    document.getElementById(
        "import-confirm"
    ).disabled = true;

    document.getElementById(
        "import-validation"
    ).hidden = true;

    document.getElementById(
        "import-success"
    ).hidden = true;

    mostrarEstadoImportacion("");
}



/* =========================================
   NAVEGACIÓN PRINCIPAL
========================================= */
const VISTAS_PRINCIPALES = [
    "flota",
    "monitorizacion",
    "importar",
    "informacion"
];

function mostrarVistaPrincipal() {
    // Leer la vista solicitada en la URL
    const vistaSolicitada = window.location.hash.slice(1);

    // Por defecto, mostrar la flota
    const vistaActiva = VISTAS_PRINCIPALES.includes(
        vistaSolicitada
    )
        ? vistaSolicitada
        : "flota";

    // Pausar la simulación al salir de Monitorización
    if (
        vistaActiva !== "monitorizacion" &&
        reproduciendo
    ) {
        detenerReproduccion();
    }

    // Mostrar solo la sección seleccionada
    VISTAS_PRINCIPALES.forEach(vista => {

        const seccion = document.getElementById(vista);

        if (seccion) {
            seccion.hidden = vista !== vistaActiva;
        }

    });

    // Actualizar el menú lateral
    document.querySelectorAll(
        'aside a[href="#flota"], ' +
        'aside a[href="#monitorizacion"], ' +
        'aside a[href="#importar"], ' +
        'aside a[href="#informacion"]'
    ).forEach(enlace => {

        const activo = enlace.getAttribute("href") ===
            `#${vistaActiva}`;

        enlace.classList.toggle("active", activo);

        if (activo) {
            enlace.setAttribute("aria-current", "page");
        } else {
            enlace.removeAttribute("aria-current");
        }

    });

}


/* =====================================================
   CONTEXTO DEL MOTOR
===================================================== */

function actualizarContextoMotor(datos) {

    document.getElementById(
        "context-motor"
    ).textContent = `Motor ${datos.motor_id}`;

    document.getElementById(
        "context-profile"
    ).textContent =
        datos.profile_name ||
        datos.profile;


    document.getElementById(
        "context-regime"
    ).textContent =
        datos.profile === "FD002" &&
        datos.operating_regime !== undefined
            ? `Régimen ${datos.operating_regime}`
            : "No aplica";

    document.getElementById(
        "context-cycle"
    ).textContent =
        `${datos.ciclo_actual} / ${datos.ultimo_ciclo_disponible}`;

    // Estado actual de Isolation Forest
    const indicadorAlerta = document.getElementById(
        "context-alert"
    );

    const alertaActiva = datos.anomalias.alerta_activa;

    indicadorAlerta.textContent = alertaActiva
        ? "Alerta activa"
        : "Sin alerta activa";

    indicadorAlerta.classList.toggle(
        "is-alert",
        alertaActiva
    );

    // Predicción de nuestra LSTM
    document.getElementById(
        "context-rul"
    ).textContent = datos.prognosis.disponible
        ? `${formatearNumero(
            datos.prognosis.rul_estimada
        )} ciclos`
        : "Datos insuficientes";

    // Mostrar información actualizada
    document.getElementById(
        "monitor-context"
    ).hidden = false;
}






/* Ejecutar cuando el HTML esté preparado */
document.addEventListener("DOMContentLoaded", () => {
    cargarFlota();

    // Mostrar la sección inicial
    mostrarVistaPrincipal();

    // Cambiar de sección cuando cambia la URL
    window.addEventListener(
        "hashchange",
        mostrarVistaPrincipal
    );

    document
        .getElementById("buscar-motor")
        .addEventListener("input", aplicarBusqueda);

    document
        .getElementById("filtro-alertas")
        .addEventListener("change", aplicarBusqueda);

    document
        .getElementById("filtro-origen")
        .addEventListener(
            "change",
            aplicarBusqueda
        );    

    document
        .getElementById("orden-motores")
        .addEventListener("change", aplicarBusqueda);

    /* REINICIAR: volver al ciclo 1 */
    document
        .getElementById("cycle-reset")
        .addEventListener("click", () => {
            irACiclo(1);
        });


    /* CICLO ANTERIOR */
    document
        .getElementById("cycle-prev")
        .addEventListener("click", () => {
            irACiclo(cicloActual - 1);
        });


    /* CICLO SIGUIENTE */
    document
        .getElementById("cycle-next")
        .addEventListener("click", () => {
            irACiclo(cicloActual + 1);
        });

    /* BARRA DESLIZANTE */

    // Actualizar el ciclo seleccionado sin ocultar los gráficos
    document
        .getElementById("cycle-slider")
        .addEventListener("input", (evento) => {

            // Detener la reproducción automática
            detenerReproduccion();

            // Invalidar peticiones antiguas
            ultimaSolicitudEstado++;
            ultimaSolicitudEvaluacion++;

            // Actualizar el ciclo seleccionado
            cicloActual = Number(evento.target.value);

            // Actualizar la barra y sus controles
            actualizarControlesCiclo();

            // Mantener visibles los indicadores y gráficos
            // anteriores hasta recibir los nuevos resultados.
        });


    // Consultar FastAPI cuando se confirma el ciclo
    document
        .getElementById("cycle-slider")
        .addEventListener("change", () => {
            consultarEstadoCiclo();
        });

    /* REPRODUCIR / PAUSAR */
    document
        .getElementById("playback-toggle")
        .addEventListener(
            "click",
            alternarReproduccion
        );

    /* CAMBIAR EL SENSOR VISUALIZADO */
    document
        .getElementById("sensor-selector")
        .addEventListener(
            "change",
            actualizarGraficoSensores
        );

    /* CAMBIAR DE PESTAÑA */
    document.querySelectorAll(
        ".monitor-tab"
    ).forEach(boton => {

        boton.addEventListener("click", () => {

            cambiarVistaMonitorizacion(
                boton.dataset.vista
            );

        });

    });

    /* ACTIVAR O DESACTIVAR EVALUACIÓN */
    document
        .getElementById("evaluation-toggle")
        .addEventListener("change", (evento) => {

            if (!motorSeleccionadoEsReferencia) {
                evento.target.checked = false;
                return;
            }

            ultimaSolicitudEvaluacion++;

            // Eliminar cualquier referencia real anterior
            datosEvaluacionActual = null;

            // Redibujar el gráfico sin la referencia antigua
            if (datosRulActual !== null) {
                const vistaActiva = document.querySelector(
                    ".monitor-panel"
                ).dataset.vista;

                if (vistaActiva === "ia") {
                    actualizarGraficoRul(datosRulActual);
                }
            }

            document.getElementById(
                "evaluation-results"
            ).hidden = true;

            document.getElementById(
                "evaluation-message"
            ).textContent = "";

            if (
                evento.target.checked &&
                motorSeleccionado !== null
            ) {
                consultarEvaluacionCiclo(
                    motorSeleccionado,
                    cicloActual,
                    ultimaSolicitudEstado
                );
            }

        });

        /* IMPORTACIÓN DE MOTORES */
        document
            .getElementById("import-validate")
            .addEventListener(
                "click",
                validarImportacion
            );

        document
            .getElementById("import-confirm")
            .addEventListener(
                "click",
                importarMotores
            );

        document
            .getElementById("import-reset")
            .addEventListener(
                "click",
                resetearFormularioImportacion
            );

        // Si se cambia el origen o el archivo, es necesario validar de nuevo
        document
            .getElementById("import-origin")
            .addEventListener(
                "input",
                invalidarValidacionImportacion
            );

        document
            .getElementById("import-file")
            .addEventListener(
                "change",
                invalidarValidacionImportacion
            );

        /* ACTUALIZACIÓN DE MOTOR EXTERNO */
        document
            .getElementById("update-confirm")
            .addEventListener(
                "click",
                actualizarMotorExterno
            );

        document
            .getElementById("update-file")
            .addEventListener(
                "change",
                () => {
                    document.getElementById(
                        "update-status"
                    ).textContent = "";

                }
            );
});