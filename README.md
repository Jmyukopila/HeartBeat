# HeartBeat

Análisis fisiológico a partir de señales de acelerómetro (BCG) capturadas con **PhyPhox**.

## ¿Qué mide HeartBeat?

HeartBeat procesa la señal del acelerómetro de tu teléfono para extraer:

- **Frecuencia Cardíaca (FC)** — latidos por minuto
- **Variabilidad Cardíaca (HRV)** — SDNN, RMSSD, pNN50, LF/HF
- **Frecuencia Respiratoria (FR)** — respiraciones por minuto
- **Calidad de señal** — SNR y SQI

Todo a partir de la **señal BCG (Ballistocardiograma)**: las vibraciones mecánicas que produce tu corazón y se transmiten al teléfono cuando lo apoyas sobre el pecho.

## Captura de datos con PhyPhox

Usa la app **PhyPhox** (gratuita, iOS/Android):

1. Abre PhyPhox → **"Aceleración con g"**
2. Apoya el teléfono sobre tu pecho (boca arriba)
3. Presiona **INICIAR** y espera 60-120 segundos
4. Presiona **DETENER** y exporta como CSV

Las columnas que HeartBeat reconoce automáticamente:

| Columna | Descripción |
|---------|-------------|
| `Time (s)` | Tiempo en segundos |
| `Acceleration X (m/s²)` | Eje X (con gravedad) |
| `Acceleration Y (m/s²)` | Eje Y (con gravedad) |
| `Acceleration Z (m/s²)` | Eje Z (con gravedad) |

> El eje Z suele capturar mejor el BCG si el teléfono está plano sobre el pecho.
> HeartBeat **elige automáticamente el eje con mejor relación señal/ruido** entre los 3.

## Estructura del proyecto

```
HeartBeat/
├── backend/
│   └── api.py              # API FastAPI (procesamiento de señal)
├── frontend/
│   └── index.html           # Dashboard interactivo
├── requirements.txt         # Dependencias Python
└── .gitignore
```

## Requisitos previos

Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

## Cómo iniciar

### Backend

Desde la raíz del proyecto (`HeartBeat/`):

```bash
python -m uvicorn backend.api:app --reload
```

Queda escuchando en `http://127.0.0.1:8000`.

### Frontend

Abrir `frontend/index.html` con doble clic en el navegador (Chrome recomendado).

No requiere servidor — es estático.

### Endpoints

| Método | Ruta       | Descripción                              |
|--------|------------|------------------------------------------|
| GET    | `/health`  | Health check                             |
| POST   | `/procesar`| Subir CSV → métricas fisiológicas        |

`POST /procesar` recibe `multipart/form-data` con campo `"file"` (archivo CSV).

Documentación interactiva: `http://127.0.0.1:8000/docs`

## Flujo de uso completo

1. Captura datos con PhyPhox ("Aceleración con g", teléfono sobre el pecho)
2. Inicia backend: `python -m uvicorn backend.api:app --reload`
3. Abre `frontend/index.html` en el navegador
4. Selecciona el archivo CSV exportado de PhyPhox
5. El dashboard muestra dos pestañas:
   - **Tu Resultado** — métricas coloquiales con explicación en lenguaje cotidiano
   - **Métricas Técnicas** — todos los valores crudos del procesamiento

## Conexión Frontend ↔ Backend

El frontend detecta si el backend está disponible (`GET /health`) y envía el CSV a `POST /procesar` mediante `fetch()`.

CORS habilitado para todos los orígenes.

## Cómo funciona el procesamiento

1. **Selección de eje** — calcula SNR cardíaco en cada eje, usa el de mejor calidad
2. **Filtrado** — pasa banda 0.7–3 Hz para señal cardíaca, 0.1–0.4 Hz para respiratoria
3. **FC** — consenso entre autocorrelación, FFT, Welch y detección de picos (promedio si coinciden en ±3 bpm)
4. **HRV** — SDNN, RMSSD, pNN50, ratio LF/HF desde los intervalos entre latidos
5. **FR** — consenso entre Welch, envolvente de Hilbert y detección de picos en banda respiratoria
6. **Interpretación** — análisis automático en lenguaje cotidiano
