from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks, welch, hilbert
from numpy.fft import rfft, rfftfreq
import io
import re

app = FastAPI(title="HeartBeat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COL_TIME = re.compile(r"time\s*\(s\)", re.IGNORECASE)
COL_AXIS = re.compile(
    r"acceleration\s*(x|y|z)\s*\(m/s", re.IGNORECASE
)

MAX_FILE_SIZE = 50 * 1024 * 1024

MIN_FS = 20
IBI_MIN = 0.3
IBI_MAX = 1.5

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _find_columns(df: pd.DataFrame):
    time_col = None
    axis_cols: dict[str, str] = {}
    for col in df.columns:
        s = col.strip()
        if COL_TIME.search(s):
            time_col = col
        m = COL_AXIS.search(s)
        if m:
            axis_cols[m.group(1).lower()] = col
    return time_col, axis_cols

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4):
    nyq = fs * 0.5
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def _cardiac_snr(sig: np.ndarray, fs: float):
    f_w, Pxx = welch(sig, fs=fs, nperseg=int(min(4 * fs, len(sig) // 2)))
    band_hr = (f_w >= 0.7) & (f_w <= 3.0)
    noise_mask = (~band_hr) & (f_w > 0)
    p_signal = Pxx[band_hr].sum() + 1e-12
    p_noise = Pxx[noise_mask].sum() + 1e-12
    return float(10 * np.log10(p_signal / p_noise))

def _detect_peaks_bcg(sig: np.ndarray, fs: float):
    dist = int(fs * IBI_MIN)
    prom = float(np.std(sig) * 1.2)
    peaks, _ = find_peaks(sig, distance=dist, prominence=prom)
    if len(peaks) < 3:
        peaks, _ = find_peaks(sig, distance=dist, prominence=prom * 0.5)
    if len(peaks) < 3:
        peaks, _ = find_peaks(-sig, distance=dist, prominence=prom * 0.5)
    return peaks

def _clean_ibis(peaks: np.ndarray, fs: float):
    if len(peaks) < 2:
        return np.array([])
    ibis = np.diff(peaks) / fs
    mask = (ibis >= IBI_MIN) & (ibis <= IBI_MAX)
    return ibis[mask]

# ---------------------------------------------------------------------------
# FC methods
# ---------------------------------------------------------------------------

def _fc_autocorr(sig: np.ndarray, fs: float):
    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
    lags = np.arange(ac.size) / fs
    mask = (lags >= IBI_MIN) & (lags <= IBI_MAX)
    pk, _ = find_peaks(ac[mask])
    if pk.size == 0:
        return np.nan
    period = lags[mask][pk[0]]
    return 60.0 / period

def _fc_fft(sig: np.ndarray, fs: float):
    Y = np.abs(rfft(sig))
    f = rfftfreq(sig.size, d=1 / fs)
    band = (f >= 0.7) & (f <= 3.0)
    if not band.any():
        return np.nan
    return 60.0 * f[band][np.argmax(Y[band])]

def _fc_welch(sig: np.ndarray, fs: float):
    nperseg = int(min(4 * fs, len(sig) // 2))
    f_w, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band = (f_w >= 0.7) & (f_w <= 3.0)
    if not band.any():
        return np.nan
    return 60.0 * f_w[band][np.argmax(Pxx[band])]

def _fc_peaks(ibis: np.ndarray):
    if len(ibis) < 3:
        return np.nan
    return 60.0 / float(np.nanmean(ibis))

def _fc_consensus(values: dict[str, float]):
    vals = np.array([v for v in values.values() if not np.isnan(v)])
    if len(vals) == 0:
        return np.nan, 0.0
    median = np.median(vals)
    agreeing = vals[np.abs(vals - median) <= 3.0]
    if len(agreeing) == 0:
        return float(median), 0.3
    confidence = len(agreeing) / len(vals)
    return float(np.mean(agreeing)), float(confidence)

# ---------------------------------------------------------------------------
# FR methods
# ---------------------------------------------------------------------------

def _fr_welch(sig: np.ndarray, fs: float):
    nperseg = int(min(4 * fs, len(sig) // 2))
    f_w, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band = (f_w >= 0.1) & (f_w <= 0.4)
    if not band.any():
        return np.nan
    return 60.0 * f_w[band][np.argmax(Pxx[band])]

def _fr_envelope(sig: np.ndarray, fs: float):
    analytic = hilbert(sig)
    env = np.abs(analytic)
    env -= np.mean(env)
    nperseg = int(min(4 * fs, len(env) // 2))
    f_w, Pxx = welch(env, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    band = (f_w >= 0.1) & (f_w <= 0.4)
    if not band.any():
        return np.nan
    return 60.0 * f_w[band][np.argmax(Pxx[band])]

def _fr_peaks(sig: np.ndarray, fs: float):
    sig_r = _bandpass(sig, fs, 0.1, 0.4)
    peaks, _ = find_peaks(sig_r, distance=int(fs * 0.5))
    ibis = _clean_ibis(peaks, fs)
    if len(ibis) < 3:
        return np.nan
    return 60.0 / float(np.nanmean(ibis))

def _fr_consensus(values: list[float]):
    vals = np.array([v for v in values if not np.isnan(v)])
    if len(vals) == 0:
        return np.nan
    return float(np.median(vals))

# ---------------------------------------------------------------------------
# HRV
# ---------------------------------------------------------------------------

def _compute_hrv(ibis: np.ndarray, fs: float):
    if len(ibis) < 3:
        return {"SDNN_ms": np.nan, "RMSSD_ms": np.nan, "pNN50": np.nan,
                "LF_Hz": np.nan, "HF_Hz": np.nan, "LF_HF_ratio": np.nan}

    sdnn = float(np.std(ibis, ddof=1) * 1000)
    diffs = np.diff(ibis)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000)
    pnn50 = float(np.mean(np.abs(diffs) * 1000 > 50) * 100)

    t_ibi = np.cumsum(ibis)
    t_reg = np.arange(0, t_ibi[-1], 1.0)
    ibi_reg = np.interp(t_reg, t_ibi[:-1], ibis[1:])
    ibi_reg -= np.mean(ibi_reg)

    f_w, Pxx = welch(ibi_reg, fs=1.0, nperseg=int(min(256, len(ibi_reg) // 2)))
    lf_band = (f_w >= 0.04) & (f_w <= 0.15)
    hf_band = (f_w >= 0.15) & (f_w <= 0.4)
    lf_power = Pxx[lf_band].sum() + 1e-12
    hf_power = Pxx[hf_band].sum() + 1e-12
    lf_peak = f_w[lf_band][np.argmax(Pxx[lf_band])] if lf_band.any() else np.nan
    hf_peak = f_w[hf_band][np.argmax(Pxx[hf_band])] if hf_band.any() else np.nan

    return {
        "SDNN_ms": sdnn,
        "RMSSD_ms": rmssd,
        "pNN50": pnn50,
        "LF_Hz": float(lf_peak),
        "HF_Hz": float(hf_peak),
        "LF_HF_ratio": float(lf_power / hf_power),
    }

# ---------------------------------------------------------------------------
# Coloquial & interpretation
# ---------------------------------------------------------------------------

def _nivel_hr(hr: float):
    if hr < 60:  return "bajo"
    if hr <= 80: return "normal"
    if hr <= 100: return "normal"
    return "alto"

def _explicacion_hr(hr: float):
    if hr < 60:
        return f"Tu corazón late {hr:.0f} veces por minuto, más lento de lo habitual. Puede indicar relajación profunda o buena condición cardiovascular."
    if hr <= 80:
        return f"Tu corazón late {hr:.0f} veces por minuto. Es un ritmo normal y saludable en reposo."
    if hr <= 100:
        return f"Tu corazón late {hr:.0f} veces por minuto, ligeramente elevado. Puede reflejar concentración, activación leve o ansiedad."
    return f"Tu corazón late {hr:.0f} veces por minuto, por encima del rango de reposo. Posible estrés, actividad física o excitación."

def _nivel_hrv(hrv: float):
    if hrv < 50:  return "bajo"
    if hrv <= 100: return "moderado"
    return "alto"

def _explicacion_hrv(hrv: float):
    if hrv < 50:
        return f"Tu variabilidad cardíaca es de {hrv:.1f} ms, considerada baja. Puede asociarse a estrés, fatiga o activación del sistema simpático."
    if hrv <= 100:
        return f"Tu variabilidad cardíaca es de {hrv:.1f} ms, un valor moderado. Refleja un equilibrio fisiológico aceptable."
    return f"Tu variabilidad cardíaca es de {hrv:.1f} ms, un valor alto. Indica buena regulación autonómica y capacidad de recuperación."

def _nivel_fr(fr: float):
    if fr < 10:  return "bajo"
    if fr <= 18: return "normal"
    return "alto"

def _explicacion_fr(fr: float):
    if fr < 10:
        return f"Respiras {fr:.1f} veces por minuto, por debajo de lo típico. Compatible con relajación profunda o sueño."
    if fr <= 18:
        return f"Respiras {fr:.1f} veces por minuto, un ritmo normal y tranquilo."
    return f"Respiras {fr:.1f} veces por minuto, elevado. Puede deberse a estrés, ansiedad o actividad física."

def _nivel_snr(snr: float, sqi: float):
    if sqi < 0.5 or snr < -5: return "mala"
    if sqi < 0.7 or snr < 5:  return "regular"
    return "buena"

def _explicacion_snr(snr: float, sqi: float):
    nivel = _nivel_snr(snr, sqi)
    if nivel == "mala":
        return "La señal tiene mucho ruido. Los resultados deben tomarse con precaución. Intenta colocar el dispositivo más firme sobre el pecho."
    if nivel == "regular":
        return "La calidad de la señal es aceptable. La mayoría de las métricas son confiables."
    return "La calidad de la señal es buena. Los resultados son confiables."

def _generar_coloquiales(tecnicas: dict):
    hr = tecnicas.get("FC_final_bpm", np.nan)
    hrv = tecnicas.get("HRV_SDNN_ms", np.nan)
    fr = tecnicas.get("FR_final_rpm", np.nan)
    snr = tecnicas.get("SNR_cardiaco_dB", np.nan)
    sqi = tecnicas.get("SQI", np.nan)

    items = []

    if not np.isnan(hr):
        items.append({
            "nombre": "Frecuencia Cardíaca",
            "icono": "❤️",
            "valor": f"{hr:.0f} bpm",
            "nivel": _nivel_hr(hr),
            "explicacion": _explicacion_hr(hr),
        })

    if not np.isnan(hrv):
        items.append({
            "nombre": "Variabilidad Cardíaca",
            "icono": "📊",
            "valor": f"{hrv:.1f} ms",
            "nivel": _nivel_hrv(hrv),
            "explicacion": _explicacion_hrv(hrv),
        })

    if not np.isnan(fr):
        items.append({
            "nombre": "Frecuencia Respiratoria",
            "icono": "🫁",
            "valor": f"{fr:.1f} resp/min",
            "nivel": _nivel_fr(fr),
            "explicacion": _explicacion_fr(fr),
        })

    if not np.isnan(snr):
        items.append({
            "nombre": "Calidad de Señal",
            "icono": "📡",
            "valor": _nivel_snr(snr, sqi).title(),
            "nivel": _nivel_snr(snr, sqi),
            "explicacion": _explicacion_snr(snr, sqi),
        })

    return items

def _interpretar_global(tecnicas: dict):
    hr = tecnicas.get("FC_final_bpm", np.nan)
    hrv = tecnicas.get("HRV_SDNN_ms", np.nan)
    fr = tecnicas.get("FR_final_rpm", np.nan)

    if np.isnan(hr) or np.isnan(hrv) or np.isnan(fr):
        return {
            "estado": "no concluyente",
            "resumen": "No hay suficientes datos para una interpretación.",
            "recomendacion": "Intenta repetir la medición con el dispositivo bien colocado.",
        }

    palabras = []
    if hr > 95 and hrv < 55 and fr > 18:
        estado = "estresado o ansioso"
        palabras = ["activación simpática elevada", "estrés", "ansiedad"]
    elif hr < 70 and hrv > 80 and fr < 15:
        estado = "relajado"
        palabras = ["calma", "recuperación", "relajación"]
    elif 70 <= hr <= 90 and hrv > 60 and 10 <= fr <= 17:
        estado = "concentrado"
        palabras = ["concentración", "estado de flujo", "atención enfocada"]
    else:
        estado = "no concluyente"
        palabras = ["el contexto individual", "no hay un patrón claro"]

    return {
        "estado": estado,
        "resumen": f"Tu organismo muestra un patrón fisiológico de {estado}.",
        "recomendacion": generar_recomendacion(estado, palabras),
    }

def generar_recomendacion(estado: str, palabras: list[str]):
    recs = {
        "estresado o ansioso": (
            "Prueba con respiración profunda (4s inhala, 6s exhala) durante 2 minutos. "
            "Considera tomar una pausa breve para reducir la activación."
        ),
        "relajado": (
            "Excelente estado. Mantén este ritmo con respiración diafragmática lenta "
            "para reforzar la coherencia cardiorrespiratoria."
        ),
        "concentrado": (
            "Buen estado de activación controlada. Si necesitas mantenerlo, realiza "
            "pausas de respiración consciente cada 30 minutos."
        ),
    }
    return recs.get(estado, "No hay una recomendación específica para este patrón.")

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _process_axis_snr(signal: np.ndarray, fs: float):
    sig = signal - np.mean(signal)
    low_q, high_q = np.quantile(sig, [0.002, 0.998])
    sig = np.clip(sig, low_q, high_q)
    sig_f = _bandpass(sig, fs, 0.7, 3.0)
    return _cardiac_snr(sig_f, fs)

def procesar_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))

    time_col, axis_cols = _find_columns(df)
    if time_col is None:
        raise HTTPException(status_code=422, detail="No se encontró columna de tiempo (Time (s))")
    if not axis_cols:
        raise HTTPException(status_code=422, detail="No se encontraron columnas de aceleración")

    t_raw = df[time_col].to_numpy()
    dt = np.median(np.diff(t_raw))
    fs = 1.0 / dt
    if fs < MIN_FS:
        raise HTTPException(status_code=422,
                            detail=f"Frecuencia de muestreo muy baja ({fs:.1f} Hz). Mínimo {MIN_FS} Hz.")

    best_axis = None
    best_snr = -np.inf
    best_col = None
    for axis_name, col_name in axis_cols.items():
        sig = df[col_name].to_numpy().astype(float)
        snr_val = _process_axis_snr(sig, fs)
        if snr_val > best_snr:
            best_snr = snr_val
            best_axis = axis_name
            best_col = col_name

    raw = df[best_col].to_numpy().astype(float)
    signal = raw - np.mean(raw)
    low_q, high_q = np.quantile(signal, [0.002, 0.998])
    signal = np.clip(signal, low_q, high_q)

    signal_c = _bandpass(signal, fs, 0.7, 3.0)

    win_sg = min(31, len(signal_c) - 1 if len(signal_c) % 2 == 0 else len(signal_c))
    if win_sg >= len(signal_c):
        win_sg = max(5, len(signal_c) - 1)
    if win_sg % 2 == 0:
        win_sg -= 1
    signal_sg = savgol_filter(signal_c, win_sg, polyorder=3)
    signal_norm = (signal_sg - np.mean(signal_sg)) / np.std(signal_sg)

    trim = min(100, len(signal_norm) // 4)
    zf = signal_norm[trim:-trim]
    t_seg = t_raw[trim:-trim] - t_raw[trim]
    fs_actual = 1.0 / np.median(np.diff(t_seg))

    fc_auto = _fc_autocorr(zf, fs_actual)
    fc_fft = _fc_fft(zf, fs_actual)
    fc_welch = _fc_welch(zf, fs_actual)

    peaks = _detect_peaks_bcg(zf, fs_actual)
    ibis = _clean_ibis(peaks, fs_actual)
    fc_pk = _fc_peaks(ibis)

    fc_final, fc_confianza = _fc_consensus({
        "autocorr": fc_auto, "fft": fc_fft, "welch": fc_welch, "peaks": fc_pk,
    })

    hrv = _compute_hrv(ibis, fs_actual)

    fr_w = _fr_welch(signal, fs)
    fr_env = _fr_envelope(zf, fs_actual)
    fr_pk = _fr_peaks(signal, fs)
    fr_final = _fr_consensus([fr_w, fr_env, fr_pk])

    f_w_full, Pxx_full = welch(zf, fs=fs_actual,
                                nperseg=int(min(4 * fs_actual, len(zf) // 2)))
    band_hr = (f_w_full >= 0.7) & (f_w_full <= 3.0)
    noise_mask = (~band_hr) & (f_w_full > 0)
    snr_val = float(10 * np.log10(
        (Pxx_full[band_hr].sum() + 1e-12) / (Pxx_full[noise_mask].sum() + 1e-12)
    ))

    n_agree = sum(1 for v in [fc_auto, fc_fft, fc_welch, fc_pk]
                  if not np.isnan(v) and abs(v - fc_final) <= 3)
    sqi = float(min(1.0, (n_agree / 4) * 0.7 + max(0, snr_val) / 20 * 0.3))

    tecnicas = {
        "fs_Hz": round(fs_actual, 1),
        "duracion_s": round(float(t_seg[-1]), 1),
        "eje_principal": best_axis.upper(),
        "SNR_cardiaco_dB": round(snr_val, 2),
        "SQI": round(sqi, 2),
        "FC_final_bpm": round(fc_final, 1) if not np.isnan(fc_final) else None,
        "FC_autocorr_bpm": round(fc_auto, 1) if not np.isnan(fc_auto) else None,
        "FC_fft_bpm": round(fc_fft, 1) if not np.isnan(fc_fft) else None,
        "FC_welch_bpm": round(fc_welch, 1) if not np.isnan(fc_welch) else None,
        "FC_peaks_bpm": round(fc_pk, 1) if not np.isnan(fc_pk) else None,
        "confianza_consenso": round(fc_confianza, 2),
        "FR_final_rpm": round(fr_final, 1) if not np.isnan(fr_final) else None,
        "FR_welch_rpm": round(fr_w, 1) if not np.isnan(fr_w) else None,
        "FR_envelope_rpm": round(fr_env, 1) if not np.isnan(fr_env) else None,
        "FR_peaks_rpm": round(fr_pk, 1) if not np.isnan(fr_pk) else None,
        "HRV_SDNN_ms": round(hrv["SDNN_ms"], 1) if not np.isnan(hrv["SDNN_ms"]) else None,
        "HRV_RMSSD_ms": round(hrv["RMSSD_ms"], 1) if not np.isnan(hrv["RMSSD_ms"]) else None,
        "HRV_pNN50": round(hrv["pNN50"], 1) if not np.isnan(hrv["pNN50"]) else None,
        "HRV_LF_Hz": round(hrv["LF_Hz"], 3) if not np.isnan(hrv["LF_Hz"]) else None,
        "HRV_HF_Hz": round(hrv["HF_Hz"], 3) if not np.isnan(hrv["HF_Hz"]) else None,
        "HRV_LF_HF_ratio": round(hrv["LF_HF_ratio"], 2) if not np.isnan(hrv["LF_HF_ratio"]) else None,
    }

    return {
        "tecnicas": tecnicas,
        "coloquiales": _generar_coloquiales(tecnicas),
        "interpretacion_global": _interpretar_global(tecnicas),
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/procesar")
async def procesar(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 50 MB)")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Solo se aceptan archivos CSV")
    try:
        results = procesar_csv(file_bytes)
        return {"status": "ok", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
