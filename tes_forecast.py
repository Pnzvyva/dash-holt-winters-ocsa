# tes_forecast.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class TESForecaster:
    """Carga datos TES, corre diagnósticos y pronostica con Holt-Winters."""

    def __init__(self):
        self.data = None
        self.last_error = None  # <- importante para la app

    # ------------------------- Carga y limpieza -------------------------
    def load_data(self, file_like):
        """Carga y limpia datos desde un Excel (archivo o buffer subido en Streamlit)."""
        try:
            self.last_error = None

            xls = pd.ExcelFile(file_like)
            sheet = "Datos" if "Datos" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)

            # Normaliza nombres de columnas
            df.columns = df.columns.str.replace(r"\(Dato diario\)", "", regex=True).str.strip()

            if "Fecha" not in df.columns:
                raise ValueError("No se encontró una columna llamada 'Fecha'.")

            # Limpieza de fechas
            df = df[df["Fecha"].notna() & (df["Fecha"].astype(str).str.strip() != "")]
            df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Fecha"]).set_index("Fecha").sort_index()

            # Conversión numérica segura
            for c in df.columns:
                s = df[c].astype(str).str.replace(",", ".", regex=False)
                s = s.replace(["-", "", "nan", "NaN"], np.nan)
                df[c] = pd.to_numeric(s, errors="coerce")

            self.data = df
            return True

        except Exception as e:
            self.last_error = str(e)
            self.data = None
            return False

    def get_data_info(self):
        if self.data is None:
            return {}
        return {
            "fecha_inicio": self.data.index.min(),
            "fecha_fin": self.data.index.max(),
            "filas": len(self.data),
            "columnas": list(self.data.columns),
            "faltantes": self.data.isnull().sum().to_dict(),
        }

    def get_numeric_columns(self):
        if self.data is None:
            return []
        return [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]

    # --------------------------- Diagnóstico ---------------------------
    def run_diagnostics(self, column):
        serie = self.data[column].dropna()
        res = {}

        adf = adfuller(serie)
        res["ADF"] = {"estadistico": adf[0], "p": adf[1], "estacionaria": adf[1] < 0.05}

        try:
            kpss_res = kpss(serie, regression="c", nlags="auto")
            res["KPSS"] = {"estadistico": kpss_res[0], "p": kpss_res[1], "estacionaria": kpss_res[1] > 0.05}
        except Exception:
            res["KPSS"] = {"error": "No se pudo calcular KPSS"}

        if 3 <= len(serie) <= 5000:
            sh = stats.shapiro(serie)
            res["Shapiro-Wilk"] = {"estadistico": sh[0], "p": sh[1], "normal": sh[1] > 0.05}

        lj = acorr_ljungbox(serie, lags=10, return_df=True)
        res["Autocorrelacion_LjungBox_p<0.05"] = bool((lj["lb_pvalue"] < 0.05).any())

        x = np.arange(len(serie))
        slope, intercept, r, p, se = stats.linregress(x, serie)
        res["Tendencia"] = {"pendiente": slope, "p": p, "hay_tendencia": p < 0.05}

        acf_vals = acf(serie, nlags=40)
        res["Estacionalidad"] = {"hay_estacionalidad": bool((np.abs(acf_vals[1:]) > 0.3).any())}

        return res

    # ---------------------------- Pronóstico ---------------------------
    def holt_winters_forecast(self, column, periodos=30, estacionalidad=12, tendencia=None, tipo_estacional=None):
        serie = self.data[column].dropna()
        modelo = ExponentialSmoothing(
            serie,
            trend=tendencia,
            seasonal=tipo_estacional,
            seasonal_periods=estacionalidad if tipo_estacional else None
        ).fit()

        fcst = modelo.forecast(periodos)
        fitted = modelo.fittedvalues

        mae = mean_absolute_error(serie[-len(fitted):], fitted)
        mse = mean_squared_error(serie[-len(fitted):], fitted)
        rmse = float(np.sqrt(mse))

        return {
            "serie": serie,
            "ajuste": fitted,
            "pronostico": fcst,
            "metricas": {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse},
            "parametros": {"tendencia": tendencia, "estacional": tipo_estacional, "periodos": estacionalidad},
        }

    # ----------------------------- Gráficos ----------------------------
    def plot_diagnostics(self, column):
        serie = self.data[column].dropna()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Diagnóstico de la serie: {column}")

        axes[0, 0].plot(serie.index, serie.values)
        axes[0, 0].set_title("Serie temporal")

        axes[0, 1].hist(serie.values, bins=20, color="lightgray", edgecolor="black")
        axes[0, 1].set_title("Distribución")

        stats.probplot(serie.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Gráfico Q-Q")

        acf_vals = acf(serie, nlags=30)
        axes[1, 1].stem(range(len(acf_vals)), acf_vals)
        axes[1, 1].set_title("Función de autocorrelación")

        plt.tight_layout()
        return fig
