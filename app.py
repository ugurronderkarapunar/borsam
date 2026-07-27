import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from prophet import Prophet
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

# ---------- Veri çekme ve teknik göstergeler ----------
def veri_cek(ticker, donem="1y", aralik="1d"):
    hisse = yf.Ticker(ticker)
    df = hisse.history(period=donem, interval=aralik)
    if df.empty:
        raise ValueError("Veri çekilemedi. Sembolü kontrol edin.")
    df.index = df.index.tz_localize(None)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_sinyal'] = macd.macd_signal()
    df['Gunluk_Getiri'] = df['Close'].pct_change()
    df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# ---------- Zaman Serisi Modelleri ----------
def prophet_tahmin(df, gun=30):
    df = df.copy()
    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    gelecek = model.make_future_dataframe(periods=gun)
    tahmin = model.predict(gelecek)
    tahmin_df = tahmin[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(gun)
    tahmin_df = tahmin_df.rename(columns={
        'ds': 'Tarih',
        'yhat': 'Tahmin',
        'yhat_lower': 'Alt',
        'yhat_upper': 'Ust'
    })
    return tahmin_df.reset_index(drop=True)

def arima_tahmin(df, gun=30):
    model = pm.auto_arima(df['Close'], seasonal=False, trace=False,
                          error_action='ignore', suppress_warnings=True,
                          stepwise=True)
    tahmin, guven_araligi = model.predict(n_periods=gun, return_conf_int=True)
    son_tarih = df.index[-1]
    gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=gun)
    tahmin_df = pd.DataFrame({
        'Tarih': gelecek_tarihler,
        'Tahmin': tahmin,
        'Alt': guven_araligi[:, 0],
        'Ust': guven_araligi[:, 1]
    })
    return tahmin_df

def holt_winters_tahmin(df, gun=30):
    model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add',
                                 seasonal_periods=5)
    fitted = model.fit()
    tahmin = fitted.forecast(gun)
    son_tarih = df.index[-1]
    gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=gun)
    tahmin_df = pd.DataFrame({
        'Tarih': gelecek_tarihler,
        'Tahmin': tahmin,
        'Alt': np.nan,
        'Ust': np.nan
    })
    return tahmin_df

# ---------- Kişisel sinyal ----------
def sinyal_uret(df, tahmin_df, risk_profili="dengeli"):
    son_kapanis = df['Close'].iloc[-1]
    gelecek_fiyat = tahmin_df['Tahmin'].iloc[-1]
    beklenen_degisim = (gelecek_fiyat - son_kapanis) / son_kapanis

    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_sinyal = df['MACD_sinyal'].iloc[-1]

    puan = 0
    if beklenen_degisim > 0.05:
        puan += 2
    elif beklenen_degisim > 0.02:
        puan += 1
    elif beklenen_degisim < -0.05:
        puan -= 2
    elif beklenen_degisim < -0.02:
        puan -= 1

    if rsi < 30:
        puan += 1.5
    elif rsi > 70:
        puan -= 1.5

    if macd > macd_sinyal:
        puan += 1
    else:
        puan -= 1

    if risk_profili == "muhafazakar":
        al_esik = 3.0
        sat_esik = -2.0
    elif risk_profili == "agresif":
        al_esik = 1.5
        sat_esik = -1.0
    else:  # dengeli
        al_esik = 2.0
        sat_esik = -1.5

    if puan >= al_esik:
        return "AL", puan, beklenen_degisim
    elif puan <= sat_esik:
        return "SAT", puan, beklenen_degisim
    else:
        return "TUT", puan, beklenen_degisim

# ---------- Streamlit Arayüz ----------
st.set_page_config(page_title="Borsa Asistanım", layout="wide")
st.title("📈 Zaman Serisi Analizli Kişisel Yatırım Asistanı")

sembol = st.text_input("Hisse Sembolü (örn: THYAO.IS, GARAN.IS)", "THYAO.IS")
risk = st.selectbox("Risk Profiliniz", ["agresif", "dengeli", "muhafazakar"])
donem = st.selectbox("Geçmiş Veri Aralığı", ["6ay","1y","2y","5y"])
model_secimi = st.selectbox("Tahmin Modeli", ["Prophet", "ARIMA", "Holt-Winters"])

# Dönem haritası (Streamlit seçenekleri İngilizce olmamalı)
donem_haritasi = {
    "6ay": "6mo",
    "1y": "1y",
    "2y": "2y",
    "5y": "5y"
}

if st.button("Analiz Et"):
    with st.spinner("Veri çekiliyor ve model eğitiliyor..."):
        try:
            df = veri_cek(sembol, donem=donem_haritasi[donem])

            if model_secimi == "Prophet":
                tahmin_df = prophet_tahmin(df, gun=30)
            elif model_secimi == "ARIMA":
                tahmin_df = arima_tahmin(df, gun=30)
            elif model_secimi == "Holt-Winters":
                tahmin_df = holt_winters_tahmin(df, gun=30)

            sinyal, puan, beklenen_degisim = sinyal_uret(df, tahmin_df, risk)

            # Sonuç metrikleri
            kolon1, kolon2, kolon3 = st.columns(3)
            kolon1.metric("Son Kapanış", f"₺{df['Close'].iloc[-1]:.2f}")
            kolon2.metric(f"30 Günlük Beklenen Değişim ({model_secimi})",
                          f"%{beklenen_degisim*100:.2f}")
            kolon3.metric("Sinyal", sinyal, delta=puan)

            # Fiyat ve tahmin grafiği
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                     name='Gerçek Fiyat', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Tahmin'],
                                     name=f'{model_secimi} Tahmini', line=dict(dash='dash', color='orange')))
            if 'Alt' in tahmin_df.columns and not tahmin_df['Alt'].isna().all():
                fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Alt'],
                                         mode='lines', line=dict(color='gray', dash='dot'),
                                         name='Alt Sınır'))
                fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Ust'],
                                         fill='tonexty', mode='lines',
                                         line=dict(color='gray', dash='dot'),
                                         name='Üst Sınır'))
            fig.update_layout(
                title=f"{sembol} – Gerçek ve Tahmini Fiyat ({model_secimi})",
                xaxis_title="Tarih",
                yaxis_title="Fiyat (₺)",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)

            # RSI Göstergesi
            st.subheader("RSI Göstergesi")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'],
                                         name='RSI', line=dict(color='purple')))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red",
                              annotation_text="Aşırı Alım 70")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green",
                              annotation_text="Aşırı Satım 30")
            rsi_fig.update_layout(
                title="RSI (Göreceli Güç Endeksi)",
                xaxis_title="Tarih",
                yaxis_title="RSI Değeri"
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

            # Tüm modelleri karşılaştır
            if st.checkbox("Tüm modelleri karşılaştır"):
                st.subheader("Model Tahminleri Karşılaştırması")
                modeller = {
                    "Prophet": prophet_tahmin(df, 30),
                    "ARIMA": arima_tahmin(df, 30),
                    "Holt-Winters": holt_winters_tahmin(df, 30)
                }
                karsilastirma_fig = go.Figure()
                karsilastirma_fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                                                       name='Gerçek Fiyat', line=dict(color='blue')))
                renkler = ['orange', 'green', 'red']
                for (isim, fdf), renk in zip(modeller.items(), renkler):
                    karsilastirma_fig.add_trace(go.Scatter(x=fdf['Tarih'], y=fdf['Tahmin'],
                                                           name=isim, line=dict(color=renk, dash='dot')))
                karsilastirma_fig.update_layout(
                    title="Tüm Modellerin Tahmin Karşılaştırması",
                    xaxis_title="Tarih",
                    yaxis_title="Fiyat (₺)"
                )
                st.plotly_chart(karsilastirma_fig, use_container_width=True)

            # Sinyal açıklaması
            if sinyal == "AL":
                st.success(f"✅ Sinyal: **AL** (Puan: {puan:.2f}) – Model {model_secimi} alım fırsatı gösteriyor.")
            elif sinyal == "SAT":
                st.error(f"❌ Sinyal: **SAT** (Puan: {puan:.2f}) – Model {model_secimi} satış baskısı öngörüyor.")
            else:
                st.warning(f"⏸️ Sinyal: **TUT** (Puan: {puan:.2f}) – Şu an beklemede kalmak daha uygun görünüyor.")

            st.info(f"🔍 Detay: Beklenen değişim %{beklenen_degisim*100:.2f} | Risk profili: **{risk}** | Model: **{model_secimi}**")

        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")

st.caption("⚠️ Bu uygulama yalnızca eğitim ve kişisel gelişim amaçlıdır. Kesinlikle yatırım tavsiyesi içermez.")
