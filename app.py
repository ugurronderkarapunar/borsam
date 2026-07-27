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
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# ---------- BIST 100 Listesi ----------
@st.cache_data(ttl=86400)
def bist100_listesi():
    url = "https://en.wikipedia.org/wiki/BIST_100"
    tablo = pd.read_html(url)[0]
    if 'Symbol' in tablo.columns:
        semboller = tablo['Symbol'].dropna().tolist()
    else:
        semboller = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "KCHOL.IS"]
    return [s for s in semboller if isinstance(s, str) and len(s) > 2]

# ---------- Veri çekme (Hisse) ----------
@st.cache_data(ttl=3600)
def veri_cek_hisse(ticker, donem="1y", aralik="1d"):
    hisse = yf.Ticker(ticker)
    df = hisse.history(period=donem, interval=aralik)
    if df.empty:
        raise ValueError(f"{ticker} için veri çekilemedi.")
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
    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    gelecek = model.make_future_dataframe(periods=gun)
    tahmin = model.predict(gelecek)
    tahmin_df = tahmin[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(gun)
    tahmin_df = tahmin_df.rename(columns={'ds': 'Tarih', 'yhat': 'Tahmin', 'yhat_lower': 'Alt', 'yhat_upper': 'Ust'})
    return tahmin_df.reset_index(drop=True)

def arima_tahmin(df, gun=30):
    model = pm.auto_arima(df['Close'], seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True)
    tahmin, guven_araligi = model.predict(n_periods=gun, return_conf_int=True)
    son_tarih = df.index[-1]
    gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=gun)
    tahmin_df = pd.DataFrame({'Tarih': gelecek_tarihler, 'Tahmin': tahmin, 'Alt': guven_araligi[:, 0], 'Ust': guven_araligi[:, 1]})
    return tahmin_df

def holt_winters_tahmin(df, gun=30):
    model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=5)
    fitted = model.fit()
    tahmin = fitted.forecast(gun)
    son_tarih = df.index[-1]
    gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=gun)
    tahmin_df = pd.DataFrame({'Tarih': gelecek_tarihler, 'Tahmin': tahmin, 'Alt': np.nan, 'Ust': np.nan})
    return tahmin_df

# ---------- Sinyal Üretimi ----------
def sinyal_uret(df, tahmin_df, risk_profili="dengeli"):
    son_kapanis = df['Close'].iloc[-1]
    gelecek_fiyat = tahmin_df['Tahmin'].iloc[-1]
    beklenen_degisim = (gelecek_fiyat - son_kapanis) / son_kapanis
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_sinyal = df['MACD_sinyal'].iloc[-1]

    puan = 0
    if beklenen_degisim > 0.05: puan += 2
    elif beklenen_degisim > 0.02: puan += 1
    elif beklenen_degisim < -0.05: puan -= 2
    elif beklenen_degisim < -0.02: puan -= 1

    if not np.isnan(rsi):
        if rsi < 30: puan += 1.5
        elif rsi > 70: puan -= 1.5

    if not np.isnan(macd) and not np.isnan(macd_sinyal):
        if macd > macd_sinyal: puan += 1
        else: puan -= 1

    if risk_profili == "muhafazakar": al_esik, sat_esik = 3.0, -2.0
    elif risk_profili == "agresif": al_esik, sat_esik = 1.5, -1.0
    else: al_esik, sat_esik = 2.0, -1.5

    if puan >= al_esik: return "AL", puan, beklenen_degisim
    elif puan <= sat_esik: return "SAT", puan, beklenen_degisim
    else: return "TUT", puan, beklenen_degisim

# ---------- Gelişmiş İstatistiksel Analiz ----------
def gelismis_analiz(df):
    st.subheader("🧪 Gelişmiş İstatistiksel Analiz (Log Getiri + ARIMA + GARCH)")
    getiri = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    st.write(f"**Veri sayısı:** {len(getiri)} gün")

    try:
        arima_model = pm.auto_arima(getiri, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True, stepwise=True, max_p=5, max_q=5)
        st.write(f"**Seçilen ARIMA modeli:** {arima_model.order}")
    except Exception as e:
        st.error(f"ARIMA seçilemedi: {e}")
        return

    try:
        kalintilar = arima_model.resid()
        lb_test = acorr_ljungbox(kalintilar.dropna(), lags=[10, 20], return_df=True)
        st.write("**Ljung-Box Testi:**")
        st.dataframe(lb_test)
        p_degeri = lb_test.iloc[0, 1]
        if p_degeri > 0.05:
            st.success("✅ Kalıntılar beyaz gürültü özelliğinde (p > 0.05).")
        else:
            st.warning("⚠️ Kalıntılarda otokorelasyon var (p ≤ 0.05).")
    except Exception as e:
        st.error(f"Kalıntı analizi başarısız: {e}")
        return

    try:
        garch_model = arch_model(kalintilar.dropna(), vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        st.write("**GARCH(1,1) Model Özeti:**")
        st.text(garch_fit.summary().tables[1].as_text())
    except Exception as e:
        st.error(f"GARCH uyumu başarısız: {e}")
        return

    try:
        tahmin_vol = garch_fit.forecast(horizon=30)
        vol_forecast = tahmin_vol.variance.values[-1, :]
        son_tarih = getiri.index[-1]
        gelecek_tarihler = pd.bdate_range(start=son_tarih + pd.Timedelta(days=1), periods=30)
        vol_df = pd.DataFrame({'Tarih': gelecek_tarihler, 'Volatilite (varyans)': vol_forecast})
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(x=vol_df['Tarih'], y=vol_df['Volatilite (varyans)'], name='Volatilite', line=dict(color='red')))
        vol_fig.update_layout(title="30 Günlük Koşullu Volatilite Tahmini", xaxis_title="Tarih", yaxis_title="Varyans")
        st.plotly_chart(vol_fig, use_container_width=True)

        ortalama_getiri = arima_model.predict(n_periods=30)
        birikimli_getiri = np.exp(np.cumsum(ortalama_getiri))
        tahmini_fiyat = df['Close'].iloc[-1] * birikimli_getiri
        fiyat_df = pd.DataFrame({'Tarih': gelecek_tarihler, 'Tahmini Fiyat': tahmini_fiyat})
        st.write("**ARIMA+GARCH ile 30 Günlük Fiyat Tahmini:**")
        st.dataframe(fiyat_df.style.format({'Tahmini Fiyat': '{:.2f}'}))

        beklenen_toplam_getiri = np.sum(ortalama_getiri)
        yillik_vol = np.sqrt(np.mean(vol_forecast)) * np.sqrt(252)
        sharpe = beklenen_toplam_getiri / (yillik_vol * np.sqrt(30/252) + 1e-6)
        st.write(f"**Tahmini 30 günlük log getiri:** {beklenen_toplam_getiri:.4f}")
        st.write(f"**Yıllıklandırılmış volatilite:** {yillik_vol:.4f}")
        st.write(f"**Basit Sharpe oranı:** {sharpe:.2f}")
        if sharpe > 0.5: st.success("📈 Risk/getiri olumlu.")
        elif sharpe < -0.5: st.error("📉 Risk/getiri olumsuz.")
        else: st.info("⚖️ Nötr.")
    except Exception as e:
        st.error(f"Volatilite tahmini başarısız: {e}")

# ---------- Ana Arayüz ----------
st.set_page_config(page_title="Borsa Asistanım", layout="wide")
st.title("📈 Zaman Serisi Analizli Kişisel Yatırım Asistanı")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    risk = st.selectbox("Risk Profiliniz", ["agresif", "dengeli", "muhafazakar"])
    model_secimi = st.selectbox("Tahmin Modeli", ["Prophet", "ARIMA", "Holt-Winters"])
    donem = st.selectbox("Geçmiş Veri Aralığı", ["6 ay", "1 yıl", "2 yıl", "5 yıl"])
    donem_haritasi = {"6 ay": "6mo", "1 yıl": "1y", "2 yıl": "2y", "5 yıl": "5y"}

hisse_listesi = bist100_listesi()
hisse_secim = st.radio("Hisse Seçim Yöntemi", ["BIST 100 Listesi", "Manuel Kod Gir"])
if hisse_secim == "BIST 100 Listesi":
    sembol = st.selectbox("Hisse Seçin", hisse_listesi)
else:
    sembol = st.text_input("Hisse Sembolü (örn: THYAO.IS)", "THYAO.IS").upper()

gelismis_on = st.checkbox("Gelişmiş İstatistiksel Analiz (ARIMA‑GARCH)")

col1, col2 = st.columns([1,1])
with col1:
    analiz_btn = st.button("🔍 Hisseyi Analiz Et")
with col2:
    toplu_tara_btn = st.button("🚀 BIST 100 Toplu Tara")

if analiz_btn:
    with st.spinner("Veri çekiliyor..."):
        try:
            df = veri_cek_hisse(sembol, donem=donem_haritasi[donem])
            if model_secimi == "Prophet":
                tahmin_df = prophet_tahmin(df)
            elif model_secimi == "ARIMA":
                tahmin_df = arima_tahmin(df)
            else:
                tahmin_df = holt_winters_tahmin(df)

            sinyal, puan, beklenen_degisim = sinyal_uret(df, tahmin_df, risk)

            c1, c2, c3 = st.columns(3)
            c1.metric("Son Kapanış", f"₺{df['Close'].iloc[-1]:.2f}")
            c2.metric(f"30 Günlük Beklenen Değişim", f"%{beklenen_degisim*100:.2f}")
            c3.metric("Sinyal", sinyal, delta=puan)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçek Fiyat'))
            fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Tahmin'], name='Tahmin', line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

            if gelismis_on:
                gelismis_analiz(df)

            if sinyal == "AL": st.success(f"✅ Sinyal: **AL** (Puan: {puan:.2f})")
            elif sinyal == "SAT": st.error(f"❌ Sinyal: **SAT** (Puan: {puan:.2f})")
            else: st.warning(f"⏸️ Sinyal: **TUT** (Puan: {puan:.2f})")

        except Exception as e:
            st.error(f"❌ Hata: {e}")

if toplu_tara_btn:
    st.subheader("📋 BIST 100 Toplu Tarama")
    with st.spinner("Taranıyor..."):
        sonuc = []
        for s in hisse_listesi:
            try:
                df = veri_cek_hisse(s, donem="6mo")
                tahmin_df = prophet_tahmin(df)
                sinyal, puan, _ = sinyal_uret(df, tahmin_df, risk)
                if sinyal in ["AL", "SAT"]:
                    sonuc.append({"Hisse": s, "Sinyal": sinyal, "Puan": round(puan,2), "Fiyat": round(df['Close'].iloc[-1],2)})
            except: pass
        if sonuc:
            sonuc_df = pd.DataFrame(sonuc)
            def renklendir(val):
                if val == "AL": return 'color: green; font-weight: bold'
                elif val == "SAT": return 'color: red; font-weight: bold'
                return ''
            styled = sonuc_df.style.map(renklendir, subset=['Sinyal'])
            st.dataframe(styled, use_container_width=True)
            st.success(f"{len(sonuc)} hisse sinyal verdi.")
        else:
            st.warning("Hiç sinyal yok.")

st.caption("⚠️ Eğitim amaçlıdır, yatırım tavsiyesi değildir.")
