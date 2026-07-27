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
import tefas
import warnings
warnings.filterwarnings("ignore")

# ---------- BIST 100 Listesi (Wikipedia) ----------
@st.cache_data(ttl=86400)
def bist100_listesi():
    url = "https://en.wikipedia.org/wiki/BIST_100"
    tablo = pd.read_html(url)[0]
    if 'Symbol' in tablo.columns:
        semboller = tablo['Symbol'].dropna().tolist()
    else:
        semboller = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "KCHOL.IS"]
    return [s for s in semboller if isinstance(s, str) and len(s) > 2]

# ---------- Fon Listesi (TEFAS) ----------
@st.cache_data(ttl=86400)
def fon_listesi():
    try:
        # tefas'tan tüm fon kodlarını al
        fonlar = tefas.get_all_funds()
        return fonlar['KOD'].tolist()
    except:
        # Yedek liste
        return ["YF1", "YF2", "YF3", "YF4", "YF5"]

# ---------- Veri Çekme (Hisse) ----------
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

# ---------- Veri Çekme (Fon) ----------
@st.cache_data(ttl=3600)
def veri_cek_fon(fon_kodu, donem="1y"):
    # TEFAS'tan günlük veri çek
    try:
        df = tefas.get_fund_history(fon_kodu, period=donem)
        if df.empty:
            raise ValueError("Fon verisi çekilemedi.")
        df = df[['TARIH', 'FIYAT']].rename(columns={'TARIH': 'Date', 'FIYAT': 'Close'})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        # Teknik göstergeler
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        # MACD için yeterli veri varsa
        if len(df) >= 26:
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_sinyal'] = macd.macd_signal()
        else:
            df['MACD'] = np.nan
            df['MACD_sinyal'] = np.nan
        df['Gunluk_Getiri'] = df['Close'].pct_change()
        df['Volatilite'] = df['Gunluk_Getiri'].rolling(window=20).std()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise ValueError(f"Fon verisi alınamadı: {e}")

# ---------- Zaman Serisi Modelleri (Hem hisse hem fon için) ----------
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

# ---------- Sinyal üretimi ----------
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

    if not np.isnan(rsi):
        if rsi < 30:
            puan += 1.5
        elif rsi > 70:
            puan -= 1.5

    if not np.isnan(macd) and not np.isnan(macd_sinyal):
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
    else:
        al_esik = 2.0
        sat_esik = -1.5

    if puan >= al_esik:
        return "AL", puan, beklenen_degisim
    elif puan <= sat_esik:
        return "SAT", puan, beklenen_degisim
    else:
        return "TUT", puan, beklenen_degisim

# ---------- Ana Uygulama ----------
st.set_page_config(page_title="Borsa & Fon Asistanım", layout="wide")
st.title("📈 Zaman Serisi Analizli Kişisel Yatırım Asistanı")
st.markdown("Bu uygulama hisse senetleri ve yatırım fonları için teknik analiz ve yapay zeka destekli tahminler sunar.")

# Sekme yapısı
sekme1, sekme2 = st.tabs(["📊 Hisse Senetleri", "💰 Yatırım Fonları"])

# Ortak ayarlar (sidebar)
with st.sidebar:
    st.header("⚙️ Genel Ayarlar")
    risk_aciklama = {
        "agresif": "Yüksek risk, küçük sinyallere duyarlı.",
        "dengeli": "Orta risk, belirgin sinyallere bakar.",
        "muhafazakar": "Düşük risk, sadece güçlü sinyaller."
    }
    risk = st.selectbox("Risk Profiliniz", ["agresif", "dengeli", "muhafazakar"],
                        help="**Agresif:** Hızlı al-sat.\n**Dengeli:** Dengeli.\n**Muhafazakar:** Uzun vadeli.")
    st.caption(risk_aciklama[risk])
    model_secimi = st.selectbox("Tahmin Modeli", ["Prophet", "ARIMA", "Holt-Winters"],
                                help="**Prophet:** Tatil/mevsim etkilerini yakalar.\n**ARIMA:** Klasik istatistik.\n**Holt-Winters:** Mevsimsel üstel düzleştirme.")
    donem = st.selectbox("Geçmiş Veri Aralığı", ["6 ay", "1 yıl", "2 yıl", "5 yıl"])
    donem_haritasi = {"6 ay": "6mo", "1 yıl": "1y", "2 yıl": "2y", "5 yıl": "5y"}

# ---------- Hisse Senetleri Sekmesi ----------
with sekme1:
    st.subheader("📊 Hisse Senedi Analizi")
    hisse_secim = st.radio("Hisse Seçim Yöntemi", ["BIST 100 Listesi", "Manuel Kod Gir"],
                           help="BIST 100'den seçebilir veya herhangi bir BIST hisse kodunu yazabilirsiniz (örn: THYAO.IS).")
    if hisse_secim == "BIST 100 Listesi":
        hisse_listesi = bist100_listesi()
        sembol = st.selectbox("Hisse Seçin", hisse_listesi)
    else:
        sembol = st.text_input("Hisse Sembolü (örn: THYAO.IS, GARAN.IS)", "THYAO.IS").upper()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        analiz_btn = st.button("🔍 Hisseyi Analiz Et", key="hisse_analiz")
    with col2:
        toplu_tara_btn = st.button("🚀 BIST 100 Toplu Tara", key="toplu_tara")

    if analiz_btn:
        with st.spinner("Veri çekiliyor ve model eğitiliyor..."):
            try:
                df = veri_cek_hisse(sembol, donem=donem_haritasi[donem])
                if model_secimi == "Prophet":
                    tahmin_df = prophet_tahmin(df, gun=30)
                elif model_secimi == "ARIMA":
                    tahmin_df = arima_tahmin(df, gun=30)
                else:
                    tahmin_df = holt_winters_tahmin(df, gun=30)

                sinyal, puan, beklenen_degisim = sinyal_uret(df, tahmin_df, risk)

                c1, c2, c3 = st.columns(3)
                c1.metric("Son Kapanış", f"₺{df['Close'].iloc[-1]:.2f}")
                c2.metric(f"30 Günlük Beklenen Değişim ({model_secimi})", f"%{beklenen_degisim*100:.2f}")
                c3.metric("Sinyal", sinyal, delta=puan)

                # Grafik
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçek Fiyat', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Tahmin'], name=f'{model_secimi} Tahmin', line=dict(dash='dash', color='orange')))
                if 'Alt' in tahmin_df.columns and not tahmin_df['Alt'].isna().all():
                    fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Alt'], mode='lines', line=dict(color='gray', dash='dot'), name='Alt Sınır'))
                    fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Ust'], fill='tonexty', mode='lines', line=dict(color='gray', dash='dot'), name='Üst Sınır'))
                fig.update_layout(title=f"{sembol} – Gerçek ve Tahmini Fiyat ({model_secimi})", xaxis_title="Tarih", yaxis_title="Fiyat (₺)")
                st.plotly_chart(fig, use_container_width=True)

                # RSI
                st.subheader("RSI (Göreceli Güç Endeksi)")
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
                st.plotly_chart(rsi_fig, use_container_width=True)

                if sinyal == "AL":
                    st.success(f"✅ Sinyal: **AL** (Puan: {puan:.2f})")
                elif sinyal == "SAT":
                    st.error(f"❌ Sinyal: **SAT** (Puan: {puan:.2f})")
                else:
                    st.warning(f"⏸️ Sinyal: **TUT** (Puan: {puan:.2f})")
            except Exception as e:
                st.error(f"❌ Hata: {e}")

    if toplu_tara_btn:
        st.subheader("📋 BIST 100 Toplu Tarama")
        with st.spinner("Taranıyor..."):
            hisseler = bist100_listesi()
            sonuc_listesi = []
            for i, s in enumerate(hisseler):
                try:
                    df = veri_cek_hisse(s, donem="6mo")
                    tahmin_df = prophet_tahmin(df, gun=30)
                    sinyal, puan, _ = sinyal_uret(df, tahmin_df, risk)
                    if sinyal in ["AL", "SAT"]:
                        sonuc_listesi.append({"Hisse": s, "Sinyal": sinyal, "Puan": round(puan,2), "Fiyat": round(df['Close'].iloc[-1],2)})
                except:
                    pass
            if sonuc_listesi:
                sonuc_df = pd.DataFrame(sonuc_listesi)
                def renklendir(val):
                    if val == "AL": return 'color: green; font-weight: bold'
                    elif val == "SAT": return 'color: red; font-weight: bold'
                    return ''
                styled = sonuc_df.style.map(renklendir, subset=['Sinyal'])
                st.dataframe(styled, use_container_width=True)
                st.success(f"{len(sonuc_listesi)} hisse sinyal verdi.")
            else:
                st.warning("Hiç sinyal yok.")

# ---------- Yatırım Fonları Sekmesi ----------
with sekme2:
    st.subheader("💰 Yatırım Fonu Analizi")
    try:
        fon_kodlari = fon_listesi()
        fon_kodu = st.selectbox("Fon Kodu Seçin", fon_kodlari, help="TEFAS'taki tüm fon kodları listelenmiştir.")
    except:
        fon_kodu = st.text_input("Fon Kodu (örn: YF1)", "YF1")
    
    if st.button("🔍 Fonu Analiz Et", key="fon_analiz"):
        with st.spinner("Fon verisi çekiliyor..."):
            try:
                df_fon = veri_cek_fon(fon_kodu, donem="1y")  # fonlar için varsayılan 1 yıl
                if len(df_fon) < 20:
                    st.error("Yeterli veri yok (en az 20 iş günü).")
                else:
                    if model_secimi == "Prophet":
                        tahmin_df = prophet_tahmin(df_fon, gun=30)
                    elif model_secimi == "ARIMA":
                        tahmin_df = arima_tahmin(df_fon, gun=30)
                    else:
                        tahmin_df = holt_winters_tahmin(df_fon, gun=30)

                    sinyal, puan, beklenen_degisim = sinyal_uret(df_fon, tahmin_df, risk)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Son Fiyat", f"₺{df_fon['Close'].iloc[-1]:.4f}")
                    c2.metric(f"30 Günlük Beklenen Değişim", f"%{beklenen_degisim*100:.2f}")
                    c3.metric("Sinyal", sinyal, delta=puan)

                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=df_fon.index, y=df_fon['Close'], name='Gerçek Fiyat', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Tahmin'], name=f'{model_secimi} Tahmin', line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)

                    if sinyal == "AL":
                        st.success(f"✅ Sinyal: **AL** (Puan: {puan:.2f})")
                    elif sinyal == "SAT":
                        st.error(f"❌ Sinyal: **SAT** (Puan: {puan:.2f})")
                    else:
                        st.warning(f"⏸️ Sinyal: **TUT** (Puan: {puan:.2f})")
            except Exception as e:
                st.error(f"❌ Hata: {e}")

# ---------- Terimler Rehberi (Her iki sekmede de görünür) ----------
with st.expander("📖 Borsa & Fon Terimleri Rehberi (Yeni Başlayanlar İçin)"):
    st.markdown("""
    **Hisse Senedi:** Şirket ortaklık payı.  
    **Yatırım Fonu:** Birçok yatırımcının parasını toplayıp profesyonel yöneticilerin hisse, tahvil vb. araçlara yatırdığı portföy. Fon kodu ile alınır.  
    **RSI:** Aşırı alım (>70) / aşırı satım (<30) göstergesi.  
    **SMA:** Belirli gün sayısının ortalama fiyatı.  
    **MACD:** Trend takip göstergesi, al/sat sinyalleri üretir.  
    **Volatilite:** Fiyat dalgalanması; yüksekse risk büyüktür.  
    **Sinyal (AL/SAT/TUT):** Modelin risk profilinize göre verdiği öneri.  
    **Risk Profili:** Agresif (cesur), Dengeli, Muhafazakâr (temkinli).  
    **Prophet / ARIMA / Holt-Winters:** Geçmiş veriden geleceği tahmin eden modeller.
    """)

st.warning("""
⚠️ **Tahminler Ne Kadar Güvenilir?**  
Bu uygulama yalnızca geçmiş verilere dayalı matematiksel tahmin sunar, **kesin sonuç vaat etmez**.  
Yatırım kararlarınızı kendi araştırmanızla verin, gerekirse uzmana danışın.
""")

st.caption("⚠️ Eğitim amaçlıdır, yatırım tavsiyesi değildir.")
