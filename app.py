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

# ---------- Sabit: BIST 100 sembollerini Wikipedia'dan çek ----------
@st.cache_data(ttl=86400)
def bist100_listesi():
    url = "https://en.wikipedia.org/wiki/BIST_100"
    tablo = pd.read_html(url)[0]
    # Tablo yapısına göre sembol sütunu "Symbol" olabilir, kontrol edip alalım
    if 'Symbol' in tablo.columns:
        semboller = tablo['Symbol'].dropna().tolist()
    else:
        # Yedek liste
        semboller = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "KCHOL.IS"]
    return [s for s in semboller if isinstance(s, str) and len(s) > 2]

# ---------- Veri çekme ve teknik göstergeler ----------
@st.cache_data(ttl=3600)
def veri_cek(ticker, donem="1y", aralik="1d"):
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
st.markdown("Bu uygulama, seçtiğiniz hisse için üç farklı zaman serisi modeliyle tahmin yapar ve risk profilinize uygun **AL/SAT/TUT** sinyali üretir. Ayrıca **tüm BIST 100 hisselerini tarama** özelliğine sahiptir.")

# ---------- Kenar Çubuğu: Ayarlar ----------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    # Hisse listesini oluştur
    try:
        hisse_listesi = bist100_listesi()
    except:
        hisse_listesi = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "KCHOL.IS"]
    sembol = st.selectbox("Hisse seçin", hisse_listesi, index=0)
    
    risk_aciklama = {
        "agresif": "Daha yüksek risk alır, küçük sinyallerde bile alım yapar. Kazanç potansiyeli yüksek, kayıp riski de fazladır.",
        "dengeli": "Orta düzey risk, sinyallerin biraz daha güçlü olmasını bekler. (Varsayılan)",
        "muhafazakar": "Düşük risk ister, yalnızca çok güçlü sinyallerde alım yapar, kayıp ihtimali en azdır."
    }
    risk = st.selectbox(
        "Risk profiliniz",
        ["agresif", "dengeli", "muhafazakar"],
        help="**Agresif:** Hızlı al-sat, yüksek volatiliteye uygun.\n**Dengeli:** Ne çok cesur ne çok temkinli.\n**Muhafazakar:** Uzun vadeli, güvenli liman arayan."
    )
    st.caption(risk_aciklama[risk])

    donem = st.selectbox("Geçmiş veri aralığı", ["6 ay", "1 yıl", "2 yıl", "5 yıl"])
    donem_haritasi = {"6 ay": "6mo", "1 yıl": "1y", "2 yıl": "2y", "5 yıl": "5y"}
    model_secimi = st.selectbox("Tahmin modeli", ["Prophet", "ARIMA", "Holt-Winters"],
                                help="**Prophet:** Facebook, tatil/mevsim etkilerini iyi yakalar.\n**ARIMA:** Klasik istatistiksel yöntem.\n**Holt-Winters:** Mevsimsel üstel düzleştirme.")
    
    # Toplu tarama butonu
    st.markdown("---")
    toplu_tara = st.button("🚀 BIST 100 Hisselerini Toplu Tara")

# ---------- Ana Bölüm ----------
if not toplu_tara:
    # Normal tek hisse analizi
    if st.button("🔍 Seçili Hisseyi Analiz Et"):
        with st.spinner("Veri çekiliyor ve model eğitiliyor..."):
            try:
                df = veri_cek(sembol, donem=donem_haritasi[donem])
                if model_secimi == "Prophet":
                    tahmin_df = prophet_tahmin(df, gun=30)
                elif model_secimi == "ARIMA":
                    tahmin_df = arima_tahmin(df, gun=30)
                else:
                    tahmin_df = holt_winters_tahmin(df, gun=30)

                sinyal, puan, beklenen_degisim = sinyal_uret(df, tahmin_df, risk)

                kolon1, kolon2, kolon3 = st.columns(3)
                kolon1.metric("Son Kapanış", f"₺{df['Close'].iloc[-1]:.2f}")
                kolon2.metric(f"30 Günlük Beklenen Değişim ({model_secimi})", f"%{beklenen_degisim*100:.2f}")
                kolon3.metric("Sinyal", sinyal, delta=puan)

                # Fiyat grafiği
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçek Fiyat', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Tahmin'],
                                         name=f'{model_secimi} Tahmini', line=dict(dash='dash', color='orange')))
                if 'Alt' in tahmin_df.columns and not tahmin_df['Alt'].isna().all():
                    fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Alt'],
                                             mode='lines', line=dict(color='gray', dash='dot'), name='Alt Sınır'))
                    fig.add_trace(go.Scatter(x=tahmin_df['Tarih'], y=tahmin_df['Ust'],
                                             fill='tonexty', mode='lines', line=dict(color='gray', dash='dot'), name='Üst Sınır'))
                fig.update_layout(title=f"{sembol} – Gerçek ve Tahmini Fiyat ({model_secimi})",
                                  xaxis_title="Tarih", yaxis_title="Fiyat (₺)")
                st.plotly_chart(fig, use_container_width=True)

                # RSI
                st.subheader("RSI (Göreceli Güç Endeksi)")
                st.caption("RSI, bir hissenin aşırı alım (>70) ya da aşırı satım (<30) bölgesinde olup olmadığını gösterir. 50 orta noktadır.")
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
                st.plotly_chart(rsi_fig, use_container_width=True)

                # Teknik gösterge özeti
                with st.expander("📊 Diğer Teknik Göstergeler"):
                    st.write(f"**SMA(20):** {df['SMA_20'].iloc[-1]:.2f} (20 günlük basit hareketli ortalama)")
                    st.write(f"**SMA(50):** {df['SMA_50'].iloc[-1]:.2f} (50 günlük hareketli ortalama)")
                    st.write(f"**MACD:** {df['MACD'].iloc[-1]:.4f} | **Sinyal:** {df['MACD_sinyal'].iloc[-1]:.4f}")
                    st.write(f"**Volatilite (20 gün):** %{df['Volatilite'].iloc[-1]*100:.2f}")

                if sinyal == "AL":
                    st.success(f"✅ Sinyal: **AL** (Puan: {puan:.2f}) – Model alım fırsatı gösteriyor.")
                elif sinyal == "SAT":
                    st.error(f"❌ Sinyal: **SAT** (Puan: {puan:.2f}) – Model satış baskısı öngörüyor.")
                else:
                    st.warning(f"⏸️ Sinyal: **TUT** (Puan: {puan:.2f}) – Beklemede kalmak daha uygun.")

                st.info(f"🔍 Beklenen değişim %{beklenen_degisim*100:.2f} | Risk: {risk} | Model: {model_secimi}")
            except Exception as e:
                st.error(f"❌ Hata: {e}")

else:
    # ---------- Toplu Tarama Modu ----------
    st.subheader("📋 BIST 100 Toplu Tarama Sonuçları")
    st.markdown("Sadece **AL** veya **SAT** sinyali veren hisseler listelenir. Risk profiliniz: **" + risk + "**")
    with st.spinner("BIST 100 hisseleri taranıyor... Bu işlem biraz zaman alabilir (birkaç dakika). Lütfen bekleyin."):
        sonuc_listesi = []
        hata_sayisi = 0
        # İlerleme çubuğu
        progress = st.progress(0)
        for i, sembol in enumerate(hisse_listesi):
            try:
                df = veri_cek(sembol, donem="6mo")  # Hızlı olsun diye 6 aylık veri
                # Model olarak sadece Prophet kullanalım (hız için)
                tahmin_df = prophet_tahmin(df, gun=30)
                sinyal, puan, _ = sinyal_uret(df, tahmin_df, risk)
                if sinyal in ["AL", "SAT"]:
                    sonuc_listesi.append({
                        "Hisse": sembol,
                        "Sinyal": sinyal,
                        "Puan": round(puan, 2),
                        "Son Fiyat": round(df['Close'].iloc[-1], 2)
                    })
            except:
                hata_sayisi += 1
            # İlerleme çubuğunu güncelle
            progress.progress((i + 1) / len(hisse_listesi))
        
        progress.empty()
        if sonuc_listesi:
            sonuc_df = pd.DataFrame(sonuc_listesi)
            # Sinyale göre renklendirme
            def renklendir(val):
                color = 'red' if val == 'SAT' else 'green'
                return f'color: {color}; font-weight: bold'
            styled_df = sonuc_df.style.applymap(renklendir, subset=['Sinyal'])
            st.dataframe(styled_df, use_container_width=True)
            st.success(f"✅ {len(sonuc_listesi)} hisse sinyal verdi. ({hata_sayisi} hisse veri çekilemedi.)")
        else:
            st.warning("Hiçbir hisse AL/SAT sinyali vermedi. Piyasa şu an sizin profilinize uygun fırsat sunmuyor olabilir.")
        st.caption("Not: Toplu tarama yalnızca Prophet modeli ve 6 aylık veri ile hızlı sonuç içindir.")

st.sidebar.caption("⚠️ Bu uygulama eğitim amaçlıdır, yatırım tavsiyesi değildir.")
