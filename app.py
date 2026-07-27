import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
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
def fetch_stock_data(ticker, period="1y", interval="1d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError("Veri çekilemedi.")
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    df.dropna(inplace=True)
    return df

# ---------- Zaman Serisi Modelleri ----------
def forecast_prophet(df, periods=30):
    df_prophet = df[['Close']].reset_index()
    df_prophet.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_df = forecast_df.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
    return forecast_df.reset_index(drop=True)

def forecast_arima(df, periods=30):
    # pmdarima ile otomatik ARIMA seçimi
    model = pm.auto_arima(df['Close'], seasonal=False, trace=False,
                          error_action='ignore', suppress_warnings=True,
                          stepwise=True)
    forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast,
        'Lower': conf_int[:, 0],
        'Upper': conf_int[:, 1]
    })
    return forecast_df

def forecast_holt_winters(df, periods=30):
    # Mevsimsellik periyodu: haftalık (5 iş günü) varsayalım, günlük veride 5
    model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add',
                                 seasonal_periods=5)
    fitted = model.fit()
    forecast = fitted.forecast(periods)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    # Güven aralıkları basit bir simülasyonla verilebilir, burada gösterilmeyecek
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast,
        'Lower': np.nan,
        'Upper': np.nan
    })
    return forecast_df

# ---------- Kişisel sinyal (artık seçilen modele göre) ----------
def generate_signal_from_forecast(df, forecast_df, risk_profile="moderate"):
    last_close = df['Close'].iloc[-1]
    future_price = forecast_df['Forecast'].iloc[-1]
    pct_change_pred = (future_price - last_close) / last_close

    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]

    score = 0
    if pct_change_pred > 0.05:
        score += 2
    elif pct_change_pred > 0.02:
        score += 1
    elif pct_change_pred < -0.05:
        score -= 2
    elif pct_change_pred < -0.02:
        score -= 1

    if rsi < 30:
        score += 1.5
    elif rsi > 70:
        score -= 1.5

    if macd > macd_signal:
        score += 1
    else:
        score -= 1

    if risk_profile == "conservative":
        buy_threshold = 3.0
        sell_threshold = -2.0
    elif risk_profile == "aggressive":
        buy_threshold = 1.5
        sell_threshold = -1.0
    else:  # moderate
        buy_threshold = 2.0
        sell_threshold = -1.5

    if score >= buy_threshold:
        return "AL", score, pct_change_pred
    elif score <= sell_threshold:
        return "SAT", score, pct_change_pred
    else:
        return "TUT", score, pct_change_pred

# ---------- Streamlit Arayüz ----------
st.set_page_config(page_title="Gelişmiş Borsa Asistanım", layout="wide")
st.title("📈 Zaman Serisi Analizli Kişisel Yatırım Asistanı")

ticker = st.text_input("Hisse Sembolü (örn: THYAO.IS, GARAN.IS)", "THYAO.IS")
risk = st.selectbox("Risk Profilin", ["aggressive", "moderate", "conservative"])
period_choice = st.selectbox("Geçmiş Veri Aralığı", ["6mo","1y","2y","5y"])
model_choice = st.selectbox("Tahmin Modeli", ["Prophet", "ARIMA", "Holt-Winters"])

if st.button("Analiz Et"):
    with st.spinner("Veri çekiliyor ve model eğitiliyor..."):
        try:
            df = fetch_stock_data(ticker, period=period_choice)

            # Seçilen modele göre tahmin
            if model_choice == "Prophet":
                forecast_df = forecast_prophet(df, periods=30)
            elif model_choice == "ARIMA":
                forecast_df = forecast_arima(df, periods=30)
            elif model_choice == "Holt-Winters":
                forecast_df = forecast_holt_winters(df, periods=30)

            # Sinyal üret
            signal, score, exp_return = generate_signal_from_forecast(df, forecast_df, risk)

            # Sonuç metrikleri
            col1, col2, col3 = st.columns(3)
            col1.metric("Son Kapanış", f"{df['Close'].iloc[-1]:.2f}")
            col2.metric(f"Beklenen 30 Günlük Değişim ({model_choice})",
                        f"%{exp_return*100:.2f}")
            col3.metric("Sinyal", signal, delta=score)

            # Grafik: Gerçek fiyat + Tahmin
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçek Fiyat'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'],
                                     name=f'{model_choice} Tahmin', line=dict(dash='dash')))
            if 'Lower' in forecast_df.columns and not forecast_df['Lower'].isna().all():
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower'],
                                         mode='lines', line=dict(color='gray', dash='dot'),
                                         name='Alt Sınır'))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper'],
                                         fill='tonexty', mode='lines', line=dict(color='gray', dash='dot'),
                                         name='Üst Sınır'))
            st.plotly_chart(fig, use_container_width=True)

            # Teknik Göstergeler
            st.subheader("RSI Göstergesi")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(rsi_fig, use_container_width=True)

            # İsteğe bağlı: Bütün modelleri karşılaştırma (opsiyonel buton)
            if st.checkbox("Tüm modelleri karşılaştır"):
                st.subheader("Model Tahminleri Karşılaştırması")
                models = {
                    "Prophet": forecast_prophet(df, 30),
                    "ARIMA": forecast_arima(df, 30),
                    "Holt-Winters": forecast_holt_winters(df, 30)
                }
                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçek'))
                colors = ['orange', 'green', 'red']
                for (name, fdf), col in zip(models.items(), colors):
                    comp_fig.add_trace(go.Scatter(x=fdf['Date'], y=fdf['Forecast'],
                                                  name=name, line=dict(color=col, dash='dot')))
                st.plotly_chart(comp_fig, use_container_width=True)

            st.info(f"🔍 Detaylı Sinyal Puanı: {score:.2f} (Risk Profili: {risk}, Model: {model_choice})")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

st.caption("⚠️ Bu uygulama yalnızca eğitim amaçlıdır, yatırım tavsiyesi içermez.")
