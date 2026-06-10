# Streamlit BTCUSDT Real-Time Prediction

This project contains a Streamlit app for real-time BTCUSDT prediction using Binance minute K-lines and an XGBoost model.

## Files

- `streamlit_online_predict.py`: Streamlit application entry point.
- `features.py`: feature generation used by the app.
- `xgb_strict_label_model.json`: XGBoost model file used for predictions.
- `requirements.txt`: required Python packages.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_online_predict.py
```

## Notes

- The app saves runtime state to `streamlit_online_predict_state.json` in the project directory.
- Do not commit the state file if you want a clean repository.
