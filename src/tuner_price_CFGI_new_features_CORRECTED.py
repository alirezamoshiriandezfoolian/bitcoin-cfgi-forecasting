import os
import optuna
import numpy as np
import tensorflow as tf

from data_loader2 import load_data
from preprocessing_price import create_sequences, fit_scalers, transform_sequences, inverse_transform_y
from data_splitter_price import split_data
from model_builder_price import build_price_model
from trainer_price import train_price_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def objective(trial):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # اگر نبود، ادامه بده

    # 1) بارگذاری داده
    df = load_data(file_name="features_for_price.csv", folder="data/processed")

    # 2) تعریف ستون‌ها
    feature_cols = [
    "close", "volume", "rs_volatility", "value",
    "CFGI_momentum",
    "value_lag1", "value_lag2", "value_lag3", "value_lag4", "value_lag5", "value_lag6", "value_lag7",
    "value_sma3", "value_sma5", "value_sma7"
    ]
    label_col = 'close'
    date_col = 'timeopen'

    # 2.1) پاک‌سازی امن: حذف NaN فقط از ستون‌های مورد استفاده
    # (اینکار لازمه چون lag/SMA باعث NaN در ابتدای سری شده)
    df = df.dropna(subset=feature_cols + [label_col])

    # 3) ساخت sequence‌ها
    sequence_length = trial.suggest_categorical(
        "sequence_length", [1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
    )
    X, y, dates = create_sequences(df, feature_cols, label_col, date_col, sequence_length)

    # 4) تقسیم داده‌ها
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_data(X, y, dates)

    # 5) نرمال‌سازی ویژگی‌ها و خروجی‌ها (فقط با train fit می‌کنیم)
    scaler_X, scaler_y = fit_scalers(X_train, y_train)
    X_train_scaled = transform_sequences(X_train, scaler_X)
    X_val_scaled   = transform_sequences(X_val, scaler_X)
    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled   = scaler_y.transform(y_val)

    # 6) ساخت مدل
    lstm_units   = trial.suggest_categorical("lstm_units", [128, 256, 512])
    num_layers   = trial.suggest_int("num_layers", 1, 2)
    dropout_rates = [
        trial.suggest_float(f"dropout_rate_{i+1}", 0.0, 0.4, step=0.1)
        for i in range(num_layers)
    ]
    dense_units  = trial.suggest_categorical("dense_units", [32, 64, 128, 256])
    activation   = trial.suggest_categorical("activation", ["prelu", "swish", "relu"])

    model = build_price_model(
        input_shape=X_train_scaled.shape[1:],
        lstm_units=lstm_units,
        dropout_rates=dropout_rates,
        dense_units=dense_units,
        num_layers=num_layers,
        activation=activation
    )

    # 7) پارامترهای آموزش
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16, 32])
    patience      = trial.suggest_int("patience", 20, 30)

    os.makedirs("outputs_new_features_price_corrected", exist_ok=True)
    model_path = f"outputs_new_features_price_corrected/trial_price_model_{trial.number}_new_features.h5"

    # 8) آموزش مدل
    model, history, _ = train_price_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=200,
        patience=patience,
        model_path=model_path
    )

    # 9) پیش‌بینی روی val و برگرداندن به مقیاس واقعی
    y_val_pred_scaled = model.predict(X_val_scaled).flatten()
    y_val_pred = inverse_transform_y(y_val_pred_scaled, scaler_y).flatten()
    y_val_real = y_val.flatten()  # y_val واقعی (غیرنرمال) برای مقایسه

    # 10) متریک‌ها
    r2 = r2_score(y_val_real, y_val_pred)
    mae = mean_absolute_error(y_val_real, y_val_pred)
    mse = mean_squared_error(y_val_real, y_val_pred)

    trial.set_user_attr("r2_price", r2)
    trial.set_user_attr("mae_price", mae)
    trial.set_user_attr("mse_price", mse)

    # 11) معیار اصلی بهینه‌سازی (همچنان val_loss)
    return min(history.history["val_loss"])
