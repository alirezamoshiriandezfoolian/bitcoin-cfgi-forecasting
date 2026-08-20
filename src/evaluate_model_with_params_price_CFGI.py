import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, mean_absolute_percentage_error

from data_loader2 import load_data
from preprocessing_price import create_sequences, fit_scalers, transform_sequences, inverse_transform_y
from data_splitter_price import split_data
from model_builder_price import build_price_model
from trainer_price import train_price_model

def evaluate_model_with_params(hyperparams: dict, model_name: str, results_csv_path: str):
    print("✅ evaluate_model_with_params شروع شد.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # اگر نبود، ادامه بده

    # 1. بارگذاری داده
    df = load_data(file_name="base_dataframe_CFGI.csv", folder="data/processed")
    feature_cols = ['close', 'volume', 'rs_volatility', 'value']
    label_col = 'close'
    date_col = 'timeopen'

    # 2. ساخت sequence
    sequence_length = hyperparams["sequence_length"]
    X, y, dates = create_sequences(df, feature_cols, label_col, date_col, sequence_length)

    # 3. تقسیم داده‌ها (بدون data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test, dates_test = split_data(X, y, dates)

    # 4. نرمال‌سازی فقط با داده‌های آموزش
    scaler_X, scaler_y = fit_scalers(X_train, y_train)
    X_train_scaled = transform_sequences(X_train, scaler_X)
    X_val_scaled = transform_sequences(X_val, scaler_X)
    X_test_scaled = transform_sequences(X_test, scaler_X)

    y_train_scaled = scaler_y.transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    # محدوده‌ی X‌های اسکیل‌شده
    print("X_train_scaled min/max:", float(X_train_scaled.min()), float(X_train_scaled.max()))
    print("X_val_scaled   min/max:", float(X_val_scaled.min()),   float(X_val_scaled.max()))
    print("X_test_scaled  min/max:", float(X_test_scaled.min()),  float(X_test_scaled.max()))

    print("y_train_scaled  min/max:", float(y_train_scaled.min()), float(y_train_scaled.max()))
    print("y_val_scaled    min/max:", float(y_val_scaled.min()), float(y_val_scaled.max()))
    print("y_test_scaled   min/max:", float(y_test_scaled.min()), float(y_test_scaled.max()))


    # 5. ساخت مدل
    model = build_price_model(
        input_shape=X_train_scaled.shape[1:],
        lstm_units=hyperparams["lstm_units"],
        dropout_rates = [
            hyperparams[f"dropout_rate_{i+1}"]
            for i in range(hyperparams["num_layers"])
        ],
        dense_units=hyperparams["dense_units"],
        num_layers=hyperparams["num_layers"],
        activation=hyperparams["activation"]
    )

    # 6. آموزش مدل روی داده‌های نرمال‌شده
    model_path = f"outputs/{model_name}.h5"
    model, history, _ = train_price_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        learning_rate=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        epochs=250,
        patience=hyperparams["patience"],
        model_path=model_path
    )

    # 7. پیش‌بینی روی داده تست و بازگردانی به مقیاس واقعی
    y_pred_scaled = model.predict(X_test_scaled).flatten().reshape(-1, 1)
    y_pred_inv = inverse_transform_y(y_pred_scaled, scaler_y)
    y_test_inv = inverse_transform_y(y_test_scaled, scaler_y)

    # 8. ذخیره نتایج در فایل debug
    os.makedirs("outputs", exist_ok=True)
    df_debug = pd.DataFrame({
        "date": dates_test.flatten(),
        "y_test_inv": y_test_inv.flatten(),
        "y_pred_inv": y_pred_inv.flatten()
    })
    df_debug.to_csv(f"outputs/debug_{model_name}.csv", index=False)

    # 9. محاسبه متریک‌ها روی مقیاس واقعی
    metrics = {
        "model_name": model_name,
        "mae": mean_absolute_error(y_test_inv, y_pred_inv),
        "mse": mean_squared_error(y_test_inv, y_pred_inv),
        "medae": median_absolute_error(y_test_inv, y_pred_inv),
        "mape": mean_absolute_percentage_error(y_test_inv, y_pred_inv),
        "r2": r2_score(y_test_inv, y_pred_inv),
    }

    metrics.update(hyperparams)

    # 10. ذخیره متریک‌ها در CSV
    df_result = pd.DataFrame([metrics])
    try:
        df_existing = pd.read_csv(results_csv_path)
        df_result = pd.concat([df_existing, df_result], ignore_index=True)
    except FileNotFoundError:
        pass
    df_result.to_csv(results_csv_path, index=False)

    print(f"✅ ارزیابی مدل '{model_name}' تکمیل و ذخیره شد.")