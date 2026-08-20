# compare_CFGI_extreme_train.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ثبات‌پذیری
import tensorflow as tf, random
np.random.seed(42); random.seed(42); tf.random.set_seed(42)

# ماژول‌های پروژه (همین‌هایی که داری)
from data_loader2 import load_data
from preprocessing_price import create_sequences, fit_scalers, transform_sequences, inverse_transform_y
from data_splitter_price import split_data
from model_builder_price import build_price_model
from trainer_price import train_price_model


def compute_fgi_prev_for_dates(fgi_df: pd.DataFrame,
                               target_dates: np.ndarray,
                               time_col: str = "timeopen",
                               value_col: str = "value") -> np.ndarray:
    """
    FGIِ روز t را به تاریخ هدف t+1 نسبت می‌دهد و سپس مقدار FGI_prev
    را بر اساس آرایه ی 'target_dates' (تاریخ‌های هدف نمونه‌ها) برمی‌گرداند.
    """
    fgi = fgi_df[[time_col, value_col]].copy()
    fgi[time_col] = pd.to_datetime(fgi[time_col])
    fgi["d_day"] = fgi[time_col].dt.floor("D")

    # جدول روزانه
    day_fgi = fgi[["d_day", value_col]].drop_duplicates("d_day")

    # lag=1 : FGI(t) رو به روز t+1 نسبت بده تا با تاریخ هدف منطبق بشه
    lag = day_fgi.copy()
    lag["d_day"] = lag["d_day"] + pd.Timedelta(days=1)
    lag = lag.rename(columns={value_col: "FGI_prev"})

    # نگاشت تاریخ هدف → FGI_prev
    mapper = dict(zip(lag["d_day"].values, lag["FGI_prev"].values))
    d_days = pd.to_datetime(pd.Series(target_dates).dt.floor("D"))
    return d_days.map(mapper).to_numpy()


def train_eval_extreme_only(
    file_name: str,
    feature_cols: List[str],
    label_col: str,
    date_col: str,
    hyperparams: Dict,
    model_name: str,
    fgi_file_name: str = "base_dataframe_CFGI.csv",
    outputs_dir: str = "outputs/cfgi_extreme_train_eval"
):
    """
    1) ساخت همه sequence ها
    2) محاسبه FGI_prev برای تاریخ هدف هر نمونه (بدون لیک: از روز قبل)
    3) فیلتر فقط نمونه‌های FGI_prev<=25 یا FGI_prev>=75
    4) split به train/val/test (زمانی)
    5) اسکیل فقط با train، آموزش فقط روی train، EarlyStopping با val
    6) ارزیابی روی test (همه‌اش روزهای شدید)
    """

    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    # 1) داده و sequence ها
    df = load_data(file_name=file_name, folder="data/processed")
    seq_len = int(hyperparams["sequence_length"])
    X, y, dates = create_sequences(df, feature_cols, label_col, date_col, sequence_length=seq_len)

    # 2) محاسبه FGI_prev برای هر نمونه (با استفاده از فایل CFGI)
    df_cfgi = load_data(file_name=fgi_file_name, folder="data/processed")
    fgi_prev = compute_fgi_prev_for_dates(df_cfgi, dates, time_col="timeopen", value_col="value")

    # 3) ماسک روزهای احساسات شدید (<=25 یا >=75)
    mask_extreme = (fgi_prev <= 25) | (fgi_prev >= 75)
    X_ext, y_ext, dates_ext, fgi_prev_ext = X[mask_extreme], y[mask_extreme], dates[mask_extreme], fgi_prev[mask_extreme]

    if len(X_ext) < 50:
        print(f"⚠️ فقط {len(X_ext)} نمونه شدید داریم — ممکنه برای آموزش کم باشه.")

    # 4) split زمانی روی زیرمجموعه شدید
    X_train, X_val, X_test, y_train, y_val, y_test, dates_test = split_data(X_ext, y_ext, dates_ext)

    # 5) اسکیل (X و y) فقط با train شدید
    scaler_X, scaler_y = fit_scalers(X_train, y_train)
    X_train_s = transform_sequences(X_train, scaler_X)
    X_val_s   = transform_sequences(X_val,   scaler_X)
    X_test_s  = transform_sequences(X_test,  scaler_X)
    y_train_s = scaler_y.transform(y_train)
    y_val_s   = scaler_y.transform(y_val)
    y_test_s  = scaler_y.transform(y_test)

    # هماهنگ‌سازی dropout با num_layers
    num_layers = int(hyperparams["num_layers"])
    raw_rates = [hyperparams.get("dropout_rate_1"),
                 hyperparams.get("dropout_rate_2"),
                 hyperparams.get("dropout_rate_3")]
    dropout_rates = [r for r in raw_rates if r is not None]
    if len(dropout_rates) == 0:
        dropout_rates = [0.0] * num_layers
    elif len(dropout_rates) < num_layers:
        dropout_rates = dropout_rates + [dropout_rates[-1]] * (num_layers - len(dropout_rates))
    elif len(dropout_rates) > num_layers:
        dropout_rates = dropout_rates[:num_layers]

    # 6) ساخت و آموزش مدل
    model = build_price_model(
        input_shape=X_train_s.shape[1:],
        lstm_units=hyperparams["lstm_units"],
        dropout_rates=dropout_rates,
        dense_units=hyperparams["dense_units"],
        num_layers=num_layers,
        activation=hyperparams["activation"]
    )

    model_path = f"{outputs_dir}/{model_name}.h5"
    model, history, _ = train_price_model(
        model=model,
        X_train=X_train_s,
        y_train=y_train_s,
        X_val=X_val_s,
        y_val=y_val_s,
        learning_rate=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams.get("epochs", 200),
        patience=hyperparams["patience"],
        model_path=model_path
    )

    # 7) پیش‌بینی روی test شدید + برگرداندن مقیاس
    y_pred_s = model.predict(X_test_s).reshape(-1, 1)
    y_pred   = inverse_transform_y(y_pred_s, scaler_y).reshape(-1)
    y_true   = inverse_transform_y(y_test_s, scaler_y).reshape(-1)

    # متریک‌ها
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    print(f"\n📊 [{model_name}] on EXTREME-only TEST")
    print(f"count={len(y_true)} | MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.6f}")

    # ذخیره دیباگ
    df_debug = pd.DataFrame({
        "date": dates_test.reshape(-1),
        "y_true": y_true,
        "y_pred": y_pred,
        "FGI_prev": compute_fgi_prev_for_dates(df_cfgi, dates_test, "timeopen", "value")
    })
    df_debug.to_csv(f"{outputs_dir}/debug_{model_name}_EXTREME.csv", index=False)

    return {"mae": mae, "rmse": rmse, "r2": r2, "count": len(y_true)}


if __name__ == "__main__":
    # ⬇️ هایپرپارامترهای بهترین‌ت (بدون FGI)
    base_hparams = {
        "sequence_length": 4,
        "lstm_units": 512,
        "dense_units": 128,
        "dropout_rate_1": 0.3,
        "num_layers": 1,
        "activation": "relu",
        "batch_size": 32,
        "learning_rate": 0.0008289739,
        "patience": 30,
        "epochs": 250
    }

    # ⬇️ هایپرپارامترهای بهترین‌ت (با FGI)
    cfgi_hparams = {
        "sequence_length": 4,
        "lstm_units": 512,
        "dense_units": 64,
        "dropout_rate_1": 0.0,
        "num_layers": 1,
        "activation": "swish",
        "batch_size": 16,
        "learning_rate": 0.0000814175,
        "patience": 30,
        "epochs": 250
    }

    out_dir = "outputs/cfgi_extreme_train_eval1_final"

    # مدل baseline (بدون FGI) — آموزش فقط روی روزهای شدید
    res_base = train_eval_extreme_only(
        file_name="base_dataframe_CFGI.csv",
        feature_cols=["close", "volume", "rs_volatility"],
        label_col="close",
        date_col="timeopen",
        hyperparams=base_hparams,
        model_name="price_baseline_EXTREMEtrain",
        fgi_file_name="base_dataframe_CFGI.csv",
        outputs_dir=out_dir
    )

    # مدل با FGI — آموزش فقط روی روزهای شدید
    res_cfgi = train_eval_extreme_only(
        file_name="base_dataframe_CFGI.csv",
        feature_cols=["close", "volume", "rs_volatility", "value"],
        label_col="close",
        date_col="timeopen",
        hyperparams=cfgi_hparams,
        model_name="price_with_cfgi_EXTREMEtrain",
        fgi_file_name="base_dataframe_CFGI.csv",
        outputs_dir=out_dir
    )

    # -------------------------------
    # ✅ مقایسهٔ جفتی روی روزهای تست «مشترک»
    # -------------------------------
    base_debug_path = f"{out_dir}/debug_price_baseline_EXTREMEtrain_EXTREME.csv"
    cfgi_debug_path = f"{out_dir}/debug_price_with_cfgi_EXTREMEtrain_EXTREME.csv"
    df_base = pd.read_csv(base_debug_path)
    df_cfgi = pd.read_csv(cfgi_debug_path)

    # هم‌سطح‌سازی تاریخ‌ها (روز)
    df_base["d_day"] = pd.to_datetime(df_base["date"]).dt.floor("D")
    df_cfgi["d_day"] = pd.to_datetime(df_cfgi["date"]).dt.floor("D")

    merged = df_base.merge(
        df_cfgi[["d_day", "y_pred"]].rename(columns={"y_pred": "y_pred_cfgi"}),
        on="d_day", how="inner"
    )
    # متریک‌ها فقط روی تاریخ‌های مشترک
    mae_base_common = mean_absolute_error(merged["y_true"], merged["y_pred"])
    mae_cfgi_common = mean_absolute_error(merged["y_true"], merged["y_pred_cfgi"])
    r2_base_common  = r2_score(merged["y_true"], merged["y_pred"]) if len(merged) > 1 else np.nan
    r2_cfgi_common  = r2_score(merged["y_true"], merged["y_pred_cfgi"]) if len(merged) > 1 else np.nan

    merged["err_base"] = merged["y_pred"] - merged["y_true"]
    merged["err_cfgi"] = merged["y_pred_cfgi"] - merged["y_true"]
    merged.to_csv(f"{out_dir}/paired_errors_extreme.csv", index=False)

    # (اختیاری) تست آماری Wilcoxon روی قدرمطلق خطاها
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(np.abs(merged["err_base"]), np.abs(merged["err_cfgi"]))
        print(f"\n[EXTREME-only] Wilcoxon on |errors|  stat={stat:.3f}, p={p:.4f}, n={len(merged)}")
    except Exception as e:
        print("Wilcoxon test skipped (scipy not available or other issue):", e)

    print("\n================ SUMMARY (EXTREME-only training) ================")
    print("Baseline (no FGI):", res_base)
    print("With FGI:", res_cfgi)

    print("\n📊 Comparison on SAME extreme test days:")
    print(f"Baseline MAE={mae_base_common:.4f}, R²={r2_base_common:.4f}")
    print(f"With FGI   MAE={mae_cfgi_common:.4f}, R²={r2_cfgi_common:.4f}")
