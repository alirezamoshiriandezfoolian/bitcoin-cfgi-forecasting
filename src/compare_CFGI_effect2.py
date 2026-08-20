# compare_cfgi_effect_by_date.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ثبات‌پذیری
import tensorflow as tf, random
np.random.seed(42); random.seed(42); tf.random.set_seed(42)

# --- ماژول‌های پروژه ---
from data_loader2 import load_data
from preprocessing_price import create_sequences, fit_scalers, transform_sequences, inverse_transform_y
from model_builder_price import build_price_model
from trainer_price import train_price_model

# ==============================
# ابزارهای کمکی
# ==============================
def split_df_by_date(df: pd.DataFrame, date_col: str = "timeopen",
                     train_ratio: float = 0.7, val_ratio: float = 0.15):
    """اسپلیت ترتیبی صرفاً بر اساس تاریخ (بدون ساخت توالی)."""
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train:n_train+n_val].copy()
    test_df  = df.iloc[n_train+n_val:].copy()
    return train_df, val_df, test_df

def _harmonize_dropout(hparams: Dict) -> Tuple[List[float], int]:
    """هماهنگ‌سازی dropout_rates با num_layers."""
    num_layers = int(hparams["num_layers"])
    raw_rates = [
        hparams.get("dropout_rate_1", None),
        hparams.get("dropout_rate_2", None),
        hparams.get("dropout_rate_3", None),
    ]
    dropout_rates = [r for r in raw_rates if r is not None]
    if len(dropout_rates) == 0:
        dropout_rates = [0.0] * num_layers
    elif len(dropout_rates) < num_layers:
        dropout_rates += [dropout_rates[-1]] * (num_layers - len(dropout_rates))
    elif len(dropout_rates) > num_layers:
        dropout_rates = dropout_rates[:num_layers]
    return dropout_rates, num_layers

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "count": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": (float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan),
    }

def attach_FGI_lag1_for_target_day(
    df_pred: pd.DataFrame,
    fgi_df: pd.DataFrame,
    date_col_pred: str = "date",
    fgi_time_col: str = "timeopen",
    fgi_value_col: str = "value",
) -> pd.DataFrame:
    """
    برای هر ردیف df_pred که تاریخ هدف (t+1) دارد، مقدار FGI روز قبل (t) را متصل می‌کند.
    """
    out = df_pred.copy()
    out[date_col_pred] = pd.to_datetime(out[date_col_pred])
    out["d_day"] = out[date_col_pred].dt.floor("D")

    tmp = fgi_df[[fgi_time_col, fgi_value_col]].copy()
    tmp[fgi_time_col] = pd.to_datetime(tmp[fgi_time_col])
    tmp["d_day"] = tmp[fgi_time_col].dt.floor("D")
    tmp = tmp[["d_day", fgi_value_col]].drop_duplicates("d_day")

    tmp["d_day"] = tmp["d_day"] + pd.Timedelta(days=1)  # lag=1 → مپ به روز هدف
    tmp = tmp.rename(columns={fgi_value_col: "FGI_t"})
    return out.merge(tmp, on="d_day", how="left")

def add_segment_col(df: pd.DataFrame) -> pd.DataFrame:
    """ستون segment را بر اساس FGI_t بساز."""
    def seg_func(x):
        if pd.isna(x): return "FGI_missing"
        if x <= 25:    return "FGI_<=25_prev"
        if x >= 75:    return "FGI_>=75_prev"
        return "FGI_25_75_prev"
    out = df.copy()
    out["segment"] = out["FGI_t"].apply(seg_func)
    return out

# ==============================
# هستهٔ آموزش/ارزیابی هر مدل (با اسپلیت تاریخ → توالی در پارتیشن‌ها)
# ==============================
def train_eval_one_model_by_date(
    df_raw: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    date_col: str,
    hyperparams: Dict,
    model_name: str,
    outputs_dir: str = "outputs"
) -> pd.DataFrame:
    """
    - اسپلیت تاریخ روی df_raw
    - ساخت توالی در هر پارتیشن با seq_len مخصوص این مدل
    - اسکیل فقط با train
    - آموزش + EarlyStopping روی val
    - پیش‌بینی روی test و inverse
    - خروجی: DataFrame با ستون‌های [date, y_true, y_pred]
    """
    os.makedirs(outputs_dir, exist_ok=True)

    # 1) اسپلیت خام بر اساس تاریخ
    tr_df, va_df, te_df = split_df_by_date(df_raw, date_col=date_col, train_ratio=0.7, val_ratio=0.15)

    # 2) ساخت توالی‌ها در هر پارتیشن
    seq_len = int(hyperparams["sequence_length"])
    X_tr, y_tr, d_tr = create_sequences(tr_df, feature_cols, label_col, date_col, seq_len)
    X_va, y_va, d_va = create_sequences(va_df, feature_cols, label_col, date_col, seq_len)
    X_te, y_te, d_te = create_sequences(te_df, feature_cols, label_col, date_col, seq_len)

    # اگر به‌علت کمبود داده/seq_len طول صفر شد
    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        raise ValueError(f"Insufficient data after sequencing for model '{model_name}'. "
                         f"Try smaller sequence_length or check dataset spans.")

    # 3) اسکیل فقط با train
    scaler_X, scaler_y = fit_scalers(X_tr, y_tr)
    X_tr_s = transform_sequences(X_tr, scaler_X)
    X_va_s = transform_sequences(X_va, scaler_X)
    X_te_s = transform_sequences(X_te, scaler_X)
    y_tr_s = scaler_y.transform(y_tr)
    y_va_s = scaler_y.transform(y_va)
    y_te_s = scaler_y.transform(y_te)

    # 4) ساخت مدل
    dropout_rates, num_layers = _harmonize_dropout(hyperparams)
    model = build_price_model(
        input_shape=X_tr_s.shape[1:],
        lstm_units=hyperparams["lstm_units"],
        dropout_rates=dropout_rates,
        dense_units=hyperparams["dense_units"],
        num_layers=num_layers,
        activation=hyperparams["activation"],
    )

    # 5) آموزش
    model_path = f"{outputs_dir}/{model_name}.h5"
    model, history, _ = train_price_model(
        model=model,
        X_train=X_tr_s, y_train=y_tr_s,
        X_val=X_va_s, y_val=y_va_s,
        learning_rate=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams.get("epochs", 200),
        patience=hyperparams["patience"],
        model_path=model_path
    )

    # 6) پیش‌بینی روی test و بازگردانی به مقیاس واقعی
    y_pred_s = model.predict(X_te_s).reshape(-1, 1)
    y_pred   = inverse_transform_y(y_pred_s, scaler_y).reshape(-1)
    y_true   = inverse_transform_y(y_te_s, scaler_y).reshape(-1)

    df_debug = pd.DataFrame({
        "date": pd.to_datetime(d_te).astype("datetime64[ns]"),
        "y_true": y_true,
        "y_pred": y_pred
    })
    df_debug.to_csv(f"{outputs_dir}/debug_{model_name}.csv", index=False)
    return df_debug

# ==============================
# مقایسهٔ دو مدل با هم‌ترازسازی تاریخ‌ها + سگمنت‌بندی
# ==============================
def compare_two_models_on_segments_by_date(
    base_hparams: Dict,
    cfgi_hparams: Dict,
    out_dir: str = "outputs/cfgi_segment_eval",
    model_base_name: str = "price_baseline",
    model_cfgi_name: str = "price_with_cfgi",
    file_name: str = "base_dataframe_CFGI.csv",
    date_col: str = "timeopen",
):
    """
    - هر دو مدل با «بهترین هایپرپارامتر خودش» آموزش می‌شوند.
    - اسپلیتِ تاریخ روی دیتافریم خام مشترک انجام می‌شود.
    - تاریخ‌های تست در خروجیِ دو مدل با inner-join هم‌تراز می‌شوند (intersection).
    - متریک‌ها (کل + سگمنت) روی همین بازهٔ مشترک محاسبه می‌شوند.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # دیتای خام مشترک
    df_raw = load_data(file_name=file_name, folder="data/processed")

    # مدل بدون FGI
    base_df_pred = train_eval_one_model_by_date(
        df_raw=df_raw,
        feature_cols=["close", "volume", "rs_volatility"],
        label_col="close",
        date_col=date_col,
        hyperparams=base_hparams,
        model_name=model_base_name,
        outputs_dir=out_dir
    )

    # مدل با FGI
    cfgi_df_pred = train_eval_one_model_by_date(
        df_raw=df_raw,
        feature_cols=["close", "volume", "rs_volatility", "value"],
        label_col="close",
        date_col=date_col,
        hyperparams=cfgi_hparams,
        model_name=model_cfgi_name,
        outputs_dir=out_dir
    )

    # هم‌ترازسازی تاریخ تست (intersection)
    base_df_pred["d_day"] = pd.to_datetime(base_df_pred["date"]).dt.floor("D")
    cfgi_df_pred["d_day"] = pd.to_datetime(cfgi_df_pred["date"]).dt.floor("D")
    merged = base_df_pred.merge(
        cfgi_df_pred[["d_day", "y_pred"]].rename(columns={"y_pred": "y_pred_cfgi"}),
        on="d_day", how="inner"
    )
    # y_true را از baseline می‌گیریم (هر دو باید یکسان باشند روی تاریخ مشترک)
    merged = merged[["d_day", "y_true", "y_pred", "y_pred_cfgi"]].copy()
    merged.to_csv(f"{out_dir}/aligned_predictions.csv", index=False)

    # اتصال FGI(t) برای سگمنت‌بندی
    merged = attach_FGI_lag1_for_target_day(
        df_pred=merged.rename(columns={"d_day": "date"}),
        fgi_df=df_raw,
        date_col_pred="date",
        fgi_time_col=date_col,
        fgi_value_col="value",
    ).rename(columns={"date": "d_day"})

    # سگمنت
    merged = add_segment_col(merged)

    # متریک‌های کل روی بازه مشترک
    overall_base  = _metrics(merged["y_true"], merged["y_pred"])
    overall_cfgi  = _metrics(merged["y_true"], merged["y_pred_cfgi"])
    pd.DataFrame([overall_base, overall_cfgi], index=["BASE", "CFGI"]).to_csv(
        f"{out_dir}/overall_metrics_aligned.csv"
    )

    # متریک‌های سگمنت
    seg_rows = []
    for s, g in merged.groupby("segment", dropna=False):
        m_base = _metrics(g["y_true"], g["y_pred"])
        m_cfgi = _metrics(g["y_true"], g["y_pred_cfgi"])
        seg_rows.append({
            "segment": s,
            **{f"BASE_{k}": v for k, v in m_base.items()},
            **{f"CFGI_{k}": v for k, v in m_cfgi.items()},
        })
    pd.DataFrame(seg_rows).to_csv(f"{out_dir}/segment_metrics_aligned.csv", index=False)

    # تست آماری جفتی (اختیاری: اگر SciPy نصب باشد)
    try:
        from scipy.stats import wilcoxon
        all_stat, all_p = wilcoxon(np.abs(merged["y_pred"] - merged["y_true"]),
                                   np.abs(merged["y_pred_cfgi"] - merged["y_true"]))
        print(f"[ALL] Wilcoxon |err|  stat={all_stat:.3f}, p={all_p:.4f}, n={len(merged)}")
        with open(f"{out_dir}/wilcoxon_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"[ALL] Wilcoxon |err|  stat={all_stat:.6f}, p={all_p:.6f}, n={len(merged)}\n")
            for s, g in merged.groupby("segment"):
                if len(g) > 10:
                    st, p = wilcoxon(np.abs(g["y_pred"] - g["y_true"]),
                                     np.abs(g["y_pred_cfgi"] - g["y_true"]))
                    f.write(f"[{s}] Wilcoxon |err|  stat={st:.6f}, p={p:.6f}, n={len(g)}\n")
    except Exception as e:
        print("Wilcoxon test skipped (scipy not available or other issue):", e)

    print(f"\n✅ Files saved under: {out_dir}")
    print(" - aligned_predictions.csv")
    print(" - overall_metrics_aligned.csv")
    print(" - segment_metrics_aligned.csv")
    print(" - (optional) wilcoxon_summary.txt")

# ==============================
# اجرای نمونه
# ==============================
if __name__ == "__main__":
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

    compare_two_models_on_segments_by_date(
        base_hparams=base_hparams,
        cfgi_hparams=cfgi_hparams,
        out_dir="outputs/cfgi_segment_eval_by_date_final",
        model_base_name="price_baseline",
        model_cfgi_name="price_with_cfgi",
        file_name="base_dataframe_CFGI.csv",
        date_col="timeopen"
    )
