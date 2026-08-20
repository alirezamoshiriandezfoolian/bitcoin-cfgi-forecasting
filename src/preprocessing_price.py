import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    date_col: str,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ساخت دنباله‌های زمانی برای ورودی مدل Bi-LSTM (قبل از نرمال‌سازی)

    Returns:
    - X: ویژگی‌ها به شکل (num_samples, sequence_length, num_features)
    - y: خروجی‌ها به شکل (num_samples, num_targets)
    - dates: تاریخ هر نمونه برای ارزیابی بعدی
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    X, y, dates = [], [], []

    for i in range(sequence_length, len(df)):
        X.append(df[feature_cols].iloc[i-sequence_length:i].values)
        y.append(df[label_col].iloc[i])
        dates.append(df[date_col].iloc[i])

    X = np.array(X)    # (num_samples, seq_len, num_features)
    y = np.array(y).reshape(-1, 1)    # (num_samples, num_targets)
    dates = np.array(dates)

    return X, y, dates

def fit_scalers(X_train, y_train):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler_X.fit(X_train_reshaped)
    scaler_y.fit(y_train.reshape(-1, 1))

    return scaler_X, scaler_y


def transform_sequences(X: np.ndarray, scaler: MinMaxScaler ) -> np.ndarray:
    """
    نرمال‌سازی کل دنباله‌ها با استفاده از اسکالر آموزش‌دیده
    """
    num_samples, seq_len, num_features = X.shape
    reshaped = X.reshape(-1, num_features)
    scaled = scaler.transform(reshaped)
    scaled_X = scaled.reshape(num_samples, seq_len, num_features)

    return scaled_X

def inverse_transform_y(y_scaled, scaler_y):
    # اگر ورودی تک‌بعدی باشه، reshape کن به (n, 1)
    if len(y_scaled.shape) == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    return scaler_y.inverse_transform(y_scaled)
