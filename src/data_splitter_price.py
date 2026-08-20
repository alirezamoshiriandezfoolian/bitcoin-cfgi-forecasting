import numpy as np
from typing import Tuple

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """
    تقسیم داده‌ها به سه بخش آموزش، اعتبارسنجی و آزمون به ترتیب زمانی (بدون shuffle)

    Returns:
    - X_train, X_val, X_test
    - y_train, y_val, y_test
    - dates_test (فقط تاریخ‌های تست برای ارزیابی نهایی)
    """
    total_size = len(X)
    train_end = int(train_ratio * total_size)
    val_end = train_end + int(val_ratio * total_size)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]
    dates_test       = dates[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test, dates_test
