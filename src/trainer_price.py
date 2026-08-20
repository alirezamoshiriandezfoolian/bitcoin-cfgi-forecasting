import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict

def train_price_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    model_path: str = "outputs/best_model_price.h5"
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, float]]:
    """
    آموزش مدل Bi-LSTM برای پیش‌بینی قیمت بیت‌کوین با استفاده از LogCosh.

    Returns:
    - model: مدل آموزش‌دیده
    - history: تاریخچه آموزش
    - extra_metrics: مقادیر MAE، MSE و R² روی validation set
    """

    # کامپایل مدل با LogCosh
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LogCosh()
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    y_val_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    r2  = r2_score(y_val, y_val_pred)

    extra_metrics = {
        "mae_price": mae,
        "mse_price": mse,
        "r2_price": r2
    }

    return model, history, extra_metrics
