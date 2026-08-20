from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.layers import LeakyReLU, PReLU, Activation
from tensorflow.keras.activations import swish

def build_price_model(
    input_shape,
    lstm_units=64,
    dropout_rates=[0.2],
    dense_units=32,
    num_layers=1,
    activation="leaky_relu"
):
    """
    ساخت مدل Bi-LSTM برای پیش‌بینی قیمت بیت‌کوین (یک خروجی)

    Parameters:
    - input_shape: شکل ورودی به مدل (sequence_length, num_features)
    - lstm_units: تعداد واحدهای LSTM در هر لایه Bi-LSTM
    - dropout_rate: نرخ dropout بعد از هر لایه
    - dense_units: نرون‌های لایه Dense
    - num_layers: تعداد لایه‌های Bi-LSTM
    - activation: نوع تابع فعال‌سازی در Dense

    Returns:
    - model: مدل Keras قابل کامپایل
    """
    # نرمال‌سازی طول dropout_rates نسبت به num_layers
    if len(dropout_rates) < num_layers:
        dropout_rates = dropout_rates + [dropout_rates[-1]] * (num_layers - len(dropout_rates))
    elif len(dropout_rates) > num_layers:
        dropout_rates = dropout_rates[:num_layers]

    inputs = Input(shape=input_shape, name="input_layer")
    x = inputs

    for i in range(num_layers):
        return_seq = i < num_layers - 1
        x = Bidirectional(
            LSTM(units=lstm_units, return_sequences=return_seq),
            name=f"bilstm_layer_{i+1}"
        )(x)
        x = Dropout(rate=dropout_rates[i], name=f"dropout_layer_{i+1}")(x)

    # Dense میانی اختیاری
    if dense_units and dense_units > 0:
        x = Dense(units=dense_units, name="dense_layer")(x)

        # اعمال activation فقط در صورت وجود Dense میانی
        if activation == "leaky_relu":
            x = LeakyReLU(alpha=0.01, name="activation_layer")(x)
        elif activation == "prelu":
            x = PReLU(name="activation_layer")(x)
        elif activation == "swish":
            x = Activation("swish", name="activation_layer")(x)
        elif activation == "relu":
            x = Activation("relu", name="activation_layer")(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    outputs = Dense(units=1, activation="linear", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Price_Model")
    return model
