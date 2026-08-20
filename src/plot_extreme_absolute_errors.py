from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path(
    "outputs/cfgi_extreme_train_eval1_final/paired_errors_extreme.csv"
)

OUTPUT_PATH = Path(
    "outputs/cfgi_extreme_train_eval1_final/absolute_errors_extreme.png"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run compare_CFGI_extreme_train2.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_cols = {"d_day", "err_base", "err_cfgi"}
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {INPUT_PATH}: "
            f"{sorted(missing_cols)}"
        )

    df["d_day"] = pd.to_datetime(df["d_day"])

    df = df.sort_values("d_day").reset_index(drop=True)

    df["abs_error_base"] = np.abs(df["err_base"])
    df["abs_error_cfgi"] = np.abs(df["err_cfgi"])

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["d_day"],
        df["abs_error_base"],
        label="Baseline model"
    )

    plt.plot(
        df["d_day"],
        df["abs_error_cfgi"],
        label="Model with CFGI"
    )

    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.title(
        "Absolute Forecast Errors on Extreme-Sentiment Test Days"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()