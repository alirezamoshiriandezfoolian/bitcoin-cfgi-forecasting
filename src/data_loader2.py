# src/data_loader.py

import pandas as pd
import os

def load_data(file_name: str, folder: str = "data/processed") -> pd.DataFrame:
    """
    بارگذاری فایل CSV یا Excel و مرتب‌سازی بر اساس ستون 'timeopen'

    Args:
        file_name (str): نام فایل (با پسوند)
        folder (str): مسیر پوشه فایل

    Returns:
        pd.DataFrame: دیتافریم مرتب‌شده
    """
    file_path = os.path.join(folder, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    
     # بررسی وجود ستون 'timeopen' با پیام شفاف
    if "timeopen" not in df.columns:
        raise KeyError(f"'timeopen' column not found. Available columns: {list(df.columns)}")

    # تبدیل ستون 'timeopen' به datetime و مرتب‌سازی
    df["timeopen"] = pd.to_datetime(df["timeopen"])
    df.sort_values("timeopen", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
