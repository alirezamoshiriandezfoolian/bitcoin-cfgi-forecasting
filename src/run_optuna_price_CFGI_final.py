import os
import optuna
import joblib
import matplotlib.pyplot as plt


from tuner_price_CFGI import objective  # 📌 نسخه ویژه مدل قیمت


# 1️⃣ ساخت پوشه خروجی در صورت نبود
os.makedirs("optuna_results_price_CFGI_final", exist_ok=True)


# 2️⃣ تعریف Study روی SQLite
study = optuna.create_study(
    study_name="bilstm_price_tuning_CFGI_final",
    direction="minimize",
    storage="sqlite:///optuna_results_price_CFGI_final/optuna_results_price_CFGI_final.db",
    load_if_exists=True
)


# 3️⃣ اجرای تیونینگ
n_trials = 750  # تعداد قابل تغییر
study.optimize(objective, n_trials=n_trials)


# 4️⃣ ذخیره بهترین پارامترها
best_params_path = "optuna_results_price_CFGI_final/best_params_price_CFGI_final.pkl"
joblib.dump(study.best_params, best_params_path)
print(f"✅ Best parameters saved to: {best_params_path}")


# 5️⃣ چاپ بهترین trial
print("\n📌 Best trial:")
print(f"  Loss: {study.best_value:.6f}")
print("  Params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")


# 6️⃣ نمودار تاریخچه loss
try:
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History (Price Model)")
    plt.tight_layout()
    plt.savefig("optuna_results_price_CFGI_final/optimization_history.png")
    plt.close()
    print("📈 Optimization history saved as PNG.")
except Exception as e:
    print("⚠️ Couldn't plot optimization history:", e)


# 7️⃣ نمودار اهمیت پارامترها
try:
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances (Price Model)")
    plt.tight_layout()
    plt.savefig("optuna_results_price_CFGI_final/param_importances.png")
    plt.close()
    print("📊 Parameter importances saved as PNG.")
except Exception as e:
    print("⚠️ Couldn't plot parameter importances:", e)


# 8️⃣ نمودار مختصات موازی
try:
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinates (Price Model)")
    plt.tight_layout()
    plt.savefig("optuna_results_price_CFGI_final/parallel_coordinates.png")
    plt.close()
    print("🧮 Parallel coordinate plot saved as PNG.")
except Exception as e:
    print("⚠️ Couldn't plot parallel coordinate:", e)