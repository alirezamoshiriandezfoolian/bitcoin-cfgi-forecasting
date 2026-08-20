import os
import optuna
import joblib
import matplotlib.pyplot as plt

from tuner_price_CFGI_new_features_CORRECTED import objective


RESULT_DIR = "optuna_results_new_features_price_corrected"
STUDY_NAME = "bilstm_price_tuning_new_features_price_corrected"

os.makedirs(RESULT_DIR, exist_ok=True)

study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="minimize",
    storage=f"sqlite:///{RESULT_DIR}/optuna_results_new_features_price_corrected.db"
)

N_TRIALS = 750

study.optimize(
    objective,
    n_trials=N_TRIALS
)

joblib.dump(
    study.best_params,
    os.path.join(
        RESULT_DIR,
        "best_params_new_features_price_corrected.pkl"
    )
)

print("\n======================================")
print("OPTIMIZATION COMPLETED")
print("======================================")
print(f"Completed trials requested: {N_TRIALS}")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best validation loss: {study.best_value:.10f}")

print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

try:
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "optimization_history.png"),
        dpi=300
    )
    plt.close()
except Exception as e:
    print(f"Optimization-history plot could not be created: {e}")

try:
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "param_importances.png"),
        dpi=300
    )
    plt.close()
except Exception as e:
    print(f"Parameter-importance plot could not be created: {e}")