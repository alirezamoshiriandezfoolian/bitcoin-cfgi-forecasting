import os

from evaluate_model_with_params_price_new_features_CORRECTED import (
    evaluate_model_with_params
)


RESULTS_PATH = "outputs/test_results_final.csv"

os.makedirs("outputs", exist_ok=True)


configs = [
    {
        "trial_number": 660,
        "sequence_length": 2,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 256,
        "activation": "prelu",
        "learning_rate": 0.0017966316237751285,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 522,
        "sequence_length": 2,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 256,
        "activation": "prelu",
        "learning_rate": 0.0015821849846848893,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 633,
        "sequence_length": 2,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 256,
        "activation": "prelu",
        "learning_rate": 0.0017236175579619793,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 582,
        "sequence_length": 2,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 256,
        "activation": "prelu",
        "learning_rate": 0.0016149678207319136,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 637,
        "sequence_length": 2,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 256,
        "activation": "prelu",
        "learning_rate": 0.0017860027300456614,
        "batch_size": 16,
        "patience": 30
    }
]


for rank, config in enumerate(configs, start=1):

    trial_number = config.pop("trial_number")

    model_name = (
        f"best_model_CFGI_new_features_corrected_tune{rank}"
    )

    print("\n======================================")
    print(
        f"Running rank {rank}/5 "
        f"(Optuna Trial {trial_number})"
    )
    print("======================================")

    evaluate_model_with_params(
        hyperparams=config,
        model_name=model_name,
        results_csv_path=RESULTS_PATH
    )


print("\n======================================")
print("ALL FIVE FINAL EVALUATIONS COMPLETED")
print("======================================")
print(f"Results saved to: {RESULTS_PATH}")