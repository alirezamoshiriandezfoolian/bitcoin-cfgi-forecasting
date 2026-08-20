import os

from evaluate_model_with_params_price_CFGI import evaluate_model_with_params


RESULTS_PATH = "outputs/test_results_final.csv"

os.makedirs("outputs", exist_ok=True)


configs = [
    {
        "trial_number": 661,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 64,
        "activation": "swish",
        "learning_rate": 8.145754627039187e-05,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 549,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 64,
        "activation": "swish",
        "learning_rate": 8.165830851380932e-05,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 505,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 64,
        "activation": "swish",
        "learning_rate": 8.267453890450604e-05,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 656,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 64,
        "activation": "swish",
        "learning_rate": 8.225917975476676e-05,
        "batch_size": 16,
        "patience": 30
    },
    {
        "trial_number": 474,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.0,
        "dense_units": 64,
        "activation": "swish",
        "learning_rate": 8.218861109504286e-05,
        "batch_size": 16,
        "patience": 30
    }
]


for rank, config in enumerate(configs, start=1):

    trial_number = config.pop("trial_number")

    model_name = f"best_model_CFGI_final_tune_{rank}"

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
print("ALL FIVE CFGI FINAL EVALUATIONS COMPLETED")
print("======================================")
print(f"Results saved to: {RESULTS_PATH}")