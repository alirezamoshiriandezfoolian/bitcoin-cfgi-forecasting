import os

from evaluate_model_with_params_price import evaluate_model_with_params


RESULTS_PATH = "outputs/test_results_final.csv"

os.makedirs("outputs", exist_ok=True)


configs = [
    {
        "trial_number": 634,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.3,
        "dense_units": 128,
        "activation": "relu",
        "learning_rate": 0.0008289739845554754,
        "batch_size": 32,
        "patience": 30
    },
    {
        "trial_number": 360,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.4,
        "dense_units": 128,
        "activation": "relu",
        "learning_rate": 0.001019707839722582,
        "batch_size": 32,
        "patience": 30
    },
    {
        "trial_number": 425,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.3,
        "dense_units": 128,
        "activation": "relu",
        "learning_rate": 0.0006547490571174007,
        "batch_size": 32,
        "patience": 30
    },
    {
        "trial_number": 723,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.3,
        "dense_units": 128,
        "activation": "relu",
        "learning_rate": 0.0007741123865387291,
        "batch_size": 32,
        "patience": 30
    },
    {
        "trial_number": 421,
        "sequence_length": 4,
        "lstm_units": 512,
        "num_layers": 1,
        "dropout_rate_1": 0.3,
        "dense_units": 128,
        "activation": "relu",
        "learning_rate": 0.0006646016716010517,
        "batch_size": 32,
        "patience": 30
    }
]


for rank, config in enumerate(configs, start=1):

    trial_number = config.pop("trial_number")

    model_name = f"best_model_base_final_{rank}"

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
print("ALL FIVE BASELINE FINAL EVALUATIONS COMPLETED")
print("======================================")
print(f"Results saved to: {RESULTS_PATH}")