# Investigating the Impact of the Crypto Fear and Greed Index on Bitcoin Price Using a Deep Learning Bi-LSTM Model

## Overview

This repository contains the data-preparation workflow, model
implementation, hyperparameter optimization procedures, evaluation
scripts, and computational outputs associated with the study
**"Investigating the Impact of the Crypto Fear and Greed Index on
Bitcoin Price Using a Deep Learning Bi-LSTM Model."**

The study examines whether the Crypto Fear and Greed Index (CFGI)
provides incremental predictive information for next-day Bitcoin
closing-price forecasting beyond market-based predictors. A
bidirectional long short-term memory (Bi-LSTM) framework is used to
compare a market-based benchmark model with sentiment-augmented
specifications incorporating the CFGI directly and through derived
temporal features.

Hyperparameters are optimized using Optuna. Additional analyses examine
whether the predictive contribution of the CFGI changes during periods
of extreme fear and extreme greed.

------------------------------------------------------------------------

## Research Design

The empirical analysis uses daily Bitcoin market and sentiment data from
**January 2020 through December 2024**.

Three main model specifications are considered:

1.  **Benchmark model**
    -   Bitcoin closing price
    -   Trading volume
    -   Rogers--Satchell volatility
2.  **Benchmark + CFGI**
    -   Benchmark variables
    -   Crypto Fear and Greed Index
3.  **Benchmark + CFGI-derived features**
    -   Benchmark variables
    -   CFGI
    -   CFGI lags (1--7 days)
    -   CFGI simple moving averages (3, 5, and 7 days)
    -   CFGI momentum

The forecasting target is the **next-day Bitcoin closing price**.

Observations are kept in chronological order throughout preprocessing
and evaluation. Samples are divided into training (70%), validation
(15%), and test (15%) subsets without shuffling. Feature and target
scalers are fitted using the training subset only and then applied
unchanged to the validation and test subsets.

------------------------------------------------------------------------

## Repository Structure

``` text
bitcoin-cfgi-forecasting/
│
├── notebook/
│   └── data_preparation/
│       ├── 01_prepare_volatility_data.ipynb
│       ├── 02_build_base_dataframe.ipynb
│       ├── 03_merge_cfgi.ipynb
│       ├── 04_create_cfgi_lags.ipynb
│       ├── 05_create_cfgi_sma.ipynb
│       ├── 06_create_momentum_features.ipynb
│       ├── 07_merge_cfgi_features.ipynb
│       └── 08_build_features_for_price.ipynb
│
├── src/
│   ├── data_loader2.py
│   ├── preprocessing_price.py
│   ├── data_splitter_price.py
│   ├── model_builder_price.py
│   ├── trainer_price.py
│   │
│   ├── tuner_price.py
│   ├── tuner_price_CFGI.py
│   ├── tuner_price_CFGI_new_features_CORRECTED.py
│   │
│   ├── run_optuna_price_base_final.py
│   ├── run_optuna_price_CFGI_final.py
│   ├── run_optuna_price_new_features_corrected.py
│   │
│   ├── evaluate_model_with_params_price.py
│   ├── evaluate_model_with_params_price_CFGI.py
│   ├── evaluate_model_with_params_price_new_features.py
│   ├── evaluate_model_with_params_price_new_features_CORRECTED.py
│   │
│   ├── run_final_evaluation_base.py
│   ├── run_evaluation_CFGI_with_baseline_params.py
│   ├── run_evaluation_new_features_with_baseline_params.py
│   ├── run_final_evaluation_CFGI.py
│   ├── run_final_evaluation_new_features_corrected.py
│   │
│   ├── compare_CFGI_effect2.py
│   ├── compare_CFGI_extreme_train2.py
│   └── plot_extreme_absolute_errors.py
│
├── outputs/
│   ├── test_results_final.csv
│   ├── cfgi_segment_eval_by_date_final/
│   └── cfgi_extreme_train_eval1_final/
│
├── optuna_results_price_base_final/
├── optuna_results_price_CFGI_final/
├── optuna_results_new_features_price_corrected/
│
├── requirements.txt
└── .gitignore
```

The `notebook/data_preparation/` directory contains the sequential
data-preparation workflow. The `src/` directory contains the
forecasting, optimization, evaluation, and sentiment-regime analysis
code. Final evaluation outputs and the Optuna study databases are
retained in their corresponding output directories.

The analysis also uses a local `data/` working directory with `raw/` and
`processed/` subdirectories. This directory is intentionally excluded
from version control because the source and derived datasets are not
redistributed in this repository.

------------------------------------------------------------------------

## Data

The empirical analysis uses daily Bitcoin market data obtained from
**CoinMarketCap** and the **Crypto Fear and Greed Index (CFGI)**
obtained from **Alternative.me**.

Because these data originate from third-party sources, the raw and
processed datasets used in the study are not redistributed in this
repository. Users wishing to reproduce the data-preparation stage should
obtain the corresponding source data from the original providers,
subject to their current access conditions and terms of use.

The supplied data-preparation notebooks document the workflow used in
the study to construct the Rogers--Satchell volatility measure, merge
the CFGI with the Bitcoin market data, generate the lagged,
moving-average, and momentum-based CFGI features, and construct the
analytical datasets used in the forecasting experiments.

For the supplied notebooks to use the same local paths as the original
workflow, create:

``` text
data/
├── raw/
└── processed/
```

and place the two source files in `data/raw/` using the filenames
expected by the notebooks:

``` text
data/raw/combined_bitcoin_price_2020_2024.xlsx
data/raw/crypto-fear_and_greed_index.xlsx
```

The first file contains the Bitcoin market observations used by the
data-preparation workflow, and the second contains the daily CFGI
observations. The repository does not redistribute either file.

------------------------------------------------------------------------

## Data Preparation

Run the notebooks in `notebook/data_preparation/` in numerical order
after the required source files have been placed in `data/raw/`:

``` text
01_prepare_volatility_data.ipynb
        ↓
02_build_base_dataframe.ipynb
        ↓
03_merge_cfgi.ipynb
        ↓
04_create_cfgi_lags.ipynb
05_create_cfgi_sma.ipynb
06_create_momentum_features.ipynb
        ↓
07_merge_cfgi_features.ipynb
        ↓
08_build_features_for_price.ipynb
```

The workflow performs the following main operations:

-   preparation of the Bitcoin market data;
-   computation of Rogers--Satchell volatility;
-   construction of the base market dataframe;
-   merging of the daily CFGI;
-   generation of CFGI lags;
-   generation of 3-, 5-, and 7-day CFGI moving averages;
-   construction of the CFGI momentum feature;
-   merging of the derived sentiment features; and
-   construction of the final feature-engineered dataset used by the
    corresponding forecasting model.

Notebook 07 generates the intermediate file:

``` text
data/processed/merged_all_features.csv
```

which is subsequently used by Notebook 08 to construct:

``` text
data/processed/features_for_price.csv
```

------------------------------------------------------------------------

## Model Specifications

All forecasting specifications use a Bi-LSTM architecture and predict
the Bitcoin closing price one day ahead.

### Benchmark

The benchmark specification uses:

``` text
close
volume
rs_volatility
```

### Benchmark + CFGI

The second specification adds:

``` text
value
```

where `value` represents the daily CFGI.

### Benchmark + CFGI-Derived Features

The third specification additionally incorporates:

``` text
CFGI_momentum

value_lag1
value_lag2
value_lag3
value_lag4
value_lag5
value_lag6
value_lag7

value_sma3
value_sma5
value_sma7
```

------------------------------------------------------------------------

## Hyperparameter Optimization

Hyperparameter optimization is performed using **Optuna**.

Separate optimization runners are provided for the three model
specifications:

``` text
src/run_optuna_price_base_final.py
src/run_optuna_price_CFGI_final.py
src/run_optuna_price_new_features_corrected.py
```

The optimization procedure is configured for **750 trials per model
specification** and minimizes validation Log-Cosh loss.

The search space includes sequence length, Bi-LSTM depth, LSTM units,
dropout rate, dense-layer size, activation function, learning rate,
batch size, and early-stopping patience.

The corresponding Optuna SQLite databases are retained in:

``` text
optuna_results_price_base_final/
optuna_results_price_CFGI_final/
optuna_results_new_features_price_corrected/
```

These databases preserve the recorded optimization studies and allow the
trial histories and selected configurations to be inspected without
rerunning the complete search.

------------------------------------------------------------------------

## Final Evaluation

The final evaluation design separates the contribution of sentiment
information from the effect of model-specific hyperparameter
optimization.

The five selected configurations are evaluated on the held-out test set
using the following metrics:

-   Mean Absolute Error (MAE)
-   Mean Squared Error (MSE)
-   Median Absolute Error (MedAE)
-   Mean Absolute Percentage Error (MAPE)
-   Coefficient of Determination (R²)

From the repository root, the main evaluation runners are:

### Benchmark

``` bash
python src/run_final_evaluation_base.py
```

### CFGI model using benchmark hyperparameters

``` bash
python src/run_evaluation_CFGI_with_baseline_params.py
```

### CFGI-derived-features model using benchmark hyperparameters

``` bash
python src/run_evaluation_new_features_with_baseline_params.py
```

### CFGI model using independently optimized hyperparameters

``` bash
python src/run_final_evaluation_CFGI.py
```

### CFGI-derived-features model using independently optimized hyperparameters

``` bash
python src/run_final_evaluation_new_features_corrected.py
```

The consolidated evaluation results are stored in:

``` text
outputs/test_results_final.csv
```

**Important:** the evaluation runners append their results to
`outputs/test_results_final.csv`. The repository already contains the
reported final results. For a clean rerun, preserve the supplied file
and run the evaluation sequence in a fresh working copy, or temporarily
rename the existing CSV before executing the five runners in the order
shown above.

------------------------------------------------------------------------

## Extreme-Sentiment Analysis

Extreme sentiment is defined as:

``` text
Extreme Fear:  CFGI ≤ 25
Extreme Greed: CFGI ≥ 75
```

Two complementary analyses are provided.

### 1. Evaluation by sentiment regime

``` bash
python src/compare_CFGI_effect2.py
```

The models are trained using the full training sample and their
test-period performance is evaluated across sentiment regimes.

Outputs are written to:

``` text
outputs/cfgi_segment_eval_by_date_final/
```

Principal retained outputs include:

``` text
aligned_predictions.csv
overall_metrics_aligned.csv
segment_metrics_aligned.csv
```

### 2. Extreme-only training and evaluation

``` bash
python src/compare_CFGI_extreme_train2.py
```

In this setting, the relevant training, validation, and test
observations are restricted to extreme-sentiment periods.

Outputs are written to:

``` text
outputs/cfgi_extreme_train_eval1_final/
```

The paired error file used for the error plot is:

``` text
paired_errors_extreme.csv
```

The plotting script:

``` bash
python src/plot_extreme_absolute_errors.py
```

reads this file and generates:

``` text
outputs/cfgi_extreme_train_eval1_final/absolute_errors_extreme.png
```

The repository also retains `Fig1_final.pdf`, the figure file used with
the manuscript materials.

------------------------------------------------------------------------

## Reproducing the Analysis

### 1. Create and activate the Python environment

Python **3.10** is recommended. The original project environment used
Python **3.10.9**.

``` bash
python -m venv venv
```

On Windows:

``` bash
venv\Scripts\activate
```

### 2. Install dependencies

``` bash
pip install -r requirements.txt
```

### 3. Obtain and prepare the source data

Obtain the Bitcoin market data from **CoinMarketCap** and the CFGI data
from **Alternative.me**, subject to the providers' current access
conditions and terms of use.

Create the local directories:

``` text
data/raw/
data/processed/
```

and place the source files in `data/raw/` with the filenames expected by
the notebooks:

``` text
combined_bitcoin_price_2020_2024.xlsx
crypto-fear_and_greed_index.xlsx
```

Then run all notebooks under:

``` text
notebook/data_preparation/
```

in numerical order from `01` through `08`.

### 4. Reproduce the final evaluations

The complete Optuna searches are not required to run the final
evaluation stage because the selected hyperparameter configurations are
specified in the final evaluation runners.

For a clean rerun, use a fresh working copy or preserve and temporarily
rename the supplied:

``` text
outputs/test_results_final.csv
```

Then run, in order:

``` bash
python src/run_final_evaluation_base.py
python src/run_evaluation_CFGI_with_baseline_params.py
python src/run_evaluation_new_features_with_baseline_params.py
python src/run_final_evaluation_CFGI.py
python src/run_final_evaluation_new_features_corrected.py
```

### 5. Reproduce the extreme-sentiment analyses

``` bash
python src/compare_CFGI_effect2.py
python src/compare_CFGI_extreme_train2.py
python src/plot_extreme_absolute_errors.py
```

### Optional: rerun the full hyperparameter optimization

The supplied SQLite databases retain the recorded Optuna studies. A new
full HPO run is computationally expensive and is not necessary for
reproducing the reported final evaluation stage.

If a fresh HPO run is desired, use a clean working copy in which the
corresponding Optuna database files are not already present, and
execute:

``` bash
python src/run_optuna_price_base_final.py
python src/run_optuna_price_CFGI_final.py
python src/run_optuna_price_new_features_corrected.py
```

The optimization runners are configured for 750 trials per model
specification.

------------------------------------------------------------------------

## Software Environment

The original project environment used Python 3.10.9 and the following
principal packages:

``` text
Python             3.10.9
TensorFlow         2.10.0
NumPy              1.24.3
pandas             2.2.3
scikit-learn       1.7.0
Optuna             4.4.0
Matplotlib         3.10.3
openpyxl           3.1.5
joblib             1.5.1
```

Exact Python-package requirements are provided in `requirements.txt`.

Random seeds are fixed where applicable to improve computational
reproducibility.

------------------------------------------------------------------------

## Outputs

### Main forecasting results

``` text
outputs/test_results_final.csv
```

contains the consolidated test-set evaluation results for the main model
configurations.

### Sentiment-regime results

``` text
outputs/cfgi_segment_eval_by_date_final/
```

contains the aligned predictions and aggregate performance results for
the sentiment-regime analysis.

### Extreme-only results

``` text
outputs/cfgi_extreme_train_eval1_final/
```

contains the retained prediction/error outputs and figure material
associated with the extreme-only analysis.

### Hyperparameter optimization

The SQLite databases under:

``` text
optuna_results_price_base_final/
optuna_results_price_CFGI_final/
optuna_results_new_features_price_corrected/
```

contain the recorded Optuna studies for the three forecasting
specifications.

Generated Keras model files and intermediate Optuna trial checkpoints
are not version-controlled because they can be regenerated from the
supplied code and configurations.

------------------------------------------------------------------------

## Main Finding

Across the examined specifications, incorporating the CFGI did not
produce a consistent improvement in next-day Bitcoin closing-price
forecasting relative to the market-based benchmark. This pattern
remained when the CFGI was included directly, when derived sentiment
features were incorporated, when the augmented models were independently
optimized, and in the additional extreme-sentiment analyses.

Within the daily-frequency Bi-LSTM forecasting framework examined in the
study, the CFGI therefore provided limited incremental predictive
information beyond the market-based variables included in the benchmark.

------------------------------------------------------------------------

## Data Availability

The source datasets are not redistributed in this repository because
they originate from third-party data providers.

Bitcoin market data were obtained from **CoinMarketCap**, and the Crypto
Fear and Greed Index was obtained from **Alternative.me**. The
repository provides the data-preparation workflow and records the local
input filenames used in the original analysis. Users should consult the
respective data providers and their applicable access conditions and
terms of use when obtaining, using, or redistributing source data.

------------------------------------------------------------------------

## Citation

If you use this repository or the associated methodology, please cite
the corresponding article.

Citation details will be added following publication.
