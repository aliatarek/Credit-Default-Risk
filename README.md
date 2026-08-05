# Credit Default Risk

This project analyzes loan application data to predict credit default risk. The target is a supervised binary classification label:

- `TARGET = 0`: applicant repaid normally.
- `TARGET = 1`: applicant had repayment difficulties.

The work follows the project phases described in `Credit Default Risk.pdf`, using the primary application dataset.

## Repository Contents

```text
applications_dataset.csv
phase_1/
  Phase_1_EDA.ipynb
phase_2/
  Phase_2_Data_Preparation_Feature_Engineering.ipynb
```

## Phase 1 - Exploratory Data Analysis

Notebook:

```text
phase_1/Phase_1_EDA.ipynb
```

Phase 1 focuses on understanding the dataset before modeling. It includes:

- dataset shape and primary key checks;
- target distribution and class imbalance;
- missing value analysis;
- categorical default-rate analysis;
- numeric analysis by target;
- correlation analysis;
- outlier inspection;
- business insights from the application data.

## Phase 2 - Data Preparation and Feature Engineering

Notebook:

```text
phase_2/Phase_2_Data_Preparation_Feature_Engineering.ipynb
```

Phase 2 prepares model-ready datasets. It includes:

- handling missing values;
- dropping high-missing columns;
- creating application-level features;
- train, validation, and test split;
- numeric imputation;
- categorical imputation;
- categorical encoding;
- saving prepared feature and label files.

## Dataset

The dataset file is:

```text
applications_dataset.csv
```

Each row represents one loan application and is identified by `SK_ID_CURR`.

## How To Run

Open the notebooks in Jupyter from the repository root:

```powershell
jupyter notebook phase_1\Phase_1_EDA.ipynb
jupyter notebook phase_2\Phase_2_Data_Preparation_Feature_Engineering.ipynb
```

Run Phase 1 first to understand the data, then run Phase 2 to prepare model-ready files.
