# Fecal Contamination in Agricultural Water

💧This project is dedicated to investigating the impact of various geospatial and meteorological factors on fecal contamination of watersheds.

📖 This dataset is sourced from the publication "**Green, H., Wilder, M., Wiedmann, M., Weller, D.L. Integrative survey of 68 non-overlapping upstate New York watersheds reveals stream features associated with aquatic fecal contamination. Front. Microbiol. 12 (2021). https://doi.org/10.3389/fmicb.2021.684533**". 
Please cite this paper when using this dataset.

# Sample Analysis

## Overview

**Prediction task**:
- Regression for predicting E.coli concentration in watersheds
- Classification for predicting the presence of fecal contamination in watersheds

**Predictor and outcome variables**:
- The detailed description of metadata for predictor and outcome variables is accessible under the file name "Fecal_Metadata.csv"
- The cleaned dataset is accessible under the file name "processed_mstdata.csv"

**Evaluation metrics**:
- The regression model was evaluated on R<sup>2</sup>, RMSE, MSE, and MAE
- The classification model was evaluated on ROC AUC, sensitivity, specificity, and F1 score
- The specific packages for calculating these metrics are accessible in the model training script under the file name "ML_runner.py"


## Installation

### Dependencies
Before you run this script, make sure you have the following installed:
- Python 3.6 or higher
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- seaborn

You can install the necessary libraries using pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
To get started with this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/FoodDatasets/Predicting-fecal-contamination-in-agricultural-water.git
cd Predicting-fecal-contamination-in-agricultural-water
```
## Supported Algorithms
-  RandomForest: Random Forest Regression
-  LinearRegression: Linear Regression
-  SVM: Support Vector Machines for regression
-  GradientBoosting: Gradient Boosting Regressor

## Machine Learning Model Execution Guide

This script allows users to select different machine learning algorithms via command line parameters to train models and evaluate them on a specified dataset.
### Required Arguments

- `--file_path` (required): The path to your CSV file.
- `--target_label` (required): The target label for the prediction model. Choose from the available target labels in the dataset.
- `--algorithm` (required): The algorithm to use. Options include `RandomForest`, `LinearRegression`, `SVM`, `GradientBoosting`.
### Available target labels:
- `combined_label` This label combine the following labels together: HF183_pa, Rum2Bac_pa, and GFD_pa, if one of them is Positive,the label is Positive


### Optional Arguments
##### GBM (Gradient Boosting Machine) and RandomForest Specific Arguments
- `--n_estimators`: The number of trees in the forest (default: 100). Applicable for `RandomForest` and `GradientBoosting`.
  

- `--max_depth`: The maximum depth of the tree (default: None). Applicable for `RandomForest` and `GradientBoosting`.
##### SVM (Support Vector Machine) Specific Arguments
- `--C`: The regularization parameter (default: 1.0). Applicable for the `SVM` algorithm.

- `--kernel`: The kernel type to be used in the algorithm (default: 'rbf'). Applicable for the `SVM` algorithm. Options include linear, poly, rbf, sigmoid, precomputed.
``` bash
python script.py --file_path path/to/mstdata.csv --target_label combined_label --algorithm GradientBoosting 

```


### Usage Example
``` bash
python ML_runner.py --file_path path/to/mstdata.csv --target_label combined_label --algorithm RandomForest --n_estimators 200


```
## Example output of four algorithms-combined label

![Model Performance Table](Images/output_new.png)

### Performance Table

| Algorithm           | Avg ROC AUC | Accuracy | Precision | Recall | F1 Score | ROC AUC (Test) |
|---------------------|-------------|----------|-----------|--------|----------|----------------|
| RandomForest        | 0.58        | 0.65     | 0.50      | 0.21   | 0.30     | 0.55           |
| SVM                 | 0.57        | 0.65     | 0.00      | 0.00   | 0.00     | 0.50           |
| NeuralNetwork       | 0.57        | 0.70     | 0.75      | 0.21   | 0.33     | 0.59           |
| GradientBoosting    | 0.57        | 0.63     | 0.46      | 0.36   | 0.40     | 0.56           |
| LogisticRegression  | 0.50        | 0.68     | 0.55      | 0.43   | 0.48     | 0.62           |




### Confusion Matrix
| Algorithm           | True Positive (TP) | False Positive (FP) | False Negative (FN) | True Negative (TN) |
|---------------------|--------------------|---------------------|---------------------|--------------------|
| RandomForest        | 3                  | 3                   | 11                  | 23                 |
| SVM                 | 0                  | 0                   | 14                  | 26                 |
| GradientBoosting    | 5                  | 6                   | 9                   | 20                 |
| LogisticRegression  | 6                  | 5                   | 8                   | 21                 |
| NeuralNetwork       | 3                  | 1                   | 11                  | 25                 |



## Example output of ecoli
| Model            | Cross_Validation_Score |   MSE  |  RMSE  |  MAE  |    R²    |
|------------------|------------------------|--------|--------|-------|----------|
| SVM              | 0.1102                 | 0.303  | 0.550  | 0.443 | -0.125   |
| RandomForest     | 0.4466                 | 0.204  | 0.452  | 0.364 |  0.242   |
| LinearRegression | -240.5672              | 0.483  | 0.695  | 0.525 | -0.793   |
| GradientBoosting | 0.0323                 | 0.304  | 0.551  | 0.449 | -0.127   |
| NeuralNetwork    | -3.5591                | 0.659  | 0.812  | 0.629 | -1.444   |


# LazyPredict Method

## Dependencies

- Python 3.7+
- pandas
- scikit-learn
- LazyPredict
- imbalanced-learn (optional, for handling class imbalance)

Install the required packages

```bash
pip install pandas scikit-learn lazypredict imbalanced-learn
```

## Usage
Run the script with the datset file, file path and target column

``` bash
python ML_TEST.py --file_path /path/to/mstdata.csv --target_label combined_label --algorithm LazyClassifier
```
### Command-line Arguments
- `--file_path`: Path to the CSV file containing the dataset (required).
- `--target_label`
- `--algorithm`
Available target labels: `combined_label`

## Example Output 
### combined_label

![Model Performance](Images/output4.png)
| Model                          | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken | Sensitivity | Specificity |
|--------------------------------|----------|-------------------|---------|----------|------------|-------------|-------------|
| GaussianNB                     | 0.78     | 0.74              | 0.74    | 0.77     | 0.01       | 0.64        | 0.85        |
| Perceptron                     | 0.75     | 0.71              | 0.71    | 0.75     | 0.00       | 0.57        | 0.85        |
| LabelSpreading                 | 0.75     | 0.66              | 0.66    | 0.72     | 0.01       | 0.36        | 0.96        |
| LabelPropagation               | 0.75     | 0.66              | 0.66    | 0.72     | 0.01       | 0.36        | 0.96        |
| RandomForestClassifier         | 0.75     | 0.66              | 0.66    | 0.72     | 0.06       | 0.36        | 0.96        |
| NearestCentroid                | 0.70     | 0.65              | 0.65    | 0.69     | 0.01       | 0.50        | 0.81        |
| ExtraTreeClassifier            | 0.70     | 0.65              | 0.65    | 0.69     | 0.01       | 0.50        | 0.81        |
| KNeighborsClassifier           | 0.68     | 0.63              | 0.63    | 0.67     | 0.14       | 0.50        | 0.77        |
| SVC                            | 0.72     | 0.62              | 0.62    | 0.68     | 0.01       | 0.29        | 0.96        |
| NuSVC                          | 0.72     | 0.62              | 0.62    | 0.68     | 0.01       | 0.29        | 0.96        |
| ExtraTreesClassifier           | 0.70     | 0.62              | 0.62    | 0.67     | 0.05       | 0.36        | 0.88        |
| BernoulliNB                    | 0.68     | 0.62              | 0.62    | 0.66     | 0.01       | 0.43        | 0.81        |
| RidgeClassifierCV              | 0.72     | 0.61              | 0.61    | 0.66     | 0.01       | 0.21        | 1.00        |
| LogisticRegression             | 0.72     | 0.61              | 0.61    | 0.66     | 0.01       | 0.21        | 1.00        |
| RidgeClassifier                | 0.72     | 0.61              | 0.61    | 0.66     | 0.01       | 0.21        | 1.00        |
| BaggingClassifier              | 0.68     | 0.60              | 0.60    | 0.65     | 0.02       | 0.36        | 0.85        |
| AdaBoostClassifier             | 0.70     | 0.59              | 0.59    | 0.64     | 0.04       | 0.21        | 0.96        |
| XGBClassifier                  | 0.65     | 0.58              | 0.58    | 0.63     | 0.12       | 0.36        | 0.81        |
| SGDClassifier                  | 0.62     | 0.58              | 0.58    | 0.62     | 0.00       | 0.43        | 0.73        |
| LinearSVC                      | 0.68     | 0.57              | 0.57    | 0.62     | 0.01       | 0.21        | 0.92        |
| LinearDiscriminantAnalysis     | 0.68     | 0.57              | 0.57    | 0.62     | 0.01       | 0.21        | 0.92        |
| LGBMClassifier                 | 0.68     | 0.57              | 0.57    | 0.62     | 0.06       | 0.21        | 0.92        |
| DecisionTreeClassifier         | 0.65     | 0.57              | 0.57    | 0.62     | 0.01       | 0.29        | 0.85        |
| QuadraticDiscriminantAnalysis  | 0.35     | 0.50              | 0.50    | 0.18     | 0.02       | 1.00        | 0.00        |
| DummyClassifier                | 0.65     | 0.50              | 0.50    | 0.51     | 0.01       | 0.00        | 1.00        |
| CalibratedClassifierCV         | 0.65     | 0.50              | 0.50    | 0.51     | 0.02       | 0.00        | 1.00        |
| PassiveAggressiveClassifier    | 0.55     | 0.49              | 0.49    | 0.54     | 0.00       | 0.29        | 0.69        |

### ecoli
| Model                           | Adjusted R-Squared | R-Squared | RMSE  | Time Taken | MSE  | R²    | MAE  |
|----------------------------------|-------------------:|----------:|------:|-----------:|-----:|------:|-----:|
| Lars                             |               49.07 |     -13.79 |  2.56 |        0.01 | 6.58 | -13.79 |  2.17 |
| KernelRidge                      |               41.57 |     -11.48 |  2.36 |        0.01 | 5.55 | -11.48 |  2.26 |
| GaussianProcessRegressor         |               21.33 |      -5.25 |  1.67 |        0.01 | 2.78 |  -5.25 |  1.39 |
| PassiveAggressiveRegressor        |                5.33 |      -0.33 |  0.77 |        0.01 | 0.59 |  -0.33 |  0.59 |
| ExtraTreeRegressor               |                5.15 |      -0.28 |  0.75 |        0.00 | 0.57 |  -0.28 |  0.60 |
| DecisionTreeRegressor            |                5.15 |      -0.28 |  0.75 |        0.00 | 0.57 |  -0.28 |  0.59 |
| ExtraTreesRegressor              |                5.06 |      -0.25 |  0.75 |        0.08 | 0.56 |  -0.25 |  0.60 |
| RANSACRegressor                  |                4.94 |      -0.21 |  0.73 |        0.06 | 0.54 |  -0.21 |  0.58 |
| DummyRegressor                   |                4.69 |      -0.14 |  0.71 |        0.00 | 0.51 |  -0.14 |  0.55 |
| ElasticNet                       |                4.69 |      -0.14 |  0.71 |        0.00 | 0.51 |  -0.14 |  0.55 |
| LassoLars                        |                4.69 |      -0.14 |  0.71 |        0.00 | 0.51 |  -0.14 |  0.55 |
| Lasso                            |                4.69 |      -0.14 |  0.71 |        0.00 | 0.51 |  -0.14 |  0.55 |
| XGBRegressor                     |                4.60 |      -0.11 |  0.70 |        0.17 | 0.49 |  -0.11 |  0.55 |
| QuantileRegressor                |                4.60 |      -0.11 |  0.70 |        0.01 | 0.49 |  -0.11 |  0.54 |
| LassoLarsCV                      |                4.48 |      -0.07 |  0.69 |        0.01 | 0.48 |  -0.07 |  0.55 |
| OrthogonalMatchingPursuitCV      |                4.47 |      -0.07 |  0.69 |        0.00 | 0.48 |  -0.07 |  0.55 |
| OrthogonalMatchingPursuit        |                4.47 |      -0.07 |  0.69 |        0.00 | 0.48 |  -0.07 |  0.55 |
| TransformedTargetRegressor       |                4.43 |      -0.06 |  0.69 |        0.00 | 0.47 |  -0.06 |  0.55 |
| LinearRegression                 |                4.43 |      -0.06 |  0.69 |        0.00 | 0.47 |  -0.06 |  0.55 |
| Ridge                            |                4.42 |      -0.05 |  0.68 |        0.00 | 0.47 |  -0.05 |  0.55 |
| SGDRegressor                     |                4.42 |      -0.05 |  0.68 |        0.01 | 0.47 |  -0.05 |  0.56 |
| LarsCV                           |                4.39 |      -0.04 |  0.68 |        0.02 | 0.46 |  -0.04 |  0.54 |
| LassoCV                          |                4.39 |      -0.04 |  0.68 |        0.03 | 0.46 |  -0.04 |  0.54 |
| ElasticNetCV                     |                4.38 |      -0.04 |  0.68 |        0.06 | 0.46 |  -0.04 |  0.54 |
| LassoLarsIC                      |                4.38 |      -0.04 |  0.68 |        0.01 | 0.46 |  -0.04 |  0.54 |
| RidgeCV                          |                4.36 |      -0.03 |  0.68 |        0.01 | 0.46 |  -0.03 |  0.55 |
| AdaBoostRegressor                |                4.35 |      -0.03 |  0.68 |        0.01 | 0.46 |  -0.03 |  0.55 |
| LinearSVR                        |                4.30 |      -0.02 |  0.67 |        0.01 | 0.45 |  -0.02 |  0.52 |
| PoissonRegressor                 |                4.24 |       0.00 |  0.67 |        0.01 | 0.44 |   0.00 |  0.54 |
| BaggingRegressor                 |                4.24 |       0.00 |  0.67 |        0.02 | 0.44 |   0.00 |  0.53 |
| BayesianRidge                    |                4.18 |       0.02 |  0.66 |        0.03 | 0.44 |   0.02 |  0.53 |
| TweedieRegressor                 |                4.18 |       0.02 |  0.66 |        0.00 | 0.44 |   0.02 |  0.53 |
| GammaRegressor                   |                4.17 |       0.02 |  0.66 |        0.08 | 0.43 |   0.02 |  0.53 |
| HuberRegressor                   |                4.15 |       0.03 |  0.66 |        0.01 | 0.43 |   0.03 |  0.52 |
| RandomForestRegressor            |                4.14 |       0.03 |  0.66 |        0.09 | 0.43 |   0.03 |  0.53 |
| GradientBoostingRegressor        |                4.10 |       0.05 |  0.65 |        0.03 | 0.42 |   0.05 |  0.52 |
| KNeighborsRegressor              |                4.06 |       0.06 |  0.65 |        0.07 | 0.42 |   0.06 |  0.51 |
| HistGradientBoostingRegressor    |                4.05 |       0.06 |  0.65 |        0.04 | 0.42 |   0.06 |  0.51 |
| LGBMRegressor                    |                4.05 |       0.06 |  0.65 |        0.05 | 0.42 |   0.06 |  0.51 |
| MLPRegressor                     |                4.01 |       0.07 |  0.64 |        0.21 | 0.41 |   0.07 |  0.50 |
| SVR                              |                3.84 |       0.13 |  0.62 |        0.01 | 0.39 |   0.13 |  0.50 |
| NuSVR                            |                3.67 |       0.18 |  0.60 |        0.01 | 0.36 |   0.18 |  0.48 |



