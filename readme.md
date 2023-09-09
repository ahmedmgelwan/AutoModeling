# AutoModeling: Automated Machine Learning Toolkit

Simplify your machine learning workflows with AutoModeling - an automated toolkit for exploratory data analysis (EDA), regression, and classification tasks. Effortlessly preprocess data, build accurate models, evaluate performance, and make predictions, all while minimizing manual effort. Streamline your data science projects and achieve optimal results with minimal coding.

## Features

- **EDA**: Explore and visualize data's basic information, summary statistics, and relationships among variables.
- **RegressionModels**: Build and evaluate various regression models, including Linear Regression, Ridge, Lasso, KNeighbors, Decision Trees, and Random Forests.
- **ClassificationModels**: Build and evaluate classification models, including Logistic Regression, KNeighbors, Decision Trees, and Random Forests.

## Installation

You can clone the AutoModeling repository by running the following command:

```bash
git clone https://github.com/ahmedmgelwan/AutoModeling.git
```

## Usage

### Using the Package

```python
# Importing classes
from AutoModeling.auto_eda import EDA
from AutoModeling.regression import RegressionModels
from AutoModeling.classification import ClassificationModels

# EDA
eda_instance = EDA("your_dataset.csv")
eda_instance.remove_outliers("numeric_column")

# Extract Data Frame
data = eda_instance.df

# Regression Modeling
regression_model = RegressionModels(data, target_column)
regression_model.model_building()
best_regression_model = regression_model.get_best_model()
regression_predictions = regression_model.predict()

# Classification Modeling
classification_model = ClassificationModels(data, target_column)
classification_model.model_building()
best_classification_model = classification_model.get_best_model()
classification_predictions = classification_model.predict()

# Saving best model
classification_model.save_best_model('file_name') # if your task is classification
```

### Using the `main.py` Script

AutoModeling now includes a convenient `main.py` script that simplifies the model-building process:

1. Run `main.py`.
2. Provide the data path, optionally remove outliers, and specify the target variable.
3. The script will automatically determine whether it's a regression or classification task.
4. It will build and evaluate models, display scores, and offer the option to save the best model.

## Example Notebooks

Check out the example notebooks for regression and classification tasks using AutoModeling:

1. [Regression Modeling Example Notebook](https://www.kaggle.com/code/ahmedmgelwan/automate-insurance-prediction-by-automodeling)
2. [Classification Modeling Example Notebook](https://www.kaggle.com/code/ahmedmgelwan/automate-loan-approval-prediction-by-automodeling)

## Contributing

Contributions are welcome!

------

Coded with ❤️ by Ahmed Gelwan