# AutoModeling: Automated Machine Learning Toolkit

Simplify your machine learning workflows with AutoModeling - an automated toolkit for exploratory data analysis (EDA), regression, and classification tasks. Effortlessly preprocess data, build accurate models, evaluate performance, and make predictions, all while minimizing manual effort. Streamline your data science projects and achieve optimal results with minimal coding.

## Features

- **EDA**: Explore and visualize data's basic information, summary statistics, and relationships among variables.
- **RegressionModels**: Build and evaluate various regression models, including Linear Regression, Ridge, Lasso, KNeighbors, Decision Trees, and Random Forests.
- **ClassificationModels**: Build and evaluate classification models, including Logistic Regression, KNeighbors, Decision Trees, and Random Forests.

## Installation

You can clone AutoModeling repo by:

```bash
git clone https://github.com/ahmedmgelwan/AutoModeling.git
```

## Usage

```python
# Importing classes
from AutoModeling.auto_eda import EDA
from AutoModeling.regression import RegressionModels
from AutoModeling.classification import ClassificationModels



# EDA
eda_instance = EDA("your_dataset.csv")
eda_instance.remove_outliers("numeric_column")

# Exatract Data Frame
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
```

## Example Notebooks

Check out the example notebooks for regression and classification tasks using AutoModeling:

1. [Regression Modeling Example Notebook](https://www.kaggle.com/code/ahmedmgelwan/automate-insurance-prediction-by-automodeling)
2. [Classification Modeling Example Notebook](https://www.kaggle.com/code/ahmedmgelwan/automate-loan-approval-prediction-by-automodeling)

## Contributing

License
This project is licensed under the MIT License - see the LICENSE file for details.License
This project is licensed under the MIT License - see the LICENSE file for details.Contributions are welcome!

------

Coded with ❤️ by Ahmed Gelwan