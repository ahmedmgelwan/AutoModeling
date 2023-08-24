import pandas as pd
from sklearn.preprocessing import  StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings

warnings.filterwarnings('ignore')

class RegressionModels:
    def __init__(self, data, target):
        self.data = data
        self.features = data.drop(columns=target)
        self.target = data[target]
        self.models = [
            LinearRegression(),
            Ridge(),
            Lasso(),
            KNeighborsRegressor(),
            DecisionTreeRegressor(),
            RandomForestRegressor()
        ]
        self.best_models = []
        self.scores = pd.DataFrame(columns=['Model', 'Training Score', 'Testing Score'])
        self.preprocessing()
        self.split()
        
    def preprocessing(self):
        self.features = pd.get_dummies(
            self.features,
            columns=self.features.select_dtypes('object').columns, drop_first=True
        )
        self.features_scaled = StandardScaler().fit_transform(self.features)
        
    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features_scaled, self.target, test_size=0.2, random_state=42
        )

    def model_building(self):
        # Define parameter grids for hyperparameter tuning
        param_grids = [
            {},  # For LinearRegression
            {'alpha': [0.01, 1.0, 10.0]},  # For Ridge
            {'alpha': [0.1, 1.0, 10.0]},  # For Lasso
            {'n_neighbors':range(3,30,2)}, # For KNN
            {'max_depth': range(1,55,5)},  # For DecisionTreeRegressor
            {'n_estimators': range(1,160,10),'max_depth': range(1,55,5)}  # For RandomForestRegressor
        ]

        for model, param_grid in zip(self.models, param_grids):
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            self.best_models.append(best_model)
            model_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5)
            self.model_evaluation(best_model, model_scores)
    
    def model_evaluation(self, model, scores):
        model_name = model.__class__.__name__
        training_score = scores.mean()
        testing_score = model.score(self.X_test, self.y_test)
        self.scores = self.scores.append({'Model': model_name, 'Training Score': training_score, 'Testing Score': testing_score}, ignore_index=True)

    def get_best_model(self):
        best_model_name = self.scores.loc[self.scores['Testing Score'].idxmax()]['Model']
        best_model_index = next((index for index, model in enumerate(self.models) if model.__class__.__name__ == best_model_name), None)
        self.best_model = self.best_models[best_model_index]
        return self.best_model
    
    def predict(self):
        if self.best_model:
            pred = self.best_model.predict(self.X_test)
        return pred
    
    def save_best_model(self,file_name='regression_model.pkl'):
        with open(file_name,'wb') as model_file:
            pickle.dump(self.best_model,model_file)
