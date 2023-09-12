import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

class ClassificationModels:
    def __init__(self, data, target):
        if data[target].isna().sum():
            data[target].dropna(inplace=True)
        self.data = data
        self.features = data.drop(columns=target)
        self.target = LabelEncoder().fit_transform(data[target])
        self.models = [
            LogisticRegression(),
            RidgeClassifier(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            RandomForestClassifier()
        ]
        self.best_models = []
        self.scores = pd.DataFrame(columns=['Model','Precision', 'Recall', 'F1-Score'])
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
            {'C':[0.01,0.1,1,10]},  # For LogisticRegression
            {'alpha': [0.01, 0.1, 1.0, 10.0]},  # For RidgeClassifier
            {'n_neighbors': range(3, 30, 2)},  # For KNeighborsClassifier
            {'max_depth': range(1, 55, 5)},  # For DecisionTreeClassifier
            {'n_estimators': range(1, 160, 10), 'max_depth': range(1, 55, 5)}  # For RandomForestClassifier
        ]

        for model, param_grid in zip(self.models, param_grids):
            print(f'Trainig {model.__class__.__name__}.')

            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            self.best_models.append(best_model)
            self.evaluate_model(best_model)
        
    def evaluate_model(self, model):
        y_pred = model.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        model_name = model.__class__.__name__
        self.scores = pd.concat([self.scores, pd.DataFrame([{'Model': model_name ,'Precision': precision, 'Recall': recall, 'F1-Score': f1}])], ignore_index=True, axis=0)
        
    def best_model_index(self):
        if len(self.scores) == 0 :
            print('You Must First train models for your data\nTrain models and then extract the best one.')
            return
        best_model_name = self.scores.loc[self.scores['F1-Score'].idxmax()]['Model']
        best_model_index = next((index for index, model in enumerate(self.models) if model.__class__.__name__ == best_model_name), None)
        return best_model_index

    def get_best_model(self):
        best_model_index = self.best_model_index()
        if best_model_index is None:
            return
        self.best_model = self.best_models[best_model_index]
        return self.best_model
    
    def predict(self):
        if self.best_model:
            pred = self.best_model.predict(self.X_test)
        return pred
    
    def save_best_model(self,file_name='classification_model.pkl'):
        with open(file_name,'wb') as model_file:
            pickle.dump(self.best_model,model_file)

