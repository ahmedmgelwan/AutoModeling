from auto_eda import EDA
from regression import RegressionModels
from classification import ClassificationModels

def main():
    data_path = input('Data Path: ')
    eda = EDA(data_path)
    df = eda.df

    rm_outliers = input('Do You Want to remove any outliers? [Y/N] - ').lower()
    while rm_outliers in ['y', 'yes']:
        col_name = input('Column [Only one column]: ')
        eda.remove_outliers(col_name)
        rm_outliers = input('Do You Want to remove any other outliers? [Y/N] - ').lower()
    target = input('Target: ')

    if df[target].dtype == 'object' or df[target].nunique() ==2:
        print('Your task is classification')
        models = ClassificationModels(df,target)
        models.model_building()
        print('All Models Scores.')
        print(models.scores)
    else:
        print('Your task is regression')
        models = RegressionModels(df,target)
        models.model_building()
        print('All Models Scores.')
        print(models.scores)
    best_model = models.get_best_model()
    print(f'Best Model is {best_model.__class__.__name__}')
    saving = input('Do you want to save this model? [Y/N] - ').lower()
    if saving in ['y','yes']:
        model_name = input('Name: ')
        if '.' not in model_name:
            model_name += '.pkl'
        models.save_best_model(model_name)
        print('Model Saved.')


if __name__ == '__main__':
    main()
