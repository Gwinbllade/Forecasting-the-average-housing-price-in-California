import pickle
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# custom files
import model_best_hyperparameters
import columns
from feature_engineering import fr_engineering




def over_sampling(ds):
    X = ds.drop('ocean_proximity', axis=1)
    y = ds['ocean_proximity']

    ros = RandomOverSampler(
        sampling_strategy='auto', 
        random_state=0,  
    )  

    X_res, y_res = ros.fit_resample(X, y)


    ds = pd.concat([X_res, y_res], axis=1)
    return ds

def model_metrics(ml_model, X_train, y_train, X_test, y_test):
    return ([str(ml_model).split("(")[0], 
                       ml_model.score(X_train, y_train), 
                       ml_model.score(X_test, y_test),
                       np.sqrt(mean_squared_error(y_train, ml_model.predict(X_train))),
                       np.sqrt(mean_squared_error(y_test, ml_model.predict(X_test))),
                       mean_absolute_error(y_train, ml_model.predict(X_train)),
                       mean_absolute_error(y_test, ml_model.predict(X_test)),
                       mean_absolute_percentage_error(y_train, ml_model.predict(X_train)),
                       mean_absolute_percentage_error(y_test, ml_model.predict(X_test)),
                      ])

    



ds = pd.read_csv("data/train_data.csv")

# Let's create a dict and impute mean values
mean_impute_values = dict()
for column in columns.mean_impute_columns:
    mean_impute_values[column] = ds[column].mean()


# save parameters 
param_dict = {'mean_impute_values':mean_impute_values}

with open('pipeline/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


ds = over_sampling(ds)


ds = fr_engineering(ds)

# Define target and features columns
y = ds[columns.y_column]
X = ds.drop(columns.y_column, axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)

# Building and train Random Forest Model
rf = RandomForestRegressor(**model_best_hyperparameters.params)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)





metrics = pd.DataFrame([ model_metrics(rf, X_train, y_train, X_test, y_test)], columns = ["Algorithm", "Train_Score", "Test_Score", "Train_RMSE",
                                         "Test_RMSE", "Train_MAE", "Test_MAE", "Train_MAPE", "Test_MAPE"]).sort_values("Test_MAPE").set_index("Algorithm")
metrics.to_csv('models/model_metrics.csv')


rankings = rf.feature_importances_.tolist()
importance = pd.DataFrame(sorted(zip(X_train.columns,rankings),reverse=True),columns=["variable","importance"]).sort_values("importance",ascending = False)
importance.to_csv("models/feature_importance.csv", index=False)

filename = 'models/finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))
