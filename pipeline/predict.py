import pickle
import pandas as pd
from feature_engineering import fr_engineering

# custom files
import columns

# read train data
ds = pd.read_csv("data/new_data.csv")



# feature engineering
ds = fr_engineering(ds)


# Define target and features columns
X = ds[columns.X_columns]

# load the model and predict
rf = pickle.load(open('models/finalized_model.sav', 'rb'))

y_pred = rf.predict(X)

ds['Churn_pred'] = rf.predict(X)
ds.to_csv('data/prediction_results.csv', index=False)