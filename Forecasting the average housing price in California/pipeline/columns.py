cat_columns =  ["ocean_proximity"]

y_column =  ['median_house_value']

X_columns = ['housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
       'households', 'median_income',
       'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']

cols_to_scale = ['housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value' ]

outliers_columns = ['total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']


mean_impute_columns = ['housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income']


drop_columns = ["longitude","latitude"]