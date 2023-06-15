# Importing Libraries
import pandas as pd
import numpy as np
import pickle

# Dealing with NaNs and transformin the skewed features

def clean_transform_data(df):
    """Returns cleaned DataFrame.
    
    Args: 
        df (pd.DataFrame) : uncleaned DataFrame
        
    Returns:
        df  (pd.DataFrame) : cleaned DataFrame
    
    """
    
    # medians of various fetures
    median={'fixed acidity': 7.0, 'volatile acidity': 0.29, 'citric acid': 0.31, 'residual sugar': 3.0,
            'chlorides': 0.047, 'free sulfur dioxide': 29.0, 'total sulfur dioxide': 118.0,
            'density': 0.99489,'pH': 3.21,'sulphates': 0.51,'alcohol': 10.3,'quality': 6.0}
    
    # replace NaN be corrsponding Median
    for i in range(1,df.shape[1]-1):
        col_name=df.columns[i]
        df[col_name] = df[col_name].fillna(median[col_name])
    
   # Applying log function to the heavily skewed features
    skewed_features=['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
    for feature in skewed_features:
        df[feature] = np.log(df[feature]) 

    return df


# request the file name (with its path if in other directory)
wine_file = input('enter the name of the wine file, with the path if required')

# reading the data from file into dataframe
wine = pd.read_csv(wine_file, sep=";")

# cleaing and transforming the data
wine=clean_transform_data(wine)

# reading the best model
opt_model = pickle.load(open("opt_model.p",'rb')) 

# applying the prediction model
predictions = opt_model.predict(wine)

# Save predictions as a file with the same name as the data file after adding "y_pred"
pred_file=wine_file[:-4]+'_pred.csv'
pd.Series(predictions).to_csv(pred_file, index=False)