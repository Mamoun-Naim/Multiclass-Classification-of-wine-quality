# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# read the two files
red_wine = pd.read_csv("winequality-red.csv", sep=";")
white_wine = pd.read_csv("winequality-white.csv", sep=";")

# adding the type of wine
red_wine.insert(0, "type", 1)
white_wine.insert(0, "type", 0)

# merging the two files
wine = red_wine.append(white_wine, ignore_index=True)

# classes balancing using SMOTE
def resample_data_smote(features, target):
    """Using SMOTE for the imbalanced data set.
        Returns balanced Data Set.
    Args: 
        features (pd.DataFrame) : unbalanced DataFrame
        target (pd.DataFrame) : unbalanced DataFrame
    Returns:
        features (pd.DataFrame) : balanced DataFrame
        target (pd.DataFrame) : balanced DataFrame
    """
    #initiate Smote
    smotesampler = SMOTE(random_state=42,k_neighbors=4)
    #fit and resample
    features, target = smotesampler.fit_resample(features, target)
    
    return features, target

# This Dictionary will be used to replace NaNs in the dataframes
median_nan={}
for col_name in wine.columns[1:]:
    median_nan[col_name]=wine[col_name].median()
    
# Applying log function to the heavily skewed features
skewed_features=['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
for feature in skewed_features:
    wine[feature] = np.log(wine[feature])

# define features and target
wine_features=wine.iloc[:,:-1]
wine_target=wine.iloc[:,-1]

# Train-test data Splitting
w_train_f, w_test_f, w_train_t, w_test_t = train_test_split(wine_features, wine_target, random_state=42, test_size=0.1)

#apply SMOTE
f, t=resample_data_smote(w_train_f, w_train_t)

# features scaling
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(f)
features_test_scaled = scaler.transform(w_test_f)
features_train_scaled = pd.DataFrame(features_train_scaled,columns=w_train_f.columns)
features_test_scaled = pd.DataFrame(features_test_scaled,columns=w_test_f.columns)

# LinearRegression
model_reg=LinearRegression()
model_reg.fit(features_train_scaled,t)

# Calculate and save model performances 
score_sum=[]
target_pred=model_reg.predict(features_train_scaled)
target_pred=np.rint(target_pred)
train_accuracy=accuracy_score(t,target_pred)

target_pred=model_reg.predict(features_test_scaled)
target_pred=np.rint(target_pred)
test_accuracy=accuracy_score(w_test_t,target_pred)

score_sum.append({'name':'LinearRegression     ','train':train_accuracy,'test':test_accuracy})


# RandomForestRegressor
model_rfr=RandomForestRegressor(random_state=42)
model_rfr.fit(features_train_scaled,t)

# Calculate and save model performances 
target_pred=model_rfr.predict(features_train_scaled)
target_pred=np.rint(target_pred)
train_accuracy=accuracy_score(t,target_pred)

target_pred=model_rfr.predict(features_test_scaled)
target_pred=np.rint(target_pred)
test_accuracy=accuracy_score(w_test_t,target_pred)

score_sum.append({'name':'RandomForestRegressor     ','train':train_accuracy,'test':test_accuracy})


# features scaling
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(w_train_f)
features_test_scaled = scaler.transform(w_test_f)
features_train_scaled = pd.DataFrame(features_train_scaled,columns=w_train_f.columns)
features_test_scaled = pd.DataFrame(features_test_scaled,columns=w_test_f.columns)


# LogisticRegression
model_log=LogisticRegression(class_weight='balanced',multi_class='multinomial',random_state=42)
model_log.fit(features_train_scaled,w_train_t)

# Calculate and save model performances 
target_pred=model_log.predict(features_train_scaled)
train_accuracy=accuracy_score(w_train_t,target_pred)

target_pred=model_log.predict(features_test_scaled)
test_accuracy=accuracy_score(w_test_t,target_pred)

score_sum.append({'name':'LogisticRegression     ','train':train_accuracy,'test':test_accuracy})


# RandomForestClassifier
model_rf=RandomForestClassifier(class_weight='balanced',random_state=42)
model_rf.fit(w_train_f,w_train_t)

# Calculate and save model performances 
target_pred=model_rf.predict(w_train_f)
train_accuracy=accuracy_score(w_train_t,target_pred)

target_pred=model_rf.predict(w_test_f)
test_accuracy=accuracy_score(w_test_t,target_pred)

score_sum.append({'name':'RandomForestClassifier     ','train':train_accuracy,'test':test_accuracy})


# ANN
warnings.filterwarnings('ignore')
model_ann = Sequential() 
hidden_layer = Dense(units=50, activation= 'relu', input_dim=w_train_f.shape[1])
output_layer = Dense(units=1,activation= 'linear')
model_ann.add(hidden_layer)
model_ann.add(output_layer)
model_ann.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
model_ann.fit(w_train_f, w_train_t, epochs=50, batch_size=64,verbose=False)

# Calculate and save model performances 
target_pred=model_ann.predict(w_train_f)
target_pred=np.rint(target_pred)
train_accuracy=accuracy_score(w_train_t,target_pred)

target_pred=model_ann.predict(w_test_f)
target_pred=np.rint(target_pred)
test_accuracy=accuracy_score(w_test_t,target_pred)

score_sum.append({'name':'         ANN        ','train':train_accuracy,'test':test_accuracy})
warnings.filterwarnings('ignore')


print('         Modell                          train accuracy             test accuracy')
print('---------------------------------------------------------------------------------')
for i in range(len(score_sum)):
    print('   {:40s}    {:.2f}                     {:.2f}'.format(score_sum[i]['name'],score_sum[i]['train'],score_sum[i]['test']))
    print('---------------------------------------------------------------------------------')


# print ordered feature_importance 
feature_imp=pd.DataFrame(model_rf.feature_importances_, index=w_train_f.columns).sort_values(by=0, ascending=False)
print(feature_imp)
# showing the feature_importance chart
fig, ax = plt.subplots(figsize=[10,10])
sns.barplot(x=feature_imp[0], y=feature_imp.index)
sns.set(font_scale=1.5)
plt.xlabel('Feature Importance Score')
plt.show()


# Omission of the feature "type" because it does not play a role in the classification
w_train_f = w_train_f.drop('type', axis=1)
w_test_f = w_test_f.drop('type', axis=1)

# Defining the appropriate pipeline
model_pipeline = Pipeline([('pca', PCA()),
                           ('rfc', RandomForestClassifier(class_weight='balanced',random_state=42))
                          ])

# n_components: number of kept components
# max_depth : Maximum number of levels in tree
# min_samples_leaf : Minimum number of samples required at each leaf node
search_space = {'pca__n_components' : [8,9,10],
                     'rfc__max_depth': np.geomspace(start=5, stop=50, num=10, dtype='int'),
                     'rfc__min_samples_leaf': np.geomspace(start=5, stop=50, num=10, dtype='int')
                    }
opt_model=GridSearchCV(estimator=model_pipeline,
                       param_grid=search_space,
                       scoring='accuracy',
                       cv=5,
                       n_jobs=-1
                      )

opt_model.fit(w_train_f, w_train_t)

# Calculate and save model performances 
print("accuracy ot the training set",opt_model.best_score_)
target_pred = opt_model.predict(w_test_f)
test_accuracy = accuracy_score(w_test_t, target_pred)
print("accuracy ot the testing set",test_accuracy)


# Accuracy with 0 tolerance
mask0=abs(w_test_t- target_pred)==0
print('with 0 quality tolerance is the accuracy {:.0f}%'.format(100*mask0.sum()/w_test_t.shape[0]))

# Accuracy with 1 tolerance
mask1=abs(w_test_t- target_pred)<=1
print('with 1 quality tolerance is the accuracy {:.0f}%'.format(100*mask1.sum()/w_test_t.shape[0]))

# Accuracy with 2 tolerance
mask2=abs(w_test_t- target_pred)<=2
print('with 2 quality tolerance is the accuracy {:.0f}%'.format(100*mask2.sum()/w_test_t.shape[0]))


# save the best model
pickle.dump(opt_model,open('opt_model.p','wb'))
