# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading the two files
red_wine = pd.read_csv("winequality-red.csv", sep=";")
white_wine = pd.read_csv("winequality-white.csv", sep=";")

# adding the type of wine
red_wine.insert(0, "type", 1)
white_wine.insert(0, "type", 0)

# concatenating the red and white wine dataframes
wine = red_wine.append(white_wine, ignore_index=True)

# display the first five rows of the dataframe
print(wine.head())

# checkig the datatypes
print(wine.info())

# display two digital places 
pd.options.display.float_format = '{:.2f}'.format                   

# display statistics of the data
print(wine.describe())

# features distribution
wine.hist(bins=50, figsize=(15,12))
plt.show()

# Applying log function to the heavily skewed features
skewed_features=['residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
for feature in skewed_features:
    wine[feature] = np.log(wine[feature])

#redrawing the features distribution
wine.hist(bins=50, figsize=(15,12))
plt.show()

# Feature distribution as function of wine-type
fig, ax = plt.subplots(nrows=12,figsize=[20,48])
wine_features_names=['type','fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                     'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
for i in range(1,wine.shape[1]):
    pd.crosstab(index= wine[wine_features_names[i]], columns=wine[wine_features_names[0]]).plot(ax=ax[i-1])
plt.show()

# investigate correlation between features
fig, ax = plt.subplots(figsize=[20,20])
corr= wine.corr()
triangle = np.triu(corr)
sns.heatmap(corr.abs(), annot=True, fmt='.2f', mask=triangle,cmap= 'coolwarm' , ax=ax)
plt.show()

# Scatterplots of each pair of features
sns.set(font_scale=1.2)
sns.pairplot(wine,corner=True);
plt.show()

# Data distribution per classes
print(wine["quality"].value_counts(normalize=True) * 100)
# Using a pie chart to show the data distribution per class
explode = (0, 0, 0, 0.2, 0.5)
plt.pie(wine["quality"].value_counts()[:5],autopct="%1.0f%%",textprops={'fontsize': 12},explode=explode,labels=['6','5','7','4','8'])
plt.show()

# detecting outlier
fig, ax = plt.subplots(nrows=11,figsize=[20,48])
i=0
for col_name in red_wine.columns[1:-1]:
    ax[i].boxplot(wine[col_name],vert=False, positions=[200],widths=50)
    ax[i].set(title=col_name)
    i+=1
plt.show()