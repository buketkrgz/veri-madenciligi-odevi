# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, ttest_ind

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                  columns=['a', 'b', 'c'])
print(df)
df = pd.read_csv("../input/spotify-top-50-songs-in-2021/spotify_top50_2021.csv")
sns.pairplot(df)
sns.set_style("whitegrid")
intensity = sum(df.energy) / len(df.energy)
df['energy_level'] = ['energized' if i > intensity else 'without energy' for i in df.energy]

sns.relplot(x='loudness', y='energy', data=df, kind='line', style='energy_level', hue='energy_level', markers=True,
            dashes=False, ci='sd')
plt.xlabel('Loudness (dB)', fontsize=20)
plt.ylabel('Energy', fontsize=20)
plt.title('Connection between the Loudness (dB) and Energy', fontsize=25)
sns.catplot(x='loudness', y='energy', data=df, kind='point', hue='energy_level')
plt.xlabel('Loudness (dB)', fontsize=20)
plt.ylabel('Energy', fontsize=20)
plt.title('Connection between the Loudness (dB) and Energy', fontsize=25)
independent_var = df[['energy', 'danceability', 'loudness', 'liveness', 'valence', 'acousticness', 'speechiness']]
dependent_var = df['popularity']
result = linear_model.LinearRegression()
result.fit(independent_var, dependent_var)

intercept = result.intercept_
reg_coef = result.coef_
print(
    'Label:  energy(x1), danceability(x2), loudness(x3), liveness(x4), valence(x5), acousticness(x6), speechiness(x7)')
print('\nIntercept value (a): %0.3f' % intercept)
print(
    '\nRegression Equation: ŷ = %0.3f + %0.3f*X1 + %0.3f*X2 + %0.3f*X3, + %0.3f*X4, + %0.3f*X5, +  %0.3f*X6, + %0.3f*X7' % (
    intercept, reg_coef[0], reg_coef[1], reg_coef[2], reg_coef[3], reg_coef[4], reg_coef[5], reg_coef[6]))
x_var = sm.add_constant(independent_var)
model = sm.OLS(dependent_var, x_var).fit()
predictions = model.predict(x_var)
print(model.summary())
X = df[['energy', 'danceability', 'loudness', 'liveness', 'valence', 'acousticness', 'speechiness']]
y = df['popularity']
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
predict = knn.predict(X)
pd.Series(predict).value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
knn.score(X_test, y_test)
