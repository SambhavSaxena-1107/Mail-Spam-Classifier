import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data .csv')

print(df)

data = df.where((pd.notnull(df)),'')

data.head(10)

data.info()

data.shape

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

X=data['Message']
y=data['Category']

print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

print(y.shape)
print(y_train.shape)
print(y_test.shape)

feature_extraction=TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.fit_transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


print(X_train_features)

Model = LogisticRegression()

Model.fit(X_train_features,y_train)

prediction_on_training_data = Model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(y_train,prediction_on_training_data)

print("accuracy on Training data:" , accuracy_on_training_data)

Model = LogisticRegression()

Model.fit(X_test_features,y_test)

prediction_test_data = Model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_test_data)

print("accuracy on Test data:" , accuracy_on_test_data)
