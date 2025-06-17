import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,ConfusionMatrixDisplay

df=pd.read_csv('combined_data.csv')
df.head()

df.rename(columns={'label':'spam1_ham0'},inplace=True)
print(df.head())
vectorizer=CountVectorizer()
X=df['text']
Y=df['spam1_ham0']

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,train_size=0.8)
X_train_vec=vectorizer.fit_transform(X_train)
X_test_vec=vectorizer.transform(X_test)
model=MultinomialNB()
model.fit(X_train_vec,y_train)
Y_pred=model.predict(X_test_vec)

acc=accuracy_score(Y_pred,y_test)
f1=f1_score(Y_pred,y_test)
recall=recall_score(Y_pred,y_test)
print('Accuracy score=',acc)
print('F1 score=',f1)
print('Recall=',recall)
con=confusion_matrix(Y_pred,y_test)
disp=ConfusionMatrixDisplay(display_labels=['spam','ham'],confusion_matrix=con)
disp.plot()
