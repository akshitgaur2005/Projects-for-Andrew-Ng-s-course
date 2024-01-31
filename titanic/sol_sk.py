import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

def main():
    X_train = pd.read_csv("train.csv")

    print(f"before:\n{X_train}")

    X_train.Sex = X_train.Sex.map({'male':0, 'female':1})
    X_train.Embarked = X_train.Embarked.map({'S': 0, 'C':1, 'Q':2})
    X_train.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)
    X_train.fillna(X_train.mean(), inplace = True)
    
    y_train = X_train['Survived']
    X_train.drop(['Survived'], axis=1, inplace=True)
    
    X, X_test, y, y_test = train_test_split(X_train, y_train, test_size = 0.25)
    
    pipe = Pipeline([('scaler', preprocessing.StandardScaler()), ('model', LogisticRegression())])
    
    
    pipe.fit(X, y)
    score = pipe.score(X_test, y_test)
    print(f"score: {score}")
    
    test = pd.read_csv("test.csv")
    test.Sex = test.Sex.map({'male':0, 'female':1})
    test.Embarked = test.Embarked.map({'S': 0, 'C':1, 'Q':2})
    test.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)
    test.fillna(test.mean(), inplace = True)
    
    preds = pipe.predict(test)
    print(f"test:\n{preds}")
    preds_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':preds})
    preds_df.to_csv("submission.csv", index=False)

    #print(f"unique: {X_train.isnull().sum()}")

    #X_train.head(10)


main()