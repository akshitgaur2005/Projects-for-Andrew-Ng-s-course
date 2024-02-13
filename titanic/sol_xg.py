import pandas as pd
from xgboost import XGBClassifier as xc

def main():
    X_train = pd.read_csv("train.csv")

    print(f"before:\n{X_train}")

    X_train.Sex = X_train.Sex.map({'male':0, 'female':1})
    X_train.Embarked = X_train.Embarked.map({'S': 0, 'C':1, 'Q':2})
    X_train.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)
    X_train.fillna(X_train.mean(), inplace = True)
    
    y_train = X_train['Survived']
    X_train.drop(['Survived'], axis=1, inplace=True)

    model = xc()

    model.fit(X_train, y_train)

    test = pd.read_csv("test.csv")
    test.Sex = test.Sex.map({'male':0, 'female':1})
    test.Embarked = test.Embarked.map({'S': 0, 'C':1, 'Q':2})
    test.drop(columns = ['Name', 'Ticket', 'Cabin'], inplace = True)
    test.fillna(test.mean(), inplace = True)
    
    preds = model.predict(test)
    print(f"test:\n{preds}")
    preds_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':preds})
    preds_df.to_csv("submission_xc.csv", index=False)

main()