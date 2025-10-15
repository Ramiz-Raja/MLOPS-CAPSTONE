import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # basic feature engineering
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].fillna('S')
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    # Select features present in sample; fill missing with 0 if needed
    for col in ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']:
        if col not in df.columns:
            df[col] = 0
    X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
