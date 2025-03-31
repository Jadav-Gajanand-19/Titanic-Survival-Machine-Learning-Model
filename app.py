import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_data=pd.read_csv("/content/train.csv")
test_data=pd.read_csv("/content/test.csv")
# women survival
women_survival=train_data.loc[train_data.Sex=="female"]["Survived"]
women_survival_rate=sum(women_survival)/len(women_survival)
print(f"Percentage of women survival : {round(women_survival_rate*100,2)}")
# men survival
men_survival=train_data.loc[train_data.Sex=="male"]["Survived"]
men_survival_rate=sum(men_survival)/len(men_survival)
print(f"Percentage of women survival : {round(men_survival_rate*100,2)}")

# prediction model

Y=train_data["Survived"]
features=["Pclass", "Sex", "Age", "SibSp", "Parch"]
X=pd.get_dummies(train_data[features])
X.fillna(X.median(), inplace=True)
x_test=pd.get_dummies(test_data[features])
x_test.fillna(x_test.median(),inplace=True)
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,Y)
predict=model.predict(x_test)
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survival':predict})
output.to_csv('/content/sample_data/survival_prediction.csv')
print("submission successful")
