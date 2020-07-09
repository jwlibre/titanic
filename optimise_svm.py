import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import pdb

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")


# PREPROCESSING - DATA IMPUTATION
# dataset for categorical imputation - 'embarked'
imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
cat_imp_train = pd.DataFrame(train_data['Embarked'], dtype='category')
cat_imp_test = pd.DataFrame(test_data['Embarked'], dtype='category')

cat_imp_train = imp_cat.fit_transform(cat_imp_train)
train_data['Embarked'] = cat_imp_train
cat_imp_test = imp_cat.fit_transform(cat_imp_test)
test_data['Embarked'] = cat_imp_test

# dataset for non-categorical imputation
imp_noncat = SimpleImputer(missing_values=np.nan, strategy='mean')
noncategoricals = ['Age','SibSp','Parch','Fare']
noncat_imp_train = train_data[noncategoricals]
noncat_imp_test = test_data[noncategoricals]
noncat_imp_train = pd.DataFrame(imp_noncat.fit_transform(noncat_imp_train), columns=noncategoricals)
noncat_imp_test = pd.DataFrame(imp_noncat.fit_transform(noncat_imp_test), columns=noncategoricals)
for key in noncategoricals:
    train_data[key] = noncat_imp_train[key]
    test_data[key] = noncat_imp_test[key]


# PREPROCESSING - FEATURE SCALING AND NORMALISATION
mean_age = train_data.Age.mean()
sigma_age = np.sqrt(train_data.Age.var())

mean_fare = train_data.Fare.mean()
sigma_fare = np.sqrt(train_data.Fare.var())

train_data["Age_scaled"] = (train_data.Age - mean_age)/sigma_age
train_data["Fare_scaled"] = (train_data.Fare - mean_fare)/sigma_fare

test_data["Age_scaled"] = (test_data.Age - mean_age)/sigma_age
test_data["Fare_scaled"] = (test_data.Fare - mean_fare)/sigma_fare


features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare_scaled', 'Age_scaled']
y = train_data["Survived"]
X = pd.get_dummies(train_data[features])


# MODEL SELECTION
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Commencing model optimisation --- ", current_time)

parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[c for c in range(1,11)], 'degree':[d for d in range(1,11)], 'gamma':[g for g in [0.01,0.1]]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X, y)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Finished model optimisation --- ", current_time)

# # dictionary of best parameters
# best_params = clf.best_params_

X_test = pd.get_dummies(test_data[features])
predictions = clf.predict(X_test) # automatically uses best parameters

print('writing output file')
# OUTPUT TO CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv("outputs/svm_optimal_gridsearch.csv", index=False)
print('done')
