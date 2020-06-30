import pandas as pd
import numpy as np
from sklearn import svm, metrics
import pdb

class Model:
    # attributes of Model object:
    # type [linear, poly, rbf, sigmoid]
    # description
    # degree
    # gamma
    # features used
    def __init__(self, type, description, degree, gamma, features):
        self.type = type
        self.description = description
        self.degree = degree
        self.gamma = gamma
        self.features = features
        self.score = 0

    # Methods of Model object
    # fit_and_evaluate_svm: returns f1 score

    def fit_and_evaluate_svm(self, X, y, X_cv, y_cv):
        if self.type == 'linear':
            clf = svm.SVC(kernel=self.type)
        elif self.type == 'poly':
            clf = svm.SVC(kernel=self.type, degree=self.degree)
        else:
            clf = svm.SVC(kernel=self.type, gamma=self.gamma)
        clf.fit(X, y)
        predictions_cv = clf.predict(X_cv)
        f1 = metrics.f1_score(y_cv, predictions_cv)
        print("{} has F1 score: {}".format(self.description,f1))
        self.score = f1
        print("score assigned to model {} = {}".format(self.description, self.score))


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Separate CV data from training data
# ensure training data is randomised first!
train_data = train_data.sample(frac=1).reset_index(drop=True)
cv_data = train_data.iloc[750:]
train_data = train_data.iloc[:750]

y = train_data["Survived"]
y_cv = cv_data["Survived"]

all_features = ['Pclass', 'Sex', 'Age',
                'SibSp', 'Parch', 'Fare', 'Embarked']

# normalise the Fare and the Age features
# be aware that these variables both contain NaN - how should I approach
# missing values?
mean_age = train_data.Age.mean()
sigma_age = np.sqrt(train_data.Age.var())

mean_fare = train_data.Fare.mean()
sigma_fare = np.sqrt(train_data.Fare.var())

train_data["Age_scaled"] = (train_data.Age - mean_age)/sigma_age
train_data["Fare_scaled"] = (train_data.Fare - mean_fare)/sigma_fare

cv_data["Age_scaled"] = (cv_data.Age - mean_age)/sigma_age
cv_data["Fare_scaled"] = (cv_data.Fare - mean_fare)/sigma_fare

test_data["Age_scaled"] = (test_data.Age - mean_age)/sigma_age
test_data["Fare_scaled"] = (test_data.Fare - mean_fare)/sigma_fare


# set up lists for storing evaluation metrics for each model,
# so we can pick the best one
# TO-DO - replace with a summary table detailing all parameters of model,
# types of model,
# features used
model_desc_list = []
f1_list = []




# SECTION 1 - Unscaled Features
features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = pd.get_dummies(train_data[features])
X_cv = pd.get_dummies(cv_data[features])

model_desc = "Linear kernel, no scalable features included"
model = Model(type='linear', description=model_desc, degree=0, gamma=0, features=features)
pdb.set_trace()

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    if k == 'linear':
        model = Model(type=k, description=model_desc, degree=0, gamma=0, features=features)
        model.fit_and_evaluate_svm(X, y, X_cv, y_cv)
        model_desc_list.append(model_desc)
        f1_list.append(model.score)
    elif k == 'poly':
        for i in range(10):
            model = Model(type=k, description=model_desc, degree=i, gamma=0, features=features)
            model.fit_and_evaluate_svm(X, y, X_cv, y_cv)
            model_desc_list.append(model_desc)
            f1_list.append(model.score)
    else:
        for g in ['auto', 'scale']:


# Model 1: linear kernel
clf = svm.SVC(kernel = 'linear')
model_desc = "Linear kernel, no scaled features"
fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)

# Model 2: let's try a polynomial, and try different degrees
for i in range(10):
    clf = svm.SVC(kernel='poly', degree=i)
    model_desc = "Polynomial with degree {}, no scaled features".format(i)
    fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)

# Model 3 and 4: let's try rbf and sigmoid, and try different gamma values
for k in ['rbf', 'sigmoid']:
    for g in ['auto', 'scale']:
        clf = svm.SVC(kernel=k, gamma=g)
        model_desc = "{} with gamma {}, no scaled features".format(k, g)
        fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)


# SECTION 2 - try normalising the Fare and Age features and include them
# in the analysis



pdb.set_trace()





# CODE FOR LOOPING OVER KERNEL TYPES - NOT USEFUL IF WE'RE GOING OOP
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# for k in kernels:
#     if k == 'poly':
#         for i in range(10):
#             clf = svm.SVC(kernel=k, degree=i)
#             model_desc = "Polynomial with degree {}, {}".format(i, features_desc)
#             fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)
#     else if k == 'rbf' or k == 'sigmoid':
#         for g in ['auto', 'scale']:
#             clf = svm.SVC(kernel=k, gamma=g)
#             model_desc = "{} with gamma {}, {}".format(k, g, features_desc)
#             fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)
#     else:
#         clf = svm.SVC(kernel = 'linear')
#         model_desc = "Linear kernel, {}".format(features_desc)
#         fit_and_evaluate_svm(clf, X, y, X_cv, y_cv, model_desc_list, f1_list, model_desc)


# OUTPUT TO CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
