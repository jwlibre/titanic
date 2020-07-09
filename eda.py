import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.impute import SimpleImputer
import pdb

class Model:
    # attributes of Model object:
    # type [linear, poly, rbf, sigmoid]
    # degree
    # gamma
    # features used
    def __init__(self, type, degree, gamma, C, features):
        self.type = type
        self.degree = degree
        self.gamma = gamma
        self.C = C
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
            clf = svm.SVC(kernel=self.type, gamma=self.gamma, C=self.C)
        clf.fit(X, y)
        predictions_cv = clf.predict(X_cv)
        f1 = metrics.f1_score(y_cv, predictions_cv)
        self.score = f1


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# ensure training data is randomised first!
train_data = train_data.sample(frac=1).reset_index(drop=True)

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

# Separate CV data from training data
cv_data = train_data.iloc[800:]
train_data = train_data.iloc[:800]

y = train_data["Survived"]
y_cv = cv_data["Survived"]

all_features = ['Pclass', 'Sex', 'Age',
                'SibSp', 'Parch', 'Fare', 'Embarked']


# normalise the Fare and the Age features
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
g_list = []
d_list = []
k_list = []
C_list = []
features_list = []
f1_list = []

features_sets = [['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare_scaled', 'Age_scaled'],
                 ['Pclass', 'Sex', 'SibSp', 'Parch'],
                 ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare_scaled', 'Age_scaled'],
                 ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']]

print('finding best model...')
# explore linear, poly, rbf and sigmoid models
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for features in features_sets:
    X = pd.get_dummies(train_data[features])
    X_cv = pd.get_dummies(cv_data[features])
    for k in kernels:
        if k == 'linear':
            model = Model(type=k, degree='NA', gamma='NA', C='NA', features=features)
            model.fit_and_evaluate_svm(X, y, X_cv, y_cv)
            f1_list.append(model.score)
            g_list.append(model.gamma)
            d_list.append(model.degree)
            C_list.append(model.C)
            features_list.append(model.features)
            k_list.append(k)
        elif k == 'poly':
            for d in range(10):
                model = Model(type=k, degree=d, gamma='NA', C='NA', features=features)
                model.fit_and_evaluate_svm(X, y, X_cv, y_cv)
                f1_list.append(model.score)
                g_list.append(model.gamma)
                d_list.append(model.degree)
                C_list.append(model.C)
                features_list.append(model.features)
                k_list.append(k)
        else:
            for g in ['auto', 'scale', 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]:
                for C in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
                    model = Model(type=k, degree='NA', gamma=g, C=C, features=features)
                    model.fit_and_evaluate_svm(X, y, X_cv, y_cv)
                    f1_list.append(model.score)
                    g_list.append(model.gamma)
                    d_list.append(model.degree)
                    C_list.append(model.C)
                    features_list.append(model.features)
                    k_list.append(k)


summary = pd.DataFrame()
summary['k'] = k_list
summary['d'] = d_list
summary['g'] = g_list
summary['C'] = C_list
summary['features'] = features_list
summary['f1'] = f1_list
summary = summary.sort_values(by='f1', ascending=False, ignore_index=True)
print(summary.head())

print('Highest performing model:')
print(summary.loc[0])

# THE TOP ROW OF summary GIVES THE BEST CONFIGURATION FOR SVM
# now, rerun this best configuration on the original training and test data
train_data = train_data.append(cv_data)
y = train_data["Survived"]

type = summary['k'][0]
print('Fitting {} model to test data...'.format(type))

if type == 'linear':
    clf = svm.SVC(kernel=type)
elif type == 'poly':
    clf = svm.SVC(kernel=type, degree=summary['d'][0])
else:
    clf = svm.SVC(kernel=type, gamma=summary['g'][0], C=summary['C'][0])

features = summary['features'][0]
X = pd.get_dummies(train_data[features])
clf.fit(X, y)

X_test = pd.get_dummies(test_data[features])
predictions = clf.predict(X_test)

print('writing output file')
# OUTPUT TO CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv("outputs/svm_optimal_1.csv", index=False)
print('done')
