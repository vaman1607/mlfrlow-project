# Importing reuired packages

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import os
import time
import mlflow

# load the dataset

dataset=pd.read_csv("train.csv")
num_cols=dataset.select_dtypes(include=["int", "float64"]).columns.tolist()
categorical_cols=dataset.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("Loan_Status")
categorical_cols.remove("Loan_ID")

## Null handling

# Filling categorical columns null values with mode

for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# Filling numerical columns null values with median values    
    
for col in num_cols:
    dataset[col].fillna(dataset[col].median(), inplace=True)    

# handling outliers

dataset[num_cols]= dataset[num_cols].apply(lambda x : x.clip(*x.quantile([0.05, 0.95])))  

# Log transformation & Domain processing

dataset["LoanAmount"]=np.log(dataset["LoanAmount"]).copy()
dataset["TotalIncome"]=dataset["ApplicantIncome"]+dataset["CoapplicantIncome"]
dataset["TotalIncome"]=np.log(dataset["TotalIncome"]).copy()

# label encoding in categorical columns

for col in categorical_cols:
    le=LabelEncoder()
    dataset[col]=le.fit_transform(dataset[col])

# label encoding in Target columns

dataset["Loan_Status"]=le.fit_transform(dataset["Loan_Status"])      


# Train-Test Split

X=dataset.drop(columns=["Loan_Status", "Loan_ID"])
y=dataset["Loan_Status"]
RANDOM_SEED=6

X_train, X_test, y_train, y_test=train_test_split(X, y, train_size=0.7, random_state=RANDOM_SEED)


# randomforest CLassifier

rf=RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest={
    'n_estimators' : [200, 400, 700],
    'max_depth' : [10, 20, 30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes' :[50,100]
}

grid_forest=GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_forest=grid_forest.fit(X_train, y_train)


# Logistic Regression

lr=LogisticRegression(random_state=RANDOM_SEED)

param_grid_lr={
    'C' : [100, 10, 1.0, 0.1, 0.01 ],
    'penalty' : ['l1', 'l2'],
    'solver' : ['liblinear']
}

grid_logreg=GridSearchCV(
    estimator=lr,
    param_grid=param_grid_lr,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_lr=grid_logreg.fit(X_train, y_train)


# DecisionTree Classifier

dt=DecisionTreeClassifier(random_state=RANDOM_SEED)

param_grid_dt={
    'max_depth':[3, 5, 7, 9, 11, 13],
    'criterion': ['gini', 'entropy']
}

grid_dt=GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_tree=grid_dt.fit(X_train, y_train)

mlflow.set_experiment("Loan_prediction")

# Model evaluation metrics

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)

    auc=metrics.auc(fpr, tpr)

    plt.figure(figsize=(9,9))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area=%0.2f'%auc)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])

    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('Turte Positive Rate', size=14)
    plt.legend(loc='lower right')

    # save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # close plot
    plt.close()

    return(accuracy, auc, f1)


def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        run_id=run.info.run_id
        mlflow.set_tag("run_id", run_id)
        pred=model.predict(X)

        #metrics
        (accuracy, f1, auc)=eval_metrics(y, pred)

        # logging best_params from Gridsearch

        mlflow.log_params(model.best_params_)

        # log the metrics

        mlflow.log_metric("Best CV Score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # logging artifacts and model

        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()

mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")        
mlflow_logging(model_lr, X_test, y_test, "LogisticRegression")
mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")

