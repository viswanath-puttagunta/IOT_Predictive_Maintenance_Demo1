"""
Module for Algorithm convenience functions to do grid search,
False Positive and False Negative analysis
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

algo_str_dict = {
    "LogisticRegression": LogisticRegression,
    "LogisticRegressionB": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,   
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "SVC": SVC
}

def recall_precision_f1(cm):
    """
    Given Confusion Matrix (cm), outputs recall, precision and f1 score
    """
    TN = cm[0][0]
    TP = cm[1][1] + 0.001  #To make sure denomintor is not zero
    FN = cm[1][0]
    FP = cm[0][1]
    recall_tpr = float(TP)/(TP+FN) 
    precision = float(TP)/(TP+FP)
    f1 = (2*precision*recall_tpr/(precision + recall_tpr))
    print "recall_sensitivity = %0.2f" % recall_tpr
    print "precision          = %0.2f" % precision
    print "f1                 = %0.2f"% f1
    return (recall_tpr, precision, f1)

def cv_roc_auc_accuracy(clf, X, y, cv=10):
    """
    Cross validation roc_auc and accuracy scores
    """
    roc_auc = cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean()
    cv_accuracy = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
    print "Cross-val-score(roc_auc) = %0.2f"%roc_auc
    print "Cross-val-score(accuracy) = %0.2f"%cv_accuracy
    return (round(roc_auc, 2), round(cv_accuracy, 2))

def cm_accuracy_rpf1(y_test, y_pred):
    """
    Returns accuracy, recall, precision, f1 scores
    for a given y_test (known values) vs y_pred (predicted values)
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print "Accuracy = %0.2f"%accuracy
    print "Confusion Matrix"
    cm = metrics.confusion_matrix(y_test, y_pred)
    print cm
    (recall, precision, f1) = recall_precision_f1(cm)
    return (round(accuracy, 2), round(recall, 2), round(precision, 2), round(f1, 2))

def do_clf(tdf, algo, algo_dd, fcols, resp):
    """
    Convinience function to do classification and reporting Cross Validation Scores
    """
    clf = algo(**algo_dd)
    X = tdf[fcols]
    y = tdf[resp]
    (roc_auc, accuracy, recall, precision, f1) = cv_roc_auc_accuracy_recall_precision(clf, X, y)
    return (roc_auc, accuracy, recall, precision, f1) 

def run_algo_analysis(df_sfeature, sfeature, fcols, algos_str, algos_dd):
    """
    Convenience function for doing custom grid search
    Returns dataframe with outputs of each run
    """
    ll = list()
    for fcol in fcols:
        ffcols = map(lambda x: sfeature + x, fcol.split(":")) 
        for algo_str in algos_str:  
            print "----------"
            print algo_str + ":" + fcol
            algo_dd = algos_dd[algo_str]
            algo = algo_str_dict[algo_str] 
            (roc_auc, accuracy, recall, precision, f1) = do_clf(df_sfeature, algo, algo_dd, ffcols, "failure")
            ll.append({"fcols": fcol, "algo":algo_str, 
                       "recall": recall, "precision": precision,
                       "f1": f1, "roc_auc": roc_auc, "accuracy": accuracy})
    dfresults = pd.DataFrame(ll)[["fcols", "algo", "recall", "precision", "f1", "roc_auc", "accuracy"]]
    dfresults = dfresults.sort_values(by=["recall", "f1", "roc_auc", "accuracy"], ascending=False)
    return dfresults

def do_clf_validate(tdf, algo_str,algo_dd, fcols, resp, random_state=4):
    """
    Convenience function to do misprediction (FP and FN) analysis
    """
    algo = algo_str_dict[algo_str]
    clf = algo(**algo_dd)
    X = tdf[fcols]
    y = tdf[resp]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    (accuracy, recall, precision, f1) = cm_accuracy_rpf1(y_test, y_pred)
    analysisdf = X_test.join(y_test)
    analysisdf.loc[:,"y_pred"] = y_pred
    return analysisdf

def cv_roc_auc_accuracy_recall_precision(clf, X, y, cv=10):
    """
    More robust performance results
    """
    roc_auc = cross_val_score(clf, X, y, cv=10, scoring='roc_auc').mean()
    cv_accuracy = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()
    recall = cross_val_score(clf, X, y, cv=10, scoring='recall').mean()
    precision = cross_val_score(clf, X, y, cv=10, scoring='precision').mean()
    f1 = cross_val_score(clf, X, y, cv=10, scoring='f1').mean()
    print "Cross-val-score(roc_auc) = %0.2f"%roc_auc
    print "Cross-val-score(accuracy) = %0.2f"%cv_accuracy
    print "Cross-val-score(recall)   = %0.2f"%recall
    print "Cross-val-score(precision)= %0.2f"%precision
    print "Cross-val-score(f1)       = %0.2f"%precision
    return (round(roc_auc, 2), round(cv_accuracy, 2),
            round(recall,2), round(precision,2), round(f1,2))

def do_clf_validate_new(tdf, algo_str,algo_dd, fcols, resp, random_state=4):
    #CV based metrics for more robust assessment
    algo = algo_str_dict[algo_str]
    clf = algo(**algo_dd)
    X = tdf[fcols]
    y = tdf[resp]
    (roc_auc, accuracy, recall, precision,f1) = cv_roc_auc_accuracy_recall_precision(clf, X, y, cv=10) 
