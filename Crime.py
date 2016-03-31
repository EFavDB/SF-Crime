# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:03:18 2015

@author: damienrj
"""

def main():
    import pandas as pd
    from sklearn import ensemble
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.grid_search import GridSearchCV
    from sklearn import preprocessing
    from sklearn.metrics import log_loss
    import numpy as np

    #Load Data with pandas, and parse the first column into datetime
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')   
   
#%% This section processes the categories into numeric values    
    le_crime = preprocessing.LabelEncoder()
    crime = le_crime.fit_transform(train.Category)
 
    days = pd.get_dummies(train.DayOfWeek)
    district = pd.get_dummies(train.PdDistrict)
    train_data = pd.concat([days, district], axis=1)

    train_data['Y']=train['Y']
    train_data['X']=train['X']
    train_data['crime']=crime

    days = pd.get_dummies(test.DayOfWeek)
    district = pd.get_dummies(test.PdDistrict)
    test_data = pd.concat([days, district], axis=1)
    
    test_data['Y']=test['Y']
    test_data['X']=test['X']

    
#%% Decide on the features and scale to zscore
    
    #Define features vector
    features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
       'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN',
       'Y', 'X']

    
    #Create a validation training set for testing    
    training, validation = train_test_split(train_data, train_size=.70)
    model = BernoulliNB()
    model.fit(training[features], training['crime'])
    predicted = np.array(model.predict_proba(validation[features]))
    log_loss(validation['crime'], predicted) 


#    Run a paramater search over depth and min samples per leaf          
#    param_grid = {'learning_rate': [0.1, 0.05, 0.01],
#                  'max_depth': [5, 10, 15],
#                  'min_samples_leaf': [3, 10, 20],
#              }
#    est = ensemble.GradientBoostingClassifier(n_estimators=200, subsample=.5)
#    # Apply search to subset of data for speed
#    gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(
#        training.iloc[0:1000][features], training.iloc[0:1000]['crime'])
#    
#    # best hyperparameter setting
#    gs_cv.best_params_    
#    clf = gs_cv.best_estimator_
#    
#    # Train model of training set to esimate performance with validation set
#    clf.fit(training[features], training['salary'])
#    result_validation = clf.predict(validation[features])
#    
#    #Use mean absolute error for metric
#    print(mean_absolute_error(validation['salary'], result_validation))
    
    #%% Train final model of whole training dataset
    model = BernoulliNB()
    model.fit(train_data[features], train_data['crime'])
    predicted = model.predict_proba(test_data[features])


    #Write results
    result=pd.DataFrame(predicted, columns=le_crime.classes_)
    result.to_csv('testResult.csv', index = True, index_label = 'Id'  )
if __name__ == '__main__': 
    main()    