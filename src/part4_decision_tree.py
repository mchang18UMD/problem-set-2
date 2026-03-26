'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

# Read in the dataframe(s) from PART 3

def decisiontree():

    print('part4_decision_tree.py')

    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')

    # Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero)

    param_grid_dt = {'max_depth': [1,2,3]}

    # Initialize the Decision Tree model. Assign this to a variable called `dt_model`.

    dt_model = DTC(random_state=5)

    # Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 

    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5)

    # Run the model

    features = ['current_charge_felony','num_fel_arrests_last_year']
    target = 'y' 

    X_train = df_arrests_train[features]
    y_train = df_arrests_train[target]

    X_test = df_arrests_test[features]

    # Run the model

    gs_cv_dt.fit(X_train,y_train)

    # What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 

    max_depth_value = gs_cv_dt.best_params_['max_depth']

    print('What was the optimal value for max_depth?', max_depth_value)

    if max_depth_value == min(param_grid_dt['max_depth']):
        print('The most regularization')
    elif max_depth_value == max(param_grid_dt['max_depth']):
        print('The least regularization')
    else:
        print('The middle regularization')

    # Now predict for the test set. Name this column `pred_dt` 

    df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)

    print(df_arrests_test['pred_dt'].head())

    df_arrests_test['pred_dt_prob'] = gs_cv_dt.predict_proba(df_arrests_test[features])[:,1] # Predicted probabilities

    # Save dataframe(s) save as .csv('s) in `data/`

    df_arrests_test.to_csv('data/df_arrests_test_decision_tree.csv', index=False)
    df_arrests_train.to_csv('data/df_arrests_train_decision_tree.csv', index=False)