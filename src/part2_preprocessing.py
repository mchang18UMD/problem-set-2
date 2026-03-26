'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages

import pandas as pd
import numpy as np

# Your code here

def Preprocessing():

    print('part2_preprocessing.py')

    pred_universe = pd.read_csv('data/pred_universe_raw.csv') # Load `pred_universe_raw.csv` into a dataframe
    arrest_events = pd.read_csv('data/arrest_events_raw.csv') # Load `arrest_events_raw.csv` into a dataframe

    df_arrests = pred_universe.merge(arrest_events, on = 'person_id',how='outer') # Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`

    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ']) # Convert 'arrest_date_univ' to datetime
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event']) # Convert 'arrest_date_event' to datetime

    # Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 

    df_arrests['days_after_arrest'] = (df_arrests['arrest_date_event'] - df_arrests['arrest_date_univ']).dt.days # Calculates days between arrest_date_event and arrest_date_univ

    df_arrests['y'] = np.where( 
    (df_arrests['days_after_arrest'] >=1) &
    (df_arrests['days_after_arrest'] <=365) &
    (df_arrests['charge_degree'] == 'felony'),
    1,
    0
    )

    # Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?

    print('What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?', df_arrests['y'].mean()) 

    # Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise.

    df_arrests['current_charge_felony'] = np.where(
    df_arrests['charge_degree'] == 'felony',
    1,
    0
    )

    # Use a print statment to print this question and its answer: What share of current charges are felonies?

    print('What share of current charges are felonies?', df_arrests['current_charge_felony'].mean())

    # Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 

    df_arrests['days_before_arrest'] = (df_arrests['arrest_date_univ'] - df_arrests['arrest_date_event']).dt.days # Calculates days between arrest_date_event and arrest_date_univ

    df_arrests['num_fel_arrests_last_year'] = np.where( 
    (df_arrests['days_before_arrest'] >=1) &
    (df_arrests['days_before_arrest'] <=365) &
    (df_arrests['charge_degree'] == 'felony'),
    1,
    0
    )

    df_arrests['num_fel_arrests_last_year'] = (df_arrests.groupby('person_id')['num_fel_arrests_last_year'].transform('sum')) # Total number of felony arrest per person

    print('What is the average number of felony arrests in the last year?',df_arrests['num_fel_arrests_last_year'].mean()) # Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?

    pred_universe = pred_universe.merge(df_arrests[['person_id','num_fel_arrests_last_year']].drop_duplicates(),on='person_id', how='left') # Add num_fel_arrests_last_year column to pred_universe from df_arrests using person_id

    print(pred_universe['num_fel_arrests_last_year'].mean()) # Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()

    print(pred_universe.head()) # Print pred_universe.head()

    df_arrests.to_csv('data/df_arrests.csv', index=False) # Save df_arrests to data folder

    return df_arrests 