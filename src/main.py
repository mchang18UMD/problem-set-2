'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot

from part1_etl import ETL
from part2_preprocessing import Preprocessing
from part3_logistic_regression import logisticregression
from part4_decision_tree import decisiontree
from part5_calibration_plot import Calibration

# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`

    ETL()

    # PART 2: Call functions/instanciate objects from preprocessing

    Preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression

    logisticregression()

    # PART 4: Call functions/instanciate objects from decision_tree

    decisiontree()

    # PART 5: Call functions/instanciate objects from calibration_plot

    Calibration()


if __name__ == "__main__":
    main()