import numpy as np
import pandas as pd

import jo_wilder
env = jo_wilder.make_env()
iter_test = env.iter_test()

counter = 0
# The API will deliver two dataframes in this specific order,
# for every session+level grouping (one group per session for each checkpoint)
for (sample_submission, test) in iter_test:
    if counter == 0:
        print(sample_submission.head())
        print(test.head())
        print(test.shape)
        
    ## users make predictions here using the test data
    sample_submission['correct'] = 1
    
    ## env.predict appends the session+level sample_submission to the overall
    ## submission
    env.predict(sample_submission)
