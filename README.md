Biased Random Forest (BRAF)
===========================

This code sample includes a solution to the BRAF algorithm discussed in the paper: 

"Biased Random Forest For Dealing With The Class Imbalance Problem" (included in this repository)

Some notes about the implementation:

1. This code was tested on Python 3.7.1
2. Instructions for running the code:

   a. 'biasedRF.py' is the main entry for this module.
   b. Please place the dataset (diabetes.csv) on the same directory as the source code and run the following command: 

       python biasedRF.py
     
   c. All plots (ROC curves and PR curves) are generated under the directory: plot (automatically generated)
     
   Note that the ROC curve within each CV fold can be combined while
   still maintaining visual clarity and therefore, ROC curves across CV iteration are
   organized within the same plot (roc.tif). However, the PR curve for each
   CV fold is plotted separately (to avoid being cluttered).
   
3. No ML libraries are used in the main algorithm (BRAF). 
   NOTE that the implementation of the decision tree is not optimized for speed. 

4. Pandas and scikit-learn are only used in data processing including data imputation

5. File descriptions:
   - biasedRF.py: main entry for the BRAF algorithm
   - data_processor.py: data pipeline; loading data & imputing data
   - evaluate.py: cross validation, performance evaluation
   - utils_plot.py: plotting utilities