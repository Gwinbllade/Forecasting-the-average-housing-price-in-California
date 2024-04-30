Business needs

    Predicting the average housing price in a California neighborhood
    
Requirements

    python 3.7

    numpy==1.17.3
    pandas==1.1.5
    sklearn==1.0.0

Running:

    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated <prediction_results.csv> 
    The file has 'Churn_pred' column with the result value.

    The input is expected  csv file in the same folder with a name <new_data.csv>. The file shoul have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train_data.csv> should contain all features columns and target for prediction Churn.
    After running the script the "param_dict.pickle"  and "finalized_model.saw" will be created.
    Run the training script:
        python train.py

    On average, the model's predictions  are 10% wrong