import mlflow 
from mlflow import log_params, log_metrics
from mlflow import sklearn as ml_sklearn
from train import train_model
from preprocessing import preprocessing, split_data_label, load_csv
import argparse
import sys
import os
import logging
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type = str, help = "please your data path"
    )
    args = parser.parse_args()
    data_path = args.data_path
    df = load_csv(data_path)
    train_df, test_df = preprocessing(df)
    label_col = 'diagnosis'
    x_train, y_train = split_data_label(train_df, label_col)
    x_test, y_test = split_data_label(test_df, label_col)
    model, model_info = train_model(x_train.values, y_train.values, 
                                    x_test.values, y_test.values)
    
    log_metrics(model_info['score'])
    log_params(model_info['params'])
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    today = datetime.today().strftime("%Y%m%d-%H%M%S")
    mlflow.set_tag('mlflow.runName', today)
    dataset = mlflow.data.from_numpy(x_train.values)

    mlflow.log_input(dataset, context = "training_v0.1")
    ml_sklearn.log_model(model, 'ml_model')
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
