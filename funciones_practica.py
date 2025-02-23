import argparse
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.datasets import load_iris

def argumentos():
    parser = argparse.ArgumentParser(description='Ejecutar modelo XGBoost con MLflow tracking.')
    parser.add_argument('--nombre_job', type=str, help='Nombre del experimento en MLflow.', required=False, default="iris-xgboost-experiment")
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='Lista de valores para n_estimators.', required=True)
    return parser.parse_args()

def load_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df['target'] = iris['target']
    return df

def data_treatment(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

    test_target = test['target']
    test[['target']].to_csv('test-target.csv', index=False)
    test.drop(columns=['target']).to_csv('test.csv', index=False)

    features = [x for x in df.columns if x != 'target']
    x_train, x_test, y_train, y_test = train_test_split(train[features], train['target'],
                                                        test_size=0.2, random_state=42, stratify=train['target'])
    return x_train, x_test, y_train, y_test

def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, n_estimators_list):
    time.sleep(5)
    mlflow.set_experiment(nombre_job)

    for n in n_estimators_list:
        with mlflow.start_run():
            clf = XGBClassifier(n_estimators=n, max_depth=3, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')

            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
            model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            logloss = log_loss(y_test, y_pred_proba)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            mlflow.log_param('n_estimators', n)
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('log_loss', logloss)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.xgboost.log_model(clf, 'Iris_XGBoost')

    print("Entrenamiento y logging en MLflow finalizado correctamente.")

if __name__ == '__main__':
    args = argumentos()
    df = load_dataset()
    x_train, x_test, y_train, y_test = data_treatment(df)
    mlflow_tracking(args.nombre_job, x_train, x_test, y_train, y_test, args.n_estimators_list)
