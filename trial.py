import sys

import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score




data = pd.read_csv("CarPrice_Assignment.csv")
numerical_variable = [feature for feature in data.columns if data[feature].dtype!="object"]
categorical_variable = [feature for feature in data.columns if data[feature].dtype=="object"]

data['name with model'] = data['CarName'].map(lambda x: x.split(maxsplit=1))
data['name'] = data['name with model'].apply(lambda x: x[0])

renaming_map = {'toyouta':'toyota','Nissan':'nissan','vokswagen':'volkswagen','maxda':'mazda','porcshce':'porsche','vw':'volkswagen'}

def rename_cars(data,feature):
    return data[feature].replace(renaming_map)

data['name'] = rename_cars(data,'name')

def is_custom(spec_list):
    for items in spec_list.split():
        if items.lower() == "custom":
            return 1
            
    else:
        return 0

data['is_custom_made'] = data['name with model'].apply(lambda x: is_custom(x[-1]))

drop_item = ['car_ID','name with model','CarName']
data.drop(drop_item,axis=1,inplace=True)

numerical_variable.remove('car_ID')

def take_transformation(data,transform_list):
    data = data.copy()
    for item in transform_list:
        data[item] = np.log(data[item])
    return data

log_variables = [
 'wheelbase',
 'carlength',
 'carwidth',
 'carheight',
 'curbweight',
 'enginesize',
 'boreratio',
 'stroke',
 'compressionratio',
 'horsepower',
 'peakrpm',
 'citympg',
 'highwaympg',
 'price']
 
data = take_transformation(data,log_variables)

categorical_variable = [feature for feature in data.columns if data[feature].dtype=="object"]

def encode_categorical_variables(data,cat_variables,target):
    for features in cat_variables:
#     cat_map = {}
        encoded = data.groupby([features])[target].mean().sort_values().index
        cat_map = {k:i for i,k in enumerate(encoded)}
        data[features] = data[features].map(cat_map)

encode_categorical_variables(data,categorical_variable,'price')

features = data.columns.tolist()
features.remove('price')

scaler = StandardScaler()
scaler.fit(data[features])
data[features] = scaler.transform(data[features])

train,test,y_train,y_test = train_test_split(data[features],data['price'])

experiment_name = "ml-flow linear model.py"
experiment_id = mlflow.create_experiment(experiment_name)

def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


if __name__=="__main__":
    
    # alpha=random.randint(1,10)/10
    # l1 = random.randint(1,10)/10

    alpha = float(sys.argv[1]) if len(sys.argv)>1 else 0.5
    l1 = float(sys.argv[2]) if len(sys.argv)>2 else 0.5

    state = 42
    model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=state)
    model.fit(train[features],y_train)
    
    #making prediction
    y_pred = model.predict(test[features])
    rmse, mae, r2 = eval_metrics(y_test, y_pred)
    
    # run_name = f"run_{index}"
    
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1)
        mlflow.log_param("random_state", state)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model") 