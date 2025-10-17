import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('model_predictions_xgboost.csv')
x_true=df.iloc[:,0]
y_pred=df.iloc[:,1]
r2=r2_score(x_true,y_pred)
mae=mean_squared_error(x_true,y_pred)
rmse=mae**0.5
print(r2)
print(rmse)