import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

# 1. 加载数据
data_path = 'output_expanded_additives_pubchem_encoded.csv'
df = pd.read_csv(data_path)

# 假设目标列为'JV_default_PCE'，特征为所有其他列
y = df['JV_default_PCE'].values
X = df.drop('JV_default_PCE', axis=1).values  # 或指定列：df.filter(regex='^(fp_|Perovskite_)')

# 处理缺失值（项目中用-1填充）
X = np.nan_to_num(X, nan=-1)

# 2. 划分数据集 (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. 训练XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, y_train)

# 4. 预测和评估
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# 测试集指标
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# 5. 保存评估结果
output_dir = 'output_20250826_130253'
os.makedirs(output_dir, exist_ok=True)
eval_df = pd.DataFrame({
    'MSE': [mse], 'RMSE': [rmse], 'MAE': [mae], 'R2': [r2]
})
eval_df.to_csv(f'{output_dir}/evaluation_results_xgboost.csv', index=False)

# 6. 保存预测结果
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
pred_df.to_csv(f'{output_dir}/model_predictions_xgboost.csv', index=False)

# 7. 特征重要性（合并fp_特征）
feature_names = df.drop('JV_default_PCE', axis=1).columns.tolist()
importance_df = pd.DataFrame({
    'raw_name': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 合并fp_特征
fp_mask = importance_df['raw_name'].str.startswith('fp_')
if fp_mask.any():
    fp_importance = importance_df.loc[fp_mask, 'importance'].sum()
    importance_df = pd.concat([
        importance_df[~fp_mask],
        pd.DataFrame({'raw_name': ['fp_all'], 'importance': [fp_importance]})
    ]).sort_values('importance', ascending=False)

importance_df.to_csv(f'{output_dir}/feature_importance_xgboost.csv', index=False)

# 8. 保存模型
joblib.dump(model, f'{output_dir}/trained_model_xgboost.pkl')

# 9. 可视化（可选）
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual PCE')
plt.ylabel('Predicted PCE')
plt.title('Actual vs Predicted')
plt.savefig(f'{output_dir}/prediction_plot.png')
plt.close()

print("训练完成！模型和输出保存到 output_20250826_130253/")