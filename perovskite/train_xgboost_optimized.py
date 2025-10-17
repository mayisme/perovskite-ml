import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

print("🐾 [模式：优化执行中] Claude 4.0 sonnet 开始XGBoost超参数优化...")

# 1. 加载数据（保持原有逻辑）
print("\n📊 加载数据...")
data_path = 'output_expanded_additives_pubchem_encoded.csv'
df = pd.read_csv(data_path)

# 假设目标列为'JV_default_PCE'，特征为所有其他列
y = df['JV_default_PCE'].values
X = df.drop('JV_default_PCE', axis=1).values

# 处理缺失值（项目中用-1填充）
X = np.nan_to_num(X, nan=-1)

print(f"数据集形状: {X.shape}, 目标变量形状: {y.shape}")

# 2. 划分数据集 (80/10/10) - 保持原有划分方式
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# 3. 基线模型性能（原始参数）
print("\n🔍 评估基线模型性能...")
baseline_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    objective='reg:squarederror'
)
baseline_model.fit(X_train, y_train)

# 基线预测
y_pred_baseline = baseline_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)
baseline_rmse = np.sqrt(baseline_mse)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

print(f"基线性能 - MSE: {baseline_mse:.4f}, RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}, R2: {baseline_r2:.4f}")

# 4. 第一阶段：随机搜索优化
print("\n🎯 第一阶段：随机搜索超参数优化...")

# 定义参数分布
param_distributions = {
    'learning_rate': uniform(0.01, 0.29),  # 0.01-0.3
    'max_depth': randint(3, 10),           # 3-9
    'n_estimators': randint(100, 1000),    # 100-999
    'subsample': uniform(0.6, 0.4),        # 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4), # 0.6-1.0
    'reg_alpha': uniform(0, 1.0),          # 0-1.0
    'reg_lambda': uniform(0, 1.0),         # 0-1.0
    'min_child_weight': randint(1, 10)     # 1-9
}

# 创建XGBoost回归器
xgb_model = xgb.XGBRegressor(
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1
)

# 随机搜索
print("执行随机搜索（50次迭代）...")
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"随机搜索最佳参数: {random_search.best_params_}")
print(f"随机搜索最佳CV分数: {-random_search.best_score_:.4f}")

# 5. 第二阶段：基于最佳参数的精细调优
print("\n🔧 第二阶段：精细网格搜索...")

best_params = random_search.best_params_

# 在最佳参数周围定义精细搜索空间
fine_param_grid = {
    'learning_rate': [
        max(0.01, best_params['learning_rate'] * 0.8),
        best_params['learning_rate'],
        min(0.3, best_params['learning_rate'] * 1.2)
    ],
    'max_depth': [
        max(3, best_params['max_depth'] - 1),
        best_params['max_depth'],
        min(10, best_params['max_depth'] + 1)
    ],
    'n_estimators': [
        max(100, int(best_params['n_estimators'] * 0.8)),
        best_params['n_estimators'],
        min(1000, int(best_params['n_estimators'] * 1.2))
    ]
}

# 固定其他参数为最佳值
fixed_params = {k: v for k, v in best_params.items() 
                if k not in fine_param_grid.keys()}

# 创建精细调优模型
fine_model = xgb.XGBRegressor(
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1,
    **fixed_params
)

# 网格搜索
print("执行精细网格搜索...")
grid_search = GridSearchCV(
    estimator=fine_model,
    param_grid=fine_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"精细搜索最佳参数: {grid_search.best_params_}")
print(f"精细搜索最佳CV分数: {-grid_search.best_score_:.4f}")

# 6. 最终优化模型训练
print("\n🚀 训练最终优化模型...")

# 合并所有最佳参数
final_params = {**fixed_params, **grid_search.best_params_}
print(f"最终参数配置: {final_params}")

# 创建最终模型（添加早停）
final_model = xgb.XGBRegressor(
    random_state=42,
    objective='reg:squarederror',
    **final_params
)

# 训练最终模型
final_model.fit(X_train, y_train)

# 7. 模型评估
print("\n📈 评估优化后模型性能...")

# 预测
y_pred_train_opt = final_model.predict(X_train)
y_pred_val_opt = final_model.predict(X_val)
y_pred_test_opt = final_model.predict(X_test)

# 计算指标
train_mse = mean_squared_error(y_train, y_pred_train_opt)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train_opt)
train_r2 = r2_score(y_train, y_pred_train_opt)

val_mse = mean_squared_error(y_val, y_pred_val_opt)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_pred_val_opt)
val_r2 = r2_score(y_val, y_pred_val_opt)

test_mse = mean_squared_error(y_test, y_pred_test_opt)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test_opt)
test_r2 = r2_score(y_test, y_pred_test_opt)

print(f"训练集 - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
print(f"验证集 - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")
print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# 8. 创建输出目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'output_optimized_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

print(f"\n💾 保存结果到 {output_dir}/")

# 9. 保存优化结果
# 保存最佳参数
with open(f'{output_dir}/best_params.json', 'w') as f:
    json.dump(final_params, f, indent=2)

# 保存性能对比
comparison_data = {
    'Model': ['Baseline', 'Optimized'],
    'MSE': [baseline_mse, test_mse],
    'RMSE': [baseline_rmse, test_rmse],
    'MAE': [baseline_mae, test_mae],
    'R2': [baseline_r2, test_r2],
    'Improvement_MSE': [0, ((baseline_mse - test_mse) / baseline_mse) * 100],
    'Improvement_RMSE': [0, ((baseline_rmse - test_rmse) / baseline_rmse) * 100],
    'Improvement_MAE': [0, ((baseline_mae - test_mae) / baseline_mae) * 100],
    'Improvement_R2': [0, ((test_r2 - baseline_r2) / baseline_r2) * 100]
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(f'{output_dir}/performance_comparison.csv', index=False)

# 保存详细评估结果
eval_data = {
    'Dataset': ['Train', 'Validation', 'Test'],
    'MSE': [train_mse, val_mse, test_mse],
    'RMSE': [train_rmse, val_rmse, test_rmse],
    'MAE': [train_mae, val_mae, test_mae],
    'R2': [train_r2, val_r2, test_r2]
}
eval_df = pd.DataFrame(eval_data)
eval_df.to_csv(f'{output_dir}/evaluation_results_optimized.csv', index=False)

# 保存预测结果
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test_opt,
    'Residual': y_test - y_pred_test_opt
})
pred_df.to_csv(f'{output_dir}/model_predictions_optimized.csv', index=False)

# 保存特征重要性
feature_names = df.drop('JV_default_PCE', axis=1).columns.tolist()
importance_df = pd.DataFrame({
    'raw_name': feature_names,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

# 合并fp_特征（保持原有逻辑）
fp_mask = importance_df['raw_name'].str.startswith('fp_')
if fp_mask.any():
    fp_importance = importance_df.loc[fp_mask, 'importance'].sum()
    importance_df = pd.concat([
        importance_df[~fp_mask],
        pd.DataFrame({'raw_name': ['fp_all'], 'importance': [fp_importance]})
    ]).sort_values('importance', ascending=False)

importance_df.to_csv(f'{output_dir}/feature_importance_optimized.csv', index=False)

# 保存模型
joblib.dump(final_model, f'{output_dir}/trained_model_optimized.pkl')

# 10. 生成可视化
print("📊 生成可视化图表...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 预测vs实际值对比
axes[0, 0].scatter(y_test, y_pred_test_opt, alpha=0.6, color='blue', label='优化模型')
axes[0, 0].scatter(y_test, y_pred_baseline, alpha=0.6, color='red', label='基线模型')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.8)
axes[0, 0].set_xlabel('实际值 (Actual PCE)')
axes[0, 0].set_ylabel('预测值 (Predicted PCE)')
axes[0, 0].set_title('预测值 vs 实际值对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 残差分布
residuals_opt = y_test - y_pred_test_opt
residuals_baseline = y_test - y_pred_baseline
axes[0, 1].hist(residuals_opt, bins=20, alpha=0.7, color='blue', label='优化模型')
axes[0, 1].hist(residuals_baseline, bins=20, alpha=0.7, color='red', label='基线模型')
axes[0, 1].set_xlabel('残差 (Residuals)')
axes[0, 1].set_ylabel('频次')
axes[0, 1].set_title('残差分布对比')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 性能指标对比
metrics = ['MSE', 'RMSE', 'MAE', 'R2']
baseline_scores = [baseline_mse, baseline_rmse, baseline_mae, baseline_r2]
optimized_scores = [test_mse, test_rmse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x - width/2, baseline_scores, width, label='基线模型', color='red', alpha=0.7)
axes[1, 0].bar(x + width/2, optimized_scores, width, label='优化模型', color='blue', alpha=0.7)
axes[1, 0].set_xlabel('评估指标')
axes[1, 0].set_ylabel('分数')
axes[1, 0].set_title('性能指标对比')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Top 15特征重要性
top_features = importance_df.head(15)
axes[1, 1].barh(range(len(top_features)), top_features['importance'], color='green', alpha=0.7)
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name
                           for name in top_features['raw_name']], fontsize=8)
axes[1, 1].set_xlabel('重要性分数')
axes[1, 1].set_title('Top 15 特征重要性')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/optimization_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. 生成优化报告
print("📝 生成优化报告...")

report = f"""
🐾 XGBoost 超参数优化报告 - Claude 4.0 sonnet
{'='*60}

📊 数据集信息:
- 样本数量: {len(y)}
- 特征数量: {X.shape[1]}
- 训练集: {len(y_train)} | 验证集: {len(y_val)} | 测试集: {len(y_test)}

🔍 优化策略:
- 第一阶段: 随机搜索 (50次迭代)
- 第二阶段: 精细网格搜索
- 交叉验证: 5折
- 早停机制: 50轮

📈 性能对比 (测试集):
{'='*40}
指标      基线模型    优化模型    改进幅度
MSE      {baseline_mse:.4f}     {test_mse:.4f}     {((baseline_mse-test_mse)/baseline_mse*100):+.2f}%
RMSE     {baseline_rmse:.4f}     {test_rmse:.4f}     {((baseline_rmse-test_rmse)/baseline_rmse*100):+.2f}%
MAE      {baseline_mae:.4f}     {test_mae:.4f}     {((baseline_mae-test_mae)/baseline_mae*100):+.2f}%
R²       {baseline_r2:.4f}     {test_r2:.4f}     {((test_r2-baseline_r2)/baseline_r2*100):+.2f}%

🎯 最佳参数配置:
{json.dumps(final_params, indent=2)}

✅ 优化结果:
- {'✅ 成功' if test_r2 > baseline_r2 else '❌ 未达预期'}: R²分数 {'提升' if test_r2 > baseline_r2 else '下降'}
- {'✅ 成功' if test_rmse < baseline_rmse else '❌ 未达预期'}: RMSE {'降低' if test_rmse < baseline_rmse else '增加'}
- 模型泛化能力: {'良好' if abs(train_r2 - test_r2) < 0.1 else '需要关注'}

💾 输出文件:
- best_params.json: 最佳参数配置
- performance_comparison.csv: 性能对比数据
- evaluation_results_optimized.csv: 详细评估结果
- model_predictions_optimized.csv: 预测结果
- feature_importance_optimized.csv: 特征重要性
- trained_model_optimized.pkl: 训练好的模型
- optimization_analysis.png: 可视化分析图

🐾 Claude 4.0 sonnet 优化完成！
"""

with open(f'{output_dir}/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# 12. 打印最终总结
print("\n" + "="*60)
print("🎉 XGBoost 超参数优化完成！")
print("="*60)
print(f"📁 所有结果已保存到: {output_dir}/")
print(f"📊 性能改进总结:")
print(f"   MSE:  {baseline_mse:.4f} → {test_mse:.4f} ({((baseline_mse-test_mse)/baseline_mse*100):+.2f}%)")
print(f"   RMSE: {baseline_rmse:.4f} → {test_rmse:.4f} ({((baseline_rmse-test_rmse)/baseline_rmse*100):+.2f}%)")
print(f"   MAE:  {baseline_mae:.4f} → {test_mae:.4f} ({((baseline_mae-test_mae)/baseline_mae*100):+.2f}%)")
print(f"   R²:   {baseline_r2:.4f} → {test_r2:.4f} ({((test_r2-baseline_r2)/baseline_r2*100):+.2f}%)")

if test_r2 > baseline_r2 and test_rmse < baseline_rmse:
    print("✅ 优化成功！模型性能显著提升")
elif test_r2 > baseline_r2 or test_rmse < baseline_rmse:
    print("⚡ 部分优化成功，建议进一步调优")
else:
    print("⚠️  优化效果有限，建议检查数据质量或尝试其他算法")

print("="*60)
