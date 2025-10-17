import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import os
import datetime
import joblib
from scipy.stats import uniform, randint

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
os.environ['PYTHONIOENCODING'] = 'utf-8'

def create_output_directory():
    """创建时间戳输出目录"""
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./output_{timestamp_str}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_and_preprocess_data():
    """加载并预处理数据"""
    try:
        df = pd.read_csv('3.csv')
    except FileNotFoundError:
        raise FileNotFoundError("找不到数据文件 '3.csv'，请确保文件存在")
    
    X = df.drop('JV_default_PCE', axis=1)
    y = df['JV_default_PCE']

    # 处理特征数据
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = pd.to_numeric(X[column], errors='coerce')
        X[column].fillna(-999, inplace=True)

    # 处理目标变量
    y = pd.to_numeric(y, errors='coerce')
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]

    # 保存原始列名映射
    original_cols = X.columns.tolist()
    X.columns = [f'feature_{i}' for i in range(X.shape[1])]
    col_map = dict(zip(X.columns, original_cols))

    print("处理后的数据集形状:", X.shape)
    return X, y, col_map

def optimize_hyperparameters(X_train, y_train):
    """使用RandomizedSearchCV优化超参数"""
    print("开始超参数优化...")
    
    # 定义参数搜索空间
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 16),
        'learning_rate': uniform(0.001, 0.399),
        'subsample': uniform(0.3, 0.7),
        'colsample_bytree': uniform(0.3, 0.7),
        'gamma': uniform(0, 10),
        'reg_alpha': uniform(0, 10),
        'reg_lambda': uniform(0, 10),
        'min_child_weight': randint(1, 10)
    }
    
    # 创建XGBoost回归器
    xgb_model = XGBRegressor(random_state=74, verbosity=0)
    
    # 使用RandomizedSearchCV进行超参数优化
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=30,  # 根据项目规范
        cv=5,       # 根据项目规范
        scoring='neg_mean_squared_error',
        n_jobs=1,   # 避免并行处理导致的编码问题
        random_state=74,
        verbose=1
    )
    
    # 执行搜索
    random_search.fit(X_train, y_train)
    
    print("最佳参数:", random_search.best_params_)
    print("最佳交叉验证得分:", -random_search.best_score_)
    
    return random_search.best_params_

def create_ensemble_model():
    """创建集成模型"""
    # 创建XGBoost回归器
    xgb_model = XGBRegressor(
        colsample_bytree=0.9702708806919009,
        gamma=5.444128910885886,
        learning_rate=0.1944089296615342,
        max_delta_step=6,
        max_depth=10,
        min_child_weight=1,
        n_estimators=476,
        reg_alpha=1.6494805207231078,
        reg_lambda=7.490422689465861,
        subsample=0.46308969697563773,
        random_state=974,
        verbosity=0
    )
    
    # 创建随机森林回归器
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=74
    )
    
    # 创建集成模型
    ensemble_model = VotingRegressor([
        ('xgb', xgb_model),
        ('rf', rf_model)
    ])
    
    return ensemble_model

def train_model(X, y, optimize_params=False, use_ensemble=False):
    """训练模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=74)
    
    if use_ensemble:
        # 使用集成模型
        print("使用集成模型进行训练...")
        model = create_ensemble_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model, X_train, X_test, y_train, y_test, y_pred
    
    # 根据是否优化参数选择参数
    if optimize_params:
        best_params = optimize_hyperparameters(X_train, y_train)
        xgb_model = XGBRegressor(**best_params, random_state=974, verbosity=0)
    else:
        best_params = {
            'colsample_bytree': 0.9702708806919009,
            'gamma': 5.444128910885886,
            'learning_rate': 0.1944089296615342,
            'max_delta_step': 6,
            'max_depth': 10,
            'min_child_weight': 1,
            'n_estimators': 476,
            'reg_alpha': 1.6494805207231078,
            'reg_lambda': 7.490422689465861,
            'subsample': 0.46308969697563773
        }
        xgb_model = XGBRegressor(**best_params, random_state=974, verbosity=0)
    
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    
    return xgb_model, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== 模型评估结果 ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def calculate_feature_importance(model, X, col_map):
    """计算并聚合特征重要性"""
    # 如果是集成模型，只获取XGBoost部分的特征重要性
    if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
        xgb_model = model.named_estimators_['xgb']
    else:
        xgb_model = model
    
    if hasattr(xgb_model, 'feature_importances_'):
        raw_importance = pd.DataFrame({
            'raw_name': [col_map[c] for c in X.columns],
            'importance': xgb_model.feature_importances_
        })
    else:
        # 如果模型没有特征重要性属性，创建一个默认的DataFrame
        raw_importance = pd.DataFrame({
            'raw_name': [col_map[c] for c in X.columns],
            'importance': np.ones(len(X.columns)) / len(X.columns)
        })

    # 把 fp_ 开头的列聚合成一个变量
    fp_mask = raw_importance['raw_name'].str.startswith('fp_')
    fp_sum = raw_importance.loc[fp_mask, 'importance'].sum()

    # 非 fp_ 的列保持原名
    non_fp = raw_importance[~fp_mask].copy()

    # 构建新的重要性表
    importance_grouped = non_fp[['raw_name', 'importance']].copy()
    if fp_sum > 0:
        importance_grouped = pd.concat([
            importance_grouped,
            pd.DataFrame({'raw_name': ['fp_all'], 'importance': [fp_sum]})
        ], ignore_index=True)

    importance_grouped = importance_grouped.sort_values(
        'importance', ascending=False).reset_index(drop=True)

    print("\n=== 修正后的特征重要性（fp_ 已聚合） ===")
    print(importance_grouped)
    return importance_grouped

def plot_feature_importance(top20, output_dir):
    """绘制特征重要性图表"""
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='raw_name', data=top20)
    plt.xlabel('特征重要性')
    plt.title('前20个最重要特征（fp_ 已聚合）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.show()

def perform_shap_analysis(model, X_train, output_dir):
    """执行SHAP分析"""
    # 如果是集成模型，只对XGBoost部分进行SHAP分析
    if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
        xgb_model = model.named_estimators_['xgb']
    else:
        xgb_model = model
    
    try:
        print("开始SHAP分析...")
        X_sample = shap.utils.sample(X_train, min(100, len(X_train)))
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        # SHAP摘要图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP 特征重要性摘要图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()

        # SHAP蜂群图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP 特征重要性蜂群图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_beeswarm_plot.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"SHAP分析失败: {e}")
        print("跳过SHAP分析...")

def save_model(model, output_dir, model_type="xgboost"):
    """保存训练好的模型"""
    model_path = os.path.join(output_dir, f"trained_model_{model_type}.pkl")
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")

def save_results(predictions_df, importance_grouped, evaluation_metrics, output_dir, model_type="xgboost"):
    """保存结果到文件"""
    predictions_df.to_csv(os.path.join(output_dir, f'model_predictions_{model_type}.csv'),
                          index=False)

    importance_grouped.to_csv(os.path.join(output_dir, f'feature_importance_{model_type}.csv'),
                              index=False)

    pd.DataFrame([evaluation_metrics]).to_csv(
        os.path.join(output_dir, f'evaluation_results_{model_type}.csv'), index=False)

    print(f"\n所有文件已保存至文件夹：{output_dir}")

def main(optimize_params=False, use_ensemble=False):
    """主函数"""
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 加载和预处理数据
    X, y, col_map = load_and_preprocess_data()
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y, optimize_params, use_ensemble)
    
    # 确定模型类型
    model_type = "ensemble" if use_ensemble else "xgboost"
    
    # 评估模型
    evaluation_metrics = evaluate_model(y_test, y_pred)
    
    # 计算特征重要性
    importance_grouped = calculate_feature_importance(model, X, col_map)
    
    # 可视化前20个特征
    top20 = importance_grouped.head(20)
    plot_feature_importance(top20, output_dir)
    
    # SHAP分析（仅对XGBoost模型）
    if not use_ensemble:
        perform_shap_analysis(model, X_train, output_dir)
    
    # 保存结果
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    
    save_results(predictions_df, importance_grouped, evaluation_metrics, output_dir, model_type)
    
    # 保存模型
    save_model(model, output_dir, model_type)

if __name__ == "__main__":
    # 可以通过修改参数来决定是否进行超参数优化或使用集成模型
    # main(optimize_params=True, use_ensemble=False)  # 使用超参数优化
    # main(optimize_params=False, use_ensemble=True)  # 使用集成模型
    main(optimize_params=False, use_ensemble=False)   # 使用默认参数