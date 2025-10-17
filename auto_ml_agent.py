"""
钙钛矿太阳能电池自动机器学习代理
基于 Perovskite_database_content_all_data.csv 数据库
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
# import lightgbm as lgb  # 暂时注释，系统可能未安装
# from catboost import CatBoostRegressor  # 暂时注释，系统可能未安装
import matplotlib.pyplot as plt
import seaborn as sns
# import shap  # 暂时注释，系统可能未安装
import warnings
warnings.filterwarnings('ignore')

class PerovskiteAutoMLAgent:
    """
    钙钛矿太阳能电池自动机器学习代理
    自动进行数据预处理、特征工程、模型选择和优化
    """

    def __init__(self, data_path='Perovskite_database_content_all_data.csv'):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

    def load_data(self):
        """加载和初步处理数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path)

        # 选择目标变量 (PCE效率)
        target_columns = ['JV_default_PCE', 'JV_reverse_scan_PCE', 'JV_forward_scan_PCE']
        target_col = None

        for col in target_columns:
            if col in self.data.columns and self.data[col].notna().sum() > 100:
                target_col = col
                break

        if target_col is None:
            raise ValueError("未找到合适的PCE目标变量")

        print(f"选择目标变量: {target_col}")

        # 分离特征和目标
        self.y = self.data[target_col].copy()
        self.X = self.data.drop(columns=[col for col in target_columns if col in self.data.columns])

        # 移除非数值列和ID列
        cols_to_drop = []
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                # 检查是否可以转换为数值
                try:
                    pd.to_numeric(self.X[col], errors='coerce')
                except:
                    cols_to_drop.append(col)
            elif col.startswith('Ref_') or col.startswith('JV_link'):
                cols_to_drop.append(col)

        self.X = self.X.drop(columns=cols_to_drop)
        print(f"数据形状: {self.X.shape}, 目标变量形状: {self.y.shape}")

    def preprocess_data(self):
        """数据预处理"""
        print("正在预处理数据...")

        # 处理缺失值
        self.X = pd.DataFrame(self.imputer.fit_transform(self.X), columns=self.X.columns)

        # 移除目标变量的缺失值
        valid_idx = self.y.notna()
        self.X = self.X[valid_idx]
        self.y = self.y[valid_idx]

        # 移除异常值 (PCE > 50% 或 < 0%)
        valid_pce = (self.y >= 0) & (self.y <= 50)
        self.X = self.X[valid_pce]
        self.y = self.y[valid_pce]

        print(f"预处理后数据形状: {self.X.shape}")

    def feature_engineering(self):
        """特征工程"""
        print("正在进行特征工程...")

        # 选择最重要的特征
        selector = SelectKBest(score_func=f_regression, k=50)
        self.X = pd.DataFrame(
            selector.fit_transform(self.X, self.y),
            columns=self.X.columns[selector.get_support()]
        )

        print(f"特征选择后形状: {self.X.shape}")

    def split_data(self, test_size=0.2, random_state=42):
        """分割数据"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"训练集: {self.X_train.shape}, 测试集: {self.X_test.shape}")

    def train_models(self):
        """训练多个模型"""
        print("正在训练模型...")

        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42),
            'CatBoost': CatBoostRegressor(verbose=False, random_state=42)
        }

        results = {}

        for name, model in models.items():
            print(f"训练 {name}...")
            model.fit(self.X_train, self.y_train)

            # 预测
            y_pred = model.predict(self.X_test)

            # 评估
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)

            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }

            print(".3f")

        # 选择最佳模型
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        self.models = results

        print(f"\n最佳模型: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")

    def explain_model(self):
        """模型解释"""
        print("正在生成模型解释...")

        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("前10个重要特征:")
            print(self.feature_importance.head(10))

    def visualize_results(self):
        """可视化结果"""
        print("正在生成可视化...")

        # 预测vs实际值散点图
        plt.figure(figsize=(10, 6))
        best_results = max(self.models.values(), key=lambda x: x['r2'])
        plt.scatter(self.y_test, best_results['predictions'], alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('实际PCE (%)')
        plt.ylabel('预测PCE (%)')
        plt.title('预测vs实际PCE效率')
        plt.grid(True)
        plt.savefig('pce_prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 特征重要性
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('重要性')
            plt.title('特征重要性排名')
            plt.gca().invert_yaxis()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("可视化图表已保存")

    def run_automl_pipeline(self):
        """运行完整的自动ML流程"""
        print("=" * 50)
        print("钙钛矿太阳能电池自动ML代理启动")
        print("=" * 50)

        try:
            self.load_data()
            self.preprocess_data()
            self.feature_engineering()
            self.split_data()
            self.train_models()
            self.explain_model()
            self.visualize_results()

            print("\n" + "=" * 50)
            print("自动ML流程完成!")
            print("=" * 50)

            # 打印最佳模型结果
            best_results = max(self.models.values(), key=lambda x: x['r2'])
            print(".3f")
            print(".3f")
            print(".3f")

        except Exception as e:
            print(f"错误: {str(e)}")
            raise

    def predict_new_data(self, new_data):
        """预测新数据"""
        if self.best_model is None:
            raise ValueError("请先运行训练流程")

        # 预处理新数据
        new_data_processed = pd.DataFrame(self.imputer.transform(new_data), columns=new_data.columns)
        new_data_processed = new_data_processed[self.X_train.columns]  # 确保列顺序一致

        return self.best_model.predict(new_data_processed)


if __name__ == "__main__":
    # 创建自动ML代理实例
    agent = PerovskiteAutoMLAgent()

    # 运行完整流程
    agent.run_automl_pipeline()