"""
钙钛矿太阳能电池机器学习模型
基于钙钛矿组成特征预测JV特性 (Voc, Jsc, FF, PCE)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

class PerovskiteMLModel:
    """
    钙钛矿太阳能电池机器学习模型
    基于材料组成预测JV特性
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
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')

        # 目标变量
        self.target_columns = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']

        # 钙钛矿层特征列
        self.perovskite_columns = [
            'Perovskite_single_crystal', 'Perovskite_dimension_0D', 'Perovskite_dimension_2D',
            'Perovskite_dimension_2D3D_mixture', 'Perovskite_dimension_3D',
            'Perovskite_dimension_3D_with_2D_capping_layer', 'Perovskite_dimension_list_of_layers',
            'Perovskite_composition_perovskite_ABC3_structure',
            'Perovskite_composition_perovskite_inspired_structure', 'Perovskite_composition_a_ions',
            'Perovskite_composition_a_ions_coefficients', 'Perovskite_composition_b_ions',
            'Perovskite_composition_b_ions_coefficients', 'Perovskite_composition_c_ions',
            'Perovskite_composition_c_ions_coefficients', 'Perovskite_composition_none_stoichiometry_components_in_excess',
            'Perovskite_composition_short_form', 'Perovskite_composition_long_form',
            'Perovskite_composition_assumption', 'Perovskite_composition_inorganic',
            'Perovskite_composition_leadfree', 'Perovskite_additives_compounds',
            'Perovskite_additives_concentrations', 'Perovskite_thickness', 'Perovskite_band_gap',
            'Perovskite_band_gap_graded', 'Perovskite_band_gap_estimation_basis', 'Perovskite_pl_max',
            'Perovskite_deposition_number_of_deposition_steps', 'Perovskite_deposition_procedure',
            'Perovskite_deposition_aggregation_state_of_reactants', 'Perovskite_deposition_synthesis_atmosphere',
            'Perovskite_deposition_synthesis_atmosphere_pressure_total',
            'Perovskite_deposition_synthesis_atmosphere_pressure_partial',
            'Perovskite_deposition_synthesis_atmosphere_relative_humidity', 'Perovskite_deposition_solvents',
            'Perovskite_deposition_solvents_mixing_ratios', 'Perovskite_deposition_solvents_supplier',
            'Perovskite_deposition_solvents_purity', 'Perovskite_deposition_reaction_solutions_compounds',
            'Perovskite_deposition_reaction_solutions_compounds_supplier',
            'Perovskite_deposition_reaction_solutions_compounds_purity',
            'Perovskite_deposition_reaction_solutions_concentrations',
            'Perovskite_deposition_reaction_solutions_volumes', 'Perovskite_deposition_reaction_solutions_age',
            'Perovskite_deposition_reaction_solutions_temperature', 'Perovskite_deposition_substrate_temperature',
            'Perovskite_deposition_quenching_induced_crystallisation', 'Perovskite_deposition_quenching_media',
            'Perovskite_deposition_quenching_media_mixing_ratios', 'Perovskite_deposition_quenching_media_volume',
            'Perovskite_deposition_quenching_media_additives_compounds',
            'Perovskite_deposition_quenching_media_additives_concentrations',
            'Perovskite_deposition_thermal_annealing_temperature', 'Perovskite_deposition_thermal_annealing_time',
            'Perovskite_deposition_thermal_annealing_atmosphere', 'Perovskite_deposition_thermal_annealing_relative_humidity',
            'Perovskite_deposition_thermal_annealing_pressure', 'Perovskite_deposition_solvent_annealing',
            'Perovskite_deposition_solvent_annealing_timing', 'Perovskite_deposition_solvent_annealing_solvent_atmosphere',
            'Perovskite_deposition_solvent_annealing_time', 'Perovskite_deposition_solvent_annealing_temperature',
            'Perovskite_deposition_after_treatment_of_formed_perovskite',
            'Perovskite_deposition_after_treatment_of_formed_perovskite_met',
            'Perovskite_storage_time_until_next_deposition_step', 'Perovskite_storage_atmosphere',
            'Perovskite_storage_relative_humidity', 'Perovskite_surface_treatment_before_next_deposition_step'
        ]

    def load_data(self):
        """加载和预处理数据"""
        print("正在加载钙钛矿数据库...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成！数据形状: {self.data.shape}")

    def preprocess_targets(self):
        """预处理目标变量"""
        print("正在预处理目标变量...")

        # 检查目标变量可用性
        available_targets = []
        for col in self.target_columns:
            if col in self.data.columns:
                missing_rate = self.data[col].isnull().sum() / len(self.data) * 100
                print(".1f")
                if missing_rate < 50:  # 缺失率小于50%的目标变量
                    available_targets.append(col)

        if not available_targets:
            raise ValueError("没有合适的PCE目标变量")

        print(f"可用的目标变量: {available_targets}")
        return available_targets

    def create_material_features(self):
        """创建材料组成特征"""
        print("正在创建材料组成特征...")

        # 选择钙钛矿层特征
        available_perovskite_cols = [col for col in self.perovskite_columns if col in self.data.columns]
        print(f"钙钛矿层可用特征: {len(available_perovskite_cols)} 列")

        # 创建特征矩阵
        X_material = self.data[available_perovskite_cols].copy()

        # 处理数值特征
        numeric_cols = X_material.select_dtypes(include=[np.number]).columns
        print(f"数值特征: {len(numeric_cols)} 列")

        # 处理类别特征
        categorical_cols = X_material.select_dtypes(include=['object']).columns
        print(f"类别特征: {len(categorical_cols)} 列")

        # 对类别特征进行编码
        X_encoded = self._encode_categorical_features(X_material)

        return X_encoded

    def _encode_categorical_features(self, X):
        """编码类别特征"""
        print("正在编码类别特征...")

        X_encoded = X.copy()

        # 离子组成编码
        ion_columns = ['Perovskite_composition_a_ions', 'Perovskite_composition_b_ions', 'Perovskite_composition_c_ions']
        for col in ion_columns:
            if col in X_encoded.columns:
                X_encoded = self._encode_ion_composition(X_encoded, col)

        # 工艺参数编码
        process_columns = ['Perovskite_deposition_procedure', 'Perovskite_deposition_solvents']
        for col in process_columns:
            if col in X_encoded.columns:
                X_encoded = self._encode_process_parameters(X_encoded, col)

        # 其他类别特征使用标签编码或独热编码
        for col in X_encoded.select_dtypes(include=['object']).columns:
            if col not in ion_columns + process_columns:
                if X_encoded[col].nunique() <= 10:  # 类别数少的用独热编码
                    dummies = pd.get_dummies(X_encoded[col], prefix=col, dummy_na=True)
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
                else:  # 类别数多的用标签编码
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        return X_encoded

    def _encode_ion_composition(self, X, col):
        """编码离子组成"""
        print(f"正在编码离子组成: {col}")

        # 提取唯一离子
        all_ions = set()
        for val in X[col].dropna():
            if isinstance(val, str):
                ions = [ion.strip() for ion in val.split(';') if ion.strip()]
                all_ions.update(ions)

        # 为每个离子创建二进制特征
        for ion in sorted(all_ions):
            ion_col = f"{col}_{ion}"
            X[ion_col] = X[col].apply(lambda x: 1 if isinstance(x, str) and ion in x else 0)

        # 移除原始列
        X = X.drop(col, axis=1)

        return X

    def _encode_process_parameters(self, X, col):
        """编码工艺参数"""
        print(f"正在编码工艺参数: {col}")

        # 统计最常见的工艺参数
        process_counts = Counter()
        for val in X[col].dropna():
            if isinstance(val, str):
                processes = [p.strip() for p in val.split(';') if p.strip()]
                for process in processes:
                    process_counts[process] += 1

        # 选择最常见的工艺参数 (出现次数>10)
        common_processes = {proc for proc, count in process_counts.items() if count > 10}

        # 为每个常见工艺创建二进制特征
        for process in sorted(common_processes):
            process_col = f"{col}_{process.replace(' ', '_')}"
            X[process_col] = X[col].apply(lambda x: 1 if isinstance(x, str) and process in x else 0)

        # 移除原始列
        X = X.drop(col, axis=1)

        return X

    def prepare_data(self):
        """准备训练数据"""
        print("正在准备训练数据...")

        # 获取目标变量
        available_targets = self.preprocess_targets()

        # 创建材料特征
        X_material = self.create_material_features()

        # 合并所有目标变量
        y_data = self.data[available_targets].copy()

        # 移除目标变量的缺失值
        valid_indices = y_data.notna().all(axis=1)
        X_final = X_material[valid_indices]
        y_final = y_data[valid_indices]

        print(f"最终数据集形状: X={X_final.shape}, y={y_final.shape}")

        # 分割数据
        self.X = X_final
        self.y = y_final

        return available_targets

    def train_models(self, targets):
        """训练模型"""
        print("正在训练机器学习模型...")

        for target in targets:
            print(f"\n训练 {target} 预测模型...")

            # 准备数据
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y[target], test_size=0.2, random_state=42
            )

            # 处理缺失值
            X_train_filled = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
            X_test_filled = pd.DataFrame(self.imputer.transform(X_test), columns=X_test.columns)

            # 特征选择
            selector = SelectKBest(score_func=f_regression, k=min(50, X_train_filled.shape[1]))
            X_train_selected = selector.fit_transform(X_train_filled, y_train)
            X_test_selected = selector.transform(X_test_filled)

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)

            # 训练模型
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            }

            target_results = {}

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                target_results[name] = {
                    'model': model,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred,
                    'y_test': y_test
                }

                print(".3f")

            self.models[target] = target_results

    def visualize_results(self):
        """可视化结果"""
        print("正在生成可视化结果...")

        n_targets = len(self.models)
        fig, axes = plt.subplots(2, n_targets, figsize=(6*n_targets, 10))

        if n_targets == 1:
            axes = axes.reshape(2, -1)

        for i, (target, results) in enumerate(self.models.items()):
            # 预测vs实际值散点图
            best_model = max(results.values(), key=lambda x: x['r2'])

            ax1 = axes[0, i] if n_targets > 1 else axes[0]
            ax1.scatter(best_model['y_test'], best_model['predictions'], alpha=0.6)
            ax1.plot([best_model['y_test'].min(), best_model['y_test'].max()],
                    [best_model['y_test'].min(), best_model['y_test'].max()], 'r--', lw=2)
            ax1.set_xlabel(f'实际{target}')
            ax1.set_ylabel(f'预测{target}')
            ax1.set_title(f'{target} 预测结果')
            ax1.grid(True, alpha=0.3)

            # 特征重要性 (RandomForest)
            if 'RandomForest' in results:
                rf_model = results['RandomForest']['model']
                if hasattr(rf_model, 'feature_importances_'):
                    ax2 = axes[1, i] if n_targets > 1 else axes[1]
                    # 选择最重要的特征
                    feature_names = [f'特征_{i}' for i in range(len(rf_model.feature_importances_))]
                    top_features_idx = np.argsort(rf_model.feature_importances_)[-10:]
                    top_features = [feature_names[i] for i in top_features_idx]
                    top_importances = rf_model.feature_importances_[top_features_idx]

                    ax2.barh(range(len(top_features)), top_importances)
                    ax2.set_yticks(range(len(top_features)))
                    ax2.set_yticklabels(top_features)
                    ax2.set_xlabel('重要性')
                    ax2.set_title(f'{target} 特征重要性')
                    ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('perovskite_ml_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("可视化图表已保存为: perovskite_ml_results.png")

    def print_summary(self):
        """打印总结报告"""
        print("\n" + "="*60)
        print("钙钛矿太阳能电池机器学习模型总结")
        print("="*60)

        print(f"数据集大小: {self.X.shape[0]} 样本, {self.X.shape[1]} 特征")

        for target, results in self.models.items():
            print(f"\n{target} 预测结果:")
            for model_name, result in results.items():
                print(".3f")

        print("\n特征编码说明:")
        print("1. 离子组成: 二进制编码 (每个离子一个特征)")
        print("2. 工艺参数: 二进制编码 (常见工艺类型)")
        print("3. 其他类别特征: 独热编码或标签编码")
        print("4. 数值特征: 标准化处理")

    def run_ml_pipeline(self):
        """运行完整的机器学习流程"""
        print("="*60)
        print("钙钛矿太阳能电池机器学习分析启动")
        print("="*60)

        try:
            self.load_data()
            targets = self.prepare_data()
            self.train_models(targets)
            self.visualize_results()
            self.print_summary()

            print("\n机器学习分析完成！")

        except Exception as e:
            print(f"错误: {str(e)}")
            raise


if __name__ == "__main__":
    # 创建机器学习模型
    ml_model = PerovskiteMLModel()

    # 运行完整分析
    ml_model.run_ml_pipeline()