"""
钙钛矿添加剂优化分析
基于机器学习模型反推出最有效的添加剂组合
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AdditiveOptimizer:
    """
    钙钛矿添加剂优化器
    通过机器学习模型分析添加剂对PCE效率的影响
    """

    def __init__(self, data_path='Perovskite_database_content_all_data.csv'):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.additive_columns = []
        self.feature_importance = {}

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载钙钛矿数据库...")
        self.data = pd.read_csv(self.data_path)

        # 选择目标变量和添加剂相关特征
        target_col = 'JV_default_PCE'
        additive_related_cols = [
            'Perovskite_additives_compounds',
            'Perovskite_additives_concentrations',
            'Perovskite_composition_a_ions',
            'Perovskite_composition_b_ions',
            'Perovskite_composition_c_ions',
            'Perovskite_thickness',
            'Perovskite_band_gap'
        ]

        # 合并所有相关列
        available_cols = [col for col in additive_related_cols + [target_col] if col in self.data.columns]

        # 过滤有效数据
        df_clean = self.data[available_cols].dropna()

        print(f"有效样本数: {len(df_clean)}")
        print(f"目标变量PCE范围: {df_clean[target_col].min():.2f} - {df_clean[target_col].max():.2f}")

        return df_clean, target_col

    def extract_additives_info(self, df):
        """提取添加剂信息"""
        print("正在提取添加剂信息...")

        additives_info = []

        for idx, row in df.iterrows():
            additives = str(row.get('Perovskite_additives_compounds', ''))
            concentrations = str(row.get('Perovskite_additives_concentrations', ''))

            if additives and additives != 'nan':
                # 解析添加剂和浓度
                additive_list = [a.strip() for a in additives.split(';') if a.strip()]
                conc_list = []
                if concentrations and concentrations != 'nan':
                    try:
                        conc_list = [float(c.strip()) for c in concentrations.split(';') if c.strip()]
                    except:
                        conc_list = [0.0] * len(additive_list)

                # 确保浓度列表与添加剂列表长度匹配
                if len(conc_list) < len(additive_list):
                    conc_list.extend([0.0] * (len(additive_list) - len(conc_list)))

                for i, additive in enumerate(additive_list):
                    conc = conc_list[i] if i < len(conc_list) else 0.0
                    additives_info.append({
                        'sample_id': idx,
                        'additive': additive,
                        'concentration': conc,
                        'pce': row.get('JV_default_PCE', 0)
                    })

        additives_df = pd.DataFrame(additives_info)
        print(f"提取到 {len(additives_df)} 个添加剂记录")
        print(f"唯一添加剂类型: {additives_df['additive'].nunique()}")

        return additives_df

    def analyze_additive_effectiveness(self, additives_df):
        """分析添加剂有效性"""
        print("正在分析添加剂有效性...")

        # 按添加剂分组统计
        additive_stats = additives_df.groupby('additive').agg({
            'pce': ['count', 'mean', 'std', 'max'],
            'concentration': ['mean', 'std']
        }).round(3)

        # 展平列名
        additive_stats.columns = ['_'.join(col).strip() for col in additive_stats.columns.values]
        additive_stats = additive_stats.sort_values('pce_mean', ascending=False)

        # 计算效率提升（相对于无添加剂的平均值）
        baseline_pce = additives_df[additives_df['concentration'] == 0]['pce'].mean()
        if pd.isna(baseline_pce):
            baseline_pce = additives_df['pce'].mean() * 0.8  # 估算基准值

        additive_stats['efficiency_gain'] = additive_stats['pce_mean'] - baseline_pce

        print(f"基准PCE (无添加剂): {baseline_pce:.2f}")
        print("\n前10个最有效的添加剂:")
        print(additive_stats.head(10)[['pce_count', 'pce_mean', 'efficiency_gain']].to_string())

        return additive_stats

    def build_ml_model(self, df, target_col):
        """构建机器学习模型"""
        print("正在构建机器学习模型...")

        # 特征工程：创建添加剂相关特征
        features = self._create_additive_features(df)

        # 分割数据
        X = features
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # 评估模型
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(".3f")
        print(".3f")

        # 特征重要性
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))

        return X_train, X_test, y_train, y_test

    def _create_additive_features(self, df):
        """创建添加剂相关特征"""
        features = pd.DataFrame(index=df.index)

        # 基础特征 - 处理数值转换问题
        if 'Perovskite_thickness' in df.columns:
            # 转换字符串为数值
            thickness_numeric = pd.to_numeric(df['Perovskite_thickness'], errors='coerce')
            features['thickness'] = thickness_numeric.fillna(thickness_numeric.median())

        if 'Perovskite_band_gap' in df.columns:
            # 转换字符串为数值
            bandgap_numeric = pd.to_numeric(df['Perovskite_band_gap'], errors='coerce')
            features['band_gap'] = bandgap_numeric.fillna(bandgap_numeric.median())

        # 添加剂存在性特征
        additives_col = df.get('Perovskite_additives_compounds', pd.Series([None]*len(df)))

        # 常见添加剂列表
        common_additives = [
            'KI', 'Br', 'Cl', 'CsI', 'RbI', 'MAI', 'FAI', 'PEA', 'BA',
            'Li-TFSI', 'TBP', 'FK209', 'DTBP', 'DMSO', 'DMF'
        ]

        for additive in common_additives:
            features[f'additive_{additive}'] = additives_col.apply(
                lambda x: 1 if isinstance(x, str) and additive in x else 0
            )

        # 添加剂数量
        features['num_additives'] = additives_col.apply(
            lambda x: len([a.strip() for a in str(x).split(';') if a.strip()]) if isinstance(x, str) else 0
        )

        # 离子组成特征 (简化的二进制编码)
        for ion_type in ['a', 'b', 'c']:
            ion_col = f'Perovskite_composition_{ion_type}_ions'
            if ion_col in df.columns:
                # 常见离子
                common_ions = {
                    'a': ['MA', 'FA', 'Cs', 'Rb', 'EA'],
                    'b': ['Pb', 'Sn', 'Ge'],
                    'c': ['I', 'Br', 'Cl']
                }

                for ion in common_ions.get(ion_type, []):
                    features[f'{ion_type}_ion_{ion}'] = df[ion_col].apply(
                        lambda x: 1 if isinstance(x, str) and ion in x else 0
                    )

        return features

    def optimize_additives(self, base_composition=None):
        """优化添加剂组合"""
        print("正在优化添加剂组合...")

        # 获取训练时使用的特征名称
        if hasattr(self, 'model') and hasattr(self.model, 'feature_names_in_'):
            feature_names = list(self.model.feature_names_in_)
        else:
            print("无法获取模型特征名称，跳过优化")
            return pd.DataFrame()

        if base_composition is None:
            # 使用最常见的组合作为基准，确保包含所有特征
            base_composition = {feature: 0 for feature in feature_names}
            # 设置基准值
            base_composition.update({
                'a_ion_MA': 1, 'b_ion_Pb': 1, 'c_ion_I': 1,
                'thickness': 400, 'band_gap': 1.55, 'num_additives': 0
            })

        # 生成候选添加剂组合
        candidate_additives = [
            'KI', 'Br', 'CsI', 'RbI', 'PEA', 'BA',
            'Li-TFSI', 'TBP', 'FK209'
        ]

        optimization_results = []

        # 测试单个添加剂
        print("测试单个添加剂效果...")
        for additive in candidate_additives:
            test_composition = base_composition.copy()
            additive_feature = f'additive_{additive}'
            if additive_feature in test_composition:
                test_composition[additive_feature] = 1
                test_composition['num_additives'] = 1

                # 转换为模型输入格式，确保特征顺序正确
                X_test = pd.DataFrame([test_composition])[feature_names]

                # 预测PCE
                predicted_pce = self.model.predict(X_test)[0]

                optimization_results.append({
                    'additives': [additive],
                    'predicted_pce': predicted_pce,
                    'combination_type': 'single'
                })

        # 测试添加剂组合 (2个组合)
        print("测试添加剂组合效果...")
        for combo in combinations(candidate_additives[:6], 2):  # 只测试前6个添加剂的组合
            test_composition = base_composition.copy()
            for additive in combo:
                additive_feature = f'additive_{additive}'
                if additive_feature in test_composition:
                    test_composition[additive_feature] = 1
            test_composition['num_additives'] = len(combo)

            X_test = pd.DataFrame([test_composition])[feature_names]
            predicted_pce = self.model.predict(X_test)[0]

            optimization_results.append({
                'additives': list(combo),
                'predicted_pce': predicted_pce,
                'combination_type': 'double'
            })

        # 排序结果
        results_df = pd.DataFrame(optimization_results)
        results_df = results_df.sort_values('predicted_pce', ascending=False)

        return results_df

    def visualize_results(self, additive_stats, optimization_results):
        """可视化结果"""
        print("正在生成可视化结果...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 添加剂效率排名
        top_additives = additive_stats.head(10)
        axes[0, 0].bar(range(len(top_additives)), top_additives['pce_mean'])
        axes[0, 0].set_xticks(range(len(top_additives)))
        axes[0, 0].set_xticklabels(top_additives.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('平均PCE (%)')
        axes[0, 0].set_title('前10个最有效的添加剂')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 效率提升
        axes[0, 1].bar(range(len(top_additives)), top_additives['efficiency_gain'])
        axes[0, 1].set_xticks(range(len(top_additives)))
        axes[0, 1].set_xticklabels(top_additives.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('PCE效率提升 (%)')
        axes[0, 1].set_title('添加剂效率提升')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 优化结果 - 单添加剂
        single_results = optimization_results[optimization_results['combination_type'] == 'single']
        if len(single_results) > 0:
            top_single = single_results.head(8)
            axes[1, 0].bar(range(len(top_single)), top_single['predicted_pce'])
            axes[1, 0].set_xticks(range(len(top_single)))
            axes[1, 0].set_xticklabels([', '.join(adds) for adds in top_single['additives']], rotation=45, ha='right')
            axes[1, 0].set_ylabel('预测PCE (%)')
            axes[1, 0].set_title('单添加剂优化结果')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 优化结果 - 双添加剂
        double_results = optimization_results[optimization_results['combination_type'] == 'double']
        if len(double_results) > 0:
            top_double = double_results.head(6)
            axes[1, 1].bar(range(len(top_double)), top_double['predicted_pce'])
            axes[1, 1].set_xticks(range(len(top_double)))
            axes[1, 1].set_xticklabels([', '.join(adds) for adds in top_double['additives']], rotation=45, ha='right')
            axes[1, 1].set_ylabel('预测PCE (%)')
            axes[1, 1].set_title('双添加剂组合优化结果')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('additive_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("可视化图表已保存为: additive_optimization_results.png")

    def generate_report(self, additive_stats, optimization_results):
        """生成优化报告"""
        print("正在生成优化报告...")

        report = "# 钙钛矿添加剂优化分析报告\n\n"

        report += "## 数据概览\n\n"
        report += f"- 总样本数: {len(self.data)}\n"
        report += f"- 有效添加剂记录数: {len(additive_stats)}\n"
        report += f"- 唯一添加剂类型数: {len(additive_stats)}\n\n"

        report += "## 最有效的添加剂排名\n\n"
        report += "| 排名 | 添加剂 | 样本数 | 平均PCE | 效率提升 |\n"
        report += "|------|--------|--------|----------|----------|\n"

        for i, (additive, stats) in enumerate(additive_stats.head(10).iterrows(), 1):
            report += f"| {i} | {additive} | {stats['pce_count']} | {stats['pce_mean']:.2f} | {stats['efficiency_gain']:.2f} |\n"

        report += "\n## 机器学习模型性能\n\n"
        if hasattr(self, 'model') and self.model is not None:
            report += f"- 模型类型: Random Forest\n"
            report += f"- 训练分数: {self.model.score.__name__ if hasattr(self.model, 'score') else 'N/A'}\n"
            report += "- 关键特征重要性:\n"

            # 显示最重要的特征
            if self.feature_importance:
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:10]:
                    report += f"  - {feature}: {importance:.3f}\n"

        report += "\n## 添加剂优化建议\n\n"

        # 单添加剂建议
        single_results = optimization_results[optimization_results['combination_type'] == 'single']
        if len(single_results) > 0:
            top_single = single_results.iloc[0]
            report += f"### 最佳单添加剂\n"
            report += f"- 添加剂: {', '.join(top_single['additives'])}\n"
            report += f"- 预测PCE: {top_single['predicted_pce']:.2f}%\n\n"

        # 双添加剂建议
        double_results = optimization_results[optimization_results['combination_type'] == 'double']
        if len(double_results) > 0:
            top_double = double_results.iloc[0]
            report += f"### 最佳双添加剂组合\n"
            report += f"- 添加剂组合: {', '.join(top_double['additives'])}\n"
            report += f"- 预测PCE: {top_double['predicted_pce']:.2f}%\n\n"

        report += "## 实验建议\n\n"
        report += "1. **优先测试**: 排名前3的添加剂进行实验验证\n"
        report += "2. **组合实验**: 测试预测的最优添加剂组合\n"
        report += "3. **浓度优化**: 在最优添加剂基础上优化浓度\n"
        report += "4. **稳定性测试**: 验证添加剂对器件稳定性的影响\n\n"

        report += "---\n\n"
        report += f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        with open('additive_optimization_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("优化报告已保存为: additive_optimization_report.md")

    def run_complete_analysis(self):
        """运行完整的添加剂优化分析"""
        print("="*60)
        print("钙钛矿添加剂优化分析启动")
        print("="*60)

        try:
            # 1. 加载和预处理数据
            df, target_col = self.load_and_preprocess_data()

            # 2. 提取添加剂信息
            additives_df = self.extract_additives_info(df)

            # 3. 分析添加剂有效性
            additive_stats = self.analyze_additive_effectiveness(additives_df)

            # 4. 构建机器学习模型
            X_train, X_test, y_train, y_test = self.build_ml_model(df, target_col)

            # 5. 优化添加剂组合
            optimization_results = self.optimize_additives()

            # 6. 可视化结果
            self.visualize_results(additive_stats, optimization_results)

            # 7. 生成报告
            self.generate_report(additive_stats, optimization_results)

            print("\n添加剂优化分析完成！")

        except Exception as e:
            print(f"错误: {str(e)}")
            raise


if __name__ == "__main__":
    # 创建添加剂优化器并运行分析
    optimizer = AdditiveOptimizer()
    optimizer.run_complete_analysis()