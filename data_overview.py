"""
钙钛矿太阳能电池数据库数据概览分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PerovskiteDataOverview:
    """
    钙钛矿太阳能电池数据库数据概览分析
    """

    def __init__(self, data_path='Perovskite_database_content_all_data.csv'):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """加载数据"""
        print("正在加载钙钛矿数据库...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成！数据形状: {self.data.shape}")
        print(f"列数: {self.data.shape[1]}, 行数: {self.data.shape[0]}")

    def basic_info(self):
        """基本信息统计"""
        print("\n" + "="*60)
        print("数据库基本信息")
        print("="*60)

        print(f"总样本数: {len(self.data)}")
        print(f"总特征数: {len(self.data.columns)}")

        # 数据类型统计
        dtype_counts = self.data.dtypes.value_counts()
        print(f"\n数据类型分布:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} 列")

        # 缺失值统计
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100

        print(f"\n缺失值统计:")
        print(f"  完全无缺失的列: {(missing_data == 0).sum()}")
        print(f"  有缺失值的列: {(missing_data > 0).sum()}")
        print(".1f")
        print(".1f")

    def target_analysis(self):
        """目标变量分析"""
        print("\n" + "="*60)
        print("目标变量分析 (PCE效率)")
        print("="*60)

        # PCE相关列
        pce_columns = [col for col in self.data.columns if 'PCE' in col and 'JV' in col]
        print(f"PCE相关列: {pce_columns}")

        for col in pce_columns:
            if col in self.data.columns:
                valid_data = self.data[col].dropna()
                if len(valid_data) > 0:
                    print(f"\n{col}:")
                    print(f"  有效样本数: {len(valid_data)}")
                    print(".3f")
                    print(".3f")
                    print(".3f")
                    print(".3f")

                    # 分布可视化
                    plt.figure(figsize=(10, 6))
                    plt.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
                    plt.xlabel(f'{col} (%)')
                    plt.ylabel('频次')
                    plt.title(f'{col} 分布直方图')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f'{col}_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()

    def feature_analysis(self):
        """特征分析"""
        print("\n" + "="*60)
        print("特征分析")
        print("="*60)

        # 按类别分组特征
        categories = {
            '参考文献': [col for col in self.data.columns if col.startswith('Ref_')],
            '电池': [col for col in self.data.columns if col.startswith('Cell_')],
            '模块': [col for col in self.data.columns if col.startswith('Module_')],
            '基板': [col for col in self.data.columns if col.startswith('Substrate_')],
            'ETL': [col for col in self.data.columns if col.startswith('ETL_')],
            '钙钛矿': [col for col in self.data.columns if col.startswith('Perovskite_')],
            'HTL': [col for col in self.data.columns if col.startswith('HTL_')],
            '背接触': [col for col in self.data.columns if col.startswith('Backcontact_')],
            '附加层': [col for col in self.data.columns if col.startswith('Add_lay_')],
            '封装': [col for col in self.data.columns if col.startswith('Encapsulation_')],
            'JV特性': [col for col in self.data.columns if col.startswith('JV_')],
            '稳定性能': [col for col in self.data.columns if col.startswith('Stabilised_performance_')],
            'EQE': [col for col in self.data.columns if col.startswith('EQE_')],
            '稳定性': [col for col in self.data.columns if col.startswith('Stability_')],
            '户外测试': [col for col in self.data.columns if col.startswith('Outdoor_')]
        }

        print("特征类别分布:")
        for category, cols in categories.items():
            if cols:
                total_missing = sum(self.data[col].isnull().sum() for col in cols)
                total_values = len(cols) * len(self.data)
                missing_rate = total_missing / total_values * 100 if total_values > 0 else 0
                print("20")

        # 数值特征统计
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(f"\n数值特征数量: {len(numeric_cols)}")

        if len(numeric_cols) > 0:
            # 数值特征的基本统计
            numeric_stats = self.data[numeric_cols].describe()
            print(f"\n数值特征统计摘要 (前10列):")
            print(numeric_stats.iloc[:, :10])

    def perovskite_composition_analysis(self):
        """钙钛矿组成分析"""
        print("\n" + "="*60)
        print("钙钛矿组成分析")
        print("="*60)

        # A位离子
        if 'Perovskite_composition_a_ions' in self.data.columns:
            a_ions = self.data['Perovskite_composition_a_ions'].dropna()
            if len(a_ions) > 0:
                a_ion_counts = Counter()
                for ions in a_ions:
                    if isinstance(ions, str):
                        # 分割多个离子
                        ion_list = [ion.strip() for ion in ions.split(';')]
                        for ion in ion_list:
                            a_ion_counts[ion] += 1

                print(f"A位离子分布 (前10个):")
                for ion, count in a_ion_counts.most_common(10):
                    print("8")

        # B位离子
        if 'Perovskite_composition_b_ions' in self.data.columns:
            b_ions = self.data['Perovskite_composition_b_ions'].dropna()
            if len(b_ions) > 0:
                b_ion_counts = Counter()
                for ions in b_ions:
                    if isinstance(ions, str):
                        ion_list = [ion.strip() for ion in ions.split(';')]
                        for ion in ion_list:
                            b_ion_counts[ion] += 1

                print(f"\nB位离子分布 (前10个):")
                for ion, count in b_ion_counts.most_common(10):
                    print("8")

        # C位离子
        if 'Perovskite_composition_c_ions' in self.data.columns:
            c_ions = self.data['Perovskite_composition_c_ions'].dropna()
            if len(c_ions) > 0:
                c_ion_counts = Counter()
                for ions in c_ions:
                    if isinstance(ions, str):
                        ion_list = [ion.strip() for ion in ions.split(';')]
                        for ion in ion_list:
                            c_ion_counts[ion] += 1

                print(f"\nC位离子分布 (前10个):")
                for ion, count in c_ion_counts.most_common(10):
                    print("8")

    def correlation_analysis(self):
        """相关性分析"""
        print("\n" + "="*60)
        print("关键特征相关性分析")
        print("="*60)

        # 选择关键数值特征
        key_features = []
        pce_cols = [col for col in self.data.columns if 'PCE' in col and 'JV' in col]
        key_features.extend(pce_cols)

        # 添加钙钛矿相关特征
        perovskite_features = [col for col in self.data.columns if col.startswith('Perovskite_') and
                              self.data[col].dtype in ['int64', 'float64']]
        key_features.extend(perovskite_features[:10])  # 取前10个

        # 计算相关性
        if len(key_features) > 1:
            corr_data = self.data[key_features].select_dtypes(include=[np.number])
            if not corr_data.empty:
                correlation_matrix = corr_data.corr()

                # 显示与PCE最相关的特征
                if pce_cols:
                    pce_col = pce_cols[0]  # 选择第一个PCE列
                    if pce_col in correlation_matrix.columns:
                        pce_corr = correlation_matrix[pce_col].abs().sort_values(ascending=False)
                        print(f"与 {pce_col} 最相关的特征 (前10个):")
                        for feature, corr in pce_corr.head(10).items():
                            print("20")

    def generate_report(self):
        """生成完整报告"""
        print("\n" + "="*60)
        print("生成数据概览报告")
        print("="*60)

        self.load_data()
        self.basic_info()
        self.target_analysis()
        self.feature_analysis()
        self.perovskite_composition_analysis()
        self.correlation_analysis()

        print("\n数据概览分析完成！")
        print("生成的图表文件:")
        print("- *_distribution.png: PCE分布直方图")


if __name__ == "__main__":
    # 创建数据概览分析器
    analyzer = PerovskiteDataOverview()

    # 生成完整报告
    analyzer.generate_report()