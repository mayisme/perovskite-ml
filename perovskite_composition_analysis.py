"""
钙钛矿太阳能电池数据库 - 钙钛矿组成和添加剂详细分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

class PerovskiteCompositionAnalyzer:
    """
    钙钛矿组成和添加剂详细分析器
    """

    def __init__(self, data_path='Perovskite_database_content_all_data.csv'):
        self.data_path = data_path
        self.data = None
        self.composition_stats = {}

    def load_data(self):
        """加载数据"""
        print("正在加载钙钛矿数据库...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成！数据形状: {self.data.shape}")

    def analyze_ion_composition(self):
        """分析离子组成"""
        print("\n" + "="*60)
        print("离子组成详细分析")
        print("="*60)

        # A位离子详细分析
        self._analyze_a_site_ions()

        # B位离子详细分析
        self._analyze_b_site_ions()

        # C位离子详细分析
        self._analyze_c_site_ions()

        # 化学计量比分析
        self._analyze_stoichiometry()

    def _analyze_a_site_ions(self):
        """A位离子分析"""
        print("\n--- A位离子组成分析 ---")

        a_ions_col = 'Perovskite_composition_a_ions'
        a_coeff_col = 'Perovskite_composition_a_ions_coefficients'

        if a_ions_col in self.data.columns:
            a_ions_data = self.data[a_ions_col].dropna()
            print(f"A位离子数据量: {len(a_ions_data)}")

            # 解析离子和系数
            a_ion_compositions = self._parse_ion_compositions(a_ions_data, self.data.get(a_coeff_col))

            # 统计最常见的A位离子
            a_ion_counts = Counter()
            a_ion_concentrations = defaultdict(list)

            for composition in a_ion_compositions:
                for ion, conc in composition.items():
                    a_ion_counts[ion] += 1
                    if conc > 0:
                        a_ion_concentrations[ion].append(conc)

            print(f"\nA位离子种类统计 (前15个):")
            for ion, count in a_ion_counts.most_common(15):
                avg_conc = np.mean(a_ion_concentrations[ion]) if a_ion_concentrations[ion] else 0
                print("12")

            # 可视化A位离子分布
            self._plot_ion_distribution(a_ion_counts, "A位离子分布", "a_site_ions.png")

    def _analyze_b_site_ions(self):
        """B位离子分析"""
        print("\n--- B位离子组成分析 ---")

        b_ions_col = 'Perovskite_composition_b_ions'
        b_coeff_col = 'Perovskite_composition_b_ions_coefficients'

        if b_ions_col in self.data.columns:
            b_ions_data = self.data[b_ions_col].dropna()
            print(f"B位离子数据量: {len(b_ions_data)}")

            b_ion_compositions = self._parse_ion_compositions(b_ions_data, self.data.get(b_coeff_col))

            b_ion_counts = Counter()
            b_ion_concentrations = defaultdict(list)

            for composition in b_ion_compositions:
                for ion, conc in composition.items():
                    b_ion_counts[ion] += 1
                    if conc > 0:
                        b_ion_concentrations[ion].append(conc)

            print(f"\nB位离子种类统计 (前15个):")
            for ion, count in b_ion_counts.most_common(15):
                avg_conc = np.mean(b_ion_concentrations[ion]) if b_ion_concentrations[ion] else 0
                print("12")

            # 可视化B位离子分布
            self._plot_ion_distribution(b_ion_counts, "B位离子分布", "b_site_ions.png")

    def _analyze_c_site_ions(self):
        """C位离子分析"""
        print("\n--- C位离子组成分析 ---")

        c_ions_col = 'Perovskite_composition_c_ions'
        c_coeff_col = 'Perovskite_composition_c_ions_coefficients'

        if c_ions_col in self.data.columns:
            c_ions_data = self.data[c_ions_col].dropna()
            print(f"C位离子数据量: {len(c_ions_data)}")

            c_ion_compositions = self._parse_ion_compositions(c_ions_data, self.data.get(c_coeff_col))

            c_ion_counts = Counter()
            c_ion_concentrations = defaultdict(list)

            for composition in c_ion_compositions:
                for ion, conc in composition.items():
                    c_ion_counts[ion] += 1
                    if conc > 0:
                        c_ion_concentrations[ion].append(conc)

            print(f"\nC位离子种类统计 (前15个):")
            for ion, count in c_ion_counts.most_common(15):
                avg_conc = np.mean(c_ion_concentrations[ion]) if c_ion_concentrations[ion] else 0
                print("12")

            # 可视化C位离子分布
            self._plot_ion_distribution(c_ion_counts, "C位离子分布", "c_site_ions.png")

    def _parse_ion_compositions(self, ions_series, coeff_series=None):
        """解析离子组成数据"""
        compositions = []

        for idx, ions_str in ions_series.items():
            if not isinstance(ions_str, str):
                continue

            composition = {}

            # 分割多个离子
            ion_list = [ion.strip() for ion in ions_str.split(';') if ion.strip()]

            # 获取对应的系数
            coeff_list = []
            if coeff_series is not None and idx in coeff_series.index:
                coeff_str = coeff_series.loc[idx]
                if isinstance(coeff_str, str) and coeff_str.strip():
                    try:
                        coeff_list = [float(c.strip()) for c in coeff_str.split(';') if c.strip()]
                    except:
                        coeff_list = []

            # 配对离子和系数
            for i, ion in enumerate(ion_list):
                coeff = coeff_list[i] if i < len(coeff_list) else 1.0
                composition[ion] = coeff

            compositions.append(composition)

        return compositions

    def _plot_ion_distribution(self, ion_counts, title, filename):
        """绘制离子分布图"""
        plt.figure(figsize=(12, 8))

        # 取前15个最常见的离子
        top_ions = dict(ion_counts.most_common(15))

        plt.bar(range(len(top_ions)), list(top_ions.values()))
        plt.xticks(range(len(top_ions)), list(top_ions.keys()), rotation=45, ha='right')
        plt.xlabel('离子类型')
        plt.ylabel('出现次数')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_stoichiometry(self):
        """分析化学计量比"""
        print("\n--- 化学计量比分析 ---")

        # 分析A位系数分布
        if 'Perovskite_composition_a_ions_coefficients' in self.data.columns:
            a_coeffs = self.data['Perovskite_composition_a_ions_coefficients'].dropna()
            self._analyze_coefficient_distribution(a_coeffs, "A位离子系数分布", "a_coefficients.png")

        # 分析B位系数分布
        if 'Perovskite_composition_b_ions_coefficients' in self.data.columns:
            b_coeffs = self.data['Perovskite_composition_b_ions_coefficients'].dropna()
            self._analyze_coefficient_distribution(b_coeffs, "B位离子系数分布", "b_coefficients.png")

        # 分析C位系数分布
        if 'Perovskite_composition_c_ions_coefficients' in self.data.columns:
            c_coeffs = self.data['Perovskite_composition_c_ions_coefficients'].dropna()
            self._analyze_coefficient_distribution(c_coeffs, "C位离子系数分布", "c_coefficients.png")

    def _analyze_coefficient_distribution(self, coeff_series, title, filename):
        """分析系数分布"""
        all_coeffs = []

        for coeff_str in coeff_series:
            if isinstance(coeff_str, str):
                try:
                    coeffs = [float(c.strip()) for c in coeff_str.split(';') if c.strip()]
                    all_coeffs.extend(coeffs)
                except:
                    continue

        if all_coeffs:
            print(f"\n{title}:")
            print(f"  样本数: {len(all_coeffs)}")
            print(".3f")
            print(".3f")
            print(".3f")

            # 绘制分布图
            plt.figure(figsize=(10, 6))
            plt.hist(all_coeffs, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('系数')
            plt.ylabel('频次')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_additives(self):
        """分析钙钛矿添加剂"""
        print("\n" + "="*60)
        print("钙钛矿添加剂分析")
        print("="*60)

        # 添加剂化合物分析
        self._analyze_additive_compounds()

        # 添加剂浓度分析
        self._analyze_additive_concentrations()

    def _analyze_additive_compounds(self):
        """分析添加剂化合物"""
        print("\n--- 添加剂化合物分析 ---")

        additives_col = 'Perovskite_additives_compounds'

        if additives_col in self.data.columns:
            additives_data = self.data[additives_col].dropna()
            print(f"添加剂数据量: {len(additives_data)}")

            # 解析添加剂
            additive_counts = Counter()

            for additives_str in additives_data:
                if isinstance(additives_str, str):
                    # 分割多个添加剂
                    additive_list = [add.strip() for add in additives_str.split(';') if add.strip()]
                    for additive in additive_list:
                        additive_counts[additive] += 1

            print(f"\n最常见的添加剂化合物 (前20个):")
            for additive, count in additive_counts.most_common(20):
                percentage = count / len(additives_data) * 100
                print("25")

            # 可视化添加剂分布
            plt.figure(figsize=(14, 8))
            top_additives = dict(additive_counts.most_common(15))
            plt.bar(range(len(top_additives)), list(top_additives.values()))
            plt.xticks(range(len(top_additives)), list(top_additives.keys()), rotation=45, ha='right')
            plt.xlabel('添加剂化合物')
            plt.ylabel('出现次数')
            plt.title('钙钛矿添加剂化合物分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('perovskite_additives.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _analyze_additive_concentrations(self):
        """分析添加剂浓度"""
        print("\n--- 添加剂浓度分析 ---")

        conc_col = 'Perovskite_additives_concentrations'

        if conc_col in self.data.columns:
            conc_data = self.data[conc_col].dropna()
            print(f"添加剂浓度数据量: {len(conc_data)}")

            all_concentrations = []

            for conc_str in conc_data:
                if isinstance(conc_str, str):
                    try:
                        # 提取数值
                        conc_values = re.findall(r'[\d.]+', conc_str)
                        concentrations = [float(c) for c in conc_values]
                        all_concentrations.extend(concentrations)
                    except:
                        continue

            if all_concentrations:
                print(f"\n添加剂浓度统计:")
                print(f"  样本数: {len(all_concentrations)}")
                print(".3f")
                print(".3f")
                print(".3f")

                # 转换为对数尺度分析
                log_conc = [np.log10(c) for c in all_concentrations if c > 0]

                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.hist(all_concentrations, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('添加剂浓度')
                plt.ylabel('频次')
                plt.title('添加剂浓度分布')
                plt.grid(True, alpha=0.3)

                plt.subplot(1, 2, 2)
                plt.hist(log_conc, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('log₁₀(添加剂浓度)')
                plt.ylabel('频次')
                plt.title('添加剂浓度对数分布')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig('additive_concentrations.png', dpi=300, bbox_inches='tight')
                plt.close()

    def analyze_perovskite_properties(self):
        """分析钙钛矿材料性质"""
        print("\n" + "="*60)
        print("钙钛矿材料性质分析")
        print("="*60)

        # 厚度分析
        self._analyze_thickness()

        # 带隙分析
        self._analyze_bandgap()

        # 组成类型分析
        self._analyze_composition_types()

    def _analyze_thickness(self):
        """分析钙钛矿厚度"""
        print("\n--- 钙钛矿厚度分析 ---")

        thickness_col = 'Perovskite_thickness'

        if thickness_col in self.data.columns:
            thickness_data = self.data[thickness_col].dropna()
            print(f"厚度数据量: {len(thickness_data)}")

            # 过滤非数值数据
            numeric_thickness = pd.to_numeric(thickness_data, errors='coerce').dropna()
            print(f"数值厚度数据量: {len(numeric_thickness)}")

            if len(numeric_thickness) > 0:
                print(f"\n厚度统计 (nm):")
                print(".3f")
                print(".3f")
                print(".3f")

                # 绘制厚度分布
                plt.figure(figsize=(10, 6))
                plt.hist(numeric_thickness, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('钙钛矿厚度 (nm)')
                plt.ylabel('频次')
                plt.title('钙钛矿厚度分布')
                plt.grid(True, alpha=0.3)
                plt.savefig('perovskite_thickness.png', dpi=300, bbox_inches='tight')
                plt.close()

    def _analyze_bandgap(self):
        """分析带隙"""
        print("\n--- 钙钛矿带隙分析 ---")

        bandgap_col = 'Perovskite_band_gap'

        if bandgap_col in self.data.columns:
            bandgap_data = self.data[bandgap_col].dropna()
            print(f"带隙数据量: {len(bandgap_data)}")

            # 过滤非数值数据
            numeric_bandgap = pd.to_numeric(bandgap_data, errors='coerce').dropna()
            print(f"数值带隙数据量: {len(numeric_bandgap)}")

            if len(numeric_bandgap) > 0:
                print(f"\n带隙统计 (eV):")
                print(".3f")
                print(".3f")
                print(".3f")

                # 绘制带隙分布
                plt.figure(figsize=(10, 6))
                plt.hist(numeric_bandgap, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('带隙 (eV)')
                plt.ylabel('频次')
                plt.title('钙钛矿带隙分布')
                plt.grid(True, alpha=0.3)
                plt.savefig('perovskite_bandgap.png', dpi=300, bbox_inches='tight')
                plt.close()

    def _analyze_composition_types(self):
        """分析组成类型"""
        print("\n--- 钙钛矿组成类型分析 ---")

        type_cols = [
            'Perovskite_composition_inorganic',
            'Perovskite_composition_leadfree',
            'Perovskite_composition_short_form'
        ]

        for col in type_cols:
            if col in self.data.columns:
                if col == 'Perovskite_composition_short_form':
                    # 文本分析
                    short_forms = self.data[col].dropna()
                    if len(short_forms) > 0:
                        form_counts = Counter(short_forms)
                        print(f"\n钙钛矿简式组成 (前10个):")
                        for form, count in form_counts.most_common(10):
                            print("20")
                else:
                    # 布尔值分析
                    value_counts = self.data[col].value_counts()
                    print(f"\n{col}:")
                    for value, count in value_counts.items():
                        percentage = count / len(self.data) * 100
                        print("15")

    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n" + "="*60)
        print("生成钙钛矿组成和添加剂综合分析报告")
        print("="*60)

        self.load_data()
        self.analyze_ion_composition()
        self.analyze_additives()
        self.analyze_perovskite_properties()

        print("\n钙钛矿组成和添加剂分析完成！")
        print("生成的图表文件:")
        print("- a_site_ions.png: A位离子分布")
        print("- b_site_ions.png: B位离子分布")
        print("- c_site_ions.png: C位离子分布")
        print("- *_coefficients.png: 离子系数分布")
        print("- perovskite_additives.png: 添加剂化合物分布")
        print("- additive_concentrations.png: 添加剂浓度分布")
        print("- perovskite_thickness.png: 钙钛矿厚度分布")
        print("- perovskite_bandgap.png: 钙钛矿带隙分布")


if __name__ == "__main__":
    # 创建钙钛矿组成分析器
    analyzer = PerovskiteCompositionAnalyzer()

    # 生成综合分析报告
    analyzer.generate_comprehensive_report()