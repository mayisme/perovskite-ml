"""
钙钛矿太阳能电池数据库 - 462列数据分类分析
生成详细的分类报告和Markdown文档
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import warnings
warnings.filterwarnings('ignore')

class ColumnClassifier:
    """
    数据库列分类器
    """

    def __init__(self, data_path='Perovskite_database_content_all_data.csv'):
        self.data_path = data_path
        self.data = None
        self.column_categories = {}
        self.category_stats = {}

    def load_data(self):
        """加载数据"""
        print("正在加载钙钛矿数据库...")
        self.data = pd.read_csv(self.data_path)
        print(f"数据加载完成！总列数: {len(self.data.columns)}")

    def classify_columns(self):
        """对所有列进行分类"""
        print("正在对462列数据进行分类...")

        # 定义分类规则
        category_rules = {
            '参考文献信息': [
                'Ref_ID', 'Ref_ID_temp', 'Ref_name_of_person_entering_the_data',
                'Ref_data_entered_by_author', 'Ref_DOI_number', 'Ref_lead_author',
                'Ref_publication_date', 'Ref_journal', 'Ref_part_of_initial_dataset',
                'Ref_original_filename_data_upload', 'Ref_free_text_comment',
                'Ref_internal_sample_id'
            ],
            '电池基本信息': [
                'Cell_stack_sequence', 'Cell_area_total', 'Cell_area_measured',
                'Cell_number_of_cells_per_substrate', 'Cell_architecture',
                'Cell_flexible', 'Cell_flexible_min_bending_radius',
                'Cell_semitransparent', 'Cell_semitransparent_AVT',
                'Cell_semitransparent_wavelength_range', 'Cell_semitransparent_raw_data'
            ],
            '模块信息': [
                'Module', 'Module_number_of_cells_in_module', 'Module_area_total',
                'Module_area_effective', 'Module_JV_data_recalculated_per_cell'
            ],
            '基板信息': [
                'Substrate_stack_sequence', 'Substrate_thickness', 'Substrate_area',
                'Substrate_supplier', 'Substrate_brand_name', 'Substrate_deposition_procedure',
                'Substrate_surface_roughness_rms', 'Substrate_etching_procedure',
                'Substrate_cleaning_procedure'
            ],
            '电子传输层': [
                'ETL_stack_sequence', 'ETL_thickness', 'ETL_additives_compounds',
                'ETL_additives_concentrations', 'ETL_deposition_procedure',
                'ETL_deposition_aggregation_state_of_reactants', 'ETL_deposition_synthesis_atmosphere',
                'ETL_deposition_synthesis_atmosphere_pressure_total',
                'ETL_deposition_synthesis_atmosphere_pressure_partial',
                'ETL_deposition_synthesis_atmosphere_relative_humidity', 'ETL_deposition_solvents',
                'ETL_deposition_solvents_mixing_ratios', 'ETL_deposition_solvents_supplier',
                'ETL_deposition_solvents_purity', 'ETL_deposition_reaction_solutions_compounds',
                'ETL_deposition_reaction_solutions_compounds_supplier',
                'ETL_deposition_reaction_solutions_compounds_purity',
                'ETL_deposition_reaction_solutions_concentrations',
                'ETL_deposition_reaction_solutions_volumes', 'ETL_deposition_reaction_solutions_age',
                'ETL_deposition_reaction_solutions_temperature', 'ETL_deposition_substrate_temperature',
                'ETL_deposition_thermal_annealing_temperature', 'ETL_deposition_thermal_annealing_time',
                'ETL_deposition_thermal_annealing_atmosphere', 'ETL_storage_time_until_next_deposition_step',
                'ETL_storage_atmosphere', 'ETL_storage_relative_humidity',
                'ETL_surface_treatment_before_next_deposition_step'
            ],
            '钙钛矿层': [
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
                'Perovskite_deaction_solutions_concentrations',
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
            ],
            '空穴传输层': [
                'HTL_stack_sequence', 'HTL_thickness_list', 'HTL_additives_compounds',
                'HTL_additives_concentrations', 'HTL_deposition_procedure',
                'HTL_deposition_aggregation_state_of_reactants', 'HTL_deposition_synthesis_atmosphere',
                'HTL_deposition_synthesis_atmosphere_pressure_total', 'HTL_deposition_synthesis_atmosphere_pressure_partial',
                'HTL_deposition_synthesis_atmosphere_relative_humidity', 'HTL_deposition_solvents',
                'HTL_deposition_solvents_mixing_ratios', 'HTL_deposition_solvents_supplier',
                'HTL_deposition_solvents_purity', 'HTL_deposition_reaction_solutions_compounds',
                'HTL_deposition_reaction_solutions_compounds_supplier', 'HTL_deposition_reaction_solutions_compounds_purity',
                'HTL_deposition_reaction_solutions_concentrations', 'HTL_deposition_reaction_solutions_volumes',
                'HTL_deposition_reaction_solutions_age', 'HTL_deposition_reaction_solutions_temperature',
                'HTL_deposition_substrate_temperature', 'HTL_deposition_thermal_annealing_temperature',
                'HTL_deposition_thermal_annealing_time', 'HTL_deposition_thermal_annealing_atmosphere',
                'HTL_storage_time_until_next_deposition_step', 'HTL_storage_atmosphere',
                'HTL_storage_relative_humidity', 'HTL_surface_treatment_before_next_deposition_step'
            ],
            '背接触': [
                'Backcontact_stack_sequence', 'Backcontact_thickness_list', 'Backcontact_additives_compounds',
                'Backcontact_additives_concentrations', 'Backcontact_deposition_procedure',
                'Backcontact_deposition_aggregation_state_of_reactants', 'Backcontact_deposition_synthesis_atmosphere',
                'Backcontact_deposition_synthesis_atmosphere_pressure_total', 'Backcontact_deposition_synthesis_atmosphere_pressure_partial',
                'Backcontact_deposition_synthesis_atmosphere_relative_humidity', 'Backcontact_deposition_solvents',
                'Backcontact_deposition_solvents_mixing_ratios', 'Backcontact_deposition_solvents_supplier',
                'Backcontact_deposition_solvents_purity', 'Backcontact_deposition_reaction_solutions_compounds',
                'Backcontact_deposition_reaction_solutions_compounds_supplier', 'Backcontact_deposition_reaction_solutions_compounds_purity',
                'Backcontact_deposition_reaction_solutions_concentrations', 'Backcontact_deposition_reaction_solutions_volumes',
                'Backcontact_deposition_reaction_solutions_age', 'Backcontact_deposition_reaction_solutions_temperature',
                'Backcontact_deposition_substrate_temperature', 'Backcontact_deposition_thermal_annealing_temperature',
                'Backcontact_deposition_thermal_annealing_time', 'Backcontact_deposition_thermal_annealing_atmosphere',
                'Backcontact_storage_time_until_next_deposition_step', 'Backcontact_storage_atmosphere',
                'Backcontact_storage_relative_humidity', 'Backcontact_surface_treatment_before_next_deposition_step'
            ],
            '附加层': [
                'Add_lay_front', 'Add_lay_front_function', 'Add_lay_front_stack_sequence',
                'Add_lay_front_thickness_list', 'Add_lay_front_additives_compounds',
                'Add_lay_front_additives_concentrations', 'Add_lay_front_deposition_procedure',
                'Add_lay_front_deposition_aggregation_state_of_reactants', 'Add_lay_front_deposition_synthesis_atmosphere',
                'Add_lay_front_deposition_synthesis_atmosphere_pressure_total',
                'Add_lay_front_deposition_synthesis_atmosphere_pressure_partial',
                'Add_lay_front_deposition_synthesis_atmosphere_relative_humidity', 'Add_lay_front_deposition_solvents',
                'Add_lay_front_deposition_solvents_mixing_ratios', 'Add_lay_front_deposition_solvents_supplier',
                'Add_lay_front_deposition_solvents_purity', 'Add_lay_front_deposition_reaction_solutions_compounds',
                'Add_lay_front_deposition_reaction_solutions_compounds_supplier',
                'Add_lay_front_deposition_reaction_solutions_compounds_purity',
                'Add_lay_front_deposition_reaction_solutions_concentrations',
                'Add_lay_front_deposition_reaction_solutions_volumes', 'Add_lay_front_deposition_reaction_solutions_age',
                'Add_lay_front_deposition_reaction_solutions_temperature', 'Add_lay_front_deposition_substrate_temperature',
                'Add_lay_front_deposition_thermal_annealing_temperature', 'Add_lay_front_deposition_thermal_annealing_time',
                'Add_lay_front_deposition_thermal_annealing_atmosphere', 'Add_lay_front_storage_time_until_next_deposition_step',
                'Add_lay_front_storage_atmosphere', 'Add_lay_front_storage_relative_humidity',
                'Add_lay_front_surface_treatment_before_next_deposition_step', 'Add_lay_back', 'Add_lay_back_function',
                'Add_lay_back_stack_sequence', 'Add_lay_back_thickness_list', 'Add_lay_back_additives_compounds',
                'Add_lay_back_additives_concentrations', 'Add_lay_back_deposition_procedure',
                'Add_lay_back_deposition_aggregation_state_of_reactants', 'Add_lay_back_deposition_synthesis_atmosphere',
                'Add_lay_back_deposition_synthesis_atmosphere_pressure_total',
                'Add_lay_back_deposition_synthesis_atmosphere_pressure_partial',
                'Add_lay_back_deposition_synthesis_atmosphere_relative_humidity', 'Add_lay_back_deposition_solvents',
                'Add_lay_back_deposition_solvents_mixing_ratios', 'Add_lay_back_deposition_solvents_supplier',
                'Add_lay_back_deposition_solvents_purity', 'Add_lay_back_deposition_reaction_solutions_compounds',
                'Add_lay_back_deposition_reaction_solutions_compounds_supplier',
                'Add_lay_back_deposition_reaction_solutions_compounds_purity',
                'Add_lay_back_deposition_reaction_solutions_concentrations',
                'Add_lay_back_deposition_reaction_solutions_volumes', 'Add_lay_back_deposition_reaction_solutions_age',
                'Add_lay_back_deposition_reaction_solutions_temperature', 'Add_lay_back_deposition_substrate_temperature',
                'Add_lay_back_deposition_thermal_annealing_temperature', 'Add_lay_back_deposition_thermal_annealing_time',
                'Add_lay_back_deposition_thermal_annealing_atmosphere', 'Add_lay_back_storage_time_until_next_deposition_step',
                'Add_lay_back_storage_atmosphere', 'Add_lay_back_storage_relative_humidity',
                'Add_lay_back_surface_treatment_before_next_deposition_step'
            ],
            '封装': [
                'Encapsulation', 'Encapsulation_stack_sequence', 'Encapsulation_edge_sealing_materials',
                'Encapsulation_atmosphere_for_encapsulation', 'Encapsulation_water_vapour_transmission_rate',
                'Encapsulation_oxygen_transmission_rate'
            ],
            'JV特性测量': [
                'JV_measured', 'JV_average_over_n_number_of_cells', 'JV_certified_values',
                'JV_certification_institute', 'JV_storage_age_of_cell', 'JV_storage_atmosphere',
                'JV_storage_relative_humidity', 'JV_test_atmosphere', 'JV_test_relative_humidity',
                'JV_test_temperature', 'JV_light_source_type', 'JV_light_source_brand_name',
                'JV_light_source_simulator_class', 'JV_light_intensity', 'JV_light_spectra',
                'JV_light_wavelength_range', 'JV_light_illumination_direction', 'JV_light_masked_cell',
                'JV_light_mask_area', 'JV_scan_speed', 'JV_scan_delay_time', 'JV_scan_integration_time',
                'JV_scan_voltage_step', 'JV_preconditioning_protocol', 'JV_preconditioning_time',
                'JV_preconditioning_potential', 'JV_preconditioning_light_intensity', 'JV_reverse_scan_Voc',
                'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_reverse_scan_PCE', 'JV_reverse_scan_Vmp',
                'JV_reverse_scan_Jmp', 'JV_reverse_scan_series_resistance', 'JV_reverse_scan_shunt_resistance',
                'JV_forward_scan_Voc', 'JV_forward_scan_Jsc', 'JV_forward_scan_FF', 'JV_forward_scan_PCE',
                'JV_forward_scan_Vmp', 'JV_forward_scan_Jmp', 'JV_forward_scan_series_resistance',
                'JV_forward_scan_shunt_resistance', 'JV_link_raw_data', 'JV_default_Voc', 'JV_default_Jsc',
                'JV_default_FF', 'JV_default_PCE', 'JV_default_Voc_scan_direction', 'JV_default_Jsc_scan_direction',
                'JV_default_FF_scan_direction', 'JV_default_PCE_scan_direction', 'JV_hysteresis_index'
            ],
            '稳定性能': [
                'Stabilised_performance_measured', 'Stabilised_performance_procedure',
                'Stabilised_performance_procedure_metrics', 'Stabilised_performance_measurement_time',
                'Stabilised_performance_PCE', 'Stabilised_performance_Vmp', 'Stabilised_performance_Jmp',
                'Stabilised_performance_link_raw_data'
            ],
            'EQE': [
                'EQE_measured', 'EQE_light_bias', 'EQE_integrated_Jsc', 'EQE_link_raw_data'
            ],
            '稳定性测试': [
                'Stability_measured', 'Stability_protocol', 'Stability_average_over_n_number_of_cells',
                'Stability_light_source_type', 'Stability_light_source_brand_name', 'Stability_light_source_simulator_class',
                'Stability_light_intensity', 'Stability_light_spectra', 'Stability_light_wavelength_range',
                'Stability_light_illumination_direction', 'Stability_light_load_condition', 'Stability_light_cycling_times',
                'Stability_light_UV_filter', 'Stability_potential_bias_load_condition', 'Stability_potential_bias_range',
                'Stability_potential_bias_passive_resistance', 'Stability_temperature_load_condition',
                'Stability_temperature_range', 'Stability_temperature_cycling_times', 'Stability_temperature_ramp_speed',
                'Stability_atmosphere', 'Stability_atmosphere_oxygen_concentration', 'Stability_relative_humidity_load_conditions',
                'Stability_relative_humidity_range', 'Stability_relative_humidity_average_value', 'Stability_time_total_exposure',
                'Stability_periodic_JV_measurements', 'Stability_periodic_JV_measurements_time_between_jv',
                'Stability_PCE_initial_value', 'Stability_PCE_burn_in_observed', 'Stability_PCE_end_of_experiment',
                'Stability_PCE_T95', 'Stability_PCE_Ts95', 'Stability_PCE_T80', 'Stability_PCE_Ts80',
                'Stability_PCE_Te80', 'Stability_PCE_Tse80', 'Stability_PCE_after_1000_h', 'Stability_lifetime_energy_yield',
                'Stability_flexible_cell_number_of_bending_cycles', 'Stability_flexible_cell_bending_radius',
                'Stability_flexible_cell_PCE_initial_value', 'Stability_flexible_cell_PCE_end_of_experiment',
                'Stability_link_raw_data_for_stability_trace'
            ],
            '户外测试': [
                'Outdoor_tested', 'Outdoor_protocol', 'Outdoor_average_over_n_number_of_cells',
                'Outdoor_location_country', 'Outdoor_location_city', 'Outdoor_location_coordinates',
                'Outdoor_location_climate_zone', 'Outdoor_installation_tilt', 'Outdoor_installation_cardinal_direction',
                'Outdoor_installation_number_of_solar_tracking_axis', 'Outdoor_time_season', 'Outdoor_time_start',
                'Outdoor_time_end', 'Outdoor_time_total_exposure', 'Outdoor_potential_bias_load_condition',
                'Outdoor_potential_bias_range', 'Outdoor_potential_bias_passive_resistance',
                'Outdoor_temperature_load_condition', 'Outdoor_temperature_range', 'Outdoor_temperature_tmodule',
                'Outdoor_periodic_JV_measurements', 'Outdoor_periodic_JV_measurements_time_between_measurements',
                'Outdoor_PCE_initial_value', 'Outdoor_PCE_burn_in_observed', 'Outdoor_PCE_end_of_experiment',
                'Outdoor_PCE_T95', 'Outdoor_PCE_Ts95', 'Outdoor_PCE_T80', 'Outdoor_PCE_Ts80',
                'Outdoor_PCE_Te80', 'Outdoor_PCE_Tse80', 'Outdoor_PCE_after_1000_h', 'Outdoor_power_generated',
                'Outdoor_link_raw_data_for_outdoor_trace', 'Outdoor_detaild_weather_data_available',
                'Outdoor_link_detailed_weather_data', 'Outdoor_spectral_data_available', 'Outdoor_link_spectral_data',
                'Outdoor_irradiance_measured', 'Outdoor_link_irradiance_data'
            ]
        }

        # 初始化分类字典
        self.column_categories = defaultdict(list)

        # 手动分类
        for category, columns in category_rules.items():
            for col in columns:
                if col in self.data.columns:
                    self.column_categories[category].append(col)

        # 自动分类未分类的列
        classified_cols = set()
        for cols in self.column_categories.values():
            classified_cols.update(cols)

        unclassified_cols = [col for col in self.data.columns if col not in classified_cols]

        if unclassified_cols:
            print(f"发现 {len(unclassified_cols)} 个未分类列:")
            for col in unclassified_cols[:10]:  # 只显示前10个
                print(f"  {col}")
            if len(unclassified_cols) > 10:
                print(f"  ... 还有 {len(unclassified_cols) - 10} 个")

        print(f"\n成功分类 {len(classified_cols)} 列，占总列数 {len(self.data.columns)} 的 {len(classified_cols)/len(self.data.columns)*100:.1f}%")

    def analyze_categories(self):
        """分析各分类的统计信息"""
        print("正在分析各分类统计信息...")

        self.category_stats = {}

        for category, columns in self.column_categories.items():
            stats = {
                'column_count': len(columns),
                'total_samples': len(self.data),
                'missing_stats': {},
                'data_types': {},
                'numeric_stats': {}
            }

            for col in columns:
                # 缺失值统计
                missing_count = self.data[col].isnull().sum()
                missing_rate = missing_count / len(self.data) * 100
                stats['missing_stats'][col] = {
                    'missing_count': missing_count,
                    'missing_rate': missing_rate
                }

                # 数据类型统计
                dtype = str(self.data[col].dtype)
                if dtype not in stats['data_types']:
                    stats['data_types'][dtype] = 0
                stats['data_types'][dtype] += 1

                # 数值型列的统计
                if self.data[col].dtype in ['int64', 'float64']:
                    valid_data = self.data[col].dropna()
                    if len(valid_data) > 0:
                        stats['numeric_stats'][col] = {
                            'count': len(valid_data),
                            'mean': valid_data.mean(),
                            'std': valid_data.std(),
                            'min': valid_data.min(),
                            'max': valid_data.max()
                        }

            self.category_stats[category] = stats

    def generate_markdown_report(self):
        """生成Markdown格式的分析报告"""
        print("正在生成Markdown分析报告...")

        md_content = f"""# 钙钛矿太阳能电池数据库 - 462列数据分类分析报告

## 数据库概览

- **总样本数**: {len(self.data):,}
- **总特征数**: {len(self.data.columns):,}
- **已分类列数**: {sum(len(cols) for cols in self.column_categories.values()):,}
- **分类覆盖率**: {sum(len(cols) for cols in self.column_categories.values())/len(self.data.columns)*100:.1f}%

## 数据类型分布

"""

        # 数据类型统计
        dtype_counts = self.data.dtypes.value_counts()
        md_content += "| 数据类型 | 列数 | 占比 |\n"
        md_content += "|---------|------|------|\n"
        for dtype, count in dtype_counts.items():
            percentage = count / len(self.data.columns) * 100
            md_content += f"| {dtype} | {count} | {percentage:.1f}% |\n"

        md_content += "\n## 分类详情\n\n"

        for category, columns in self.column_categories.items():
            md_content += f"### {category} ({len(columns)} 列)\n\n"

            stats = self.category_stats[category]

            # 分类概述
            md_content += f"- **列数**: {stats['column_count']}\n"
            md_content += f"- **数据类型分布**: {stats['data_types']}\n"

            # 缺失值统计
            total_missing = sum(s['missing_count'] for s in stats['missing_stats'].values())
            avg_missing_rate = sum(s['missing_rate'] for s in stats['missing_stats'].values()) / len(stats['missing_stats'])
            md_content += f"- **平均缺失率**: {avg_missing_rate:.1f}%\n"
            md_content += f"- **总缺失值数**: {total_missing:,}\n\n"

            # 数值特征统计
            if stats['numeric_stats']:
                md_content += "#### 数值特征统计\n\n"
                md_content += "| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |\n"
                md_content += "|------|--------|------|--------|--------|--------|\n"

                for col, num_stats in list(stats['numeric_stats'].items())[:10]:  # 只显示前10个
                    md_content += f"| {col} | {num_stats['count']} | {num_stats['mean']:.2f} | {num_stats['std']:.2f} | {num_stats['min']:.2f} | {num_stats['max']:.2f} |\n"

                if len(stats['numeric_stats']) > 10:
                    md_content += f"| ...还有{len(stats['numeric_stats'])-10}个特征 | - | - | - | - | - |\n"

                md_content += "\n"

            # 列名列表
            md_content += "#### 包含的列\n\n"
            for i, col in enumerate(columns, 1):
                missing_rate = stats['missing_stats'][col]['missing_rate']
                md_content += f"{i}. **{col}** (缺失率: {missing_rate:.1f}%)\n"

            md_content += "\n---\n\n"

        # 总结
        md_content += "## 总结与洞察\n\n"

        # 分类占比
        md_content += "### 各分类占比\n\n"
        category_counts = {cat: len(cols) for cat, cols in self.column_categories.items()}
        total_classified = sum(category_counts.values())

        md_content += "| 分类 | 列数 | 占比 |\n"
        md_content += "|------|------|------|\n"
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_classified * 100
            md_content += f"| {category} | {count} | {percentage:.1f}% |\n"

        md_content += "\n### 关键发现\n\n"

        # PCE相关列
        pce_cols = [col for col in self.data.columns if 'PCE' in col and 'JV' in col]
        md_content += f"1. **目标变量**: 发现 {len(pce_cols)} 个PCE相关列，主要用于预测太阳能电池效率\n\n"

        # 钙钛矿组成相关
        perovskite_cols = [col for col in self.data.columns if col.startswith('Perovskite_')]
        md_content += f"2. **材料组成**: {len(perovskite_cols)} 个钙钛矿相关特征，涵盖组成、厚度、带隙等关键性质\n\n"

        # 工艺参数
        process_cols = []
        for cat in ['电子传输层', '空穴传输层', '背接触', '附加层']:
            if cat in self.column_categories:
                process_cols.extend(self.column_categories[cat])
        md_content += f"3. **工艺参数**: {len(process_cols)} 个工艺相关特征，详细记录了制备过程中的各种参数\n\n"

        # 性能测试
        test_cols = []
        for cat in ['JV特性测量', '稳定性能', 'EQE', '稳定性测试', '户外测试']:
            if cat in self.column_categories:
                test_cols.extend(self.column_categories[cat])
        md_content += f"4. **性能测试**: {len(test_cols)} 个测试相关特征，全面覆盖了器件性能评估\n\n"

        # 缺失值分析
        high_missing_cols = []
        for cat, stats in self.category_stats.items():
            for col, missing_info in stats['missing_stats'].items():
                if missing_info['missing_rate'] > 50:
                    high_missing_cols.append((col, missing_info['missing_rate']))

        md_content += f"5. **数据质量**: 发现 {len(high_missing_cols)} 个缺失率>50%的列，需要特别关注数据预处理\n\n"

        md_content += "### 自动ML应用建议\n\n"
        md_content += "1. **特征选择**: 基于相关性和缺失率进行特征选择\n"
        md_content += "2. **数据预处理**: 处理高缺失率特征，填充或删除\n"
        md_content += "3. **目标变量**: 优先使用 JV_default_PCE 作为主要预测目标\n"
        md_content += "4. **特征工程**: 重点关注钙钛矿组成、工艺参数与性能的相关性\n"
        md_content += "5. **模型选择**: 推荐使用集成方法处理高维混合数据\n\n"

        md_content += "---\n\n"
        md_content += f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        md_content += "*数据来源: Perovskite_database_content_all_data.csv*\n"

        # 保存Markdown文件
        with open('perovskite_database_analysis.md', 'w', encoding='utf-8') as f:
            f.write(md_content)

        print("Markdown分析报告已保存为: perovskite_database_analysis.md")

    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("="*60)
        print("钙钛矿太阳能电池数据库 - 462列数据分类分析")
        print("="*60)

        self.load_data()
        self.classify_columns()
        self.analyze_categories()
        self.generate_markdown_report()

        print("\n分析完成！")
        print("="*60)


if __name__ == "__main__":
    # 创建列分类器并运行分析
    classifier = ColumnClassifier()
    classifier.run_complete_analysis()