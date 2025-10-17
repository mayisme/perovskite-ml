# 钙钛矿太阳能电池数据库 - 462列数据分类分析报告

## 数据库概览

- **总样本数**: 43,398
- **总特征数**: 410
- **已分类列数**: 409
- **分类覆盖率**: 99.8%

## 数据类型分布

| 数据类型 | 列数 | 占比 |
|---------|------|------|
| object | 282 | 68.8% |
| float64 | 119 | 29.0% |
| bool | 6 | 1.5% |
| int64 | 3 | 0.7% |

## 分类详情

### 参考文献信息 (12 列)

- **列数**: 12
- **数据类型分布**: {'int64': 2, 'object': 8, 'bool': 2}
- **平均缺失率**: 12.7%
- **总缺失值数**: 65,958

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Ref_ID | 43398 | 21706.54 | 12539.63 | 1.00 | 43771.00 |
| Ref_ID_temp | 43398 | 20713.31 | 12508.70 | 1.00 | 42404.00 |

#### 包含的列

1. **Ref_ID** (缺失率: 0.0%)
2. **Ref_ID_temp** (缺失率: 0.0%)
3. **Ref_name_of_person_entering_the_data** (缺失率: 0.2%)
4. **Ref_data_entered_by_author** (缺失率: 0.0%)
5. **Ref_DOI_number** (缺失率: 0.7%)
6. **Ref_lead_author** (缺失率: 0.1%)
7. **Ref_publication_date** (缺失率: 0.0%)
8. **Ref_journal** (缺失率: 0.0%)
9. **Ref_part_of_initial_dataset** (缺失率: 0.0%)
10. **Ref_original_filename_data_upload** (缺失率: 0.0%)
11. **Ref_free_text_comment** (缺失率: 97.0%)
12. **Ref_internal_sample_id** (缺失率: 54.1%)

---

### 电池基本信息 (11 列)

- **列数**: 11
- **数据类型分布**: {'object': 6, 'float64': 5}
- **平均缺失率**: 36.4%
- **总缺失值数**: 173,791

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Cell_area_total | 1253 | 1.25 | 6.42 | 0.01 | 100.00 |
| Cell_area_measured | 42317 | 0.19 | 1.10 | 0.00 | 70.00 |
| Cell_number_of_cells_per_substrate | 43388 | 0.07 | 0.76 | 0.00 | 20.00 |
| Cell_semitransparent_AVT | 9 | 55.00 | 0.00 | 55.00 | 55.00 |

#### 包含的列

1. **Cell_stack_sequence** (缺失率: 0.8%)
2. **Cell_area_total** (缺失率: 97.1%)
3. **Cell_area_measured** (缺失率: 2.5%)
4. **Cell_number_of_cells_per_substrate** (缺失率: 0.0%)
5. **Cell_architecture** (缺失率: 0.0%)
6. **Cell_flexible** (缺失率: 0.0%)
7. **Cell_flexible_min_bending_radius** (缺失率: 100.0%)
8. **Cell_semitransparent** (缺失率: 0.0%)
9. **Cell_semitransparent_AVT** (缺失率: 100.0%)
10. **Cell_semitransparent_wavelength_range** (缺失率: 0.0%)
11. **Cell_semitransparent_raw_data** (缺失率: 100.0%)

---

### 模块信息 (5 列)

- **列数**: 5
- **数据类型分布**: {'object': 2, 'float64': 3}
- **平均缺失率**: 39.8%
- **总缺失值数**: 86,331

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Module_number_of_cells_in_module | 43388 | 0.06 | 0.89 | 0.00 | 36.00 |
| Module_area_total | 349 | 37.95 | 74.72 | 0.00 | 435.00 |
| Module_area_effective | 146 | 19.80 | 26.54 | 0.02 | 151.88 |

#### 包含的列

1. **Module** (缺失率: 0.0%)
2. **Module_number_of_cells_in_module** (缺失率: 0.0%)
3. **Module_area_total** (缺失率: 99.2%)
4. **Module_area_effective** (缺失率: 99.7%)
5. **Module_JV_data_recalculated_per_cell** (缺失率: 0.0%)

---

### 基板信息 (9 列)

- **列数**: 9
- **数据类型分布**: {'object': 7, 'float64': 2}
- **平均缺失率**: 76.5%
- **总缺失值数**: 298,745

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Substrate_area | 743 | 5.35 | 8.99 | 0.02 | 100.00 |

#### 包含的列

1. **Substrate_stack_sequence** (缺失率: 0.8%)
2. **Substrate_thickness** (缺失率: 98.7%)
3. **Substrate_area** (缺失率: 98.3%)
4. **Substrate_supplier** (缺失率: 97.4%)
5. **Substrate_brand_name** (缺失率: 98.3%)
6. **Substrate_deposition_procedure** (缺失率: 0.8%)
7. **Substrate_surface_roughness_rms** (缺失率: 100.0%)
8. **Substrate_etching_procedure** (缺失率: 98.0%)
9. **Substrate_cleaning_procedure** (缺失率: 96.1%)

---

### 电子传输层 (29 列)

- **列数**: 29
- **数据类型分布**: {'object': 27, 'float64': 2}
- **平均缺失率**: 36.5%
- **总缺失值数**: 459,530

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| ETL_storage_time_until_next_deposition_step | 224 | 31.57 | 33.47 | 0.10 | 100.00 |
| ETL_storage_relative_humidity | 30 | 21.30 | 17.12 | 0.90 | 40.00 |

#### 包含的列

1. **ETL_stack_sequence** (缺失率: 0.0%)
2. **ETL_thickness** (缺失率: 56.5%)
3. **ETL_additives_compounds** (缺失率: 8.8%)
4. **ETL_additives_concentrations** (缺失率: 98.4%)
5. **ETL_deposition_procedure** (缺失率: 0.8%)
6. **ETL_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
7. **ETL_deposition_synthesis_atmosphere** (缺失率: 0.8%)
8. **ETL_deposition_synthesis_atmosphere_pressure_total** (缺失率: 98.2%)
9. **ETL_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 98.4%)
10. **ETL_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 98.5%)
11. **ETL_deposition_solvents** (缺失率: 0.8%)
12. **ETL_deposition_solvents_mixing_ratios** (缺失率: 96.9%)
13. **ETL_deposition_solvents_supplier** (缺失率: 0.8%)
14. **ETL_deposition_solvents_purity** (缺失率: 0.8%)
15. **ETL_deposition_reaction_solutions_compounds** (缺失率: 95.4%)
16. **ETL_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
17. **ETL_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
18. **ETL_deposition_reaction_solutions_concentrations** (缺失率: 96.1%)
19. **ETL_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
20. **ETL_deposition_reaction_solutions_age** (缺失率: 0.8%)
21. **ETL_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
22. **ETL_deposition_substrate_temperature** (缺失率: 0.8%)
23. **ETL_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
24. **ETL_deposition_thermal_annealing_time** (缺失率: 0.8%)
25. **ETL_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
26. **ETL_storage_time_until_next_deposition_step** (缺失率: 99.5%)
27. **ETL_storage_atmosphere** (缺失率: 0.8%)
28. **ETL_storage_relative_humidity** (缺失率: 99.9%)
29. **ETL_surface_treatment_before_next_deposition_step** (缺失率: 99.6%)

---

### 钙钛矿层 (68 列)

- **列数**: 68
- **数据类型分布**: {'object': 64, 'bool': 2, 'float64': 2}
- **平均缺失率**: 28.7%
- **总缺失值数**: 847,929

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Perovskite_deposition_number_of_deposition_steps | 43388 | 1.25 | 0.54 | 0.00 | 12.00 |
| Perovskite_storage_relative_humidity | 67 | 49.13 | 21.13 | 10.00 | 90.00 |

#### 包含的列

1. **Perovskite_single_crystal** (缺失率: 0.0%)
2. **Perovskite_dimension_0D** (缺失率: 0.0%)
3. **Perovskite_dimension_2D** (缺失率: 0.0%)
4. **Perovskite_dimension_2D3D_mixture** (缺失率: 0.0%)
5. **Perovskite_dimension_3D** (缺失率: 0.0%)
6. **Perovskite_dimension_3D_with_2D_capping_layer** (缺失率: 0.0%)
7. **Perovskite_dimension_list_of_layers** (缺失率: 1.5%)
8. **Perovskite_composition_perovskite_ABC3_structure** (缺失率: 0.0%)
9. **Perovskite_composition_perovskite_inspired_structure** (缺失率: 0.0%)
10. **Perovskite_composition_a_ions** (缺失率: 0.3%)
11. **Perovskite_composition_a_ions_coefficients** (缺失率: 0.3%)
12. **Perovskite_composition_b_ions** (缺失率: 0.2%)
13. **Perovskite_composition_b_ions_coefficients** (缺失率: 0.3%)
14. **Perovskite_composition_c_ions** (缺失率: 0.2%)
15. **Perovskite_composition_c_ions_coefficients** (缺失率: 0.3%)
16. **Perovskite_composition_none_stoichiometry_components_in_excess** (缺失率: 83.1%)
17. **Perovskite_composition_short_form** (缺失率: 0.1%)
18. **Perovskite_composition_long_form** (缺失率: 0.1%)
19. **Perovskite_composition_assumption** (缺失率: 95.3%)
20. **Perovskite_composition_inorganic** (缺失率: 0.0%)
21. **Perovskite_composition_leadfree** (缺失率: 0.0%)
22. **Perovskite_additives_compounds** (缺失率: 66.7%)
23. **Perovskite_additives_concentrations** (缺失率: 88.3%)
24. **Perovskite_thickness** (缺失率: 68.4%)
25. **Perovskite_band_gap** (缺失率: 24.7%)
26. **Perovskite_band_gap_graded** (缺失率: 0.0%)
27. **Perovskite_band_gap_estimation_basis** (缺失率: 44.9%)
28. **Perovskite_pl_max** (缺失率: 76.1%)
29. **Perovskite_deposition_number_of_deposition_steps** (缺失率: 0.0%)
30. **Perovskite_deposition_procedure** (缺失率: 0.8%)
31. **Perovskite_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
32. **Perovskite_deposition_synthesis_atmosphere** (缺失率: 0.8%)
33. **Perovskite_deposition_synthesis_atmosphere_pressure_total** (缺失率: 98.2%)
34. **Perovskite_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 98.2%)
35. **Perovskite_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 98.4%)
36. **Perovskite_deposition_solvents** (缺失率: 0.8%)
37. **Perovskite_deposition_solvents_mixing_ratios** (缺失率: 6.1%)
38. **Perovskite_deposition_solvents_supplier** (缺失率: 0.8%)
39. **Perovskite_deposition_solvents_purity** (缺失率: 0.8%)
40. **Perovskite_deposition_reaction_solutions_compounds** (缺失率: 94.7%)
41. **Perovskite_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
42. **Perovskite_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
43. **Perovskite_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
44. **Perovskite_deposition_reaction_solutions_age** (缺失率: 0.8%)
45. **Perovskite_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
46. **Perovskite_deposition_substrate_temperature** (缺失率: 0.8%)
47. **Perovskite_deposition_quenching_induced_crystallisation** (缺失率: 0.0%)
48. **Perovskite_deposition_quenching_media** (缺失率: 0.8%)
49. **Perovskite_deposition_quenching_media_mixing_ratios** (缺失率: 98.3%)
50. **Perovskite_deposition_quenching_media_volume** (缺失率: 0.8%)
51. **Perovskite_deposition_quenching_media_additives_compounds** (缺失率: 97.9%)
52. **Perovskite_deposition_quenching_media_additives_concentrations** (缺失率: 99.8%)
53. **Perovskite_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
54. **Perovskite_deposition_thermal_annealing_time** (缺失率: 0.8%)
55. **Perovskite_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
56. **Perovskite_deposition_thermal_annealing_relative_humidity** (缺失率: 99.6%)
57. **Perovskite_deposition_thermal_annealing_pressure** (缺失率: 98.9%)
58. **Perovskite_deposition_solvent_annealing** (缺失率: 0.0%)
59. **Perovskite_deposition_solvent_annealing_timing** (缺失率: 99.6%)
60. **Perovskite_deposition_solvent_annealing_solvent_atmosphere** (缺失率: 0.8%)
61. **Perovskite_deposition_solvent_annealing_time** (缺失率: 0.8%)
62. **Perovskite_deposition_solvent_annealing_temperature** (缺失率: 0.8%)
63. **Perovskite_deposition_after_treatment_of_formed_perovskite** (缺失率: 96.3%)
64. **Perovskite_deposition_after_treatment_of_formed_perovskite_met** (缺失率: 99.7%)
65. **Perovskite_storage_time_until_next_deposition_step** (缺失率: 0.8%)
66. **Perovskite_storage_atmosphere** (缺失率: 0.8%)
67. **Perovskite_storage_relative_humidity** (缺失率: 99.8%)
68. **Perovskite_surface_treatment_before_next_deposition_step** (缺失率: 100.0%)

---

### 空穴传输层 (29 列)

- **列数**: 29
- **数据类型分布**: {'object': 28, 'float64': 1}
- **平均缺失率**: 35.1%
- **总缺失值数**: 441,207

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| HTL_storage_relative_humidity | 22 | 15.69 | 9.28 | 0.05 | 25.00 |

#### 包含的列

1. **HTL_stack_sequence** (缺失率: 0.0%)
2. **HTL_thickness_list** (缺失率: 76.6%)
3. **HTL_additives_compounds** (缺失率: 43.5%)
4. **HTL_additives_concentrations** (缺失率: 97.4%)
5. **HTL_deposition_procedure** (缺失率: 0.8%)
6. **HTL_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
7. **HTL_deposition_synthesis_atmosphere** (缺失率: 0.8%)
8. **HTL_deposition_synthesis_atmosphere_pressure_total** (缺失率: 97.8%)
9. **HTL_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 97.9%)
10. **HTL_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 99.4%)
11. **HTL_deposition_solvents** (缺失率: 0.8%)
12. **HTL_deposition_solvents_mixing_ratios** (缺失率: 98.6%)
13. **HTL_deposition_solvents_supplier** (缺失率: 0.8%)
14. **HTL_deposition_solvents_purity** (缺失率: 0.8%)
15. **HTL_deposition_reaction_solutions_compounds** (缺失率: 95.6%)
16. **HTL_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
17. **HTL_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
18. **HTL_deposition_reaction_solutions_concentrations** (缺失率: 96.5%)
19. **HTL_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
20. **HTL_deposition_reaction_solutions_age** (缺失率: 0.8%)
21. **HTL_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
22. **HTL_deposition_substrate_temperature** (缺失率: 0.8%)
23. **HTL_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
24. **HTL_deposition_thermal_annealing_time** (缺失率: 0.8%)
25. **HTL_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
26. **HTL_storage_time_until_next_deposition_step** (缺失率: 0.8%)
27. **HTL_storage_atmosphere** (缺失率: 0.8%)
28. **HTL_storage_relative_humidity** (缺失率: 99.9%)
29. **HTL_surface_treatment_before_next_deposition_step** (缺失率: 99.9%)

---

### 背接触 (29 列)

- **列数**: 29
- **数据类型分布**: {'object': 28, 'float64': 1}
- **平均缺失率**: 35.2%
- **总缺失值数**: 443,482

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Backcontact_storage_relative_humidity | 76 | 32.37 | 19.38 | 5.00 | 50.00 |

#### 包含的列

1. **Backcontact_stack_sequence** (缺失率: 0.8%)
2. **Backcontact_thickness_list** (缺失率: 19.3%)
3. **Backcontact_additives_compounds** (缺失率: 95.8%)
4. **Backcontact_additives_concentrations** (缺失率: 100.0%)
5. **Backcontact_deposition_procedure** (缺失率: 0.8%)
6. **Backcontact_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
7. **Backcontact_deposition_synthesis_atmosphere** (缺失率: 0.8%)
8. **Backcontact_deposition_synthesis_atmosphere_pressure_total** (缺失率: 98.0%)
9. **Backcontact_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 98.7%)
10. **Backcontact_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 99.9%)
11. **Backcontact_deposition_solvents** (缺失率: 0.8%)
12. **Backcontact_deposition_solvents_mixing_ratios** (缺失率: 99.6%)
13. **Backcontact_deposition_solvents_supplier** (缺失率: 0.8%)
14. **Backcontact_deposition_solvents_purity** (缺失率: 0.8%)
15. **Backcontact_deposition_reaction_solutions_compounds** (缺失率: 96.6%)
16. **Backcontact_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
17. **Backcontact_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
18. **Backcontact_deposition_reaction_solutions_concentrations** (缺失率: 100.0%)
19. **Backcontact_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
20. **Backcontact_deposition_reaction_solutions_age** (缺失率: 0.8%)
21. **Backcontact_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
22. **Backcontact_deposition_substrate_temperature** (缺失率: 0.8%)
23. **Backcontact_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
24. **Backcontact_deposition_thermal_annealing_time** (缺失率: 0.8%)
25. **Backcontact_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
26. **Backcontact_storage_time_until_next_deposition_step** (缺失率: 0.8%)
27. **Backcontact_storage_atmosphere** (缺失率: 0.8%)
28. **Backcontact_storage_relative_humidity** (缺失率: 99.8%)
29. **Backcontact_surface_treatment_before_next_deposition_step** (缺失率: 100.0%)

---

### 附加层 (62 列)

- **列数**: 62
- **数据类型分布**: {'object': 40, 'float64': 22}
- **平均缺失率**: 39.2%
- **总缺失值数**: 1,053,674

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Add_lay_front_thickness_list | 53 | 80.75 | 8.29 | 50.00 | 100.00 |
| Add_lay_back_thickness_list | 4 | 97.50 | 8.66 | 90.00 | 105.00 |

#### 包含的列

1. **Add_lay_front** (缺失率: 0.0%)
2. **Add_lay_front_function** (缺失率: 99.6%)
3. **Add_lay_front_stack_sequence** (缺失率: 0.8%)
4. **Add_lay_front_thickness_list** (缺失率: 99.9%)
5. **Add_lay_front_additives_compounds** (缺失率: 100.0%)
6. **Add_lay_front_additives_concentrations** (缺失率: 100.0%)
7. **Add_lay_front_deposition_procedure** (缺失率: 0.8%)
8. **Add_lay_front_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
9. **Add_lay_front_deposition_synthesis_atmosphere** (缺失率: 0.8%)
10. **Add_lay_front_deposition_synthesis_atmosphere_pressure_total** (缺失率: 100.0%)
11. **Add_lay_front_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 100.0%)
12. **Add_lay_front_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 100.0%)
13. **Add_lay_front_deposition_solvents** (缺失率: 0.8%)
14. **Add_lay_front_deposition_solvents_mixing_ratios** (缺失率: 100.0%)
15. **Add_lay_front_deposition_solvents_supplier** (缺失率: 0.8%)
16. **Add_lay_front_deposition_solvents_purity** (缺失率: 0.8%)
17. **Add_lay_front_deposition_reaction_solutions_compounds** (缺失率: 100.0%)
18. **Add_lay_front_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
19. **Add_lay_front_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
20. **Add_lay_front_deposition_reaction_solutions_concentrations** (缺失率: 100.0%)
21. **Add_lay_front_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
22. **Add_lay_front_deposition_reaction_solutions_age** (缺失率: 0.8%)
23. **Add_lay_front_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
24. **Add_lay_front_deposition_substrate_temperature** (缺失率: 0.8%)
25. **Add_lay_front_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
26. **Add_lay_front_deposition_thermal_annealing_time** (缺失率: 0.8%)
27. **Add_lay_front_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
28. **Add_lay_front_storage_time_until_next_deposition_step** (缺失率: 0.8%)
29. **Add_lay_front_storage_atmosphere** (缺失率: 0.8%)
30. **Add_lay_front_storage_relative_humidity** (缺失率: 100.0%)
31. **Add_lay_front_surface_treatment_before_next_deposition_step** (缺失率: 100.0%)
32. **Add_lay_back** (缺失率: 0.0%)
33. **Add_lay_back_function** (缺失率: 99.9%)
34. **Add_lay_back_stack_sequence** (缺失率: 0.8%)
35. **Add_lay_back_thickness_list** (缺失率: 100.0%)
36. **Add_lay_back_additives_compounds** (缺失率: 100.0%)
37. **Add_lay_back_additives_concentrations** (缺失率: 100.0%)
38. **Add_lay_back_deposition_procedure** (缺失率: 0.8%)
39. **Add_lay_back_deposition_aggregation_state_of_reactants** (缺失率: 0.8%)
40. **Add_lay_back_deposition_synthesis_atmosphere** (缺失率: 0.8%)
41. **Add_lay_back_deposition_synthesis_atmosphere_pressure_total** (缺失率: 100.0%)
42. **Add_lay_back_deposition_synthesis_atmosphere_pressure_partial** (缺失率: 100.0%)
43. **Add_lay_back_deposition_synthesis_atmosphere_relative_humidity** (缺失率: 100.0%)
44. **Add_lay_back_deposition_solvents** (缺失率: 0.8%)
45. **Add_lay_back_deposition_solvents_mixing_ratios** (缺失率: 100.0%)
46. **Add_lay_back_deposition_solvents_supplier** (缺失率: 0.8%)
47. **Add_lay_back_deposition_solvents_purity** (缺失率: 0.8%)
48. **Add_lay_back_deposition_reaction_solutions_compounds** (缺失率: 100.0%)
49. **Add_lay_back_deposition_reaction_solutions_compounds_supplier** (缺失率: 0.8%)
50. **Add_lay_back_deposition_reaction_solutions_compounds_purity** (缺失率: 0.8%)
51. **Add_lay_back_deposition_reaction_solutions_concentrations** (缺失率: 100.0%)
52. **Add_lay_back_deposition_reaction_solutions_volumes** (缺失率: 0.8%)
53. **Add_lay_back_deposition_reaction_solutions_age** (缺失率: 0.8%)
54. **Add_lay_back_deposition_reaction_solutions_temperature** (缺失率: 0.8%)
55. **Add_lay_back_deposition_substrate_temperature** (缺失率: 0.8%)
56. **Add_lay_back_deposition_thermal_annealing_temperature** (缺失率: 0.8%)
57. **Add_lay_back_deposition_thermal_annealing_time** (缺失率: 0.8%)
58. **Add_lay_back_deposition_thermal_annealing_atmosphere** (缺失率: 0.8%)
59. **Add_lay_back_storage_time_until_next_deposition_step** (缺失率: 0.8%)
60. **Add_lay_back_storage_atmosphere** (缺失率: 0.8%)
61. **Add_lay_back_storage_relative_humidity** (缺失率: 100.0%)
62. **Add_lay_back_surface_treatment_before_next_deposition_step** (缺失率: 100.0%)

---

### 封装 (6 列)

- **列数**: 6
- **数据类型分布**: {'object': 4, 'float64': 2}
- **平均缺失率**: 33.7%
- **总缺失值数**: 87,817

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Encapsulation_water_vapour_transmission_rate | 16 | 0.94 | 2.72 | 0.00 | 11.00 |
| Encapsulation_oxygen_transmission_rate | 2 | 1071.75 | 0.00 | 1071.75 | 1071.75 |

#### 包含的列

1. **Encapsulation** (缺失率: 0.0%)
2. **Encapsulation_stack_sequence** (缺失率: 0.8%)
3. **Encapsulation_edge_sealing_materials** (缺失率: 0.8%)
4. **Encapsulation_atmosphere_for_encapsulation** (缺失率: 0.8%)
5. **Encapsulation_water_vapour_transmission_rate** (缺失率: 100.0%)
6. **Encapsulation_oxygen_transmission_rate** (缺失率: 100.0%)

---

### JV特性测量 (53 列)

- **列数**: 53
- **数据类型分布**: {'bool': 2, 'int64': 1, 'object': 17, 'float64': 33}
- **平均缺失率**: 56.5%
- **总缺失值数**: 1,300,359

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| JV_average_over_n_number_of_cells | 43398 | 5.61 | 13.67 | 0.00 | 767.00 |
| JV_storage_relative_humidity | 30 | 14.47 | 18.19 | 0.05 | 65.00 |
| JV_test_relative_humidity | 331 | 39.25 | 14.18 | 10.00 | 75.00 |
| JV_test_temperature | 579 | 26.95 | 32.80 | -193.00 | 200.00 |
| JV_light_intensity | 43272 | 99.95 | 19.09 | 0.00 | 1800.00 |
| JV_light_mask_area | 1283 | 0.31 | 2.59 | 0.00 | 66.95 |
| JV_scan_speed | 16729 | 662.65 | 39170.21 | 0.01 | 5000000.00 |
| JV_scan_delay_time | 459 | 76.68 | 130.68 | 0.30 | 1000.00 |
| JV_scan_integration_time | 224 | 20.03 | 2.69 | 3.00 | 30.00 |
| JV_scan_voltage_step | 564 | 21.17 | 15.85 | 0.02 | 50.00 |
| ...还有24个特征 | - | - | - | - | - |

#### 包含的列

1. **JV_measured** (缺失率: 0.0%)
2. **JV_average_over_n_number_of_cells** (缺失率: 0.0%)
3. **JV_certified_values** (缺失率: 0.0%)
4. **JV_certification_institute** (缺失率: 99.8%)
5. **JV_storage_age_of_cell** (缺失率: 0.8%)
6. **JV_storage_atmosphere** (缺失率: 0.8%)
7. **JV_storage_relative_humidity** (缺失率: 99.9%)
8. **JV_test_atmosphere** (缺失率: 0.8%)
9. **JV_test_relative_humidity** (缺失率: 99.2%)
10. **JV_test_temperature** (缺失率: 98.7%)
11. **JV_light_source_type** (缺失率: 94.7%)
12. **JV_light_source_brand_name** (缺失率: 96.7%)
13. **JV_light_source_simulator_class** (缺失率: 97.7%)
14. **JV_light_intensity** (缺失率: 0.3%)
15. **JV_light_spectra** (缺失率: 6.6%)
16. **JV_light_wavelength_range** (缺失率: 0.8%)
17. **JV_light_illumination_direction** (缺失率: 96.2%)
18. **JV_light_masked_cell** (缺失率: 0.0%)
19. **JV_light_mask_area** (缺失率: 97.0%)
20. **JV_scan_speed** (缺失率: 61.5%)
21. **JV_scan_delay_time** (缺失率: 98.9%)
22. **JV_scan_integration_time** (缺失率: 99.5%)
23. **JV_scan_voltage_step** (缺失率: 98.7%)
24. **JV_preconditioning_protocol** (缺失率: 97.7%)
25. **JV_preconditioning_time** (缺失率: 99.5%)
26. **JV_preconditioning_potential** (缺失率: 99.7%)
27. **JV_preconditioning_light_intensity** (缺失率: 99.6%)
28. **JV_reverse_scan_Voc** (缺失率: 6.8%)
29. **JV_reverse_scan_Jsc** (缺失率: 6.7%)
30. **JV_reverse_scan_FF** (缺失率: 7.1%)
31. **JV_reverse_scan_PCE** (缺失率: 4.1%)
32. **JV_reverse_scan_Vmp** (缺失率: 99.4%)
33. **JV_reverse_scan_Jmp** (缺失率: 99.4%)
34. **JV_reverse_scan_series_resistance** (缺失率: 99.4%)
35. **JV_reverse_scan_shunt_resistance** (缺失率: 99.4%)
36. **JV_forward_scan_Voc** (缺失率: 78.7%)
37. **JV_forward_scan_Jsc** (缺失率: 78.7%)
38. **JV_forward_scan_FF** (缺失率: 78.8%)
39. **JV_forward_scan_PCE** (缺失率: 78.3%)
40. **JV_forward_scan_Vmp** (缺失率: 99.6%)
41. **JV_forward_scan_Jmp** (缺失率: 99.6%)
42. **JV_forward_scan_series_resistance** (缺失率: 99.4%)
43. **JV_forward_scan_shunt_resistance** (缺失率: 99.5%)
44. **JV_link_raw_data** (缺失率: 100.0%)
45. **JV_default_Voc** (缺失率: 5.1%)
46. **JV_default_Jsc** (缺失率: 5.0%)
47. **JV_default_FF** (缺失率: 5.4%)
48. **JV_default_PCE** (缺失率: 2.3%)
49. **JV_default_Voc_scan_direction** (缺失率: 5.1%)
50. **JV_default_Jsc_scan_direction** (缺失率: 5.0%)
51. **JV_default_FF_scan_direction** (缺失率: 5.4%)
52. **JV_default_PCE_scan_direction** (缺失率: 2.3%)
53. **JV_hysteresis_index** (缺失率: 80.6%)

---

### 稳定性能 (8 列)

- **列数**: 8
- **数据类型分布**: {'object': 3, 'float64': 5}
- **平均缺失率**: 86.2%
- **总缺失值数**: 299,224

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Stabilised_performance_procedure_metrics | 143 | 0.84 | 0.12 | 0.60 | 1.35 |
| Stabilised_performance_measurement_time | 245 | 42.94 | 231.14 | 0.50 | 3000.00 |
| Stabilised_performance_PCE | 3442 | 14.66 | 4.73 | 0.00 | 25.20 |
| Stabilised_performance_Vmp | 221 | 0.84 | 0.18 | 0.00 | 1.37 |
| Stabilised_performance_Jmp | 225 | 14.68 | 7.49 | 0.42 | 25.14 |

#### 包含的列

1. **Stabilised_performance_measured** (缺失率: 0.0%)
2. **Stabilised_performance_procedure** (缺失率: 99.3%)
3. **Stabilised_performance_procedure_metrics** (缺失率: 99.7%)
4. **Stabilised_performance_measurement_time** (缺失率: 99.4%)
5. **Stabilised_performance_PCE** (缺失率: 92.1%)
6. **Stabilised_performance_Vmp** (缺失率: 99.5%)
7. **Stabilised_performance_Jmp** (缺失率: 99.5%)
8. **Stabilised_performance_link_raw_data** (缺失率: 100.0%)

---

### EQE (4 列)

- **列数**: 4
- **数据类型分布**: {'object': 2, 'float64': 2}
- **平均缺失率**: 71.7%
- **总缺失值数**: 124,545

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| EQE_light_bias | 58 | 54.83 | 20.28 | 10.00 | 100.00 |
| EQE_integrated_Jsc | 5597 | 18.68 | 4.40 | 0.10 | 42.33 |

#### 包含的列

1. **EQE_measured** (缺失率: 0.0%)
2. **EQE_light_bias** (缺失率: 99.9%)
3. **EQE_integrated_Jsc** (缺失率: 87.1%)
4. **EQE_link_raw_data** (缺失率: 100.0%)

---

### 稳定性测试 (44 列)

- **列数**: 44
- **数据类型分布**: {'object': 23, 'float64': 21}
- **平均缺失率**: 65.3%
- **总缺失值数**: 1,247,210

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Stability_average_over_n_number_of_cells | 43388 | 1.02 | 0.62 | 1.00 | 40.00 |
| Stability_light_intensity | 7298 | 24.19 | 47.35 | 0.00 | 1000.00 |
| Stability_potential_bias_passive_resistance | 13 | 709.23 | 427.11 | 510.00 | 2000.00 |
| Stability_atmosphere_oxygen_concentration | 16 | 14.69 | 23.97 | 1.00 | 100.00 |
| Stability_relative_humidity_average_value | 5389 | 33.13 | 25.03 | 0.00 | 100.00 |
| Stability_time_total_exposure | 7405 | 734.58 | 1165.74 | 0.03 | 43800.00 |
| Stability_PCE_initial_value | 3047 | 15.36 | 11.76 | 0.00 | 100.00 |
| Stability_PCE_end_of_experiment | 7376 | 67.54 | 32.77 | 0.00 | 780.00 |
| Stability_PCE_T95 | 229 | 311.39 | 499.44 | 0.00 | 3260.00 |
| Stability_PCE_Ts95 | 27 | 163.07 | 223.13 | 1.80 | 1000.00 |
| ...还有9个特征 | - | - | - | - | - |

#### 包含的列

1. **Stability_measured** (缺失率: 0.0%)
2. **Stability_protocol** (缺失率: 82.9%)
3. **Stability_average_over_n_number_of_cells** (缺失率: 0.0%)
4. **Stability_light_source_type** (缺失率: 83.0%)
5. **Stability_light_source_brand_name** (缺失率: 100.0%)
6. **Stability_light_source_simulator_class** (缺失率: 100.0%)
7. **Stability_light_intensity** (缺失率: 83.2%)
8. **Stability_light_spectra** (缺失率: 98.4%)
9. **Stability_light_wavelength_range** (缺失率: 0.8%)
10. **Stability_light_illumination_direction** (缺失率: 100.0%)
11. **Stability_light_load_condition** (缺失率: 99.7%)
12. **Stability_light_cycling_times** (缺失率: 0.8%)
13. **Stability_light_UV_filter** (缺失率: 0.0%)
14. **Stability_potential_bias_load_condition** (缺失率: 82.9%)
15. **Stability_potential_bias_range** (缺失率: 0.8%)
16. **Stability_potential_bias_passive_resistance** (缺失率: 100.0%)
17. **Stability_temperature_load_condition** (缺失率: 99.1%)
18. **Stability_temperature_range** (缺失率: 0.8%)
19. **Stability_temperature_cycling_times** (缺失率: 0.8%)
20. **Stability_temperature_ramp_speed** (缺失率: 100.0%)
21. **Stability_atmosphere** (缺失率: 0.8%)
22. **Stability_atmosphere_oxygen_concentration** (缺失率: 100.0%)
23. **Stability_relative_humidity_load_conditions** (缺失率: 99.4%)
24. **Stability_relative_humidity_range** (缺失率: 0.8%)
25. **Stability_relative_humidity_average_value** (缺失率: 87.6%)
26. **Stability_time_total_exposure** (缺失率: 82.9%)
27. **Stability_periodic_JV_measurements** (缺失率: 0.0%)
28. **Stability_periodic_JV_measurements_time_between_jv** (缺失率: 0.8%)
29. **Stability_PCE_initial_value** (缺失率: 93.0%)
30. **Stability_PCE_burn_in_observed** (缺失率: 0.0%)
31. **Stability_PCE_end_of_experiment** (缺失率: 83.0%)
32. **Stability_PCE_T95** (缺失率: 99.5%)
33. **Stability_PCE_Ts95** (缺失率: 99.9%)
34. **Stability_PCE_T80** (缺失率: 95.8%)
35. **Stability_PCE_Ts80** (缺失率: 99.8%)
36. **Stability_PCE_Te80** (缺失率: 99.9%)
37. **Stability_PCE_Tse80** (缺失率: 100.0%)
38. **Stability_PCE_after_1000_h** (缺失率: 97.7%)
39. **Stability_lifetime_energy_yield** (缺失率: 100.0%)
40. **Stability_flexible_cell_number_of_bending_cycles** (缺失率: 0.0%)
41. **Stability_flexible_cell_bending_radius** (缺失率: 100.0%)
42. **Stability_flexible_cell_PCE_initial_value** (缺失率: 100.0%)
43. **Stability_flexible_cell_PCE_end_of_experiment** (缺失率: 100.0%)
44. **Stability_link_raw_data_for_stability_trace** (缺失率: 100.0%)

---

### 户外测试 (40 列)

- **列数**: 40
- **数据类型分布**: {'object': 22, 'float64': 18}
- **平均缺失率**: 67.6%
- **总缺失值数**: 1,173,321

#### 数值特征统计

| 特征 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |
|------|--------|------|--------|--------|--------|
| Outdoor_average_over_n_number_of_cells | 43388 | 1.00 | 0.01 | 1.00 | 3.00 |
| Outdoor_installation_tilt | 11 | 31.27 | 9.84 | 10.00 | 45.00 |
| Outdoor_installation_cardinal_direction | 3 | 180.00 | 0.00 | 180.00 | 180.00 |
| Outdoor_installation_number_of_solar_tracking_axis | 43388 | 0.00 | 0.00 | 0.00 | 0.00 |
| Outdoor_time_total_exposure | 41 | 27.46 | 46.86 | 0.07 | 180.00 |
| Outdoor_temperature_tmodule | 4 | 52.50 | 15.00 | 30.00 | 60.00 |
| Outdoor_periodic_JV_measurements_time_between_measurements | 1 | 1.00 | nan | 1.00 | 1.00 |
| Outdoor_PCE_initial_value | 21 | 11.50 | 5.14 | 4.35 | 19.30 |
| Outdoor_PCE_end_of_experiment | 40 | 39.97 | 40.46 | 0.00 | 111.00 |
| Outdoor_PCE_T95 | 3 | 8.00 | 3.46 | 4.00 | 10.00 |
| ...还有5个特征 | - | - | - | - | - |

#### 包含的列

1. **Outdoor_tested** (缺失率: 0.0%)
2. **Outdoor_protocol** (缺失率: 100.0%)
3. **Outdoor_average_over_n_number_of_cells** (缺失率: 0.0%)
4. **Outdoor_location_country** (缺失率: 99.9%)
5. **Outdoor_location_city** (缺失率: 100.0%)
6. **Outdoor_location_coordinates** (缺失率: 0.8%)
7. **Outdoor_location_climate_zone** (缺失率: 100.0%)
8. **Outdoor_installation_tilt** (缺失率: 100.0%)
9. **Outdoor_installation_cardinal_direction** (缺失率: 100.0%)
10. **Outdoor_installation_number_of_solar_tracking_axis** (缺失率: 0.0%)
11. **Outdoor_time_season** (缺失率: 100.0%)
12. **Outdoor_time_start** (缺失率: 0.8%)
13. **Outdoor_time_end** (缺失率: 0.8%)
14. **Outdoor_time_total_exposure** (缺失率: 99.9%)
15. **Outdoor_potential_bias_load_condition** (缺失率: 100.0%)
16. **Outdoor_potential_bias_range** (缺失率: 0.8%)
17. **Outdoor_potential_bias_passive_resistance** (缺失率: 100.0%)
18. **Outdoor_temperature_load_condition** (缺失率: 100.0%)
19. **Outdoor_temperature_range** (缺失率: 0.8%)
20. **Outdoor_temperature_tmodule** (缺失率: 100.0%)
21. **Outdoor_periodic_JV_measurements** (缺失率: 0.0%)
22. **Outdoor_periodic_JV_measurements_time_between_measurements** (缺失率: 100.0%)
23. **Outdoor_PCE_initial_value** (缺失率: 100.0%)
24. **Outdoor_PCE_burn_in_observed** (缺失率: 0.0%)
25. **Outdoor_PCE_end_of_experiment** (缺失率: 99.9%)
26. **Outdoor_PCE_T95** (缺失率: 100.0%)
27. **Outdoor_PCE_Ts95** (缺失率: 100.0%)
28. **Outdoor_PCE_T80** (缺失率: 100.0%)
29. **Outdoor_PCE_Ts80** (缺失率: 100.0%)
30. **Outdoor_PCE_Te80** (缺失率: 100.0%)
31. **Outdoor_PCE_Tse80** (缺失率: 100.0%)
32. **Outdoor_PCE_after_1000_h** (缺失率: 100.0%)
33. **Outdoor_power_generated** (缺失率: 100.0%)
34. **Outdoor_link_raw_data_for_outdoor_trace** (缺失率: 100.0%)
35. **Outdoor_detaild_weather_data_available** (缺失率: 0.0%)
36. **Outdoor_link_detailed_weather_data** (缺失率: 100.0%)
37. **Outdoor_spectral_data_available** (缺失率: 0.0%)
38. **Outdoor_link_spectral_data** (缺失率: 100.0%)
39. **Outdoor_irradiance_measured** (缺失率: 0.0%)
40. **Outdoor_link_irradiance_data** (缺失率: 100.0%)

---

## 总结与洞察

### 各分类占比

| 分类 | 列数 | 占比 |
|------|------|------|
| 钙钛矿层 | 68 | 16.6% |
| 附加层 | 62 | 15.2% |
| JV特性测量 | 53 | 13.0% |
| 稳定性测试 | 44 | 10.8% |
| 户外测试 | 40 | 9.8% |
| 电子传输层 | 29 | 7.1% |
| 空穴传输层 | 29 | 7.1% |
| 背接触 | 29 | 7.1% |
| 参考文献信息 | 12 | 2.9% |
| 电池基本信息 | 11 | 2.7% |
| 基板信息 | 9 | 2.2% |
| 稳定性能 | 8 | 2.0% |
| 封装 | 6 | 1.5% |
| 模块信息 | 5 | 1.2% |
| EQE | 4 | 1.0% |

### 关键发现

1. **目标变量**: 发现 4 个PCE相关列，主要用于预测太阳能电池效率

2. **材料组成**: 69 个钙钛矿相关特征，涵盖组成、厚度、带隙等关键性质

3. **工艺参数**: 149 个工艺相关特征，详细记录了制备过程中的各种参数

4. **性能测试**: 149 个测试相关特征，全面覆盖了器件性能评估

5. **数据质量**: 发现 190 个缺失率>50%的列，需要特别关注数据预处理

### 自动ML应用建议

1. **特征选择**: 基于相关性和缺失率进行特征选择
2. **数据预处理**: 处理高缺失率特征，填充或删除
3. **目标变量**: 优先使用 JV_default_PCE 作为主要预测目标
4. **特征工程**: 重点关注钙钛矿组成、工艺参数与性能的相关性
5. **模型选择**: 推荐使用集成方法处理高维混合数据

---

*报告生成时间: 2025-10-15 21:17:10*
*数据来源: Perovskite_database_content_all_data.csv*
