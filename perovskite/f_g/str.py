import shap
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union
import sys
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# ------------------------------------------------------------------
# 1. 加载官能团映射
# ------------------------------------------------------------------
def load_functional_group_mapping(csv_path: str = "bit_to_functional_group.csv") -> Dict[int, str]:
    """
    加载官能团映射文件
    """
    try:
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            try:
                bit_idx = int(row['fp_bit'].replace('fp_', ''))
                mapping[bit_idx] = row['functional_groups']
            except (ValueError, KeyError):
                continue
        return mapping
    except FileNotFoundError:
        print(f"警告: 未找到官能团映射文件 {csv_path}")
        return {}

def load_data_and_model():
    """
    加载数据集和训练好的模型，确保特征一致性
    """
    # 读取数据集，注意处理混合类型警告
    df = pd.read_csv('3.csv', low_memory=False)
    
    # 移除目标变量列，保留所有特征
    feature_cols = [col for col in df.columns if col != 'JV_default_PCE']
    X = df[feature_cols]
    
    # 加载训练好的模型
    model = joblib.load(r'output_20250826_130253/trained_model_xgboost.pkl')
    
    print(f"数据集特征数量: {len(feature_cols)}")
    print(f"指纹特征数量: {len([col for col in feature_cols if col.startswith('fp_')])}")
    
    return df, model, X, feature_cols

# ------------------------------------------------------------------
# 3. 指纹生成函数（与训练时一致）
# ------------------------------------------------------------------
def smiles_to_fp_list(smiles: str, fp_size: int = 2048) -> List[int]:
    """
    将SMILES字符串转换为指纹列表
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * fp_size
    fp = mfpgen.GetFingerprint(mol)
    return list(fp)

# ------------------------------------------------------------------
# 4. SHAP 分析器（处理所有特征）
# ------------------------------------------------------------------
class FP_SHAP_Analyzer:
    def __init__(self, model, feature_names: List[str], X_base: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.fp_indices = [i for i, name in enumerate(feature_names) if name.startswith('fp_')]
        self.non_fp_indices = [i for i, name in enumerate(feature_names) if not name.startswith('fp_')]
        self.explainer = shap.TreeExplainer(model)
        self.X_base = X_base
        self.expected_fp_size = len(self.fp_indices)
        print(f"模型期望指纹长度: {self.expected_fp_size}")

    def prepare_input_vector(self, smiles: str) -> np.ndarray:
        """
        准备完整的输入向量，包括指纹和非指纹特征
        """
        # 生成新指纹
        new_fp = smiles_to_fp_list(smiles, fp_size=self.expected_fp_size)
        
        # 确保指纹长度匹配
        if len(new_fp) != self.expected_fp_size:
            raise ValueError(f"指纹长度不匹配：期望 {self.expected_fp_size}，得到 {len(new_fp)}")
        
        # 创建完整特征向量（复制基准数据并替换指纹部分）
        X_new = self.X_base.copy()
        X_new[:, self.fp_indices] = new_fp
        
        return X_new

    def analyze_smiles(self, smiles: str, mapping: Dict[int, str]) -> Union[float, Dict[str, float]]:
        """
        分析SMILES字符串的预测和SHAP值
        """
        # 准备输入数据
        X_new = self.prepare_input_vector(smiles)
        
        # 预测
        pred = float(self.model.predict(X_new)[0])
        
        # 计算SHAP值
        try:
            shap_values = self.explainer.shap_values(X_new)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # 聚合到官能团
            group_imp = self._aggregate_shap_by_functional_group(shap_values[0] if len(shap_values.shape) > 1 else shap_values, mapping)
        except Exception as e:
            print(f"SHAP分析失败: {str(e)}")
            print("将列出基于特征名称的官能团重要性顺序:")
            group_imp = self._list_functional_groups_by_features(mapping)
        
        return pred, group_imp

    def _aggregate_shap_by_functional_group(self, shap_values: np.ndarray, mapping: Dict[int, str]) -> Dict[str, float]:
        """
        将SHAP值按官能团聚合
        """
        group_imp = {}
        # 确保我们使用正确的索引访问shap_values
        for i, feature_idx in enumerate(self.fp_indices):
            try:
                # 检查索引是否在shap_values范围内
                if feature_idx >= len(shap_values):
                    raise IndexError(f"Index {feature_idx} is out of bounds for shap_values with size {len(shap_values)}")
                
                bit_idx = int(self.feature_names[feature_idx].replace('fp_', ''))
                group = mapping.get(bit_idx, f"未知位点_{bit_idx}")
                # 使用i作为shap_values的索引，因为shap_values是针对当前样本的值数组
                imp = np.abs(shap_values[feature_idx])  # 已经是一维数组，直接访问
                group_imp[group] = group_imp.get(group, 0.0) + imp
            except (ValueError, IndexError) as e:
                print(f"处理官能团时出错: {e}")
                continue
        return group_imp

    def _list_functional_groups_by_features(self, mapping: Dict[int, str]) -> Dict[str, float]:
        """
        当SHAP分析失败时，列出基于特征名称的官能团列表
        """
        group_list = {}
        for feature_idx in self.fp_indices:
            try:
                bit_idx = int(self.feature_names[feature_idx].replace('fp_', ''))
                group = mapping.get(bit_idx, f"未知位点_{bit_idx}")
                # 当无法计算SHAP值时，给每个官能团分配相同的默认重要性值
                group_list[group] = 1.0
            except (ValueError, IndexError) as e:
                print(f"处理官能团时出错: {e}")
                continue
        return group_list

    def plot_group_importance(self, group_imp: Dict[str, float], title: str = "Functional Group SHAP"):
        """
        绘制官能团重要性图
        """
        # 过滤掉贡献很小的官能团
        filtered_imp = {k: v for k, v in group_imp.items() if v > 0.001}
        
        if not filtered_imp:
            print("没有显著的官能团贡献可可视化")
            return None
            
        sorted_items = sorted(filtered_imp.items(), key=lambda x: x[1], reverse=True)[:15]
        groups, values = zip(*sorted_items)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(groups)), values, color='skyblue')
        plt.yticks(range(len(groups)), groups)
        plt.xlabel("平均绝对SHAP值", fontsize=12)
        plt.title(title, fontsize=14)
        plt.gca().invert_yaxis()  # 最重要的在顶部
        plt.tight_layout()
        
        return plt.gcf()

# ------------------------------------------------------------------
# 5. 单次预测
# ------------------------------------------------------------------

def single_predict(smiles: str, row_index: int = 0):
    """
    单次预测函数，支持用户选择基准行
    """
    print("正在加载数据和模型...")
    df, model, X_full, feature_names = load_data_and_model()
    
    # 使用指定行作为基准（包含所有特征）
    if row_index >= len(X_full):
        print(f"警告: 指定的行索引 {row_index} 超出范围，使用第0行作为基准")
        row_index = 0
    X_base = X_full.values[row_index:row_index+1]
    
    # 加载官能团映射
    mapping = load_functional_group_mapping()
    
    # 初始化分析器
    analyzer = FP_SHAP_Analyzer(model, feature_names, X_base)
    
    try:
        print(f"分析SMILES: {smiles}")
        pred, imp = analyzer.analyze_smiles(smiles, mapping)
        
        print(f"预测PCE: {pred:.2f}%")
        print("\n官能团贡献度 (SHAP值):")
        print("-" * 40)
        
        # 如果没有足够的数据进行SHAP分析，则生成官能团特征重要性排序
        if len(imp) < 5:  # 当重要官能团数量较少时
            print("SHAP分析数据不足，生成基于特征名称的官能团重要性排序:")
            all_functional_groups = set()
            for feature_idx in analyzer.fp_indices:
                try:
                    bit_idx = int(analyzer.feature_names[feature_idx].replace('fp_', ''))
                    group = mapping.get(bit_idx, f"未知位点_{bit_idx}")
                    all_functional_groups.add(group)
                except (ValueError, IndexError) as e:
                    continue
            
            # 为每个官能团分配相同的重要性值
            imp = {group: 1.0 for group in list(all_functional_groups)[:15]}
        
        for fg, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {fg:<25} {v:.4f}")
        
        # 可视化
        fig = analyzer.plot_group_importance(imp, title=f"SMILES: {smiles}")
        if fig:
            plt.show()
        else:
            print("没有足够的官能团贡献数据生成图表")
            
    except Exception as e:
        print(f"预测失败: {str(e)}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------------
# 6. 批量预测
# ------------------------------------------------------------------

def batch_predict(smiles_list: List[str], row_index: int = 0):
    """
    批量预测函数，支持用户选择基准行
    """
    print("正在加载数据和模型...")
    df, model, X_full, feature_names = load_data_and_model()
    
    # 使用指定行作为基准
    if row_index >= len(X_full):
        print(f"警告: 指定的行索引 {row_index} 超出范围，使用第0行作为基准")
        row_index = 0
    X_base = X_full.values[row_index:row_index+1]
    
    # 加载官能团映射
    mapping = load_functional_group_mapping()
    
    # 初始化分析器
    analyzer = FP_SHAP_Analyzer(model, feature_names, X_base)
    
    all_imp = []
    preds = []
    successful_smiles = []
    
    for smi in smiles_list:
        try:
            p, imp = analyzer.analyze_smiles(smi, mapping)
            preds.append(p)
            all_imp.append(imp)
            successful_smiles.append(smi)
            print(f"成功分析: {smi} -> {p:.2f}%")
        except Exception as e:
            print(f"跳过 {smi} -> 错误: {e}")
            continue

    if not preds:
        print("没有成功的预测")
        return

    # 计算平均预测值
    avg_pred = np.mean(preds)
    print(f"\n平均预测PCE: {avg_pred:.2f}%")
    
    # 平均官能团贡献
    avg_imp = {}
    for imp in all_imp:
        for fg, v in imp.items():
            avg_imp[fg] = avg_imp.get(fg, 0) + v / len(all_imp)
    
    # 如果没有足够的数据进行SHAP分析，则生成官能团特征重要性排序
    if len(avg_imp) < 5:  # 当重要官能团数量较少时
        print("SHAP分析数据不足，生成基于特征名称的官能团重要性排序:")
        all_functional_groups = set()
        for feature_idx in analyzer.fp_indices:
            try:
                bit_idx = int(analyzer.feature_names[feature_idx].replace('fp_', ''))
                group = mapping.get(bit_idx, f"未知位点_{bit_idx}")
                all_functional_groups.add(group)
            except (ValueError, IndexError) as e:
                continue
        
        # 为每个官能团分配相同的重要性值
        avg_imp = {group: 1.0 for group in list(all_functional_groups)[:15]}
    
    print("\n平均官能团贡献度:")
    print("-" * 40)
    for fg, v in sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fg:<25} {v:.4f}")
    
    # 可视化平均贡献
    fig = analyzer.plot_group_importance(avg_imp, title="批量平均官能团贡献")
    if fig:
        plt.show()

def batch_predict_from_file(file_path: str, row_index: int = 0):
    """
    从文件读取SMILES进行批量预测，支持用户选择基准行
    文件格式支持：
    1. 每行一个任务（可以包含用分号分隔的多个SMILES）
    2. 用逗号分隔不同任务（可以在一行中）
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # 按逗号分隔不同任务
            tasks = [task.strip() for task in content.split(',') if task.strip()]
            
        print(f"从文件 {file_path} 读取到 {len(tasks)} 个任务")
        
        # 处理每个任务
        smiles_tasks = []
        for i, task in enumerate(tasks):
            # 每个任务中可能包含用分号分隔的多个SMILES
            smiles_list = [smi.strip() for smi in task.split(';') if smi.strip()]
            if smiles_list:
                smiles_tasks.append(smiles_list)
                print(f"任务 {i+1}: {len(smiles_list)} 个SMILES - {smiles_list}")
        
        if not smiles_tasks:
            print("文件中没有有效的SMILES任务")
            return
            
        print("正在加载数据和模型...")
        df, model, X_full, feature_names = load_data_and_model()
        
        # 使用指定行作为基准
        if row_index >= len(X_full):
            print(f"警告: 指定的行索引 {row_index} 超出范围，使用第0行作为基准")
            row_index = 0
        X_base = X_full.values[row_index:row_index+1]
        
        # 加载官能团映射
        mapping = load_functional_group_mapping()
        
        # 初始化分析器
        analyzer = FP_SHAP_Analyzer(model, feature_names, X_base)
        
        all_task_imp = []
        task_preds = []
        
        # 处理每个任务
        for task_idx, smiles_list in enumerate(smiles_tasks):
            print(f"\n处理任务 {task_idx+1}/{len(smiles_tasks)}: {smiles_list}")
            
            try:
                # 对于同一任务中的多个SMILES，合并它们的指纹（使用OR操作）
                combined_fp = None
                task_success = True
                task_shap_values = []
                
                # 先生成所有SMILES的指纹和SHAP值
                for smi in smiles_list:
                    try:
                        # 准备输入数据
                        X_new = analyzer.prepare_input_vector(smi)
                        
                        # 预测
                        pred = float(model.predict(X_new)[0])
                        
                        # 计算SHAP值
                        shap_values = analyzer.explainer.shap_values(X_new)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                            
                        # 确保shap_values是一维数组
                        if len(shap_values.shape) > 1:
                            shap_values = shap_values[0]
                            
                        task_shap_values.append((smi, pred, shap_values))
                        
                    except Exception as e:
                        print(f"  跳过SMILES '{smi}' -> 错误: {e}")
                        continue
                
                if not task_shap_values:
                    print(f"  任务 {task_idx+1} 中没有有效的SMILES")
                    continue
                
                # 合并同一任务中多个SMILES的指纹（使用OR操作）
                # 先获取第一个有效的指纹作为基础
                first_shap = task_shap_values[0][2]  # 第一个SMILES的SHAP值
                combined_shap = np.copy(first_shap)
                
                # 对于fp_特征，使用OR操作合并
                for smi, pred, shap_values in task_shap_values[1:]:
                    # 对fp_特征使用OR操作合并
                    for i in analyzer.fp_indices:
                        # 使用最大值模拟OR操作（因为指纹是0或1）
                        combined_shap[i] = max(combined_shap[i], shap_values[i])
                
                # 计算任务的平均预测值
                task_pred = np.mean([pred for _, pred, _ in task_shap_values])
                
                # 聚合到官能团
                try:
                    group_imp = analyzer._aggregate_shap_by_functional_group(combined_shap, mapping)
                except Exception as e:
                    print(f"SHAP分析失败: {str(e)}")
                    print("将列出基于特征名称的官能团重要性顺序:")
                    group_imp = analyzer._list_functional_groups_by_features(mapping)
                
                task_preds.append(task_pred)
                all_task_imp.append(group_imp)
                
                print(f"  任务 {task_idx+1} 预测PCE: {task_pred:.2f}%")
                
            except Exception as e:
                print(f"  处理任务 {task_idx+1} 失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if not task_preds:
            print("没有成功的任务")
            return
            
        # 计算平均预测值
        avg_pred = np.mean(task_preds)
        print(f"\n平均预测PCE: {avg_pred:.2f}%")
        
        # 平均官能团贡献
        avg_imp = {}
        for imp in all_task_imp:
            for fg, v in imp.items():
                avg_imp[fg] = avg_imp.get(fg, 0) + v / len(all_task_imp)
        
        # 如果没有足够的数据进行SHAP分析，则生成官能团特征重要性排序
        if len(avg_imp) < 5:  # 当重要官能团数量较少时
            print("SHAP分析数据不足，生成基于特征名称的官能团重要性排序:")
            all_functional_groups = set()
            for feature_idx in analyzer.fp_indices:
                try:
                    bit_idx = int(analyzer.feature_names[feature_idx].replace('fp_', ''))
                    group = mapping.get(bit_idx, f"未知位点_{bit_idx}")
                    all_functional_groups.add(group)
                except (ValueError, IndexError) as e:
                    continue
            
            # 为每个官能团分配相同的重要性值
            avg_imp = {group: 1.0 for group in list(all_functional_groups)[:15]}
        
        print("\n平均官能团贡献度:")
        print("-" * 40)
        for fg, v in sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {fg:<25} {v:.4f}")
        
        # 可视化平均贡献
        fig = analyzer.plot_group_importance(avg_imp, title="批量任务平均官能团贡献")
        if fig:
            plt.show()
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()

# ------------------------------------------------------------------
# 7. 命令行入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python str.py single <SMILES> [row_index]")
        print("  python str.py batch <SMILES1;SMILES2;...> [row_index]")
        print("  python str.py file <file_path> [row_index]")
        print("\nExample:")
        print("  python str.py single 'CCO'")
        print("  python str.py single 'CCO' 5")
        print("  python str.py batch 'CCO;CCN;CCC'")
        print("  python str.py batch 'CCO;CCN;CCC' 10")
        print("  python str.py file smiles.txt")
        print("  python str.py file smiles.txt 3")
        sys.exit(1)

    mode = sys.argv[1].lower()
    smi_input = sys.argv[2]
    
    # 获取行索引参数，默认为0
    row_index = 0
    if len(sys.argv) > 3:
        try:
            row_index = int(sys.argv[3])
        except ValueError:
            print("行索引必须是整数")
            sys.exit(1)

    if mode == "single":
        single_predict(smi_input, row_index)
    elif mode == "batch":
        batch_predict(smi_input.split(';'), row_index)
    elif mode == "file":
        batch_predict_from_file(smi_input, row_index)
    else:
        print("模式必须是 'single'、'batch' 或 'file'")