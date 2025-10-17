import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import joblib
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import queue
import os
from pathlib import Path

# 页面设置
st.set_page_config(
    page_title="分子性质预测与模型分析平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化指纹生成器
def initialize_fingerprint_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

ZERO_FP_LIST = [0] * 2048

def smiles_to_fp_list(smiles: str, mfpgen) -> List[int]:
    """将SMILES字符串转换为2048位摩根指纹的整数列表"""
    # 预处理SMILES字符串，处理一些常见问题
    smiles = smiles.strip()
    
    # 如果字符串为空，返回零指纹
    if not smiles:
        return ZERO_FP_LIST
    
    # 直接尝试解析SMILES
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 检查分子是否有效（例如，没有不合理的价态）
            try:
                Chem.SanitizeMol(mol)
                fp = mfpgen.GetFingerprint(mol)
                return list(fp)
            except Exception as sanitize_error:
                st.warning(f"SMILES '{smiles}' 分子无法被规范化: {str(sanitize_error)}")
        else:
            st.warning(f"无效的SMILES: {smiles}")
    except Exception as e:
        # 记录错误但不中断程序
        st.warning(f"解析SMILES时出错 '{smiles}': {str(e)}")
        pass
    
    # 如果直接解析失败，尝试其他处理方式
    try:
        # 移除末尾可能存在的逗号（只移除最后的逗号）
        if smiles.endswith(','):
            cleaned_smiles = smiles.rstrip(',')
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is not None:
                # 检查分子是否有效
                try:
                    Chem.SanitizeMol(mol)
                    fp = mfpgen.GetFingerprint(mol)
                    return list(fp)
                except Exception as sanitize_error:
                    st.warning(f"清理后的SMILES '{cleaned_smiles}' 分子无法被规范化: {str(sanitize_error)}")
            else:
                st.warning(f"清理后仍然无效的SMILES: {cleaned_smiles}")
    except Exception as e:
        # 记录错误但不中断程序
        st.warning(f"清理SMILES时出错 '{smiles}': {str(e)}")
        pass
    
    # 如果还是失败，返回零指纹
    st.warning(f"无法从SMILES生成指纹: {smiles}，使用零指纹")
    return ZERO_FP_LIST

def merge_multiple_fps(smiles_cell, mfpgen) -> List[int]:
    """处理可能包含多个SMILES字符串的单元格，支持点分分子和分号分隔的多个SMILES"""
    # 处理输入为字符串的情况
    smiles_input = str(smiles_cell).strip()
    
    # 如果输入为空，返回零指纹
    if not smiles_input:
        return ZERO_FP_LIST
    
    # 根据您的说明：
    # - 使用分号(;)区分不同的预测任务
    # - 使用点(.)连接同一预测任务中的不同分子片段
    # - 使用逗号(,)区分同一行中的多个预测任务（在tab1中使用）
    task_groups = [task.strip() for task in smiles_input.split(';') if task.strip()]
    
    if not task_groups:
        return ZERO_FP_LIST
    
    valid_fps = []
    error_messages = []
    
    for task in task_groups:
        # 每个任务组中可能包含多个分子片段（如Br.CC[NH3+]c1ccccc1）
        # 这些片段需要被单独转换为指纹后再合并
        fragments = [frag.strip() for frag in task.split('.') if frag.strip()]
        
        fragment_fps = []
        for frag in fragments:
            try:
                fp = smiles_to_fp_list(frag, mfpgen)
                # 只有非零指纹才加入（避免全零指纹干扰）
                if any(fp):
                    fragment_fps.append(fp)
                else:
                    error_messages.append(f"无法为分子片段生成有效指纹: {frag}")
            except Exception as e:
                error_msg = f"处理分子片段 '{frag}' 时出错: {str(e)}"
                error_messages.append(error_msg)
                # 忽略单个分子片段的错误，继续处理其他片段
                continue
        
        # 如果成功生成了所有分子片段的指纹，将它们合并
        if fragment_fps:
            arr_fps = np.array(fragment_fps)
            merged_fp = np.bitwise_or.reduce(arr_fps, axis=0)
            valid_fps.append(merged_fp.tolist())
        else:
            error_messages.append(f"任务 '{task}' 中的所有分子片段都无法生成有效指纹")
    
    # 如果有错误信息，汇总显示
    if error_messages:
        st.warning(f"在处理多个SMILES时遇到以下问题:\n" + "\n".join(error_messages))
    
    # 如果有有效的指纹，合并它们
    if valid_fps:
        arr_fps = np.array(valid_fps)
        merged_fp = np.bitwise_or.reduce(arr_fps, axis=0)
        return merged_fp.tolist()
    else:
        # 如果没有有效指纹，返回零指纹
        return ZERO_FP_LIST

# 加载数据和模型
@st.cache_resource
def load_data_and_model():
    """加载数据和模型，使用缓存提高性能"""
    try:
        df = pd.read_csv('3.csv')
        model = joblib.load(r'trained_model_xgboost.pkl')
        return df, model
    except Exception as e:
        st.error(f"加载数据或模型失败: {e}")
        return None, None

# 加载评估数据
@st.cache_resource
def load_evaluation_data():
    """加载模型评估相关数据"""
    try:
        eval_df = pd.read_csv('output_20250826_130253/evaluation_results_xgboost.csv')
        feat_df = pd.read_csv('output_20250826_130253/feature_importance_xgboost.csv')
        pred_df = pd.read_csv('output_20250826_130253/model_predictions_xgboost.csv')
        return eval_df, feat_df, pred_df
    except Exception as e:
        st.error(f"加载评估数据失败: {e}")
        return None, None, None

# 处理特征重要性数据 - 合并fp_开头的特征
def process_feature_importance(feat_df):
    """处理特征重要性数据，合并fp_开头的特征"""
    # 分离fp特征和非fp特征
    fp_features = feat_df[feat_df['raw_name'].str.startswith('fp_')]
    non_fp_features = feat_df[~feat_df['raw_name'].str.startswith('fp_')]
    
    # 计算fp特征的总重要性
    fp_total_importance = fp_features['importance'].sum()
    
    # 创建新的特征重要性DataFrame
    processed_features = non_fp_features.copy()
    
    # 添加合并后的fp特征
    fp_row = pd.DataFrame({
        'raw_name': ['fp_all_combined'],
        'importance': [fp_total_importance]
    })
    
    processed_features = pd.concat([processed_features, fp_row], ignore_index=True)
    
    # 按重要性排序
    processed_features = processed_features.sort_values('importance', ascending=False)
    
    return processed_features

# 生成SHAP分析图
def generate_shap_plot(model, X_data, feature_names):
    """生成SHAP特征重要性摘要图"""
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值（使用子采样以提高性能）
        sample_size = min(100, len(X_data))
        X_sample = X_data[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # 确保shap_values是正确的形状
        if isinstance(shap_values, list):
            # 对于某些模型，shap_values可能是一个列表
            shap_values = shap_values[0]
        
        # 处理特征名称 - 合并fp_开头的特征
        fp_mask = [name.startswith('fp_') for name in feature_names]
        non_fp_indices = [i for i, is_fp in enumerate(fp_mask) if not is_fp]
        fp_indices = [i for i, is_fp in enumerate(fp_mask) if is_fp]
        
        # 如果有fp特征，合并它们
        if fp_indices:
            # 合并fp特征的SHAP值
            fp_shap_sum = np.sum(shap_values[:, fp_indices], axis=1)
            
            # 创建新的SHAP值数组
            new_shap_values = np.zeros((sample_size, len(non_fp_indices) + 1))
            new_shap_values[:, 0] = fp_shap_sum  # 第一个位置放合并的fp特征
            new_shap_values[:, 1:] = shap_values[:, non_fp_indices]
            
            # 创建新的特征名称
            new_feature_names = ['fp_all_combined'] + [feature_names[i] for i in non_fp_indices]
            
            # 确保数据也是相应形状
            X_sample_modified = np.zeros((sample_size, len(non_fp_indices) + 1))
            X_sample_modified[:, 0] = np.sum(X_sample[:, fp_indices], axis=1)  # 合并的fp特征
            X_sample_modified[:, 1:] = X_sample[:, non_fp_indices]
        else:
            new_shap_values = shap_values
            new_feature_names = feature_names
            X_sample_modified = X_sample
        
        # 创建输出目录（如果不存在）
        if not os.path.exists('shap_plots'):
            os.makedirs('shap_plots')
        
        # 1. 创建SHAP摘要图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            new_shap_values, 
            X_sample_modified, 
            feature_names=new_feature_names,
            max_display=20,  # 显示前20个最重要的特征
            show=False
        )
        
        plt.title('SHAP特征重要性摘要图（fp特征已合并）')
        plt.tight_layout()
        
        # 保存图像
        summary_plot_path = os.path.join('shap_plots', 'shap_summary_plot.png')
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 创建SHAP热力图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            new_shap_values,
            X_sample_modified,
            feature_names=new_feature_names,
            plot_type="compact_dot",
            max_display=20,
            show=False
        )
        
        plt.title('SHAP特征重要性热力图（fp特征已合并）')
        plt.tight_layout()
        
        # 保存热力图
        heatmap_plot_path = os.path.join('shap_plots', 'shap_heatmap_plot.png')
        plt.savefig(heatmap_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 创建SHAP蜂群图
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            new_shap_values,
            X_sample_modified,
            feature_names=new_feature_names,
            plot_type="violin",
            max_display=20,
            show=False
        )
        
        plt.title('SHAP特征重要性小提琴图（fp特征已合并）')
        plt.tight_layout()
        
        # 保存蜂群图
        violin_plot_path = os.path.join('shap_plots', 'shap_violin_plot.png')
        plt.savefig(violin_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 创建特征重要性箱线图
        # 计算每个特征的绝对SHAP值（表示重要性）
        abs_shap_values = np.abs(new_shap_values)
        feature_importance = np.mean(abs_shap_values, axis=0)
        
        # 创建箱线图数据
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 为前10个最重要的特征创建箱线图
        top_features_idx = np.argsort(feature_importance)[-10:]  # 取前10个特征
        box_data = [new_shap_values[:, i] for i in top_features_idx]
        box_labels = [new_feature_names[i] for i in top_features_idx]
        
        # 绘制箱线图
        bp = ax.boxplot(box_data, labels=box_labels, vert=False, patch_artist=True)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('SHAP值')
        ax.set_title('前10个最重要特征的SHAP值分布箱线图')
        plt.tight_layout()
        
        # 保存箱线图
        boxplot_path = os.path.join('shap_plots', 'shap_boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'summary': summary_plot_path,
            'heatmap': heatmap_plot_path,
            'violin': violin_plot_path,
            'boxplot': boxplot_path
        }
        
    except Exception as e:
        st.error(f"SHAP分析失败: {e}")
        return None

# 选择基准行
def select_baseline_row(df, pce_threshold=15.0):
    """根据PCE性能选择基准行"""
    candidates = df[df['JV_default_PCE'] >= pce_threshold]
    
    if len(candidates) == 0:
        st.warning(f"没有找到PCE高于{pce_threshold}的行，将使用PCE最高的行作为基准")
        return df.loc[df['JV_default_PCE'].idxmax()].copy()
    
    median_pce = candidates['JV_default_PCE'].median()
    closest_idx = (candidates['JV_default_PCE'] - median_pce).abs().idxmin()
    
    baseline_row = candidates.loc[closest_idx].copy()
    st.success(f"选择的基准行ID: {closest_idx}, 原始PCE: {baseline_row['JV_default_PCE']:.2f}%")
    
    return baseline_row

# 预测函数
def predict_with_custom_smiles(baseline_row, smiles, model, mfpgen):
    """用自定义SMILES修改基准行的指纹列并进行预测"""
    try:
        new_fp = merge_multiple_fps(smiles, mfpgen)
        
        # 检查指纹是否全为零（表示所有SMILES都无效）
        if not any(new_fp):
            # 提供更详细的错误信息
            if ';' in str(smiles):
                raise ValueError("无法从提供的SMILES生成有效指纹，可能是因为所有SMILES格式都不正确。"
                               "请检查每个SMILES字符串是否有效，特别是用分号分隔的部分。")
            else:
                raise ValueError("无法从提供的SMILES生成有效指纹，可能是因为SMILES格式不正确。"
                               "请检查输入的SMILES字符串是否符合规范。")
        
        modified_row = baseline_row.copy()
        fp_cols = [col for col in modified_row.index if col.startswith('fp_')]
        
        # 检查特征列数量是否匹配
        if len(fp_cols) != len(new_fp):
            raise ValueError(f"指纹长度不匹配: 期望 {len(fp_cols)} 位，实际得到 {len(new_fp)} 位。"
                           f"请确保使用的指纹生成器参数与训练时一致。")
            
        modified_row[fp_cols] = new_fp
        
        if 'JV_default_PCE' in modified_row.index:
            X_pred = modified_row.drop('JV_default_PCE').values.reshape(1, -1)
        else:
            X_pred = modified_row.values.reshape(1, -1)
        
        predicted_pce = model.predict(X_pred)[0]
        
        # 确保返回的是Python原生float类型，而不是numpy.float类型
        return float(predicted_pce), modified_row
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise ValueError(f"预测过程中发生错误: {str(e)}")

# 批量处理函数
def process_batch_file(file_content, baseline_row, model, mfpgen, progress_queue):
    """处理批量SMILES文件"""
    results = []
    errors = []
    lines = file_content.decode('utf-8').split('\n')
    
    # 计算总任务数（考虑逗号分隔的任务）
    total_tasks = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # 每行中用逗号分隔的任务数
        tasks_in_line = [task.strip() for task in line.split(',') if task.strip()]
        total_tasks += len(tasks_in_line)
    
    processed = 0
    task_id = 1
    
    # 显示总任务数
    progress_queue.put((0, total_tasks, f"开始处理 {total_tasks} 个任务"))
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # 根据您的说明：使用逗号(,)区分不同的预测任务，使用分号(;)区分同一预测任务中的不同SMILES字符串
        # 将每行按逗号分割成多个任务
        tasks = [task.strip() for task in line.split(',') if task.strip()]
        
        for task_smiles in tasks:
            try:
                predicted_pce, modified_row = predict_with_custom_smiles(
                    baseline_row, task_smiles, model, mfpgen
                )
                
                result = {
                    "task_id": task_id,
                    "smiles": task_smiles,
                    "predicted_pce": float(round(predicted_pce, 2)),  # 确保是Python float类型
                    "baseline_pce": float(round(baseline_row['JV_default_PCE'], 2)),  # 确保是Python float类型
                    "change": float(round(predicted_pce - baseline_row['JV_default_PCE'], 2)),  # 确保是Python float类型
                    "modified_row": modified_row.to_dict()
                }
                
                results.append(result)
                processed += 1
                task_id += 1
                progress_queue.put((processed, total_tasks, f"处理中: {task_smiles}"))
                
            except Exception as e:
                error_msg = f"处理任务 '{task_smiles}' 时出错: {e}"
                errors.append(error_msg)
                processed += 1
                task_id += 1
                progress_queue.put((processed, total_tasks, f"处理中: {task_smiles} (失败)"))
                
                # 发送错误信息到进度队列
                progress_queue.put(("error", error_msg))
    
    # 发送错误信息
    if errors:
        for error in errors:
            progress_queue.put(("error", error))
    
    results.sort(key=lambda x: x['change'], reverse=True)
    progress_queue.put(("complete", f"成功处理 {len(results)} 个任务，{len(errors)} 个任务失败"))
    
    return results

# 后台处理线程
def process_in_background(file_content, baseline_row, model, mfpgen, progress_queue):
    """后台处理线程函数"""
    try:
        results = process_batch_file(file_content, baseline_row, model, mfpgen, progress_queue)
        progress_queue.put(("results", results))
    except Exception as e:
        progress_queue.put(("error", f"处理失败: {e}"))

# 进度更新函数
def update_progress(progress_queue, total_tasks):
    """更新处理进度"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        try:
            message = progress_queue.get(timeout=30)
            
            if message[0] == "complete":
                progress_bar.progress(1.0)
                status_text.success(message[1])
                break
            elif message[0] == "error":
                progress_bar.empty()
                status_text.error(message[1])
                break
            elif message[0] == "results":
                st.session_state.batch_results = message[1]
                break
            else:
                processed, total, status = message
                percent = processed / total
                progress_bar.progress(percent)
                status_text.info(f"{status} - 进度: {percent*100:.1f}% ({processed}/{total})")
                
        except queue.Empty:
            break

# 导出JSON函数
def export_to_json(results):
    """将结果导出为JSON文件"""
    export_data = []
    for result in results:
        export_result = result.copy()
        export_result.pop('modified_row', None)
        
        # 确保所有numpy数据类型都被转换为Python原生类型
        for key, value in export_result.items():
            if isinstance(value, (np.integer, np.floating)):
                export_result[key] = value.item()
            elif isinstance(value, np.ndarray):
                export_result[key] = value.tolist()
        
        export_data.append(export_result)
    
    return json.dumps(export_data, indent=2)

# 特征重要性可视化
def plot_feature_importance(feat_df, top_n=20):
    """绘制特征重要性图"""
    top_features = feat_df.nlargest(top_n, 'importance')
    
    fig = px.bar(
        top_features,
        x='importance',
        y='raw_name',
        orientation='h',
        title=f'Top {top_n} Feature Importances (fp特征已合并)',
        labels={'importance': 'Importance', 'raw_name': 'Features'}
    )
    
    fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
    return fig

# 模型性能评估可视化
def plot_model_performance(eval_df, pred_df):
    """绘制模型性能评估图"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Actual vs Predicted', 'Residuals Distribution', 'Prediction Error', 'Performance Metrics'),
        specs=[[{"colspan": 2}, None], [{}, {}]]
    )
    
    # 实际值vs预测值
    fig.add_trace(go.Scatter(x=pred_df['Actual'], y=pred_df['Predicted'], mode='markers', name='Predictions'), row=1, col=1)
    
    min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
    max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')), row=1, col=1)
    
    # 残差分布
    residuals = pred_df['Actual'] - pred_df['Predicted']
    fig.add_trace(go.Histogram(x=residuals, nbinsx=20, name='Residuals'), row=2, col=1)
    
    # 性能指标
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [eval_df['MSE'].iloc[0], eval_df['RMSE'].iloc[0], eval_df['MAE'].iloc[0]]
    fig.add_trace(go.Bar(x=metrics, y=values, name='Error Metrics'), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False, title_text="Model Performance Evaluation")
    return fig

# 主应用
def main():
    st.title("🧪 分子性质预测与模型分析平台")
    st.markdown("---")
    
    # 初始化会话状态
    if 'baseline_row' not in st.session_state:
        st.session_state.baseline_row = None
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'mfpgen' not in st.session_state:
        st.session_state.mfpgen = initialize_fingerprint_generator()
    if 'progress_queue' not in st.session_state:
        st.session_state.progress_queue = queue.Queue()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'shap_plot_generated' not in st.session_state:
        st.session_state.shap_plot_generated = False
    if 'shap_plot_paths' not in st.session_state:
        st.session_state.shap_plot_paths = {}
    
    # 加载数据
    df, model = load_data_and_model()
    eval_df, feat_df, pred_df = load_evaluation_data()
    
    # 处理特征重要性数据（合并fp特征）
    if feat_df is not None:
        processed_feat_df = process_feature_importance(feat_df)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        if df is not None and model is not None:
            pce_threshold = st.slider("PCE阈值 (%)", min_value=0.0, max_value=float(df['JV_default_PCE'].max()), value=15.0, step=0.1)
            
            if st.button("选择基准行"):
                with st.spinner("正在选择基准行..."):
                    st.session_state.baseline_row = select_baseline_row(df, pce_threshold)
        
        st.markdown("---")
        st.header("📊 分析选项")
        show_visualization = st.checkbox("显示预测结果可视化", value=True)
        show_analysis = st.checkbox("显示模型分析", value=True)
        
        if show_analysis and df is not None and model is not None:
            if st.button("生成SHAP分析图"):
                with st.spinner("正在生成SHAP分析图..."):
                    # 准备数据用于SHAP分析
                    X_data = df.drop('JV_default_PCE', axis=1).values
                    feature_names = df.drop('JV_default_PCE', axis=1).columns.tolist()
                    
                    shap_plot_paths = generate_shap_plot(model, X_data, feature_names)
                    if shap_plot_paths:
                        st.session_state.shap_plot_paths = shap_plot_paths
                        st.session_state.shap_plot_generated = True
                        st.success("SHAP分析图生成成功！")
    
    # 主界面
    if df is None or model is None:
        st.error("无法加载数据或模型，请检查文件路径")
        return
    
    # 模型性能评估
    if show_analysis and eval_df is not None:
        st.subheader("🏆 模型性能评估")
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("MSE", f"{eval_df['MSE'].iloc[0]:.4f}")
        with col2: st.metric("RMSE", f"{eval_df['RMSE'].iloc[0]:.4f}")
        with col3: st.metric("MAE", f"{eval_df['MAE'].iloc[0]:.4f}")
        with col4: st.metric("R²", f"{eval_df['R2'].iloc[0]:.4f}")
        
        if pred_df is not None:
            st.plotly_chart(plot_model_performance(eval_df, pred_df), use_container_width=True)
    
    # 特征重要性分析（使用处理后的数据）
    if show_analysis and processed_feat_df is not None:
        st.subheader("🔑 特征重要性分析（fp特征已合并）")
        top_n = st.slider("显示特征数量", 5, 50, 20, key="feat_slider")
        st.plotly_chart(plot_feature_importance(processed_feat_df, top_n), use_container_width=True)
        
        if st.button("下载处理后的特征重要性数据"):
            processed_feat_df.to_csv('processed_feature_importance.csv', index=False)
            st.success("处理后的特征重要性数据已保存")
    
    # SHAP分析图显示
    if show_analysis and st.session_state.shap_plot_generated:
        st.subheader("📊 SHAP特征重要性分析")
        try:
            # 创建选项卡显示不同类型的图表
            summary_tab, heatmap_tab, violin_tab, boxplot_tab = st.tabs([
                "摘要图", "热力图", "小提琴图", "箱线图"
            ])
            
            with summary_tab:
                st.image(st.session_state.shap_plot_paths['summary'])
                st.caption("SHAP特征重要性摘要图展示了每个特征对模型输出的影响。x轴表示SHAP值，红色表示高影响，蓝色表示低影响。")
            
            with heatmap_tab:
                st.image(st.session_state.shap_plot_paths['heatmap'])
                st.caption("SHAP热力图显示了特征值与SHAP值之间的关系，帮助理解特征如何影响模型预测。")
                
            with violin_tab:
                st.image(st.session_state.shap_plot_paths['violin'])
                st.caption("SHAP小提琴图展示了每个特征的SHAP值分布情况，更直观地显示特征重要性。")
                
            with boxplot_tab:
                st.image(st.session_state.shap_plot_paths['boxplot'])
                st.caption("SHAP箱线图显示了前10个最重要特征的SHAP值分布，可以观察到数据的四分位数和异常值。")
            
            # 提供所有图表的下载选项
            st.subheader("💾 下载SHAP分析图")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                with open(st.session_state.shap_plot_paths['summary'], "rb") as file:
                    st.download_button(
                        label="下载摘要图",
                        data=file,
                        file_name="shap_summary_plot.png",
                        mime="image/png"
                    )
            with col2:
                with open(st.session_state.shap_plot_paths['heatmap'], "rb") as file:
                    st.download_button(
                        label="下载热力图",
                        data=file,
                        file_name="shap_heatmap_plot.png",
                        mime="image/png"
                    )
            with col3:
                with open(st.session_state.shap_plot_paths['violin'], "rb") as file:
                    st.download_button(
                        label="下载小提琴图",
                        data=file,
                        file_name="shap_violin_plot.png",
                        mime="image/png"
                    )
            with col4:
                with open(st.session_state.shap_plot_paths['boxplot'], "rb") as file:
                    st.download_button(
                        label="下载箱线图",
                        data=file,
                        file_name="shap_boxplot.png",
                        mime="image/png"
                    )
        except Exception as e:
            st.error(f"显示SHAP图时出错: {e}")
    
    st.markdown("---")
    
    # 预测功能
    if st.session_state.baseline_row is not None:
        baseline_row = st.session_state.baseline_row
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("基准行PCE", f"{baseline_row['JV_default_PCE']:.2f}%")
        with col2: st.metric("基准行ID", baseline_row.name)
        with col3: st.metric("数据维度", f"{df.shape[0]}行 × {df.shape[1]}列")
        
        # 选项卡布局
        tab1, tab2, tab3 = st.tabs(["单次预测", "批量预测", "数据导出"])
        
        with tab1:
            st.header("🔍 单次预测")
            smiles_input = st.text_area("输入SMILES字符串", height=100, placeholder="输入单个SMILES或多个SMILES（用分号;分隔）")
            
            if st.button("开始预测", key="predict_single"):
                if smiles_input.strip():
                    with st.spinner("预测中..."):
                        try:
                            # 检查是否有逗号分隔的多个任务
                            tasks = [task.strip() for task in smiles_input.split(',') if task.strip()]
                            
                            if len(tasks) > 1:
                                # 处理多个任务
                                results = []
                                errors = []
                                for i, task_smiles in enumerate(tasks):
                                    try:
                                        predicted_pce, modified_row = predict_with_custom_smiles(
                                            baseline_row, task_smiles, model, st.session_state.mfpgen
                                        )
                                        results.append({
                                            "task_id": i + 1,
                                            "smiles": task_smiles,
                                            "predicted_pce": float(round(predicted_pce, 2)),
                                            "baseline_pce": float(round(baseline_row['JV_default_PCE'], 2)),
                                            "change": float(round(predicted_pce - baseline_row['JV_default_PCE'], 2))
                                        })
                                    except Exception as e:
                                        error_msg = f"任务 {i+1} ('{task_smiles}') 处理失败: {str(e)}"
                                        errors.append(error_msg)
                                        st.error(error_msg)  # 显示具体错误信息
                                
                                # 显示错误信息
                                if errors:
                                    for error in errors:
                                        st.warning(error)
                                
                                # 显示结果表格
                                if results:
                                    st.success(f"成功处理 {len(results)} 个任务")
                                    results_df = pd.DataFrame(results)
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # 显示最佳结果
                                    if results:
                                        best_result = max(results, key=lambda x: x['change'])
                                        st.subheader("最佳结果")
                                        st.success(f"任务ID: {best_result['task_id']}, SMILES: {best_result['smiles']}")
                                        st.success(f"预测PCE: **{best_result['predicted_pce']:.2f}%**")
                                        st.info(f"变化: **{best_result['change']:+.2f}%**")
                                else:
                                    st.error("所有任务都处理失败，请检查SMILES格式")
                            else:
                                # 处理单个任务
                                predicted_pce, modified_row = predict_with_custom_smiles(
                                    baseline_row, smiles_input, model, st.session_state.mfpgen
                                )
                                col1, col2 = st.columns(2)
                                with col1: st.success(f"预测PCE: **{predicted_pce:.2f}%**")
                                with col2: st.info(f"变化: **{predicted_pce - baseline_row['JV_default_PCE']:+.2f}%**")
                        except Exception as e:
                            st.error(f"预测失败: {e}")
                else:
                    st.warning("请输入SMILES字符串")
        
        with tab2:
            st.header("📁 批量预测")
            uploaded_file = st.file_uploader("上传SMILES文件", type=['txt'], help="每行一个任务，多个SMILES用;分隔，不同任务用换行分隔")
            
            if uploaded_file is not None and not st.session_state.processing:
                if st.button("开始批量预测", key="predict_batch"):
                    st.session_state.processing = True
                    progress_queue = st.session_state.progress_queue
                    
                    # 启动后台处理线程
                    thread = threading.Thread(
                        target=process_in_background,
                        args=(uploaded_file.getvalue(), baseline_row, model, st.session_state.mfpgen, progress_queue),
                        daemon=True
                    )
                    thread.start()
                    
                    # 显示进度
                    update_progress(progress_queue, len(uploaded_file.getvalue().decode('utf-8').split('\n')))
                    st.session_state.processing = False
            
            # 显示批量结果
            if st.session_state.batch_results:
                results = st.session_state.batch_results
                st.success(f"成功处理 {len(results)} 个任务")
                
                results_df = pd.DataFrame([{
                    '任务ID': r['task_id'],
                    'SMILES': r['smiles'],
                    '预测PCE (%)': r['predicted_pce'],
                    '变化 (%)': r['change']
                } for r in results])
                
                st.dataframe(results_df, use_container_width=True)
                
                # 可视化
                if show_visualization:
                    st.subheader("📈 预测结果可视化")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_hist = px.histogram(results_df, x='预测PCE (%)', nbins=20, title='预测PCE分布')
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        changes = [r['change'] for r in results]
                        fig_changes = px.histogram(results_df, x='变化 (%)', nbins=20, title='PCE变化分布')
                        st.plotly_chart(fig_changes, use_container_width=True)
                        
                        st.metric("平均变化", f"{np.mean(changes):+.2f}%")
                        st.metric("最大提升", f"{max(changes):+.2f}%")
                        st.metric("最大下降", f"{min(changes):+.2f}%")
        
        with tab3:
            st.header("💾 数据导出")
            
            if st.session_state.batch_results:
                results = st.session_state.batch_results
                
                # 导出JSON
                json_data = export_to_json(results)
                st.download_button("下载预测结果 (JSON)", data=json_data, file_name="predictions.json", mime="application/json")
                
                # 导出CSV
                csv_data = pd.DataFrame([{
                    'task_id': r['task_id'],
                    'smiles': r['smiles'],
                    'predicted_pce': r['predicted_pce'],
                    'baseline_pce': r['baseline_pce'],
                    'change': r['change']
                } for r in results]).to_csv(index=False)
                
                st.download_button("下载预测结果 (CSV)", data=csv_data, file_name="predictions.csv", mime="text/csv")
            else:
                st.info("暂无批量预测结果可供导出")
            
            # 导出基准行数据
            if st.session_state.baseline_row is not None:
                baseline_df = pd.DataFrame([st.session_state.baseline_row])
                baseline_csv = baseline_df.to_csv(index=False)
                st.download_button("下载基准行数据", data=baseline_csv, file_name="baseline.csv", mime="text/csv")
    
    else:
        st.info("请在侧边栏选择基准行以开始预测")

if __name__ == "__main__":
    main()