import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
from zhipuai import ZhipuAI  # 替换为新的导入方式
import json  # 新增：导入 json 模块

# 确保输出文件夹存在
output_folder = 'd:/newml/data_splits'
os.makedirs(output_folder, exist_ok=True)

# 加载数据
file_path = 'd:/newml/output_expanded_additives_pubchem_encoded.csv'
df = pd.read_csv(file_path)

# 确保 JV_default_PCE 列为数值类型
df['JV_default_PCE'] = pd.to_numeric(df['JV_default_PCE'], errors='coerce')

# 清理列名，确保只包含 ASCII 字符
df.columns = [str(col).encode('ascii', errors='ignore').decode() for col in df.columns]

# 清理字符串类型的列，确保只包含 ASCII 字符
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(lambda x: str(x).encode('ascii', errors='ignore').decode())

# 准备特征和目标变量
X = df.drop(columns=['JV_default_PCE'])
y = df['JV_default_PCE']
X = X.fillna(-1)  # 确保没有缺失值

# 按照 7:2:1 的比例划分数据集，随机种子保持不变
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=833)  # 70% 训练集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=43)  # 20% 验证集，10% 测试集

# 保存划分后的数据集到不同的CSV文件
X_train.to_csv(os.path.join(output_folder, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_folder, 'y_train.csv'), index=False)
X_val.to_csv(os.path.join(output_folder, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(output_folder, 'y_val.csv'), index=False)
X_test.to_csv(os.path.join(output_folder, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_folder, 'y_test.csv'), index=False)

# 加载保存的模型
model_path = 'd:/newml/best_model.cbm'
model = CatBoostRegressor()
model.load_model(model_path)

# Streamlit 应用
st.title("Perovskite Additives Performance Prediction")

# 新增：在全局作用域中初始化 prediction 变量
prediction = None

# 新增：使用容器优化布局
with st.container():
    col1, col2 = st.columns([3, 2])
    with col1:
        # 基准样本选择优化
        baseline_df = df[df['JV_default_PCE'] > 20].head(10)
        selected_baseline = st.selectbox("选择基准样本", baseline_df.index, 
                                       help="选择性能超过20的基准样本作为参考")
        
        # 获取选定的基准样本
        baseline_sample = baseline_df.loc[selected_baseline]

        # 新增：基准样本性能展示样式优化
        st.markdown(f"""
        <div style="background:#f0f2f6;padding:10px;border-radius:5px">
            📊 基准性能值：<span style="color:#0068c9;font-weight:bold">{baseline_sample['JV_default_PCE']:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 新增：特征选择说明
        st.markdown("### 特征选择")
        st.caption("请先选择需要调整的特征")

# 新增：特征输入分组
with st.expander("🔧 特征调整", expanded=True):
    # 初始化可编辑的特征列表
    editable_features_compounds = [col for col in baseline_sample.index if col.startswith('Perovskite_additives_compounds')]
    editable_features_concentrations = [col for col in baseline_sample.index if col.startswith('Perovskite_additives_concentrations')]

    col_compound, col_concentration = st.columns(2)
    with col_compound:
        selected_compound = st.selectbox("化合物特征", editable_features_compounds,
                                       index=0, key='compound_select')
    with col_concentration:
        selected_concentration = st.selectbox("浓度特征", editable_features_concentrations,
                                            index=0, key='concentration_select')

    # 输入字段布局优化
    input_col1, input_col2 = st.columns(2)
    input_values = {}
    with input_col1:
        input_values[selected_compound] = st.number_input(
            f"{selected_compound}", 
            value=int(baseline_sample[selected_compound]),
            min_value=0, max_value=100, step=1,
            help="调整化合物特征值 (0-100)"
        )
    with input_col2:
        input_values[selected_concentration] = st.number_input(
            f"{selected_concentration}",
            value=int(baseline_sample[selected_concentration]),
            min_value=0, max_value=100, step=1,
            help="调整浓度特征值 (0-100)"
        )

# 新增：预测按钮样式优化
predict_col, _ = st.columns([2, 8])
with predict_col:
    if st.button("🚀 开始预测", type="primary", use_container_width=True):
        # 创建预测输入数据
        input_data = baseline_sample.copy()
        for feature, value in input_values.items():
            input_data[feature] = value
        
        # 进行预测
        input_data = input_data.drop('JV_default_PCE')  # 移除目标变量
        input_data = pd.DataFrame([input_data])  # 转换为DataFrame
        prediction = model.predict(input_data)[0]
        
        # 显示预测结果
        st.success(f"预测的目标性能 (JV_default_PCE): {prediction:.4f}")

        # 新增：计算模型性能指标
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # 新增：保存预测结果、模型性能指标、基准样本和用户修改项到临时文件
        temp_data = {
            'prediction': prediction,
            'train_mse': train_mse, 'val_mse': val_mse, 'test_mse': test_mse,
            'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
            'baseline_sample': baseline_sample.to_dict(),
            'input_values': input_values
        }
        
        temp_file_path = 'e:/newml/temp_prediction_data.json'
        with open(temp_file_path, 'w') as f:
            json.dump(temp_data, f)

        # 导出预测数据
        prediction_df = input_data.copy()
        prediction_df['Predicted_JV_default_PCE'] = prediction
        csv_prediction = prediction_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "导出预测数据",
            data=csv_prediction,
            file_name="prediction_result.csv",
            mime="text/csv"
        )
        
        # 导出原始baseline样本
        baseline_export_df = pd.DataFrame([baseline_sample])
        csv_baseline = baseline_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "导出原始Baseline",
            data=csv_baseline,
            file_name="baseline_sample.csv",
            mime="text/csv"
        )

        # SHAP 分析
        st.header("SHAP 分析")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # 可视化 SHAP 值
        st.subheader("SHAP 值")
        shap.force_plot(explainer.expected_value, shap_values, input_data, show=False)
        st.pyplot(plt.gcf())
        plt.close()

        # 正负贡献可视化
        st.subheader("正负贡献")
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        plt.close()

        # 新增：生成实验设计方案
        st.header("实验设计助手")
        st.subheader("正在生成实验设计方案...")
        progress_bar = st.progress(0)

        # 从临时文件中读取数据
        temp_file_path = 'e:/newml/temp_prediction_data.json'
        with open(temp_file_path, 'r') as f:
            temp_data = json.load(f)

        prediction = temp_data['prediction']
        baseline_sample = pd.Series(temp_data['baseline_sample'])
        input_values = temp_data['input_values']

        # 新增：初始化ZhipuAI客户端
        client = ZhipuAI(api_key="4dc712e71da74ead8f16651dccbf71bf.BPESIGBsJhKo9rYz")

        # 修改：优化后的prompt结构，增加格式要求，明确中文输出
        response = client.chat.completions.create(
            model="glm-4-air-250414",
            messages=[
                {"role": "system", "content": "你是一个材料科学实验设计专家，请用中文生成规范的实验方案"},
                {"role": "user", "content": f"""请基于以下实验数据生成详细的中文实验设计方案：
# 基础数据
1. 基准性能值：{baseline_sample['JV_default_PCE']:.4f}
2. 调整特征：{', '.join(input_values.keys())}
3. 特征调整值：{', '.join(map(str, input_values.values()))}
4. 预测性能值：{prediction:.4f}

# 方案要求
## 实验步骤
- 分步骤描述具体操作流程
- 包含材料准备和仪器设置

## 参数设置
- 说明每个调整参数的设置依据
- 结合领域知识解释参数合理性

## 结果预测
- 预测可能的结果范围（置信区间）
- 分析不同结果的可能性

## 验证方法
- 提出3种不同的验证方案
- 包含对照实验设计

## 风险评估
- 列出3个主要风险点
- 每个风险点需提供应对措施

请使用以下Markdown格式模板：
### 实验设计方案
#### 1. 实验步骤
...
#### 2. 参数设置依据
...
"""}
            ]
        )

        # 更新进度条
        progress_bar.progress(100)

        # 修改：增大文本区域高度并添加导出功能
        st.subheader("生成的实验设计方案")
        experiment_design = response.choices[0].message.content
        st.text_area("实验方案内容", 
                   value=experiment_design, 
                   height=400,  # 高度从300增加到400
                   key="experiment_design")
        
        # 新增：实验方案导出功能
        st.download_button(
            label="📥 导出实验方案",
            data=experiment_design,
            file_name="experiment_design.md",
            mime="text/markdown",
            key="export_design"
        )

# 预测结果展示优化
if prediction is not None:
    st.markdown(f"""
    <div style="background:#e6f4ea;padding:20px;border-radius:10px;margin-top:20px">
        <h3 style="color:#137333">📈 预测结果</h3>
        <p style="font-size:1.2rem">预测性能值：<span style="color:#137333;font-weight:bold">{prediction:.4f}</span></p>
    </div>
    """, unsafe_allow_html=True)

    # 导出按钮布局优化
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "📥 导出预测数据",
            data=csv_prediction,
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True
        )
    with export_col2:
        st.download_button(
            "📥 导出原始Baseline",
            data=csv_baseline,
            file_name="baseline_sample.csv",
            mime="text/csv",
            use_container_width=True
        )

# 侧边栏优化
st.sidebar.markdown("## 全局SHAP分析配置")
selected_feature = st.sidebar.selectbox("分析特征", X.columns, 
                                      help="选择需要分析的特征重要性")
if selected_feature:
    with st.sidebar.expander("分析说明"):
        st.write("SHAP值表示特征对预测结果的贡献程度：")
        st.markdown("- 🔴 正值：提升预测结果")
        st.markdown("- 🔵 负值：降低预测结果")
    
    # 新增：分析加载提示
    with st.spinner("正在生成SHAP分析..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # 确保 shap_values 是一个矩阵
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, 1)
        
        st.sidebar.subheader(f"{selected_feature} 的 SHAP 分析")
        shap.dependence_plot(selected_feature, shap_values, X.values, feature_names=X.columns.tolist(), show=False)
        st.sidebar.pyplot(plt.gcf())
        plt.close()
