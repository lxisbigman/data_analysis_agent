##code_executes.py
import re
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import logging  # 新增：引入logging模块
from contextlib import redirect_stdout
from io import StringIO
from memory.save_memory import init_long_term_memory, get_history_sessions, load_memory, save_memory,build_prompt_with_memory

# 初始化日志（确保能打印详细错误）
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_and_fix_python_code(query, df, max_retries=3):
    """模仿Agent的3次重试机制，智能错误驱动修复Python绘图代码"""

    # 🔥 步骤1：自动数据清洗（防NaN）
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    text_cols = df_clean.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_clean[col] = df_clean[col].fillna("未知")
    df_clean = df_clean.dropna(how='all')
    logger.info(f"数据清洗完成：{df.shape} → {df_clean.shape}")
    df = df_clean

    # 🔥 步骤2：基础提示（含清洗后数据）
    base_prompt = f"""{query}
[系统提示：
✅ 已清洗{len(df)}行数据，字段：{list(df.columns)}
✅ 缺失值已处理（数值=均值，文本=未知）
⚠️ 强制要求：
1. 每张图：plt.figure(figsize=(10,6)) → 绘图 → plt.savefig('plot_X.png') → plt.close()
2. 必须设置：plt.title('明确标题')、plt.xlabel('X轴含义')、plt.ylabel('Y轴含义')
3. 导入：import numpy as np（雷达图用）
4. 字段名：{list(df.columns)}（不要猜测！）
示例：
import numpy as np
plt.figure(figsize=(10, 6))
plt.bar(df['city'], df['comprehensive_score'])
plt.title('各城市综合评分对比')  # 明确标题
plt.xlabel('城市名称')  # X轴含义
plt.ylabel('综合评分（0-100）')  # Y轴含义
plt.savefig('plot_0.png')
plt.close()
print("图片总数: 1")]"""

    history = []
    final_code = None
    final_result = None

    for attempt in range(max_retries):
        st.markdown(f"### 🔄 尝试 {attempt + 1}/{max_retries}")

        # 🔥 步骤3：动态提示（错误驱动）
        if attempt == 0:
            current_prompt = base_prompt
        else:
            error_summary = history[-1]["content"][:200]  # 上次错误摘要
            current_prompt = f"""{base_prompt}
[错误修复指导：
上次错误：{error_summary}
请精准修改：
- 如果TypeError：检查plt.figure()和plt.close()
- 如果KeyError：检查字段名{list(df.columns)}
- 如果ValueError：检查数据类型或NaN]"""

        with st.spinner(f"生成代码..."):
            prompt = build_prompt_with_memory(current_prompt)
            chat_result = st.session_state.llm.invoke(prompt.format_messages())
            code_content = chat_result.generations[0].message.content

        if "```python" in code_content:
            code_block = code_content.split("```python")[1].split("```")[0].strip()
            st.code(code_block, language="python")

            # 🔥 步骤4：执行+精准错误反馈
            exec_result = execute_python_code(code_block, df)

            if exec_result["expected_count"] > 0 and len(exec_result["saved_images"]) > 0:
                final_code = code_block
                final_result = exec_result
                st.success(f"✅ 第{attempt + 1}次成功！生成{len(exec_result['saved_images'])}张图")
                break
            else:
                # 🔥 关键：详细错误反馈给LLM
                error_msg = exec_result['result']
                st.warning(f"❌ 第{attempt + 1}次失败：{error_msg[:150]}...")

                # 构建精准修复提示
                fix_prompt = f"""
**精准错误**：{error_msg}
**数据状态**：{len(df)}行，字段{list(df.columns)}
**请修改**：
1. 复制上次代码，**只改错误部分**
2. 如果TypeError：加plt.figure()/plt.close()
3. 如果KeyError：改字段名{list(df.columns)}
4. 如果ValueError：加df['列名'].fillna()"""

                history.append({"role": "assistant", "content": code_content})
                history.append({"role": "user", "content": fix_prompt})

                if attempt == max_retries - 1:
                    st.error("❌ 3次尝试失败")
                    st.info("**调试建议**：检查数据字段和LLM输出")
        else:
            st.warning(f"❌ 第{attempt + 1}次：未生成Python代码")

    return final_code, final_result
# ================= 代码执行与图片处理 =================
def execute_python_code(code, df):
    fake_data_patterns = [r"pd\.DataFrame\(", r"{'办事处名称':", r"{'A占比':", r"{'CD占比':"]
    for pattern in fake_data_patterns:
        if re.search(pattern, code):
            error_msg = "错误：检测到构造假数据的行为。请使用从数据库加载的df数据，不要手动创建数据。"
            logger.warning(error_msg)
            return {"result": error_msg, "saved_images": [], "expected_count": 0, "chart_metadata_list": []}

    locals_dict = {"df": df, "pd": pd, "plt": plt}
    output = StringIO()
    saved_images = []
    image_count = 0
    chart_metadata_list = []

    try:
        code = re.sub(r"```python|```", "", code).strip()
        logger.info(f"即将执行的Python代码：\n{code}")

        with redirect_stdout(output):
            exec(code, globals(), locals_dict)

        result = output.getvalue().strip()
        logger.info(f"代码执行成功，输出结果：\n{result}")

        count_match = re.search(r"图片总数: (\d+)", result)
        if count_match:
            image_count = int(count_match.group(1))
            saved_images = [f"plot_{i}.png" for i in range(image_count)]
            logger.info(f"预期生成 {image_count} 张图片，路径：{saved_images}")

        chart_metadata_list = parse_chart_metadata(code, df, saved_images)

        existing_images = [img for img in saved_images if os.path.exists(img)]
        for img in saved_images:
            if img not in existing_images:
                warning_msg = f"图片 {img} 生成失败（未找到文件）"
                st.warning(warning_msg)
                logger.warning(warning_msg)
        saved_images = existing_images

        chart_metadata_list = [
            meta for meta in chart_metadata_list
            if meta["image_path"] in existing_images
        ]

    except Exception as e:
        error_detail = f"执行错误：{type(e).__name__} - {str(e)}\n"
        error_detail += f"出错代码：\n{code}\n"
        error_detail += f"请检查代码中是否存在字段名错误（如df['不存在的字段']）或语法错误"
        result = error_detail
        logger.error(f"代码执行失败：{error_detail}")
        st.error(f"代码执行失败：\n{error_detail}")

    finally:
        plt.close('all')

    return {
        "result": result,
        "saved_images": saved_images,
        "expected_count": image_count,
        "chart_metadata_list": chart_metadata_list
    }
def parse_chart_metadata(code, df, image_paths):
    """解析绘图代码，提取每个图表的元数据"""
    metadata_list = []
    code_blocks = re.split(r"plt\.savefig\('plot_(\d+)\.png'\)", code)
    # 过滤空字符串
    code_blocks = [b.strip() for b in code_blocks if b.strip()]

    for i, img_path in enumerate(image_paths):
        # 找到当前图片对应的代码块
        block_index = 2 * i  # 每个savefig分割后，代码块在偶数索引
        if block_index >= len(code_blocks):
            continue
        chart_code = code_blocks[block_index]

        # 1. 提取图表类型（bar/line/hist/scatter等）
        chart_type = "unknown"
        plot_patterns = {
            r"plt\.bar\(": "bar",
            r"plt\.plot\(": "line",
            r"plt\.hist\(": "histogram",
            r"plt\.scatter\(": "scatter",
            r"plt\.pie\(": "pie",
            r"plt\.boxplot\(": "boxplot"
        }
        for pattern, typ in plot_patterns.items():
            if re.search(pattern, chart_code):
                chart_type = typ
                break

        title_match = re.search(r"plt\.title\(['\"](.*?)['\"]\)", chart_code)
        title = title_match.group(1) if title_match else f"未命名图表_{i}"

        # 3. 提取X轴标签
        xlabel_match = re.search(r"plt\.xlabel\(['\"](.*?)['\"]\)", chart_code)
        x_axis = xlabel_match.group(1) if xlabel_match else "未指定X轴"

        # 4. 提取Y轴标签
        ylabel_match = re.search(r"plt\.ylabel\(['\"](.*?)['\"]\)", chart_code)
        y_axis = ylabel_match.group(1) if ylabel_match else "未指定Y轴"

        # 5. 提取使用的字段（df['字段名']）
        field_matches = re.findall(r"df\['(.*?)'\]", chart_code)
        used_fields = list(set(field_matches))  # 去重
        # 过滤不存在的字段
        valid_fields = [f for f in used_fields if f in df.columns]

        # 6. 提取示例数据（前3行）
        sample_data = []
        if valid_fields:
            try:
                # 取前3行有效字段的数据
                sample_df = df[valid_fields].head(3)
                sample_data = sample_df.values.tolist()
            except Exception as e:
                logger.warning(f"提取示例数据失败：{str(e)}")
                sample_data = []

        # 构建元数据
        metadata = {
            "image_path": img_path,
            "chart_type": chart_type,
            "title": title,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "data_source": {
                "columns": valid_fields,
                "sample_data": sample_data
            },
            "description": f"{chart_type}图，展示{title.lower()}的分布情况"
        }
        metadata_list.append(metadata)

    return metadata_list