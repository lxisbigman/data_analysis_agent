# analysis_agent.py（修改run方法的返回结果）
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from tools.code_execute_v1 import generate_and_fix_python_code
import streamlit as st

logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Heiti TC", "Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class AnalysisAgent:
    def run(self, user_query: str, raw_df: pd.DataFrame) -> dict:
        try:
            if raw_df is None or raw_df.empty:
                raise ValueError("原始数据为空，无法执行统计分析")
            # 显示分析文本
            st.markdown(f"### 📈统计分析")

            # 1. 调用代码生成函数（关键：确认返回的 execution_result 包含所有字段）
            final_code, execution_result = generate_and_fix_python_code(user_query, raw_df)

            # 2. 强制补全 execution_result 的核心字段（避免缺失）
            # 补全图表元数据（名字、关键信息）
            if "chart_metadata_list" not in execution_result or not execution_result["chart_metadata_list"]:
                execution_result["chart_metadata_list"] = []
                # 从 saved_images 中提取文件名作为默认图表名
                for idx, img_path in enumerate(execution_result.get("saved_images", [])):
                    execution_result["chart_metadata_list"].append({
                        "chart_index": idx + 1,
                        "title": os.path.basename(img_path).replace("plot_", "").replace(".png", ""),
                        "description": f"该图表基于用户查询「{user_query}」生成，展示了关键数据分布",
                        "image_path": img_path
                    })
            # 补全分析结果文本
            if "result" not in execution_result or not execution_result["result"]:
                execution_result["result"] = f"基于 {len(raw_df)} 行数据完成统计分析，共生成 {len(execution_result.get('saved_images', []))} 张图表"
            # 补全图表路径
            if "saved_images" not in execution_result:
                execution_result["saved_images"] = []

            # 3. 处理图表路径（转为绝对路径，确保跨Agent访问）
            chart_paths = execution_result.get("saved_images", [])
            valid_chart_paths = [
                os.path.abspath(img)
                for img in chart_paths
                if os.path.exists(img) and os.path.getsize(img) > 0
            ]

            # 4. 无有效图表时生成默认直方图（同步生成默认元数据）
            if not valid_chart_paths:
                logger.warning("未生成有效图表，生成默认概览图")
                numeric_cols = raw_df.select_dtypes(include=['number']).columns
                if numeric_cols.any():
                    fig, ax = plt.subplots(figsize=(8, 4))
                    raw_df[numeric_cols[0]].dropna().hist(ax=ax, bins=10, color='#1f77b4', alpha=0.7)
                    title = f"原始数据-{numeric_cols[0]}分布（默认概览图）"
                    ax.set_title(title)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel("频数")
                    default_img_path = os.path.abspath(f"plot_default_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
                    plt.tight_layout()
                    plt.savefig(default_img_path, dpi=100)
                    plt.close()
                    valid_chart_paths.append(default_img_path)
                    chart_paths = valid_chart_paths

                    # 为默认图生成元数据
                    default_metadata = {
                        "image_path": default_img_path,
                        "chart_index": 1,
                        "chart_type": "直方图",
                        "title": title,
                        "x_axis": numeric_cols[0],
                        "y_axis": "频数",
                        "used_fields": [numeric_cols[0]],
                        "sample_data": raw_df[[numeric_cols[0]]].head(3).values.tolist(),
                        "description": f"该直方图展示了{title}的分布情况。"
                    }
                    execution_result["chart_metadata_list"] = [default_metadata]
                    execution_result["expected_count"] = 1  # 这里只给默认图设置了expected_count

            # -------------------------- 关键修复：将expected_count写入full_execution_result --------------------------
            # 计算最终的expected_count（与AnalysisAgent顶层字段一致）
            final_expected_count = execution_result.get("expected_count", len(valid_chart_paths))
            # 将expected_count合并到full_execution_result中，供StatReportAgent使用
            execution_result["expected_count"] = final_expected_count
            # --------------------------------------------------------------------------------------------------------

            # 5. 返回结果（顶层expected_count可保留，也可删除，因为已写入full_execution_result）
            return {
                "success": True,
                "final_code": final_code,
                "full_execution_result": execution_result,  # 含expected_count
                "analysis_text": execution_result["result"],
                "chart_paths": valid_chart_paths,
                "processed_df": execution_result.get("processed_df", pd.DataFrame()),
                "expected_count": final_expected_count,  # 恢复顶层字段！供下游访问
                "error": None
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"统计分析失败：{error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def display_result(self, analysis_result: dict):
        logger.info("AnalysisAgent.display_result：开始渲染统计分析结果")
        logger.info(
            f"analysis_result关键字段：success={analysis_result.get('success', False)}, chart_paths={len(analysis_result.get('chart_paths', []))}")
        if not analysis_result.get("success", False):
            st.error("❌ 统计分析失败，未显示结果")
            logger.warning("统计分析失败，跳过显示")
            return

        # 显示图表
        valid_chart_paths = analysis_result.get("chart_paths", [])
        if valid_chart_paths:
            st.markdown("#### 📊 分析图表")
            cols = st.columns(2)
            metadata_count = len(analysis_result["full_execution_result"].get("chart_metadata_list", []))
            st.info(f"共生成 {len(valid_chart_paths)} 张图表，元数据 {metadata_count} 条（应相等）")

            for idx, img_path in enumerate(valid_chart_paths):
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    with cols[idx % 2]:
                        caption = f"图表 {idx + 1}: {os.path.basename(img_path)}"
                        st.image(
                            img_path,
                            caption=caption,
                            use_container_width=True
                        )
                else:
                    st.warning(f"⚠️ 图表 {idx + 1} 路径无效：{img_path}")

        # 检查预期和实际图表数量
        expected = analysis_result.get("expected_count", 0)
        if expected == 0:
            expected = analysis_result["full_execution_result"].get("expected_count", 0)
        actual = len(valid_chart_paths)
        if expected > 0 and actual != expected:
            st.warning(f"⚠️ 图片数量：预期 {expected}，实际 {actual}")
        else:
            st.success(f"✅ 图表数量验证通过：预期 {expected}，实际 {actual}")

        logger.info("AnalysisAgent.display_result：渲染完成")