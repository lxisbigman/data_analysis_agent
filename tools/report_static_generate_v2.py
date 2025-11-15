###report_static_generate_v2.py
import logging
import pandas as pd
import os
from typing import Dict, Any, List
from tools.LLM_invoke import DeepSeekLLM
from tools.report_generate import DataProcessor
import streamlit as st
from datetime import datetime
import json
from env.DB_env import DB_CONFIG
from tools.DB_connect import DatabaseConnector  # 导入数据库连接类

logger = logging.getLogger(__name__)


class StatReportAgent:
    def __init__(self):
        self.llm = DeepSeekLLM()
        self.processor = DataProcessor()
        # 初始化并检查数据库连接
        self.db_connector = DatabaseConnector()
        success, msg = self.db_connector.connect()
        if not success:
            raise ConnectionError(f"StatReportAgent初始化失败：数据库连接失败，{msg}")

    def _clean_llm_response(self, content: str) -> str:
        """清理LLM返回内容中的<think>标签和多余空白"""
        import re

        if not content:
            return content

        # 第一步：移除 <think>...</think> 标签及其内容
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 第二步：清理多余的空行和空白
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned.strip())

        # 第三步：如果清理后内容过短，返回原始内容（避免过度清理）
        if len(cleaned) < 10 and len(content) > 50:
            # 可能是清理过度，尝试只移除标签但保留内容
            cleaned = re.sub(r'</?think>', '', content)
            cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned.strip())

        return cleaned
    def _get_table_meta_from_db(self, table_name: str) -> Dict[str, Any]:
        """获取表元信息：直接复用get_full_db_structure结果，无重复查询"""
        try:
            # 1. 从完整数据库结构中获取所有信息（无重复查询）
            db_structure = self.db_connector.get_full_db_structure(schema=DB_CONFIG["schema"])

            # 2. 检查表是否存在于schema中
            if table_name not in db_structure["tables"]:
                raise ValueError(f"表 {table_name} 不在 schema {DB_CONFIG['schema']} 中")
            table_info = db_structure["tables"][table_name]

            # 3. 提取表注释（直接复用，无需再查）
            table_comment = table_info["comment"] or "无表注释"

            # 4. 提取字段信息（外层key是字段名，内层是类型/注释/序号）
            field_info = []
            for field_name, field_detail in table_info["columns"].items():
                field_info.append({
                    "name": field_name,  # 字段名=外层key
                    "type": field_detail["data_type"],  # 对应get_full_db_structure的data_type
                    "comment": field_detail["comment"],  # 直接复用已查好的字段注释
                    "position": field_detail["ordinal_position"]  # 直接复用序号
                })

            return {
                "table_name": table_name,
                "table_comment": table_comment,
                "fields": field_info,
                "success": True
            }
        except Exception as e:
            logger.error(f"获取表{table_name}元信息失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _integrate_multi_table_logic(self, table_names: List[str], df: pd.DataFrame, user_query: str) -> str:
        """整合多表元信息，生成联合业务逻辑"""
        all_table_meta = []
        for table in table_names:
            meta = self._get_table_meta_from_db(table)
            if meta["success"]:
                all_table_meta.append(meta)
            else:
                st.warning(f"表{table}的元信息获取失败，可能影响多表逻辑分析")

        # 提取表核心信息
        table_summaries = []
        for meta in all_table_meta:
            key_fields = [f["name"] for f in meta["fields"][:3]]
            table_summaries.append(
                f"- 表名：{meta['table_name']}\n"
                f"  注释：{meta['table_comment']}\n"
                f"  关键字段：{key_fields}"
            )

        # 调用LLM分析多表逻辑
        prompt = f"""
以下是多个关联表的信息，以及它们的联合查询结果，请分析业务逻辑：

【涉及表信息】
{chr(10).join(table_summaries)}

【联合查询结果字段】
{df.columns.tolist()}

【用户查询】
{user_query}

请说明：
1. 这些表共同支撑什么业务场景？
2. 表之间通过什么字段关联（如外键）？
3. 联合查询结果能反映什么业务问题？
"""
        try:
            response = self.llm.invoke(prompt)
            raw_content = response.generations[0].message.content
            cleaned_content = self._clean_llm_response(raw_content)
            return cleaned_content
        except Exception as e:
            logger.error(f"多表业务逻辑整合失败：{str(e)}")
            return f"多表联合分析：涉及表{table_names}，字段{df.columns.tolist()}"

    def generate_stat_report(self, df: pd.DataFrame, execution_result: Dict, images: List[str],
                             query: str = "", table_names: List[str] = [],
                             chart_metadata_list: List[Dict] = []) -> Dict[str, Any]:
        """生成业务化报告（支持单表/多表）"""
        try:
            valid_table_names = [t.strip() for t in table_names if t.strip()]
            if not valid_table_names:
                raise ValueError("必须指定至少一个非空的数据库表名才能生成业务报告")

            # -------------------------- 1. 业务逻辑分析（已正常，跳过）--------------------------
            with st.expander("🔍正在分析数据背后的业务逻辑", expanded=False):
                with st.spinner("🔍 正在分析数据背后的业务逻辑..."):
                    if len(valid_table_names) == 1:
                        business_logic = self._integrate_business_logic(valid_table_names[0], df, query)
                    else:
                        business_logic = self._integrate_multi_table_logic(valid_table_names, df, query)
                st.markdown("### 📌 数据业务逻辑分析（基于表结构和注释自动推理）")
                st.markdown(business_logic)
                # 校验业务逻辑结果（防止极端情况为空）
                if not business_logic.strip():
                    raise ValueError("业务逻辑分析结果为空，无法继续生成报告")
                logger.info("✅ 业务逻辑分析完成，结果长度：%d字符", len(business_logic.strip()))

                # -------------------------- 2. 字段分布分析（关键校验：防止无数值字段）--------------------------
                with st.spinner("🔍 分析字段分布特征..."):
                    field_analysis = self._analyze_field_distributions(df)
                # 校验：若字段分析为空（无数值字段），抛出明确错误
                if not field_analysis.strip():
                    raise ValueError(
                        f"字段分布分析未生成有效结果，可能原因：1. 原始数据无数值字段；2. 数值字段全为缺失值。原始数据数值字段：{df.select_dtypes(include=['number']).columns.tolist()}")
                logger.info("✅ 字段分布分析完成，结果长度：%d字符", len(field_analysis.strip()))

                # -------------------------- 3. 异常检测（关键校验：防止无数值字段）--------------------------
                with st.spinner("🚨 检测数据异常..."):
                    anomaly_analysis = self._enhanced_anomaly_detection(df)
                # 校验：若异常检测结果为空，抛出明确错误
                if not anomaly_analysis.strip():
                    raise ValueError(
                        f"异常检测未生成有效结果，可能原因：原始数据无数值字段。原始数据数值字段：{df.select_dtypes(include=['number']).columns.tolist()}")
                logger.info("✅ 异常检测完成，结果长度：%d字符", len(anomaly_analysis.strip()))

                # -------------------------- 4. 图表解读（关键校验：防止无有效图片）--------------------------
                with st.spinner("📊 解读统计图表..."):
                    valid_images = [img for img in images if os.path.exists(img) and os.path.getsize(img) > 0]
                    # 核心修改：调用时传入 chart_metadata_list
                    chart_analysis = self._analyze_charts_intelligently(valid_images, df, query, chart_metadata_list)
                # 校验：若无有效图片且图表解读为空，抛出警告（但不终止，用文字说明）
                if not valid_images:
                    st.warning("⚠️ 未检测到有效统计图表，报告中将不包含图表解读内容")
                    chart_analysis = "未生成有效统计图表，故无图表解读内容"
                logger.info("✅ 图表解读完成（有效图片：%d/%d），结果长度：%d字符", len(valid_images), len(images),
                            len(chart_analysis.strip()))

            # -------------------------- 5. 生成完整报告（LLM调用严格校验）--------------------------
            with st.spinner("📝 生成最终统计报告..."):
                integrated_prompt = f"""
    请直接生成最终报告，不要包含任何思考过程、分析步骤或<think>标签。直接输出完整的报告内容。
    基于以下分析组件，生成一份完整、专业的数据分析报告：

    ## 用户查询
    {query}

    ## 数据业务逻辑分析
    {business_logic}

    ## 字段分布分析
    {field_analysis}

    ## 异常检测结果
    {anomaly_analysis}

    ## 图表深度解读
    {chart_analysis}

    ## 统计分析执行摘要
    {execution_result.get('result', '无统计分析执行结果')}

    请严格按照以下要求生成报告：
    1. 结构完整：包含「执行摘要、数据概况（原始数据规模+字段说明）、关键发现（基于上述分析）、图表解读（无图表则说明）、异常分析、业务建议」6个部分；
    2. 数据驱动：每个结论必须引用原始数据（如“原始数据共1000行，省份字段包含3个地区”）或分析结果（如“综合得分呈正态分布，均值85.2”）；
    3. 语言专业易懂：避免技术术语，建议具体可操作（如“每周监控各省份综合得分，低于80分需排查原因”）；
    4. 字数控制：报告总字数不少于800字，每个部分至少1段内容。
    """
                # LLM调用+重试+严格结果校验
                max_retries = 2
                report_content = None
                for retry in range(max_retries + 1):
                    try:
                        logger.info(f"📞 调用LLM生成最终报告（第{retry + 1}次），prompt长度：%d字符", len(integrated_prompt))
                        response = self.llm.invoke(integrated_prompt)
                        # 校验LLM返回格式：必须有generations且content非空
                        if not hasattr(response, "generations") or len(response.generations) == 0:
                            raise ValueError(f"LLM返回无效格式：无generations字段，原始响应：{str(response)[:200]}")
                        # report_content = response.generations[0].message.content.strip()
                        # --- 修复后代码（推荐）---
                        raw_content = response.generations[0].message.content

                        # Step 1: 移除 <think>...</think> 及其内容
                        import re
                        cleaned_content = re.sub(r'<think>[\s\S]*?</think>', '', raw_content, flags=re.DOTALL)

                        # Step 2: 清理多余空行
                        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content.strip())

                        report_content = cleaned_content
                        if not report_content or len(report_content) < 500:  # 过滤过短内容
                            raise ValueError(
                                f"LLM生成报告内容无效（长度：{len(report_content)}字符），内容预览：{report_content[:100]}")
                        logger.info(f"✅ LLM生成报告成功（第{retry + 1}次），内容长度：%d字符", len(report_content))
                        break
                    except Exception as e:
                        error_msg = f"LLM生成报告第{retry + 1}次失败：{str(e)[:300]}"
                        logger.error(error_msg)
                        if retry == max_retries:
                            raise ValueError(f"LLM生成报告{max_retries + 1}次均失败，原因：{error_msg}")
                        st.warning(f"{error_msg}，正在重试...")

                # 格式化报告
                full_report = self._format_final_report(report_content, df, query)
                if not full_report.strip():
                    raise ValueError("格式化报告后内容为空")

                # 生成docx文件（校验文件流）
                doc_stream = self.processor.save_report_to_docx(full_report, valid_images)
                if not doc_stream or (hasattr(doc_stream, "getvalue") and len(doc_stream.getvalue()) < 100):
                    raise ValueError("生成Word报告失败：文件流为空或无效")
                logger.info("✅ Word报告生成成功，文件大小：%d字节", len(doc_stream.getvalue()))

            return {
                "full_report": full_report,
                "doc_stream": doc_stream,
                "business_logic": business_logic,
                "error": None
            }
        except Exception as e:
            # 捕获具体错误信息，避免返回None
            error_msg = str(e) if str(e).strip() else "未知错误（可能是LLM生成报告为空或中间步骤无有效结果）"
            logger.error(f"❌ 生成统计报告失败：{error_msg}", exc_info=True)  # 打印堆栈，方便定位
            return {"error": error_msg, "full_report": "", "business_logic": business_logic}

    def _format_final_report(self, content: str, df: pd.DataFrame, query: str) -> str:
        """格式化最终报告"""
        header = f"""# 智能数据分析报告

**分析时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析主题**：{query}
**数据规模**：{df.shape[0]} 行 × {df.shape[1]} 列
**主要字段**：{', '.join(df.columns.tolist()[:8])}

---
"""
        footer = f"""

---
*报告生成：智能数据分析系统*
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return header + content + footer

    def _analyze_charts_intelligently(self, valid_images: List[str], df: pd.DataFrame,
                                      query: str, chart_metadata_list: List[Dict]) -> str:  # 保留原参数
        """使用图表元数据（类型/坐标轴/数据）智能分析图表内容（优化对齐逻辑）"""
        chart_analyses = []
        valid_images_abs = [os.path.abspath(img) for img in valid_images]  # 统一转为绝对路径，避免路径格式不匹配

        if not chart_metadata_list:  # 元数据为空时，保留原降级逻辑
            st.warning("⚠️ 未获取到图表元数据，将基于图片路径推测内容（精度可能下降）")
            for idx, img_path in enumerate(valid_images, 1):
                if not os.path.exists(img_path):
                    continue
                # 原降级逻辑：仅靠文件名和数据字段推测（保留不变）
                chart_prompt = f"""
    请分析以下图表所展示的数据洞察：
    - 图表文件：{os.path.basename(img_path)}
    - 数据背景：{query}
    - 数据字段：{list(df.columns)}
    - 数据规模：{df.shape}
    请回答：1. 可能的图表类型（如柱状图/折线图）；2. 展示的核心趋势；3. 业务意义；4. 需关注的异常点。
    """
                try:
                    response = self.llm.invoke(chart_prompt)
                    analysis = response.generations[0].message.content
                    chart_analyses.append(f"### 图表{idx}（基于文件名推测）\n{analysis}")
                except Exception as e:
                    logger.warning(f"图表{idx}分析失败: {str(e)}")
                    chart_analyses.append(f"### 图表{idx}分析\n文件: {os.path.basename(img_path)} - 自动分析暂不可用")
            return "\n\n".join(chart_analyses)

        # 元数据不为空时：核心优化——确保元数据与图片严格对齐
        # 1. 建立元数据与有效图片的映射（通过绝对路径关联，避免路径格式问题）
        meta_image_map = {}
        for meta in chart_metadata_list:
            meta_img_abs = os.path.abspath(meta.get("image_path", ""))  # 元数据路径转绝对路径
            if meta_img_abs in valid_images_abs:
                meta_image_map[meta_img_abs] = meta  # 用绝对路径作为key，确保匹配
            else:
                logger.warning(f"元数据图片路径未找到：{meta.get('image_path')}（已转为绝对路径：{meta_img_abs}）")

        # 2. 遍历有效图片，基于匹配的元数据生成解读
        for img_abs, img_relative in zip(valid_images_abs, valid_images):
            meta = meta_image_map.get(img_abs)
            idx = len(chart_analyses) + 1  # 图表序号（按有效图片顺序）

            if not meta:
                # 无匹配元数据：降级为原逻辑，但标注“无元数据”
                st.warning(f"图表{idx}（{os.path.basename(img_relative)}）无匹配元数据，将基于文件名推测")
                chart_prompt = f"""
    请分析以下图表所展示的数据洞察：
    - 图表文件：{os.path.basename(img_relative)}
    - 数据背景：{query}
    - 数据字段：{list(df.columns)}
    - 数据规模：{df.shape}
    请回答：1. 可能的图表类型；2. 核心趋势；3. 业务意义；4. 异常点。
    """
                try:
                    response = self.llm.invoke(chart_prompt)
                    analysis = response.generations[0].message.content
                    chart_analyses.append(f"### 图表{idx}（无元数据，基于文件名推测）\n{analysis}")
                except Exception as e:
                    chart_analyses.append(
                        f"### 图表{idx}分析\n文件: {os.path.basename(img_relative)} - 自动分析暂不可用")
                continue

            # 3. 有匹配元数据：基于元数据生成精准Prompt（核心优化点）
            # 从元数据中提取完整信息（确保与图表内容一致）
            chart_title = meta.get("title", f"图表{idx}")
            chart_type = meta.get("chart_type", "未知类型")
            x_axis = meta.get("x_axis", "未指定X轴")
            y_axis = meta.get("y_axis", "未指定Y轴")
            used_fields = meta.get("used_fields", [])
            sample_data = meta.get("sample_data", [])
            img_basename = os.path.basename(img_relative)

            # 构建精准Prompt：LLM明确知道图表的标题、类型、坐标轴、数据字段
            chart_prompt = f"""
    请基于以下**明确的图表信息**，分析专业数据洞察：

    【图表基础信息】
    - 图表标题：{chart_title}
    - 图表类型：{chart_type}（如柱状图/折线图/饼图）
    - X轴含义：{x_axis}
    - Y轴含义：{y_axis}
    - 使用数据字段：{used_fields}（来自原始数据df）
    - 数据示例（前3行）：{sample_data}
    - 图表文件：{img_basename}
    - 分析背景：{query}
    - 整体数据规模：{df.shape[0]}行 × {df.shape[1]}列

    【分析要求】
    1. 明确说明该图表展示的**核心数据关系**（如“安徽省各城市网络综合评分的柱状图对比”）；
    2. 基于图表类型和数据示例，指出**关键趋势/分布特征**（如“合肥评分最高，黄山最低，呈阶梯分布”）；
    3. 结合分析背景（{query}），解释该趋势的**业务意义**（如“反映区域网络建设不均衡，需重点优化低分城市”）；
    4. 若存在异常点（如示例数据中的极端值），需指出并推测**可能原因**；
    5. 语言必须与图表标题和类型严格对应，避免泛泛而谈。
    """
            try:
                response = self.llm.invoke(chart_prompt)
                analysis = response.generations[0].message.content.strip()
                # 明确标注图表标题和类型，让报告解读与图表一一对应
                chart_analyses.append(f"### 图表{idx}：{chart_title}（{chart_type}）\n{analysis}")
            except Exception as e:
                logger.warning(f"图表{idx}（{chart_title}）分析失败: {str(e)}")
                chart_analyses.append(f"### 图表{idx}：{chart_title}（{chart_type}）\n分析暂不可用（错误：{str(e)[:50]}）")

        return "\n\n".join(chart_analyses)
    def _parse_data_meta(self, table_meta: Dict, df: pd.DataFrame) -> str:
        """解析字段元信息，过滤不存在的字段"""
        field_meta = []
        for field in table_meta["fields"]:
            field_name = field["name"]
            if field_name not in df.columns:
                logger.warning(f"字段{field_name}不在数据中，跳过分析")
                continue

            field_type = field["type"]
            field_comment = field["comment"]
            sample_values = df[field_name].dropna().head(3).tolist()
            sample_values = [round(v, 2) if isinstance(v, (int, float)) else v for v in sample_values]

            field_meta.append(
                f"- 字段名：{field_name}\n"
                f"  类型：{field_type}\n"
                f"  注释：{field_comment}\n"
                f"  示例值：{sample_values}"
            )

        # 调用LLM推测字段业务含义
        meta_prompt = f"""
请基于以下表注释和字段信息（尤其是字段注释），推测每个字段的业务含义，要求：
1. 优先信任字段注释；
2. 结合字段名、类型、示例值补充说明；
3. 关联表注释（{table_meta['table_comment']}）推测业务场景。

【表信息】
- 表名：{table_meta['table_name']}
- 表注释：{table_meta['table_comment']}
- 数据规模：{df.shape[0]}行 × {df.shape[1]}列

【字段详情】
{chr(10).join(field_meta)}

【输出格式】
### 字段业务含义推测
1. {field_name}：业务含义（结合注释和示例值说明）
...
"""
        try:
            response = self.llm.invoke(meta_prompt)
            raw_content = response.generations[0].message.content
            cleaned_content = self._clean_llm_response(raw_content)
            return cleaned_content
        except Exception as e:
            logger.error(f"解析字段业务含义失败：{str(e)}")
            # 降级处理
            simple_meta = ["### 字段业务含义推测"]
            for field in table_meta["fields"]:
                if field["name"] in df.columns:
                    sample = df[field["name"]].dropna().head(1).tolist()
                    simple_meta.append(f"- {field['name']}：{field['comment']}（示例值：{sample}）")
            return "\n".join(simple_meta)

    def _integrate_business_logic(self, table_name: str, df: pd.DataFrame, user_query: str) -> str:
        """整合单表元信息，生成业务逻辑"""
        # 获取表元信息（复用get_full_db_structure结果）
        table_meta = self._get_table_meta_from_db(table_name)
        if not table_meta["success"]:
            st.error(f"获取表元信息失败，影响业务逻辑推理：{table_meta['error']}")
            return f"表{table_name}业务逻辑：未获取到完整元信息，仅知包含字段{df.columns.tolist()}"

        # 解析字段业务含义
        field_meta = self._parse_data_meta(table_meta, df)

        # 挖掘数值关联逻辑（若未定义该方法，先降级为默认文本）
        try:
            numeric_relation = self._mine_numeric_relation(df, field_meta)
        except AttributeError:
            numeric_relation = "未检测到数值字段关联逻辑（可补充_mine_numeric_relation方法完善）"

        # 调用LLM整合业务逻辑
        integrate_prompt = f"""
请基于以下信息，整合出这张表的完整业务逻辑：
1. 表名：{table_meta.get('table_name', table_name)}
2. 表注释：{table_meta.get('table_comment', '无表注释')}
3. 字段业务含义：{field_meta}
4. 数值关联逻辑：{numeric_relation}
5. 用户查询：{user_query}

【输出要求】
- 说明业务场景（如"用于监控各区域网络质量"）；
- 解释字段如何支撑场景；
- 结合数值关联说明业务因果关系。
"""
        try:
            response = self.llm.invoke(integrate_prompt)
            return response.generations[0].message.content
        except Exception as e:
            logger.error(f"整合业务逻辑失败：{str(e)}")
            return f"表{table_name}业务逻辑：{table_meta.get('table_comment', '无表注释')}，包含字段{df.columns.tolist()}"

    def _generate_ai_report(self, df: pd.DataFrame, execution_result: Dict, images: List[str], query: str) -> str:
        """使用大模型智能生成完整报告（降级备用）"""
        data_context = self._prepare_data_context(df, execution_result, images, query)
        prompt = self._build_report_prompt(data_context, query)

        try:
            response = self.llm.invoke(prompt)
            return self._clean_report_content(response.generations[0].message.content, df)
        except Exception as e:
            logger.error(f"大模型生成报告失败: {str(e)}")
            return self._generate_fallback_report(df, execution_result, images, query)

    def _prepare_data_context(self, df: pd.DataFrame, execution_result: Dict, images: List[str], query: str) -> Dict[
        str, Any]:
        """准备详细的数据上下文（供报告生成使用）"""
        # 数值字段统计
        numeric_stats = {}
        for col in df.select_dtypes(include=['number']).columns:
            data = df[col].dropna()
            if len(data) > 0:
                numeric_stats[col] = {
                    'count': len(data), 'mean': data.mean(), 'std': data.std(),
                    'min': data.min(), 'max': data.max(), 'median': data.median(),
                    'skew': data.skew(), 'missing': df[col].isnull().sum()
                }

        # 分类字段统计
        categorical_stats = {}
        for col in df.select_dtypes(include=['object']).columns[:5]:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': df[col].nunique(),
                'top_values': value_counts.head(3).to_dict(),
                'missing': df[col].isnull().sum()
            }

        # 图表信息
        chart_info = [
            {'index': idx, 'filename': os.path.basename(p), 'path': p}
            for idx, p in enumerate(images, 1) if os.path.exists(p)
        ]

        return {
            'query': query, 'data_shape': df.shape,
            'data_quality': {
                'completeness': (df.count().sum()) / (df.shape[0] * df.shape[1]) * 100,
                'duplicates': df.duplicated().sum(),
                'total_missing': df.isnull().sum().sum()
            },
            'numeric_stats': numeric_stats, 'categorical_stats': categorical_stats,
            'anomalies': self._detect_anomalies(df),
            'execution_result': execution_result.get('result', ''),
            'charts': chart_info, 'analysis_code': execution_result.get('analysis_code', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _analyze_field_distributions(self, df: pd.DataFrame) -> str:
        """智能分析字段分布特征"""
        distribution_insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()  # 显式转列表

        # 若无数值字段，返回明确提示
        if not numeric_cols:
            return "### 字段分布分析\n未检测到原始数据中的数值字段，无法进行数值分布分析。请检查原始数据的数据类型（如是否有字段应设为数值型却被识别为字符串）。"

        # 分析前8个数值字段
        for col in numeric_cols[:8]:
            data = df[col].dropna()
            if len(data) == 0:
                distribution_insights.append(f"### {col}字段分析\n该字段全为缺失值，无法进行分布分析。")
                continue

            # 计算统计特征（原逻辑不变）
            stats = {
                'mean': data.mean(), 'std': data.std(), 'min': data.min(), 'max': data.max(),
                'median': data.median(), 'skewness': data.skew(),
                'missing_rate': df[col].isnull().sum() / len(df) * 100
            }

            # 判断分布类型（原逻辑不变）
            if abs(stats['skewness']) < 0.5:
                distribution_type = "正态分布"
            elif stats['skewness'] > 1:
                distribution_type = "严重右偏"
            elif stats['skewness'] < -1:
                distribution_type = "严重左偏"
            else:
                distribution_type = "偏态分布"

            # LLM分析（原逻辑不变）
            field_prompt = f"""
    分析字段 '{col}' 的分布特征：
    - 字段类型：数值型
    - 有效数据量：{len(data)}行（缺失率：{stats['missing_rate']:.1f}%）
    - 均值: {stats['mean']:.2f}
    - 标准差: {stats['std']:.2f}
    - 数值范围: [{stats['min']:.2f}, {stats['max']:.2f}]
    - 中位数: {stats['median']:.2f}
    - 分布形态: {distribution_type}（偏度：{stats['skewness']:.2f}）

    请从业务角度回答：
    1. 这个分布特征说明该字段的数据特点（如“综合得分均值85，说明整体水平良好”）；
    2. 缺失率{stats['missing_rate']:.1f}%是否会影响分析结论？
    3. 对业务决策有什么启示（如“需重点关注得分低于70的样本”）？
    """
            try:
                response = self.llm.invoke(field_prompt)
                distribution_insights.append(f"### {col}字段分析\n{response.generations[0].message.content.strip()}")
            except Exception as e:
                logger.warning(f"字段{col}分布分析失败: {str(e)}")
                basic_info = f"字段{col}（数值型）呈现{distribution_type}，均值{stats['mean']:.2f}，范围[{stats['min']:.2f}, {stats['max']:.2f}]，缺失率{stats['missing_rate']:.1f}%。"
                distribution_insights.append(f"### {col}字段分析\n{basic_info}（自动分析暂不可用）")

        return "\n\n".join(distribution_insights)

    def _build_report_prompt(self, context: Dict, query: str) -> str:
        """构建报告生成提示词"""
        # 整理图表描述
        chart_descriptions = "\n".join([f"- 图表{c['index']}: {c['filename']}" for c in context['charts']])

        # 整理数值字段统计
        numeric_desc = "\n".join([
            f"- {col}: 均值{stats['mean']:.2f}, 标准差{stats['std']:.2f}, 范围[{stats['min']:.2f}, {stats['max']:.2f}]"
            for col, stats in list(context['numeric_stats'].items())[:8]
        ])

        # 整理分类字段统计
        categorical_desc = "\n".join([
            f"- {col}: {stats['unique_count']}个类别, 主要值: {', '.join(list(stats['top_values'].keys())[:2])}"
            for col, stats in list(context['categorical_stats'].items())[:3]
        ])

        return f"""
请基于以下数据分析结果，生成专业数据分析报告：

## 分析背景
用户查询：{query}
分析时间：{context['timestamp']}
数据规模：{context['data_shape'][0]}行 × {context['data_shape'][1]}列

## 数据质量概况
- 完整率：{context['data_quality']['completeness']:.1f}%
- 重复数据：{context['data_quality']['duplicates']}条
- 缺失值总数：{context['data_quality']['total_missing']}个

## 关键字段统计
**数值字段：**
{numeric_desc}

**分类字段：**
{categorical_desc}

## 异常情况
{context['anomalies']}

## 分析执行结果
{context['execution_result'][:1000]}

## 生成图表
{chart_descriptions}

## 报告要求
包含执行摘要、数据概况、关键发现、图表解读、异常分析、业务建议，语言专业易懂，建议具体可操作。
"""

    def _detect_anomalies(self, df: pd.DataFrame) -> str:
        """使用大模型智能检测异常"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return "无数值字段，无法进行异常检测"

        # 整理字段统计信息
        anomaly_prompt = f"数据形状：{df.shape}\n数值字段：{list(numeric_cols)}\n各字段统计信息：\n"
        for col in numeric_cols[:5]:
            data = df[col].dropna()
            if len(data) > 0:
                stats = data.describe()
                anomaly_prompt += f"- {col}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 范围=[{stats['min']:.2f}, {stats['max']:.2f}], 缺失值={df[col].isnull().sum()}\n"

        anomaly_prompt += """
请从专业数据分析角度指出：
1. 哪些字段可能存在异常值？
2. 数据分布有什么异常特征？
3. 数据质量需要注意什么问题？
"""

        try:
            response = self.llm.invoke(anomaly_prompt)
            return response.generations[0].message.content
        except Exception as e:
            return "异常检测分析暂不可用"

    def _clean_report_content(self, content: str, df: pd.DataFrame) -> str:
        """清理和格式化报告内容"""
        header = f"""# 数据分析报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据规模**：{len(df)} 条记录

---
"""
        footer = f"""

---
*本报告由智能数据分析系统生成*
*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return header + content + footer

    def _generate_fallback_report(self, df: pd.DataFrame, execution_result: Dict, images: List[str], query: str) -> str:
        """降级报告生成方案"""
        return f"""
# 数据分析报告

**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析主题**：{query}

## 数据概览
- 规模：{df.shape[0]} 行 × {df.shape[1]} 列
- 完整率：{(df.count().sum()) / (df.shape[0] * df.shape[1]) * 100:.1f}%
- 主要字段：{', '.join(df.columns.tolist()[:5])}

## 分析结果
{execution_result.get('result', '分析完成')}

## 生成图表
共生成 {len(images)} 张图表，展示关键数据模式。

## 核心发现
基于数据分析，发现数据主要特征和趋势，建议关注关键指标变化。

## 后续建议
1. 定期监控核心指标；
2. 优化数据质量；
3. 基于分析结果调整业务决策。
"""

    def _enhanced_anomaly_detection(self, df: pd.DataFrame) -> str:
        """结合统计方法和LLM的异常检测"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=['number']).columns[:6]  # 检查前6个数值字段

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 10:
                continue

            # IQR异常检测
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            if len(outliers) > 0:
                anomalies.append({
                    'field': col, 'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(data)) * 100,
                    'outlier_values': outliers.head(5).tolist(),
                    'bounds': [lower_bound, upper_bound]
                })

        # 调用LLM分析异常业务意义
        if anomalies:
            anomaly_prompt = f"发现以下数据异常：\n{json.dumps(anomalies, ensure_ascii=False, indent=2)}\n请分析：1. 业务含义；2. 数据错误还是业务异常；3. 处理建议。"
            try:
                response = self.llm.invoke(anomaly_prompt)
                return f"## 异常检测结果\n\n{response.generations[0].message.content}"
            except Exception as e:
                # 降级统计描述
                anomaly_desc = "## 异常检测结果\n\n"
                for a in anomalies:
                    anomaly_desc += f"- {a['field']}: 发现{a['outlier_count']}个异常值({a['outlier_percentage']:.1f}%)\n"
                return anomaly_desc
        else:
            return "## 异常检测结果\n\n未发现显著的统计异常值。"

    # 补充：若之前依赖_mine_numeric_relation方法，可添加默认实现（根据实际需求调整）
    def _mine_numeric_relation(self, df: pd.DataFrame, field_meta: str) -> str:
        """挖掘数值字段关联逻辑（默认实现，可根据业务完善）"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return "数值字段不足，无法挖掘关联逻辑"

        # 计算字段间相关性（示例逻辑）
        corr = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] > 0.7:  # 相关性>0.7视为强关联
                    high_corr_pairs.append(f"{corr.columns[i]}与{corr.columns[j]}（相关系数：{corr.iloc[i, j]:.2f}）")

        if high_corr_pairs:
            return f"强关联数值字段：{', '.join(high_corr_pairs)}"
        else:
            return "未发现数值字段间的强关联关系"