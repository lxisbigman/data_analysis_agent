###report_agent
import uuid
import logging
from datetime import datetime
import streamlit as st
from tools.report_static_generate_v2 import StatReportAgent
from tools.report_generate import DataProcessor
from tools.DB_connect import DatabaseConnector
import os
import _io
import pandas as pd

logger = logging.getLogger(__name__)

# 复用原代码的表映射（确保模板报告查表正确）
TABLE_COMMENT_MAPPING = {
    "办事处数据": "office_data",
    "优秀网络": "good_data",
    "CD级分析": "cd_level_net_analysis",
    "硬件整改": "hard_change",
    "维度分析": "dimension_change",
    "网元质量画像结果": "net_quality",
    "老旧设备退网": "old_equipment_backnet"
}
TABLE_COMMENT_MAPPING_REVERSE = {v: k for k, v in TABLE_COMMENT_MAPPING.items()}


class ReportAgent:
    def __init__(self):
        self.stat_agent = StatReportAgent()  # 统计报告生成器（复用原代码）
        self.report_processor = DataProcessor()  # 模板报告生成器（复用原代码）
        self.db_connector = DatabaseConnector()  # 数据库连接（模板报告查表用）

    def run(self, user_query: str, extra_info: dict, retrieval_result: dict = None, full_execution_result: dict = None) -> dict:
        """
        执行报告生成（自动区分统计报告和模板报告）
        :param user_query: 用户原始查询
        :param extra_info: 意图识别的额外信息（含模板类型）
        :param retrieval_result: 数据检索Agent的结果（统计报告需用）
        :param full_execution_result: 统计分析的完整结果（含元数据，统计报告需用）
        :return: 报告结果字典
        """
        try:
            # 分支1：统计报告（含“统计”关键词）
            if "统计" in user_query and "报告" in user_query:
                if not retrieval_result or not retrieval_result.get("success"):
                    raise ValueError("统计报告需先执行数据检索")
                if not full_execution_result:
                    raise ValueError("未获取到统计分析结果（缺少full_execution_result）")
                return self._generate_stat_report(user_query, retrieval_result, full_execution_result)

            # 分支2：模板报告（IPRAN/RAN，无“统计”关键词）
            else:
                template_intent = extra_info.get("template_intent", "ipran")
                template_confidence = extra_info.get("template_confidence", 0.5)
                return self._generate_template_report(template_intent, template_confidence)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"报告生成失败：{error_msg}")
            return {
                "success": False,
                "report_content": "",
                "doc_stream": None,
                "report_type": "",
                "error": error_msg
            }

    def _generate_stat_report(self, user_query: str, retrieval_result: dict, full_execution_result: dict) -> dict:
        """生成统计报告（核心优化：确保真实数据完整传递）"""
        raw_df = retrieval_result["raw_df"]
        related_tables = retrieval_result["related_tables"]

        # 1. 从 full_execution_result 中提取关键信息（图表+分析文本）
        chart_metadata_list = full_execution_result.get("chart_metadata_list", [])
        analysis_conclusion = full_execution_result.get("result", "未生成具体分析结论")
        chart_paths = [os.path.abspath(img) for img in full_execution_result.get("saved_images", []) if
                       os.path.exists(img) and os.path.getsize(img) > 0]

        # 2. 强制对齐元数据与图表路径（核心修复：保留完整元数据，尤其是data字段）
        aligned_metadata = []
        for idx, path in enumerate(chart_paths):
            # 优先匹配已有元数据
            meta = next((m for m in chart_metadata_list if os.path.abspath(m.get("image_path", "")) == path), None)
            if meta:
                # 补全缺失的chart_index
                if "chart_index" not in meta:
                    meta["chart_index"] = idx + 1
                    logger.warning(f"图表{idx + 1}元数据缺失chart_index，自动补充为 {idx + 1}")
                # 补全缺失的chart_type
                if "chart_type" not in meta:
                    meta["chart_type"] = "未知类型"
                    logger.warning(f"图表{idx + 1}元数据缺失chart_type，自动补充为未知类型")
                # 验证是否包含真实数据字段（data）
                if "data" not in meta:
                    logger.warning(f"图表{idx + 1}元数据缺失data字段，可能导致解读依赖推测")
                # 保留完整元数据（关键：不丢弃任何字段）
                aligned_metadata.append(meta)
            else:
                # 生成默认元数据（包含基础数据示例）
                sample_data = raw_df.head(3).to_dict(orient="records") if not raw_df.empty else []
                default_meta = {
                    "chart_index": idx + 1,
                    "chart_type": "未知类型",
                    "title": f"图表{idx + 1}：{user_query[:10]}相关分析",
                    "description": f"基于{len(raw_df)}行原始数据生成，展示关键统计指标",
                    "image_path": path,
                    "data": sample_data,  # 新增：默认元数据添加简单数据示例
                    "x_axis": raw_df.columns[0] if not raw_df.empty else "未知",
                    "y_axis": "统计值"
                }
                aligned_metadata.append(default_meta)

        # 3. 调试日志：确认真实数据传递情况
        logger.info(f"统计报告生成：图表{len(chart_paths)}张，元数据{len(aligned_metadata)}条")
        if aligned_metadata:
            logger.info(f"元数据示例（图表1）：{aligned_metadata[0].keys()}")
            if "data" in aligned_metadata[0]:
                logger.info(f"图表1真实数据示例：{aligned_metadata[0]['data'][:2]}")  # 打印前2条数据

        # 4. 调用StatReportAgent（显式传递所有关键信息）
        report = self.stat_agent.generate_stat_report(
            df=raw_df,
            execution_result=full_execution_result,
            images=chart_paths,
            query=user_query,
            table_names=related_tables,
            chart_metadata_list=aligned_metadata
        )

        if report.get("error"):
            raise ValueError(f"StatReportAgent生成失败：{report['error']}")

        # 修复统计报告的doc_stream验证
        # 修复统计报告的doc_stream验证
        doc_stream = report.get("doc_stream")
        if doc_stream:
            # 强制处理BytesIO对象（核心修复）
            if isinstance(doc_stream, _io.BytesIO):  # 显式判断BytesIO类型
                try:
                    doc_stream = doc_stream.getvalue()  # 提取字节流
                    logger.info(f"成功将BytesIO转换为字节流，长度：{len(doc_stream)}字节")
                except Exception as e:
                    logger.error(f"BytesIO转换字节流失败：{e}")
                    doc_stream = None
            elif not isinstance(doc_stream, bytes):  # 其他非字节流类型
                logger.error(f"统计报告doc_stream类型错误：{type(doc_stream)}，无法转换")
                doc_stream = None
        else:
            logger.warning("StatReportAgent未返回doc_stream，尝试重新生成")
            # 降级方案：调用模板报告的保存方法兜底
            try:
                doc_stream = self.report_processor.save_report_to_docx(report["full_report"])
                logger.info(f"兜底生成Word字节流，长度：{len(doc_stream)}字节")
            except Exception as e:
                logger.error(f"兜底生成统计报告Word失败：{e}")
                doc_stream = None

        return {
            "success": True,
            "report_content": report["full_report"],
            "doc_stream": doc_stream,
            "report_type": "statistical",
            "chart_metadata_list": aligned_metadata,
            "error": None
        }

    def _generate_template_report(self, template_intent: str, template_confidence: float) -> dict:
        """生成模板报告（复用原代码逻辑，不做修改）"""
        template_display_name = "IPRAN" if template_intent == "ipran" else "RAN"

        # 1. 加载模板所需的固定表数据
        required_tables = self.report_processor.get_tables_for_product(template_intent)
        table_comments = [TABLE_COMMENT_MAPPING_REVERSE.get(tab, tab) for tab in required_tables]

        # 2. 确保数据库连接有效
        if not self.db_connector.connected:
            success, msg = self.db_connector.connect()
            if not success:
                raise ValueError(f"数据库连接失败：{msg}")
        report_tables = self.db_connector.get_tables_by_comments(table_comments)

        # 3. 检查缺失表
        loaded_tables = list(report_tables.keys())
        missing_tables = [tab for tab in required_tables if tab not in loaded_tables]
        if missing_tables:
            logger.warning(f"模板报告缺失表：{missing_tables}")

        # 4. 生成模板报告
        if template_intent == "ipran":
            report = self.report_processor.generate_ipran_report(report_tables)
            report_title = "IPRAN网络画像报告"
        else:
            report = self.report_processor.generate_ran_report(report_tables)
            report_title = "RAN网络画像报告"

        if report.get("error"):
            raise ValueError(report["error"])

        try:
            # 1. 调用保存方法（返回 BytesIO 流对象）
            doc_stream = self.report_processor.save_report_to_docx(report["full_report"])

            # 2. 复用统计报告的成功逻辑：处理 BytesIO 转字节流
            if isinstance(doc_stream, _io.BytesIO):
                try:
                    doc_stream = doc_stream.getvalue()  # 关键：提取字节流
                    logger.info(f"模板报告：成功将BytesIO转为字节流，长度：{len(doc_stream)}字节")
                except Exception as e:
                    logger.error(f"模板报告BytesIO转换失败：{e}")
                    raise ValueError(f"转换异常：{e}")

            # 3. 后续校验不变（现在doc_stream已经是bytes了）
            if not isinstance(doc_stream, bytes):
                raise ValueError(f"Word文档不是字节流，类型：{type(doc_stream)}")
            if len(doc_stream) < 30:
                logger.warning(f"Word文档字节流过短（{len(doc_stream)}字节），可能为空报告")

        except Exception as e:
            logger.error(f"Word文档生成失败：{str(e)}")
            doc_stream = None

        return {
            "success": True,
            "report_content": report["full_report"],
            "doc_stream": doc_stream,
            "report_type": template_intent,
            "template_info": {
                "name": template_display_name,
                "confidence": template_confidence,
                "loaded_tables": loaded_tables,
                "missing_tables": missing_tables
            },
            "reasoning": report.get("reasoning", ""),
            "error": None
        }
    def display_result(self, report_result: dict):
        """展示报告结果（优化统计报告的真实数据展示）"""
        if not report_result["success"]:
            st.error(f"❌ 报告生成失败：{report_result['error']}")
            return

        report_type = report_result["report_type"]
        report_content = report_result["report_content"]
        doc_stream = report_result["doc_stream"]

        # 1. 模板报告展示（复用原逻辑）
        if report_type in ["ipran", "ran"]:
            template_info = report_result["template_info"]
            st.info(f"模板类型：{template_info['name']}（置信度：{template_info['confidence']:.2f}）")
            st.info(f"已加载表：{', '.join(template_info['loaded_tables'])}")
            if template_info["missing_tables"]:
                st.warning(f"缺失表：{template_info['missing_tables']}")
            if report_result.get("reasoning"):
                with st.expander("📊 LLM 智能推理过程", expanded=False):
                    st.markdown(report_result["reasoning"])
            st.markdown(f"### {template_info['name']}网络画像报告")

        # 2. 统计报告展示（优化：显示真实数据摘要）
        else:
            st.markdown("### 📊 统计分析报告")
            st.info("报告基于以下图表生成（含真实数据摘要）：")
            for meta in report_result.get("chart_metadata_list", []):
                chart_index = meta.get("chart_index", "未知序号")
                chart_type = meta.get("chart_type", "未知类型")
                title = meta.get("title", "未知标题")
                # 显示真实数据摘要（关键：让用户看到真实数据）
                data_sample = meta.get("data", [])[:2]  # 只显示前2条数据，避免冗余
                st.text(f"图表{chart_index}：{title}（{chart_type}）| 数据示例：{data_sample}")

        # 3. 通用报告内容与下载按钮
        st.markdown(report_content)
        if doc_stream:
            report_display_name = "统计" if report_type == "statistical" else report_result["template_info"]["name"]
            unique_key = f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
            st.download_button(
                label=f"📥 下载{report_display_name}报告（Word）",
                data=doc_stream,
                file_name=f"{report_display_name}报告_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=unique_key
            )


# 修复：移除嵌套的display_result方法（原代码存在方法嵌套bug）