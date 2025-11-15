# retrieve_agent.py
import pandas as pd
import logging
from env.DB_env import DB_CONFIG
from tools.text2sql import dynamic_permission_agent
from tools.DB_connect import DatabaseConnector
import streamlit as st

logger = logging.getLogger(__name__)


class RetrieveAgent:
    def __init__(self):
        self.db_connector = DatabaseConnector()  # 复用原代码的数据库连接工具

    def run(self, user_query: str, extra_info: dict) -> dict:
        """
        执行数据检索
        :param user_query: 用户原始查询
        :param extra_info: 意图识别的额外信息（含schema）
        :return: 检索结果字典（含原始数据、推理过程、错误信息等）
        """
        try:
            # 1. 从额外信息中获取schema（复用原代码逻辑）
            schema = extra_info.get("schema", DB_CONFIG["schema"])

            # 2. 调用text2sql智能体执行检索（复用原代码核心逻辑）
            logger.info(f"数据检索Agent：开始处理查询 -> {user_query}")
            agent_result = dynamic_permission_agent(
                user_query=user_query,
                schema=schema,
                max_retries=3
            )

            result_data = agent_result.get("result", {})
            related_tables = agent_result.get("tables", [])
            # 新增日志：打印关键信息
            logger.info(f"检索SQL：{agent_result.get('generated_sql', '')}")
            logger.info(f"识别的相关表：{related_tables}")
            logger.info(f"查询返回数据量：{len(result_data.get('data', []))}")
            # 3. 解析检索结果（复用原代码的错误判断）
            if not related_tables or result_data.get("status") != "success":
                raise ValueError("未识别到表或数据检索失败")

            # 4. 转换为原始数据DataFrame（复用原代码的raw_df逻辑）
            raw_df = pd.DataFrame(result_data.get("data", []))
            if raw_df.empty:
                raise ValueError("原始数据为空")

            # 5. 整理返回结果（含原代码的推理过程、SQL等）
            return {
                "success": True,
                "raw_df": raw_df,
                "related_tables": related_tables,
                "generated_sql": agent_result.get("generated_sql", ""),
                "reasoning_process": agent_result.get("reasoning_process", ""),
                "database_structure": agent_result.get("database_structure_summary", ""),
                "permission_types": agent_result.get("permission_types_defined", ""),
                "user_query": user_query,  # 新增：存入用户原始查询
                "error": None
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"数据检索失败：{error_msg}")
            return {
                "success": False,
                "raw_df": None,
                "error": error_msg
            }

    def display_result(self, retrieval_result: dict):
        logger.info("RetrieveAgent.display_result：开始渲染检索结果")
        logger.info(
            f"retrieval_result关键字段：success={retrieval_result.get('success', False)}, raw_df存在={retrieval_result.get('raw_df') is not None}")
        if not retrieval_result.get("success", False):
            st.error("❌ 检索失败，未显示结果")
            logger.warning("检索失败，跳过显示")
            return
        raw_df = retrieval_result.get("raw_df")
        if raw_df is None or raw_df.empty:
            st.warning("⚠️ 检索数据为空，请检查查询或数据库内容")
            logger.warning("raw_df为空")
            return
        related_tables = retrieval_result.get("related_tables", [])
        user_query = retrieval_result.get("user_query", "未记录")
        logger.info(f"渲染数据：行数={len(raw_df)}, 字段数={len(raw_df.columns)}, 相关表={related_tables}")

        # 1. 核心数据概览
        st.markdown("### 📊 核心数据概览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总行数", len(raw_df))
        with col2:
            st.metric("总字段数", len(raw_df.columns))
        with col3:
            st.metric("缺失值总数", raw_df.isnull().sum().sum())
        st.subheader("数据预览（前5行）")
        st.dataframe(raw_df.head(), use_container_width=True)
        st.subheader("字段列表")
        st.text(", ".join(raw_df.columns.tolist()))

        # 2. 详细检索信息
        with st.expander("📋 数据检索完整详情", expanded=False):
            st.markdown("#### 1. 用户原始需求")
            st.text(user_query)
            st.markdown("#### 2. 相关数据表")
            if related_tables:
                st.text(", ".join(related_tables))
            else:
                st.info("未识别到相关表")
            st.markdown("#### 3. 数据库结构概览")
            db_struct = retrieval_result.get("database_structure", "无")
            if db_struct:
                for line in db_struct.split(";"):
                    if line.strip():
                        st.text(f"- {line.strip()}")
            else:
                st.info("未获取数据库结构信息")
            st.markdown("#### 4. 检索推理过程")
            reasoning = retrieval_result.get("reasoning_process", "无")
            if reasoning:
                st.text_area("推理步骤", reasoning.replace("；", "；\n"), height=150)
            else:
                st.info("无推理过程记录")
            st.markdown("#### 5. 执行的SQL语句")
            sql = retrieval_result.get("generated_sql", "无")
            if sql:
                st.code(sql, language="sql")
            else:
                st.info("未生成SQL语句")
            st.markdown("#### 6. 权限类型")
            permission = retrieval_result.get("permission_types", "无")
            if permission:
                st.markdown(f"> {permission}")
            else:
                st.info("未定义权限类型")
        logger.info("RetrieveAgent.display_result：渲染完成")
        result_content = {
            "role": "assistant",
            "content": f"""
        ### 📊 核心数据概览
        - 总行数：{len(raw_df)}
        - 总字段数：{len(raw_df.columns)}
        - 缺失值总数：{raw_df.isnull().sum().sum()}
        - 相关表：{', '.join(related_tables)}
        """,
            "type": "retrieval",  # 标记类型为检索结果
            "raw_df_preview": raw_df.head().to_html(),  # 数据预览（HTML格式，支持重新渲染）
            "related_tables": related_tables,
            "user_query": user_query
        }
        # 追加到消息列表
        # st.session_state.messages.append(result_content)
        logger.info("RetrieveAgent.display_result：渲染完成并已存入消息")
