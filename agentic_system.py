###agentic_system.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, Optional
import logging
import streamlit as st
import os
from agents.retrieve_agent import RetrieveAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)
#langgraph节点管理

# 1. 定义统一状态类（Agent间的数据桥梁）
class AgentState(TypedDict):
    # 基础信息
    user_query: str
    intent: str
    extra_info: Dict[str, Any]
    # 数据检索Agent的结果
    retrieval_result: Dict[str, Any] = {}
    # 统计分析Agent的结果（完整保存，不丢失字段）
    analysis_result: Dict[str, Any] = {}
    # 报告生成Agent的结果
    report_result: Dict[str, Any] = {}
    # 全局错误信息
    error: Optional[str] = None


# 2. 协调器类（管理工作流）
class AgentOrchestrator:
    def __init__(self):
        # 初始化三个独立Agent
        self.retrieve_agent = RetrieveAgent()
        self.analysis_agent = AnalysisAgent()
        self.report_agent = ReportAgent()
        # 构建LangGraph工作流
        self.graph = self._build_workflow()

    def _build_workflow(self):
        """构建工作流：定义节点、跳转规则"""
        workflow = StateGraph(AgentState)

        # 定义节点（每个节点对应一个Agent的执行逻辑）
        workflow.add_node("retrieve_data", self._run_retrieve_agent)  # 数据检索节点
        workflow.add_node("analyze_data", self._run_analysis_agent)  # 统计分析节点
        workflow.add_node("generate_report", self._run_report_agent)  # 报告生成节点
        workflow.add_node("handle_error", self._handle_error)  # 错误处理节点
        workflow.add_node("finalize", self._finalize_result)  # 结果汇总节点

        # 入口节点：根据意图决定第一步
        workflow.set_entry_point("decide_first_step")
        workflow.add_node("decide_first_step", self._decide_first_step)

        # 1. 第一步决策后的跳转
        workflow.add_conditional_edges(
            "decide_first_step",
            self._get_next_after_first_step,
            {
                "retrieve_data": "retrieve_data",
                "generate_report": "generate_report",
                "handle_error": "handle_error"
            }
        )

        # 2. 数据检索后的跳转
        workflow.add_conditional_edges(
            "retrieve_data",
            self._get_next_after_retrieval,
            {
                "analyze_data": "analyze_data",
                "generate_report": "generate_report",
                "finalize": "finalize",
                "handle_error": "handle_error"
            }
        )

        # 3. 统计分析后的跳转
        workflow.add_conditional_edges(
            "analyze_data",
            self._get_next_after_analysis,
            {
                "generate_report": "generate_report",
                "finalize": "finalize",
                "handle_error": "handle_error"
            }
        )

        # 4. 报告生成后的跳转
        workflow.add_edge("generate_report", "finalize")
        # 5. 错误处理后结束
        workflow.add_edge("handle_error", END)
        # 6. 结果汇总后结束
        workflow.add_edge("finalize", END)

        return workflow.compile()

    # ------------------------------ 节点执行逻辑 ------------------------------
    def _decide_first_step(self, state: AgentState) -> AgentState:
        """第一步决策：根据意图判断先执行哪个Agent"""
        intent = state["intent"]
        user_query = state["user_query"]

        # 模板报告（无需检索/分析）：直接生成报告
        if intent == "报告生成" and "统计" not in user_query:
            state["error"] = None
        # 其他意图（数据检索、统计分析、统计报告）：先检索数据
        elif intent in ["数据检索", "数据统计分析", "报告生成"]:
            state["error"] = None
        else:
            state["error"] = f"不支持的意图：{intent}"
        return state

    def _run_retrieve_agent(self, state: AgentState) -> AgentState:
        logger.info("=== 进入 _run_retrieve_agent 节点 ===")
        retrieval_result = self.retrieve_agent.run(
            user_query=state["user_query"],
            extra_info=state["extra_info"]
        )
        state["retrieval_result"] = retrieval_result
        state["error"] = None if retrieval_result["success"] else retrieval_result["error"]

        if retrieval_result["success"]:
            # 实时展示UI
            st.markdown("### 🔍 数据检索基础信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总行数", len(retrieval_result["raw_df"]))
            with col2:
                st.metric("总字段数", len(retrieval_result["raw_df"].columns))
            with col3:
                st.metric("缺失值总数", retrieval_result["raw_df"].isnull().sum().sum())
            st.subheader("数据预览（前5行）")
            st.dataframe(retrieval_result["raw_df"].head(), width='stretch')  # 修复警告：用width='stretch'

            with st.expander("📋 检索完整详情", expanded=False):
                st.markdown("#### 原始需求")
                st.text(state["user_query"])
                st.markdown("#### 执行SQL")
                st.code(retrieval_result["generated_sql"], language="sql")
                st.markdown("#### 推理过程")
                st.text(retrieval_result["reasoning_process"])

            st.success(f"✅ 数据检索完成：涉及表 {retrieval_result['related_tables']}")

            # 追加到消息列表（持久化）
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""
### 🔍 数据检索结果
- 涉及表：{retrieval_result['related_tables']}
- 数据量：{len(retrieval_result['raw_df'])} 行 / {len(retrieval_result['raw_df'].columns)} 列
- 缺失值：{retrieval_result['raw_df'].isnull().sum().sum()} 个
- 预览：前5行数据已展示
""",
                "type": "retrieval",
                "details": {
                    "raw_df_preview": retrieval_result["raw_df"].head().to_html(),
                    "generated_sql": retrieval_result["generated_sql"],
                    "reasoning_process": retrieval_result["reasoning_process"]
                }
            })
        else:
            st.error(f"❌ 数据检索失败：{retrieval_result['error']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 数据检索失败：{retrieval_result['error']}",
                "type": "text"
            })

        return state

    def _run_analysis_agent(self, state: AgentState) -> AgentState:
        logger.info("=== 进入 _run_analysis_agent 节点 ===")
        retrieval_result = state["retrieval_result"]
        analysis_result = self.analysis_agent.run(
            user_query=state["user_query"],
            raw_df=retrieval_result["raw_df"]
        )

        # 核心修复：完整保存分析结果（包含full_execution_result）
        state["analysis_result"] = analysis_result  # 不再手动提取字段，避免丢失
        state["error"] = None if analysis_result["success"] else analysis_result["error"]

        if analysis_result["success"]:
            # 实时展示UI
            st.markdown("### 📈 统计分析结果")
            valid_chart_paths = analysis_result["chart_paths"]
            cols = st.columns(2)
            for idx, img_path in enumerate(valid_chart_paths):
                with cols[idx % 2]:
                    st.image(img_path, caption=f"图表 {idx + 1}", use_container_width=True)

            expected = analysis_result["expected_count"]
            actual = len(valid_chart_paths)
            if expected != actual:
                st.warning(f"⚠️ 图片数量不匹配：预期 {expected} 张，实际 {actual} 张")
            else:
                st.success(f"✅ 图表生成完成：共 {actual} 张")

            st.markdown("#### 分析结论")
            st.markdown(analysis_result["analysis_text"])

            # 追加到消息列表（持久化）
            chart_captions = [f"图表 {idx + 1}：{os.path.basename(path)}" for idx, path in enumerate(valid_chart_paths)]
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"""
### 📊 统计分析结果
- 图表总数：{actual} 张（{', '.join(chart_captions)}）
- 分析结论：{analysis_result['analysis_text']}
- 数量验证：{'通过' if expected == actual else f'失败（预期{expected}，实际{actual}）'}
""",
                "type": "analysis",
                "chart_paths": valid_chart_paths,
                "analysis_text": analysis_result["analysis_text"]
            })
        else:
            st.error(f"❌ 统计分析失败：{analysis_result['error']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 统计分析失败：{analysis_result['error']}",
                "type": "text"
            })

        return state

    def _run_report_agent(self, state: AgentState) -> AgentState:
        # 从state中提取所有必要参数
        user_query = state["user_query"]
        extra_info = state["extra_info"]
        retrieval_result = state.get("retrieval_result", {})
        analysis_result = state.get("analysis_result", {})  # 获取完整的分析结果

        # 核心修复：直接从完整分析结果中提取full_execution_result
        full_execution_result = analysis_result.get("full_execution_result", {})

        # 调用ReportAgent生成报告
        report_result = self.report_agent.run(
            user_query=user_query,
            extra_info=extra_info,
            retrieval_result=retrieval_result,
            full_execution_result=full_execution_result  # 传递完整统计结果
        )

        state["report_result"] = report_result
        state["error"] = None if report_result["success"] else report_result["error"]

        if report_result["success"] and "统计" in state["user_query"]:
            # 实时展示统计报告UI
            st.markdown("### 📊 统计分析报告")
            st.info(f"基于 {len(report_result.get('chart_metadata_list', []))} 张图表生成")
            st.markdown(report_result["report_content"])

            # 追加到消息列表（持久化）
            st.session_state.messages.append({
                "role": "assistant",
                "content": report_result["report_content"],
                "type": "report",
                "report_type": "statistical",
                "doc_stream": report_result["doc_stream"]
            })
        return state

    def _finalize_result(self, state: AgentState) -> AgentState:
        logger.info("=== 进入 _finalize_result 节点 ===")
        intent = state["intent"]
        report_result = state.get("report_result", {})

        # 处理模板报告（非统计类）
        if intent == "报告生成" and "统计" not in state["user_query"] and report_result.get("success"):
            template_info = report_result["template_info"]
            # 实时展示模板报告UI
            st.markdown("### 📝 模板报告")
            st.info(f"模板类型：{template_info['name']}（置信度：{template_info['confidence']:.2f}）")
            st.info(f"已加载表：{', '.join(template_info['loaded_tables'])}")
            if template_info["missing_tables"]:
                st.warning(f"缺失表：{template_info['missing_tables']}")
            if report_result.get("reasoning"):
                with st.expander("📊 LLM 推理过程", expanded=False):
                    st.markdown(report_result["reasoning"])
            st.markdown(report_result["report_content"])

            # 追加到消息列表（持久化）
            st.session_state.messages.append({
                "role": "assistant",
                "content": report_result["report_content"],
                "type": "report",
                "report_type": report_result["report_type"],
                "doc_stream": report_result["doc_stream"],
                "template_info": template_info
            })
        return state

    def _handle_error(self, state: AgentState) -> AgentState:
        """错误处理：显示错误信息"""
        st.error(f"❌ 执行失败：{state['error']}")
        logger.error(f"工作流错误：{state['error']}")
        # 同步错误信息到会话消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ 执行失败：{state['error']}",
            "type": "text"
        })
        return state

    # ------------------------------ 跳转规则 ------------------------------
    def _get_next_after_first_step(self, state: AgentState) -> str:
        if state["error"]:
            return "handle_error"
        intent = state["intent"]
        user_query = state["user_query"]
        if intent == "报告生成" and "统计" not in user_query:
            return "generate_report"
        else:
            return "retrieve_data"

    def _get_next_after_retrieval(self, state: AgentState) -> str:
        if state["error"]:
            return "handle_error"
        intent = state["intent"]
        user_query = state["user_query"]
        if intent == "数据检索":
            return "finalize"
        elif intent == "数据统计分析" or (intent == "报告生成" and "统计" in user_query):
            return "analyze_data"
        else:
            return "finalize"

    def _get_next_after_analysis(self, state: AgentState) -> str:
        if state["error"]:
            return "handle_error"
        intent = state["intent"]
        user_query = state["user_query"]
        if intent == "数据统计分析":
            return "finalize"
        elif intent == "报告生成" and "统计" in user_query:
            return "generate_report"
        else:
            return "finalize"

    # ------------------------------ 对外接口 ------------------------------
    def run(self, user_query: str, intent: str, extra_info: Dict[str, Any]) -> None:
        """对外统一接口：初始化状态并执行工作流"""
        initial_state = AgentState(
            user_query=user_query,
            intent=intent,
            extra_info=extra_info
        )
        self.graph.invoke(initial_state)