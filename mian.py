# main.py
import streamlit as st
import os
import matplotlib.pyplot as plt
import uuid
import logging
from tools.DB_connect import DatabaseConnector
from tools.report_generate import DataProcessor
from tools.LLM_invoke import DeepSeekLLM
from intent.intent_rec import IntentRecognizer
from memory.save_memory import init_long_term_memory, get_history_sessions, load_memory, save_memory
from typing import Dict, Any
import pandas as pd
# 导入协调器
from agentic_system import AgentOrchestrator
from io import StringIO
from intent.feedback_utils import save_intent_feedback
import time
from datetime import datetime
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 Matplotlib 中文字体
plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Heiti TC", "Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 配置常量
MEMORY_DIR = "chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)


# ================= 初始化会话状态 =================
def init_session_state():
    # 1. 初始化Session ID（仅首次生成）
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
        st.query_params["session_id"] = st.session_state["session_id"]

    # 2. 初始化核心状态（仅首次设置默认值）
    defaults = {
        'df': None,
        'messages': [],
        'llm': DeepSeekLLM(),
        'data_loaded': False,
        'report_processor': DataProcessor(),
        'db_connector': DatabaseConnector(),
        'current_table_data': None,
        'last_intent': None,
        'current_query': None,
        'current_intent': None,
        'current_extra_info': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    init_long_term_memory()
    logger.info(f"✅ 会话初始化完成，Session ID：{st.session_state.session_id}")


# ================= 新增：生成执行计划显示 =================
def display_execution_plan(intent: str, query: str, extra_info: Dict[str, Any]):
    """显示 Agentic System 的决策过程和执行计划"""
    reasoning = extra_info.get("reasoning", "未提供推理过程")
    llm_raw = extra_info.get("llm_raw", {})

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**最终意图**：{intent}")

    if intent == "报告生成" and "template_intent" in extra_info:
        st.markdown(
            f"**报告模板**：{extra_info['template_intent']} (置信度：{extra_info.get('template_confidence', 0):.2f})")

    # 2. 显示执行计划（基于意图预计算工作流路径）
    st.markdown("### 📋 **Agentic System 执行计划**")
    plan_map = {
        "报告生成": {
            "统计": ["1. 🔍 数据检索 → 2. 📊 统计分析 → 3. 📝 生成报告"],
            "模板": ["1. 📝 直接生成模板报告（无需数据处理）"]
        },
        "数据统计分析": ["1. 🔍 数据检索 → 2. 📊 统计分析 → 3. 📈 显示图表与洞察"],
        "数据检索": ["1. 🔍 数据检索 → 2. 📋 显示原始数据与推理过程"],
        "闲聊": ["1. 💬 简单响应（无需 Agent 执行）"]
    }

    if intent == "报告生成":
        if "统计" in query.lower():
            plan = plan_map["报告生成"]["统计"]
        else:
            plan = plan_map["报告生成"]["模板"]
    else:
        plan = plan_map.get(intent, ["未知意图：直接结束"])

    for step in plan:
        st.markdown(f"- {step}")

    st.markdown("---")
    st.info("🔄 决策完成，正在执行计划...")


# ================= 新增：意图反馈按钮 =================
# ================= 意图反馈函数（最终优化版）=================
def add_intent_feedback(session_id: str, current_intent: str, query: str):
    # 用状态锁避免重运行时重复渲染反馈表单
    if st.session_state.get("feedback_submitted"):
        return

    st.markdown("---")
    st.markdown("### 🔍 意图识别反馈")
    st.info(f"查询：{query}\n识别意图：{current_intent}")

    with st.form(key=f"feedback_form_{session_id}"):
        col1, col2 = st.columns(2)
        correct = st.form_submit_button("✅ 意图正确", type="primary")
        wrong = st.form_submit_button("❌ 意图错误")

        # 1. 先执行保存逻辑（表单提交时优先执行）
        if correct:
            logger.info("🟢 意图正确，执行保存...")
            save_intent_feedback(session_id, query, current_intent, True)
            st.success("反馈保存成功！")
            # 2. 标记反馈已提交，避免 rerun 后重复渲染
            st.session_state["feedback_submitted"] = True
            # 3. 可选：延迟刷新，让用户看到提示
            time.sleep(1)

        if wrong:
            logger.info("🔴 意图错误，执行保存...")
            save_intent_feedback(session_id, query, current_intent, False)
            st.success("反馈保存成功！")
            st.session_state["feedback_submitted"] = True
            time.sleep(1)
# ================= 主应用 =================
def main():
    st.set_page_config(page_title="数析报告智能体", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
        <style>
        /* ========== 1. 移除宽度限制 ========== */
        section.main > div:has(> div.stChatFloatingButton) {
            max-width: 100% !important;
        }
        .css-1y0t6an, .main > div, .block-container {
            max-width: 100% !important;
            padding-left: 20rem !important;
            padding-right: 20rem !important;
        }

        /* ========== 2. 正确选择器：基于 data-role ========== */
        /* 用户消息 - 右对齐 */
        div[data-testid="stVerticalBlock"] 
          div.element-container 
          div[data-testid="stChatMessage"][data-role="user"] {
            margin-left: auto !important;
            margin-right: 0 !important;
            max-width: 70% !important;
            background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            float: right !important;
            clear: both !important;
        }

        /* 助手消息 - 左对齐 */
        div[data-testid="stVerticalBlock"] 
          div.element-container 
          div[data-testid="stChatMessage"][data-role="assistant"] {
            margin-right: auto !important;
            margin-left: 0 !important;
            max-width: 70% !important;
            background: linear-gradient(135deg, #f5f5f5, #e0e0e0) !important;
            border-radius: 18px 18px 18px 4px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            float: left !important;
            clear: both !important;
        }

        /* 清除浮动 */
        div[data-testid="stVerticalBlock"]::after {
            content: "";
            display: table;
            clear: both;
        }

        /* ========== 3. 文本优化 ========== */
        div[data-testid="stChatMessage"] p {
            margin: 0 !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }

        /* ========== 4. 输入框美化 ========== */
        div.stChatInput {
            padding: 1rem 2rem !important;
            background-color: #fafafa;
            border-top: 1px solid #eee;
        }
        input[data-testid="stChatInput"] {
            border-radius: 25px !important;
            padding: 12px 20px !important;
            border: 1px solid #ccc !important;
            font-size: 1rem !important;
        }

        /* ========== 5. 响应式 ========== */
        @media (max-width: 768px) {
            .css-1y0t6an, .main > div {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            div[data-testid="stChatMessage"] {
                max-width: 90% !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)

    init_session_state()
    # 初始化协调器
    orchestrator = AgentOrchestrator()

    # 页面标题
    col1, col2 = st.columns([1, 4])
    with col2:
        st.markdown('<h1 class="main-header">📊 Data Assistant</h1>', unsafe_allow_html=True)
    st.markdown("基于远程大模型的数据分析工具：连接数据库 → 提问 → 智能响应（支持长期记忆）")
    # 新增：处理反馈保存（在session state中存在反馈时执行）
    if "feedback" in st.session_state and st.session_state["feedback"]:
        feedback = st.session_state["feedback"]
        try:
            save_intent_feedback(
                session_id=feedback["session_id"],
                query=feedback["query"],
                intent=feedback["intent"],
                is_correct=feedback["is_correct"]
            )
            logger.info(f"📌 反馈保存完成")
        except Exception as e:
            logger.error(f"❌ 反馈保存失败：{str(e)}")
        # 清除反馈状态，避免重复保存
        del st.session_state["feedback"]
    # 侧边栏（保留原代码逻辑）
    with st.sidebar:
        st.markdown("### 🛢️ 数据库连接")
        if not st.session_state.db_connector.connected:
            if st.button("连接数据库", use_container_width=True):
                success, msg = st.session_state.db_connector.connect()
                if success:
                    st.success(msg)
                    tables, msg = st.session_state.db_connector.get_tables_with_comments()
                    if tables:
                        with st.expander("查看数据库表信息"):
                            for table, comment in tables.items():
                                st.write(f"- {table}: {comment or '无注释'}")
                else:
                    st.error(msg)
        else:
            st.success("已连接到数据库")
            tables, msg = st.session_state.db_connector.get_tables_with_comments()
            if tables:
                with st.expander("数据库表信息"):
                    for table, comment in tables.items():
                        st.write(f"- {table}: {comment or '无注释'}")

        # 清除所有数据按钮
        if st.button("🗑️ 清除所有数据", use_container_width=True):
            st.session_state.df = None
            st.session_state.messages = []
            st.session_state.data_loaded = False
            st.session_state.current_table_data = None
            # 删除生成的图片文件
            for filename in os.listdir("."):
                if filename.startswith("plot_") and filename.endswith(".png"):
                    try:
                        os.remove(filename)
                    except Exception as e:
                        logger.warning(f"删除图片文件失败: {filename}, 错误: {e}")
            st.success("已清除所有数据和图片")
            st.rerun()

        # 会话记忆管理
        st.markdown("### 会话记忆")
        history_sessions = get_history_sessions()
        selected_session = st.selectbox("选择历史会话", options=["新建会话"] + history_sessions, index=0)
        if selected_session != "新建会话" and selected_session != st.session_state.get("session_id"):
            st.session_state.session_id = selected_session
            memory = load_memory(selected_session)
            st.session_state.messages = memory["messages"]
            st.success(f"已加载会话：{selected_session}")
            st.rerun()
        if st.button("💾 保存当前会话", use_container_width=True):
            save_memory()
            st.success(f"会话已保存（ID：{st.session_state.session_id}）")

        # 模型信息
        st.markdown("### 🤖 模型信息")
        st.info(f"远程模型: DeepSeek")

    # 主内容区：提问分析
    st.markdown("### 💬 提问分析")
    if not st.session_state.db_connector.connected:
        st.info("请先在左侧连接数据库")
        return

    # 循环渲染所有历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # 1. 图片类型
            if msg["type"] == "image":
                st.image(msg["content"], caption=os.path.basename(msg["content"]))

            # 2. 报告类型
            # 报告类型消息渲染处修改
            # 报告类型消息渲染处修改
            elif msg["type"] == "report":
                st.markdown(f"### {msg.get('report_type', '网络画像')}报告")
                st.markdown(msg["content"])

                # 简化版下载按钮：只要doc_stream存在且是字节流就显示
                if "doc_stream" in msg and msg["doc_stream"] is not None:
                    if isinstance(msg["doc_stream"], bytes):
                        # 强制生成唯一key（避免Streamlit组件冲突）
                        btn_key = f"download_{msg['report_type']}_{uuid.uuid4().hex[:10]}"
                        st.download_button(
                            label="📥 下载Word报告",
                            data=msg["doc_stream"],
                            file_name=f"{msg['report_type']}_报告_{datetime.now().strftime('%Y%m%d')}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=btn_key
                        )
                    else:
                        st.warning(f"文档格式异常（类型：{type(msg['doc_stream'])}）")
                else:
                    st.warning("未找到可用的报告文件")

            # 3. 数据检索类型
            elif msg["type"] == "retrieval":
                details = msg["details"]
                if not details:
                    st.warning("缺少检索详情数据")
                    continue

                # 数据检索基础信息
                st.markdown("### 🔍 数据检索基础信息")
                col1, col2, col3 = st.columns(3)

                try:
                    raw_df_preview = pd.read_html(StringIO(details["raw_df_preview"]))[0]
                except Exception as e:
                    st.error(f"解析数据预览失败：{str(e)}")
                    raw_df_preview = pd.DataFrame()

                with col1:
                    st.metric("总行数", len(raw_df_preview) if not raw_df_preview.empty else 0)
                with col2:
                    st.metric("总字段数", len(raw_df_preview.columns) if not raw_df_preview.empty else 0)
                with col3:
                    st.metric("缺失值总数", raw_df_preview.isnull().sum().sum() if not raw_df_preview.empty else 0)

                # 数据预览表格
                st.subheader("数据预览（前5行）")
                st.dataframe(raw_df_preview, width="stretch")

                # 完整详情展开面板
                with st.expander("📋 检索完整详情", expanded=False):
                    st.markdown("#### 原始需求")
                    user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
                    st.text(user_queries[-1] if user_queries else "未找到原始查询")

                    st.markdown("#### 执行SQL")
                    st.code(details.get("generated_sql", "无SQL语句"), language="sql")

                    st.markdown("#### 推理过程")
                    st.text(details.get("reasoning_process", "无推理过程"))

                st.success(f"✅ 数据检索完成：涉及表 {details.get('related_tables', '未知')}")

            # 4. 统计分析类型
            elif msg["type"] == "analysis":
                st.markdown("### 📈 统计分析结果")
                st.markdown(msg["content"])

            # 5. 其他文本类型
            else:
                st.markdown(msg["content"])
    # 独立渲染反馈表单（主内容区，无嵌套）
    if (st.session_state.get("current_intent") and
            st.session_state.get("current_query") and
            not st.session_state.get("feedback_submitted")):
        add_intent_feedback(
            st.session_state["session_id"],
            st.session_state["current_intent"],
            st.session_state["current_query"]
        )

    # 处理用户输入
    if query := st.chat_input("输入你的问题（例如：生成报告，分析数据等）"):
        # 保存当前查询
        st.session_state.current_query = query

        st.session_state.messages.append({"role": "user", "content": query, "type": "text"})
        with st.chat_message("user"):
            st.markdown(query)

        # 1. 意图识别
        intent_recognizer = IntentRecognizer()
        intent, confidence, extra_info = intent_recognizer.detect_intent(query)
        st.session_state.current_intent = intent
        st.session_state.current_extra_info = extra_info

        system_msg = {
            "role": "system",
            "content": f"检测到意图: {intent} (置信度: {confidence:.2f})",
            "type": "text"
        }
        st.session_state.messages.append(system_msg)
        with st.chat_message("system"):
            st.markdown(system_msg["content"])

        if st.session_state.last_intent is not None and intent != st.session_state.last_intent:
            logger.info(f"意图变更：从 {st.session_state.last_intent} 到 {intent}")
            intent_change_msg = {
                "role": "system",
                "content": f"检测到意图变更：从 {st.session_state.last_intent} 变为 {intent}",
                "type": "text"
            }
            st.session_state.messages.append(intent_change_msg)
            with st.chat_message("system"):
                st.markdown(intent_change_msg["content"])
            st.session_state.df = None
            st.session_state.current_table_data = None

        st.session_state.last_intent = intent

        # 3. 调用协调器执行任务
        with st.chat_message("assistant"):
            try:
                if intent == "需要澄清":
                    response_content = st.session_state.messages[-1]["content"]
                    st.markdown(response_content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "type": "text"
                    })
                elif intent == "闲聊":
                    # 1. 固定引导语：保留原有功能说明
                    base_chat_response = "您好！我是数据分析助手，可以帮您：\n\n" \
                                         "1. **查询数据**：如 '检索2025年安徽网络画像数据'\n" \
                                         "2. **生成报告**：如 '生成网络画像报告'\n" \
                                         "3. **数据分析**：如 '分析各城市综合得分分布'\n\n" \
                                         "针对您刚才的问题，我的回复如下："
                    st.markdown(base_chat_response)

                    # 2. 大模型兜底：提取纯文本回复，适配字符串拼接
                    try:
                        # 构造单条字符串prompt（不变）
                        llm_prompt = f"""
                            你是数据分析助手的闲聊模块，需满足以下要求：
                            1. 回复友好自然，符合日常对话语气，不要太官方；
                            2. 若用户问题与数据分析无关，正常闲聊即可，不用提及功能；
                            3. 若用户问题涉及数据分析需求，简要引导使用核心功能（数据查询/报告生成/统计分析），不用展开。
                            用户现在的问题：{query}
                            """
                        # 调用大模型（返回ChatResult类型，需提取text）
                        with st.spinner("正在思考您的问题..."):
                            chat_result = st.session_state.llm.invoke(llm_prompt)  # 返回ChatResult对象

                        # 核心修复：从ChatResult中提取纯字符串回复
                        # 1. 先获取generations列表的第一个元素（默认取第一个回复）
                        if chat_result.generations and len(chat_result.generations) > 0:
                            # llm_response = chat_result.generations[0].text
                            llm_response = chat_result.generations[0].message.content.strip()
                        else:
                            llm_response = "抱歉，我暂时没找到合适的回复~"

                        # 3. 显示大模型回复（纯字符串，可正常拼接）
                        st.markdown(f"\n{llm_response}")

                        # 4. 合并存入会话消息（纯字符串拼接，无报错）
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": base_chat_response + "\n\n" + llm_response,
                            "type": "text"
                        })

                    # 5. 异常处理：覆盖所有可能的错误场景
                    except Exception as e:
                        error_msg = f"\n⚠️ 暂时无法获取智能回复：{str(e)}，您可以尝试询问数据分析相关问题，或稍后再试~"
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": base_chat_response + error_msg,
                            "type": "text"
                        })
                else:
                    # 显示决策过程
                    display_execution_plan(intent, query, extra_info)

                    # 直接执行任务，不等待反馈
                    with st.spinner("🔄 正在执行计划..."):
                        logger.info(f"🚀 开始执行任务：{intent}，查询：{query}")
                        orchestrator.run(
                            user_query=query,
                            intent=intent,
                            extra_info=extra_info
                        )
                        logger.info(f"✅ 任务执行完成：{intent}")

                    task_result = f"✅ 任务执行完成（意图：{intent}）"
                    st.markdown(task_result)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": task_result,
                        "type": "text"
                    })

                    # 强制调用反馈函数，添加日志验证
                    logger.info(
                        f"🔍 准备调用反馈函数，参数：session_id={st.session_state.session_id}, intent={intent}, query={query}")
                    add_intent_feedback(st.session_state.session_id, intent, query)
                    logger.info(f"📌 反馈函数调用完成")

            except Exception as e:
                error_msg = f"处理失败：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"错误：{str(e)}",
                    "type": "text"
                })


if __name__ == "__main__":
    main()