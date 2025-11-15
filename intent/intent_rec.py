#intent_rec.py
import streamlit as st
import json
import re
from typing import Tuple, Dict, Any
import logging
import requests
from env.LLM_env import DEEPSEEK_URL, DEEPSEEK_API_KEY, qwen3_url  # 复用LLM配置
import sys
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ================= 复用LLM调用函数（与dynamic_permission_agent一致） =================
def query_llm(prompt, system_prompt, history=None):
    """调用LLM，支持备用地址切换"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # 尝试主LLM
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, data=json.dumps({
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }))
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"主LLM调用失败: {str(e)}，切换到备用LLM")
        # 切换备用LLM
        resp = requests.post(qwen3_url, headers=headers, data=json.dumps({
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }))
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()


# ================= 优化后的意图识别模块 =================
class IntentRecognizer:
    """用户意图识别（基于大模型的精准识别 + 关键词兜底）"""

    def __init__(self):
        # 主意图定义（保留原有配置）
        self.intent_definitions = {
            "报告生成": {
                "description": "用户明确请求生成完整的、格式化的报告文档，通常包含总结、汇总、导出等关键词，输出是结构化的Word/PDF文档。",
                "positive_keywords": ["生成报告", "创建报告", "制作报告", "导出报告", "word报告", "pdf报告", "总结报告",
                                      "汇总报告"],
                "negative_keywords": ["图表", "统计图", "饼状图", "柱状图", "趋势图", "分析图", "可视化"],
                "examples": ["生成一份IPRAN网络画像报告", "创建硬件整改的Word报告", "导出办事处数据的总结文档"]
            },
            "数据统计分析": {
                "description": "用户请求对数据进行计算、统计、分析、可视化，生成图表或数值结果，关注数据洞察而非文档输出。",
                "positive_keywords": ["分析", "统计", "计算", "趋势", "占比", "比例", "分布", "图表", "饼状图",
                                      "柱状图", "可视化"],
                "negative_keywords": ["word", "pdf", "文档", "报告文档"],
                "examples": ["分析办事处数据的A占比", "统计优秀网络的趋势并生成柱状图", "计算CD级网络的数量分布"]
            },
            "数据检索": {
                "description": "用户单纯查询数据、查看原始记录、搜索特定信息，不涉及复杂的统计分析或报告生成。",
                "positive_keywords": ["查询", "查看", "搜索", "查找", "显示", "列出", "有哪些", "检索", "获取", "提取"],
                "negative_keywords": ["分析", "统计", "图表", "报告", "总结"],
                "examples": ["查询办事处数据", "查看CD级网络列表", "搜索硬件整改记录", "检索2025年安徽的网络画像数据"]
            },
            "闲聊": {
                "description": "用户进行非数据相关的对话，如问候、功能介绍、技术咨询或无关问题。",
                "positive_keywords": ["你好", "你好吗", "你是谁", "功能", "帮助", "怎么用", "天气", "聊天"],
                "negative_keywords": ["数据", "报告", "分析", "查询", "统计"],
                "examples": ["你好，系统有什么功能？", "今天天气怎么样？", "介绍一下你自己"]
            }
        }

        # 模板选择子意图定义（保留原有配置）
        self.template_intent_definitions = {
            "ipran": {
                "description": "IPRAN网络相关的报告模板",
                "keywords": ["ipran", "传输", "承载网", "接入网", "传输网络"],
                "examples": ["生成ipran报告", "ipran网络质量分析"]
            },
            "ran": {
                "description": "RAN无线接入网络相关的报告模板",
                "keywords": ["ran", "无线", "基站", "空口", "无线网络", "5g"],
                "examples": ["生成ran报告", "无线网络分析"]
            }
        }
        # ================= 新增：加载意图优化规则 =================
        self.optimize_config = self._load_optimize_config()

    def _load_optimize_config(self):
        """加载意图优化配置文件（用户反馈的规则）"""
        import os
        config_path = "intent_optimize_config.json"
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.warning(f"意图优化配置文件不存在：{config_path}，将使用默认空规则")
            return {"intent_correction_rules": []}
        # 读取配置文件
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 验证配置格式
            if "intent_correction_rules" not in config:
                config["intent_correction_rules"] = []
            logger.info(f"成功加载意图优化规则，共 {len(config['intent_correction_rules'])} 条")
            return config
        except Exception as e:
            logger.error(f"加载意图优化配置失败：{str(e)}，将使用默认空规则")
            return {"intent_correction_rules": []}
    def _build_context(self) -> str:
        """构建对话上下文（保留原有优化）"""
        history = st.session_state.get("messages", [])[-3:]
        history_text = "\n".join([f"{msg['role']}: {msg['content'][:50]}" for msg in history])

        table_info = ""
        if st.session_state.get("db_connector") and st.session_state.db_connector.connected:
            tables, _ = st.session_state.db_connector.get_tables_with_comments()
            if tables:
                table_info = ", ".join([f"{table}" for table in tables.keys()])

        data_info = ""
        if st.session_state.get("df") is not None:
            data_info = f"当前数据：{st.session_state.df.shape[0]}行 x {st.session_state.df.shape[1]}列"

        return f"""
历史对话（最近3条）：{history_text or '无'}
数据库表：{table_info or '无'}
{data_info}
"""

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """从响应文本中提取JSON内容（保留原有增强逻辑）"""
        logger.info(f"原始response_text = {response_text}")
        if not response_text or not isinstance(response_text, str):
            logger.warning("Response text 为空或非字符串")
            return {}

        # 提取intent字段兜底
        intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        if intent_match:
            fallback_intent = intent_match.group(1)
            logger.info(f"部分 JSON 修复：提取 intent = {fallback_intent}")

        try:
            # 直接解析
            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"直接 JSON 解析失败: {e}")

            # 从代码块提取
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # 提取花括号内容
            brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if brace_match:
                try:
                    return json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    pass

            # 部分修复返回
            if intent_match:
                return {
                    "intent": fallback_intent,
                    "confidence": 0.5,
                    "reasoning": "部分响应修复"
                }

            logger.warning(f"无法从响应中提取JSON: {response_text[:200]}...")
            return {}

    def _keyword_fallback(self, query: str) -> Tuple[str, float]:
        """关键词强规则兜底（保留原有配置）"""
        q = query.lower()
        rules = [
            (["报告", "word", "pdf", "文档", "导出报告", "生成报告"], "报告生成", 0.85),
            (["检索", "获取", "提取", "查询", "查看", "搜索", "查找", "显示", "列出"], "数据检索", 0.90),
            (
            ["分析", "统计", "计算", "趋势", "占比", "分布", "图表", "柱状图", "饼图", "可视化"], "数据统计分析", 0.85),
            (["你好", "hello", "天气", "聊天", "功能", "帮助", "怎么用"], "闲聊", 0.80),
        ]

        for keywords, intent, confidence in rules:
            if any(k in q for k in keywords):
                logger.info(f"关键词兜底命中 → {intent} (置信度: {confidence})")
                return intent, confidence

        logger.info("关键词兜底未命中 → 默认数据统计分析")
        return "数据统计分析", 0.6

    def _enhance_intent_with_keywords(self, intent: str, confidence: float, query: str) -> Tuple[str, float]:
        """基于关键词增强意图判断（保留原有逻辑）"""
        query_lower = query.lower()

        # 正向关键词增强
        for intent_name, definition in self.intent_definitions.items():
            for keyword in definition.get("positive_keywords", []):
                if keyword.lower() in query_lower:
                    if intent_name == intent:
                        confidence = min(confidence + 0.2, 0.95)
                    else:
                        logger.info(f"关键词'{keyword}'指向意图'{intent_name}'，但当前识别为'{intent}'")

        # 负向关键词降低置信度
        for intent_name, definition in self.intent_definitions.items():
            if intent_name == intent:
                for negative_keyword in definition.get("negative_keywords", []):
                    if negative_keyword.lower() in query_lower:
                        confidence = max(confidence - 0.3, 0.1)
                        logger.info(f"负向关键词'{negative_keyword}'降低意图'{intent}'的置信度")

        # 特定场景修正
        if intent == "报告生成" and any(
                keyword in query_lower for keyword in ["图表", "饼状图", "柱状图", "趋势图", "可视化"]):
            intent = "数据统计分析"
            confidence = 0.8
            logger.info(f"检测到图表关键词，将意图从'报告生成'修正为'数据统计分析'")

        if intent == "数据检索" and any(
                keyword in query_lower for keyword in ["分析", "统计", "计算", "趋势", "占比"]):
            intent = "数据统计分析"
            confidence = 0.8
            logger.info(f"检测到统计分析关键词，将意图从'数据检索'修正为'数据统计分析'")

        return intent, confidence

    def _build_intent_descriptions(self) -> str:
        """构建意图描述（保留原有逻辑）"""
        intent_descriptions = ""
        for intent, details in self.intent_definitions.items():
            intent_descriptions += f"""
{intent}:
- 描述: {details['description']}
- 典型关键词: {', '.join(details.get('positive_keywords', []))}
- 排除关键词: {', '.join(details.get('negative_keywords', []))}
- 示例: {', '.join(details['examples'])}
"""
        return intent_descriptions

    def detect_intent(self, query: str) -> Tuple[str, float, Dict[str, Any]]:
        """混合意图识别：改用新LLM调用方式 + 保留原有核心逻辑"""
        extra_info = {}
        logger.info(2222222)

        try:
            # ================= 新增：优先匹配用户反馈优化规则 =================
            logger.info("开始匹配意图优化规则...")
            for rule in self.optimize_config["intent_correction_rules"]:
                # 检查规则必填字段
                if not all(key in rule for key in ["keywords", "wrong_intent", "correct_intent"]):
                    logger.warning(f"规则格式不完整，跳过：{rule}")
                    continue
                # 检查用户查询是否包含规则中的关键词
                if any(keyword.lower() in query.lower() for keyword in rule["keywords"]):
                    # 命中规则，直接返回正确意图
                    final_intent = rule["correct_intent"]
                    final_confidence = 1.0  # 规则匹配置信度设为最高
                    final_reasoning = f"命中用户反馈优化规则：查询包含关键词[{', '.join(rule['keywords'])}]，修正意图从[{rule['wrong_intent']}]为[{final_intent}]"
                    logger.info(
                        f"✅ 规则匹配成功 → {final_intent} (置信度: {final_confidence:.2f})，理由: {final_reasoning}")
                    # 组装额外信息
                    extra_info["reasoning"] = final_reasoning
                    extra_info["llm_raw"] = {"intent": rule["wrong_intent"], "confidence": 0.0}  # 标记为规则修正
                    # 模板识别（如果是报告生成意图）
                    if final_intent == "报告生成":
                        template_intent, template_confidence = self.detect_template_intent(query)
                        extra_info["template_intent"] = template_intent
                        extra_info["template_confidence"] = template_confidence
                    return final_intent, final_confidence, extra_info
            logger.info(11111111)
            # 构建上下文和意图描述
            context = self._build_context()
            logger.info(4444444)
            intent_descriptions = self._build_intent_descriptions()
            logger.info(55555555)

            # 构建System Prompt（保留原有规则）
            system_prompt = f"""
你是一个专业的数据分析意图识别专家。请严格根据以下定义判断用户意图。

=== 可用意图 ===
{intent_descriptions}

=== 强制规则（优先级最高） ===
1. 包含【检索、获取、提取】关键字 → **必须**是「数据检索」
2. 包含【报告、word、pdf、文档】 → **必须**是「报告生成」  
3. 包含【分析、统计、图表】 → **必须**是「数据统计分析」
4. 包含【你好、天气、聊天】 → **必须**是「闲聊」

=== 重要判断规则 ===
1. 【报告生成】必须明确要求生成"报告"、"文档"、"Word"、"PDF"等格式化输出
2. 【数据统计分析】涉及数据分析、计算、统计、图表生成、趋势分析等
3. 【数据检索】只是简单的查询、查看、搜索数据，不涉及复杂分析
4. 【闲聊】与技术、数据无关的对话

=== 特别注意 ===
- 「检索2025年安徽的网络画像数据」→ **数据检索**（示例）
- 任何包含时间、地点的具体数据查询 → **数据检索**
- 不要因为查询具体而误判为闲聊！

上下文信息（简要）：{context}

请严格按照以下JSON格式返回，不要包含任何其他文本：
{{
    "intent": "意图名称",
    "confidence": 置信度(0.0-1.0),
    "reasoning": "简要理由"
}}
"""

            # 调用LLM（改用新的query_llm函数）
            logger.info("开始调用LLM进行意图识别")
            result_text = query_llm(
                prompt=query,
                system_prompt=system_prompt,
                history=None  # 意图识别暂不依赖历史对话，如需可从st.session_state获取
            )
            logger.info(33333333)  # 确认LLM调用成功

            # 解析LLM响应
            logger.info(f"意图识别原始响应: {result_text}")
            result_json = self._extract_json_from_response(result_text)
            llm_intent = result_json.get("intent", "数据统计分析")
            llm_confidence = float(result_json.get("confidence", 0.5))
            reasoning = result_json.get("reasoning", "")

            # 记录LLM原始判断
            logger.info(f"LLM识别: {llm_intent} (置信度: {llm_confidence:.2f}), 理由: {reasoning}")

            # 置信度判断 + 兜底
            if llm_confidence >= 0.6 and llm_intent in self.intent_definitions:
                final_intent = llm_intent
                final_confidence = llm_confidence
                final_reasoning = reasoning
                logger.info(f"✅ 大模型识别成功 → {final_intent} (置信度: {final_confidence:.2f})")
            else:
                logger.info(f"⚠️ LLM置信度低 ({llm_confidence:.2f})，进入关键词兜底...")
                final_intent, final_confidence = self._keyword_fallback(query)
                final_reasoning = f"LLM置信度低 ({llm_confidence:.2f})，关键词兜底识别为：{final_intent}"
                logger.info(f"🔄 关键词兜底 → {final_intent} (置信度: {final_confidence:.2f})")

            # 关键词增强
            final_intent, final_confidence = self._enhance_intent_with_keywords(final_intent, final_confidence, query)

            # 低置信度二次验证
            if final_confidence < 0.6:
                verify_system_prompt = f"""
用户输入：{query}
当前识别意图：{final_intent} (置信度：{final_confidence:.2f})

请重新仔细分析，这个输入更接近哪种意图？
A. 报告生成 - 明确要求生成报告文档
B. 数据统计分析 - 要求数据分析、图表生成  
C. 数据检索 - 简单查询数据
D. 闲聊 - 与技术无关的对话

请只返回A、B、C或D中的一个字母：
"""
                try:
                    verify_result = query_llm(
                        prompt=query,
                        system_prompt=verify_system_prompt,
                        history=None
                    )
                    choice = verify_result.strip().upper()
                    intent_map = {"A": "报告生成", "B": "数据统计分析", "C": "数据检索", "D": "闲聊"}
                    if choice in intent_map:
                        new_intent = intent_map[choice]
                        if new_intent != final_intent:
                            final_intent = new_intent
                            final_confidence = 0.7
                            final_reasoning = f"二次验证修正 → {final_intent}"
                            logger.info(f"🔄 二次验证修正意图: {final_intent}")
                except Exception as e:
                    logger.error(f"二次验证失败: {e}")

            # 模板识别（报告生成意图时）
            if final_intent == "报告生成":
                template_intent, template_confidence = self.detect_template_intent(query)
                extra_info["template_intent"] = template_intent
                extra_info["template_confidence"] = template_confidence
                logger.info(f"模板类型识别: {template_intent} (置信度: {template_confidence})")

            # 组装额外信息
            extra_info["reasoning"] = final_reasoning
            extra_info["llm_raw"] = {"intent": llm_intent, "confidence": llm_confidence}

            return final_intent, final_confidence, extra_info

        except Exception as e:
            logger.error(f"意图识别失败：{str(e)}")
            logger.error(f"完整异常堆栈：{repr(e)}")
            # 异常兜底
            intent, conf = self._keyword_fallback(query)
            return intent, conf, {"reasoning": f"异常兜底: {str(e)}", "error": str(e)}

    def detect_template_intent(self, query: str) -> Tuple[str, float]:
        """检测模板选择子意图（改用新LLM调用方式）"""
        try:
            template_descriptions = "\n".join(
                [f"- {template}: {details['description']}\n  关键词: {', '.join(details['keywords'])}"
                 for template, details in self.template_intent_definitions.items()])

            system_prompt = f"""
你是一个模板选择助手，根据用户输入推断报告模板类型。

可用模板：
{template_descriptions}

用户输入：{query}

请返回JSON格式：
{{
    "template": "模板名称", 
    "confidence": 置信度(0-1)
}}
"""

            # 调用LLM
            result_text = query_llm(
                prompt=query,
                system_prompt=system_prompt,
                history=None
            )
            result_json = self._extract_json_from_response(result_text)

            template = result_json.get("template", "ipran")
            confidence = float(result_json.get("confidence", 0.5))

            return template, confidence

        except Exception as e:
            logger.error(f"模板意图识别失败：{str(e)}")
            return "ipran", 0.3