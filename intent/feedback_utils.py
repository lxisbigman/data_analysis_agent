# feedback_utils.py
import json
import os
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | feedback_utils | %(message)s'
)
logger = logging.getLogger(__name__)

# 绝对路径存储
FEEDBACK_DIR = os.path.abspath("intent_feedback")
OPTIMIZE_CONFIG = "intent_optimize_config.json"  # 输出文件
os.makedirs(FEEDBACK_DIR, exist_ok=True)
logger.info(f"反馈存储目录：{FEEDBACK_DIR}")


def generate_optimize_rules():
    """自动扫描所有反馈文件，生成优化规则"""
    logger.info("开始生成意图优化规则...")
    error_map = defaultdict(list)  # wrong_intent -> [queries]

    # 1. 扫描所有反馈文件
    feedback_files = [
        f for f in os.listdir(FEEDBACK_DIR)
        if f.startswith("intent_feedback_") and f.endswith(".json")
    ]

    if not feedback_files:
        logger.info("暂无反馈文件，跳过规则生成")
        return

    # 2. 读取所有反馈
    for filename in feedback_files:
        filepath = os.path.join(FEEDBACK_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                if not item.get("is_correct", True):
                    wrong = item["recognized_intent"]  # 注意字段名！
                    query = item["query"]
                    error_map[wrong].append(query)
        except Exception as e:
            logger.error(f"读取 {filename} 失败: {e}")

    # 3. 生成规则
    rules = []
    for wrong_intent, queries in error_map.items():
        word_freq = defaultdict(int)
        for q in queries:
            for word in q.split():
                if len(word) > 1:
                    word_freq[word] += 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        keywords = [w[0] for w in top_words]

        # 智能推断 correct_intent（可扩展）
        correct_intent = _infer_correct_intent(keywords, queries)
        if keywords:
            rules.append({
                "keywords": keywords,
                "wrong_intent": wrong_intent,
                "correct_intent": correct_intent,
                "source": "auto_generated_from_feedback",
                "query_examples": queries[:2]
            })

    # 4. 写入配置文件
    config = {"intent_correction_rules": rules}
    with open(OPTIMIZE_CONFIG, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f"生成 {len(rules)} 条优化规则 → {OPTIMIZE_CONFIG}")


def _infer_correct_intent(keywords, queries):
    """简单推断正确意图（可扩展为LLM）"""
    q = " ".join(queries).lower()
    if any(k in q for k in ["报告", "word", "pdf"]):
        return "报告生成"
    if any(k in q for k in ["分析", "统计", "图表"]):
        return "数据统计分析"
    if any(k in q for k in ["查询", "检索", "查看"]):
        return "数据检索"
    return "数据统计分析"  # 默认


def save_intent_feedback(session_id: str, query: str, intent: str, is_correct: bool):
    logger.info(f"进入保存反馈函数：session_id={session_id}")
    try:
        # 1. 校验输入
        if not all([session_id, query, intent]):
            logger.error("session_id/query/intent 为空，保存失败")
            return

        # 2. 构造反馈数据
        feedback = {
            "session_id": session_id,
            "query": query,
            "recognized_intent": intent,  # 与 generate_rules 字段一致
            "is_correct": is_correct,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 3. 构造文件路径
        date_str = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(FEEDBACK_DIR, f"intent_feedback_{date_str}.json")

        # 4. 读取 + 追加
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

        existing_data.append(feedback)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        logger.info(f"反馈保存成功 → {file_path}")

        # 关键：保存后立即生成规则
        generate_optimize_rules()

    except Exception as e:
        logger.error(f"保存反馈异常：{str(e)}", exc_info=True)