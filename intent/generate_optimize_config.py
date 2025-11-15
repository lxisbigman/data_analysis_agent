# generate_optimize_config.py
import json
from collections import defaultdict

def generate_rules():
    feedback_file = "intent_feedback/intent_feedback_20251103.json"  # 你的文件
    output_file = "intent_optimize_config.json"

    # 1. 读取反馈
    with open(feedback_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 统计错误意图
    error_map = defaultdict(list)  # wrong_intent -> [queries]
    for item in data:
        if not item.get("is_correct", True):
            wrong = item["intent_pred"]
            query = item["user_query"]
            error_map[wrong].append(query)

    # 3. 生成规则（简单关键词提取）
    rules = []
    for wrong_intent, queries in error_map.items():
        # 提取高频词作为关键词
        word_freq = defaultdict(int)
        for q in queries:
            for word in q.split():
                if len(word) > 1:
                    word_freq[word] += 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        keywords = [w[0] for w in top_words]

        # 假设用户反馈时会选“正确意图”，这里用最常见的“数据统计分析”做示例
        correct_intent = "数据统计分析"  # 可扩展：让用户选正确意图
        if keywords:
            rules.append({
                "keywords": keywords,
                "wrong_intent": wrong_intent,
                "correct_intent": correct_intent,
                "source": "auto_generated_from_feedback"
            })

    # 4. 写入配置文件
    config = {"intent_correction_rules": rules}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"生成 {len(rules)} 条优化规则 → {output_file}")

if __name__ == "__main__":
    generate_rules()