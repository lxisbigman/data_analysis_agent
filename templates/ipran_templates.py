#save_memory.py
import streamlit as st
import os
import json
import uuid
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate


# 配置常量
MEMORY_DIR = "../chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)
# ================= 记忆功能 =================
def init_long_term_memory():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    if 'memory' not in st.session_state:
        st.session_state.memory = load_memory(st.session_state.session_id)


def load_memory(session_id):
    file_path = os.path.join(MEMORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"messages": [], "data_meta": {}}


def save_memory():
    memory = {
        "messages": st.session_state.messages,
        "data_meta": {
            "columns": list(st.session_state.df.columns) if st.session_state.df is not None else [],
            "shape": st.session_state.df.shape if st.session_state.df is not None else ()
        }
    }
    file_path = os.path.join(MEMORY_DIR, f"{st.session_state.session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def get_history_sessions():
    return sorted([f[:-5] for f in os.listdir(MEMORY_DIR) if f.endswith(".json")], reverse=True)


def build_prompt_with_memory(query):
    history_messages = st.session_state.messages[-5:]
    history_text = "\n".join([f"{msg['role']}: {msg['content'][:100]}" for msg in history_messages if msg['role'] != 'system'])
    system_prompt = f"""
你是一个数据分析助手，需严格遵循以下规则：
1. **必须使用提供的df数据**（从数据库加载，已包含在环境中），**禁止自行构造任何假数据**（如 pd.DataFrame(...) 或其他方式创建数据）。
2. 生成图片时，必须按顺序命名为 plot_0.png, plot_1.png, ..., plot_N.png（N为图片总数-1）。
3. 绘图完成后，必须打印图片总数，格式固定为："图片总数: X"（X为数字，如"图片总数: 3"）。
4. 代码中需包含 plt.close() 关闭每个画布，避免叠加。
5. 仅使用 df、pd、plt，不使用其他库。
6. 如果用户查询涉及“CD级分析”，默认使用 cd_level_net_analysis 表，优先选择数值字段（如“A占比”）绘制柱状图或折线图。

数据信息（必须使用这些真实数据）：
- 表名：cd_level_net_analysis
- 列名：{list(st.session_state.df.columns) if st.session_state.df is not None else []}
- 形状：{st.session_state.df.shape if st.session_state.df is not None else ()}
- 示例字段说明（仅适用于 cd_level_net_analysis 表）：
  - A占比：数值型，表示某个指标的百分比（如 33.33）。
  - 类别：字符串型，表示分类标签（如 'CD级'）。

示例代码（生成柱状图）：
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(df['类别'], df['A占比'])
plt.xlabel('类别')
plt.ylabel('A占比 (%)')
plt.title('CD级分析柱状图')
plt.savefig('plot_0.png')
plt.close()
print("图片总数: 1")
saved_images = ['plot_0.png']
expected_count = 1
用户问题：{query}
历史对话：{history_text}
返回仅包含 Python 代码的响应，格式为：
 # 代码内容
 """
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", query)])