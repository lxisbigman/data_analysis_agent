####DB_connect.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from typing import List, Optional, Any, Mapping, Dict, Tuple
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from env.DB_env import DB_CONFIG
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置 Matplotlib 中文字体
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 配置常量
MEMORY_DIR = "chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

TABLE_COMMENT_MAPPING = {
    "办事处数据": "office_data",
    "优秀网络": "good_data",
    "CD级分析": "cd_level_net_analysis",
    "硬件整改": "hard_change",
    "维度分析": "dimension_change",
    "网元质量画像结果": "net_quality",
    "老旧设备退网": "old_equipment_backnet"
}


# ================= 数据库连接模块 =================
class DatabaseConnector:
    """数据库连接与数据获取工具"""

    def __init__(self):
        self.engine = None
        self.inspector = None
        self.connected = False

    def connect(self):
        try:
            connection_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            self.engine = create_engine(connection_str)
            self.inspector = inspect(self.engine)
            with self.engine.connect():
                self.connected = True
                return True, "数据库连接成功"
        except SQLAlchemyError as e:
            return False, f"数据库连接失败: {str(e)}"

    def get_tables_with_comments(self):
        if not self.connected:
            return None, "未建立数据库连接"
        try:
            tables = self.inspector.get_table_names(schema=DB_CONFIG["schema"])
            table_info = {}
            for table in tables:
                comment_query = text(f"""
                    SELECT obj_description('{DB_CONFIG['schema']}.{table}'::regclass) AS table_comment;
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(comment_query)
                    comment = result.scalar() or ""
                table_info[table] = comment.strip()
            return table_info, "获取表信息成功"
        except SQLAlchemyError as e:
            return None, f"获取表信息失败: {str(e)}"

    def get_table_by_comment(self, target_comment):
        tables, msg = self.get_tables_with_comments()
        if not tables:
            return None, msg
        for table, comment in tables.items():
            if comment == target_comment:
                return table, f"找到表: {table}"
        for table, comment in tables.items():
            if target_comment in comment:
                return table, f"找到相似表: {table} (注释: {comment})"
        return None, f"未找到注释为'{target_comment}'的表"

    def load_table_data(self, table_name):
        if not self.connected:
            return None, "未建立数据库连接"
        try:
            query = text(f"SELECT * FROM {DB_CONFIG['schema']}.{table_name}")
            df = pd.read_sql(query, self.engine)
            return df, f"成功加载表 {table_name}，共 {len(df)} 行数据"
        except SQLAlchemyError as e:
            return None, f"加载表 {table_name} 失败: {str(e)}"

    def get_tables_by_comments(self, target_comments: List[str]) -> Dict[str, pd.DataFrame]:
        loaded_data = {}
        table_info, msg = self.get_tables_with_comments()
        if not table_info:
            raise ValueError(f"无法获取表信息: {msg}")
        for comment in target_comments:
            matched_table = None
            for table, desc in table_info.items():
                if desc.strip() == comment:
                    matched_table = table
                    break
            if not matched_table:
                raise ValueError(f"未找到注释为「{comment}」的表")
            df, load_msg = self.load_table_data(matched_table)
            if df is None:
                raise ValueError(f"加载表失败: {load_msg}")
            business_key = TABLE_COMMENT_MAPPING.get(comment)
            if business_key:
                loaded_data[business_key] = df
                st.info(f"已加载: {comment}（表名: {matched_table}）- {len(df)}行数据")
            else:
                st.warning(f"注释「{comment}」没有对应的业务键映射，已跳过")
        return loaded_data

    def get_full_db_structure(self,schema=DB_CONFIG['schema']):
        """获取“无遗漏”的数据库结构：仅客观呈现，不添加任何业务预设"""

        inspector = inspect(self.engine)
        structure = {
            "schema": schema,
            "tables": {},  # 表级信息：{表名: {columns: {列名: {type, comment}}, comment, foreign_keys}}
            "foreign_key_graph": defaultdict(list)  # 外键图谱：{源表: [{目标表, 源列, 目标列}]}
        }

        # 1. 获取schema下所有表，确保无遗漏
        all_tables = inspector.get_table_names(schema=schema)
        with self.engine.connect() as conn:
            for table in all_tables:
                # 1.1 表注释（客观获取，不加工）
                table_comment = conn.execute(
                    text(f"SELECT obj_description('{schema}.{table}'::regclass) AS c;")
                ).scalar() or ""

                # 1.2 列信息（含列注释，客观获取）
                columns = {}
                col_info_list = conn.execute(text(f"""
                    SELECT column_name, data_type, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = '{schema}' AND table_name = '{table}'
                    ORDER BY ordinal_position;
                """)).mappings().fetchall()
                for col in col_info_list:
                    col_name = col["column_name"]
                    # 列注释
                    col_comment = conn.execute(
                        text(f"SELECT col_description('{schema}.{table}'::regclass, {col['ordinal_position']}) AS c;")
                    ).scalar() or ""
                    columns[col_name] = {
                        "data_type": col["data_type"],
                        "comment": col_comment,
                        "ordinal_position": col["ordinal_position"]
                    }

                # 1.3 外键关系（构建图谱，不预设关联意义）
                foreign_keys = inspector.get_foreign_keys(table, schema=schema)
                table_fks = []
                for fk in foreign_keys:
                    for src_col, tgt_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                        # 记录外键关系到图谱
                        structure["foreign_key_graph"][table].append({
                            "target_table": fk["referred_table"],
                            "source_column": src_col,
                            "target_column": tgt_col
                        })
                        # 记录当前表的外键详情
                        table_fks.append({
                            "constraint_name": fk["name"],
                            "source_column": src_col,
                            "target_table": fk["referred_table"],
                            "target_column": tgt_col
                        })

                # 1.4 组装当前表信息（纯客观数据，无业务标注）
                structure["tables"][table] = {
                    "comment": table_comment,
                    "columns": columns,
                    "foreign_keys": table_fks
                }

        return structure