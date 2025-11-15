####text2sql

import pandas as pd
from sqlalchemy import create_engine, inspect
import requests
from env.DB_env import DB_CONFIG  # 复用数据库配置（与DatabaseConnector一致）
import re
import json
from collections import defaultdict
from sqlalchemy import text  # 保留sqlalchemy的text，用于SQL执行
import logging
from typing import List
from env.LLM_env import DEEPSEEK_URL, DEEPSEEK_API_KEY, qwen3_url
# 正确导入日期类型
from datetime import datetime, date

# 配置日志（与现有系统风格统一）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库连接（仅配置，无业务逻辑）
user = DB_CONFIG['user']
passward = DB_CONFIG['password']
host = DB_CONFIG['host']
port = DB_CONFIG['port']
database = DB_CONFIG['database']
schema = DB_CONFIG['schema']
engine = create_engine(f'postgresql://{user}:{passward}@{host}:{port}/{database}')


def get_full_db_structure(schema=DB_CONFIG['schema']):
    """获取“无遗漏”的数据库结构：仅客观呈现，不添加任何业务预设"""
    logger.info(f"=== 开始获取 {schema}  schema 表结构 ===")

    inspector = inspect(engine)
    structure = {
        "schema": schema,
        "tables": {},  # 表级信息：{表名: {columns: {列名: {type, comment}}, comment, foreign_keys}}
        "foreign_key_graph": defaultdict(list)  # 外键图谱：{源表: [{目标表, 源列, 目标列}]}
    }

    # 1. 获取schema下所有表，确保无遗漏
    all_tables = inspector.get_table_names(schema=schema)
    logger.info(f"获取到 {schema} 下共 {len(all_tables)} 张表：{all_tables}")
    with engine.connect() as conn:
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
    logger.info(f"=== 表结构获取完成：{schema} 下 {len(structure['tables'])} 张表，外键关系 {len(structure['foreign_key_graph'])} 条 ===")
    return structure


def query_llm(prompt, system_prompt, history=None):
    """调用LLM，不注入业务逻辑"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, data=json.dumps({
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2500
        }))
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"LLM调用失败: {str(e)}")
        resp = requests.post(qwen3_url, headers=headers, data=json.dumps({
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2500
        }))
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()


def validate_sql_table(sql, real_tables):
    """纯校验工具：检查SQL中的表是否存在于真实表列表"""
    extracted_tables = re.findall(r'(?:FROM|JOIN)\s+["`\']?(\w+)["`\']?\s+', sql, re.IGNORECASE)
    invalid = [t for t in extracted_tables if t.lower() not in [rt.lower() for rt in real_tables]]
    if invalid:
        return False, f"无效表名：{invalid}，真实表名：{real_tables}"
    return True, "表名校验通过"


def execute_sql(sql):
    """处理两种日期类型的 execute_sql 函数"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if result.returns_rows:
                rows = []
                for row in result.mappings().fetchall():
                    row_dict = dict(row)
                    # 同时处理 datetime（带时间）和 date（纯日期）
                    for key, value in row_dict.items():
                        if isinstance(value, (datetime, date)):
                            row_dict[key] = value.isoformat()
                    rows.append(row_dict)
                return {"status": "success", "data": rows}
            else:
                return {"status": "success", "msg": "执行成功（无返回数据）"}
    except Exception as e:
        return {"status": "error", "error": f"SQL执行失败: {str(e)}"}


def validate_user_input(user_query):
    has_chinese_name = re.search(r'[\u4e00-\u9fff]{2,4}', user_query) is not None
    has_employee_id = re.search(r'\d{6,}', user_query) is not None
    return has_chinese_name and has_employee_id


def dynamic_permission_agent(user_query, max_retries=3, schema=schema):
    """核心智能体：全流程动态推理+SQL自动修复功能，新增返回表名"""
    logger.info(f"\n=== 启动 dynamic_permission_agent ===")
    logger.info(f"用户查询：{user_query}")
    logger.info(f"最大重试次数：{max_retries}，目标schema：{schema}")
    # 步骤1：获取纯客观的数据库结构（无任何业务标注）
    db_struct = get_full_db_structure(schema=schema)
    real_tables = list(db_struct["tables"].keys())
    print(f"步骤1：获取到{schema}下{len(real_tables)}张表的结构（含外键、注释）")

    # 步骤2：引导LLM完成“需求→表→关系→SQL”的全流程推理，新增关键规则
    system_prompt = f"""
        你是一个“数据库自主推理专家”，需严格按以下步骤处理用户查询，**不依赖任何预设业务逻辑**，并能根据错误信息修复SQL：
        【数据库结构（纯客观信息）】
        {json.dumps(db_struct, ensure_ascii=False, indent=2)}

        【推理与修复步骤（必须按顺序执行）】
        1. 需求解析：分析用户查询的核心诉求（基于表注释判断关键词）；
        2. 表筛选：结合需求和表注释筛选相关表；
        3. 关系推理：基于外键图谱梳理表关联；
        4. SQL设计：生成符合要求的SQL（带schema、表别名、条件筛选），且**必须满足：**
           - 字段名**禁止使用中文别名**，不允许用`AS`给字段重命名（如`SELECT province AS "省份"`是错误的）；
           - 保留数据库原始字段名（英文），直接使用`SELECT 字段名`；
           - 表可以使用英文别名（如`FROM schema.table t`），但字段必须用原始名；
           - 模糊匹配强制规则：使用LIKE时**前后必须加%**（如`LIKE '%2025%'`，禁止`LIKE '2025%'`或`LIKE '%2025'`）；
           - 日期字段处理：若日期包含在字符串字段中（如“安徽安庆移动无线4G网络_2025/6/1”），需用SPLIT_PART拆分后匹配。
        5. 错误修复（关键新增步骤）：
           - 若收到执行错误（如“表不存在”“权限不足”“语法错误”“查询无结果”），必须先分析错误原因；
           - 针对性修复：
             - 表不存在：移除该表或替换为存在的表（参考real_tables列表）；
             - 权限不足：排除无权限的表，仅保留可访问的表；
             - 语法错误：修正SQL语法（如字段名错误、JOIN条件错误）；
             - 字段不存在：检查列名是否与表结构匹配，替换为正确字段名；
             - 查询无结果：调整筛选条件（如补全LIKE的%、扩大日期范围、模糊匹配省份）。

        【输出格式（必须包含推理过程和最终SQL）】
        {{
            "reasoning": "分5步说明：1.需求解析；2.表筛选；3.关系推理；4.SQL设计；5.（若有错误）错误原因及修复思路",
            "sql": "修复后的SQL语句（带schema）",
            "permission_types_defined": "定义的权限类型及对应表"
        }}
        【特别说明】
        若执行“前置要求”中的输入校验不通过（缺少名字或工号），无需遵循上述JSON输出格式，直接返回提示语即可。
        """

    # 步骤3：多轮推理+错误驱动修复+执行（含无数据重试）
    history = []
    final_result = None
    final_sql = None
    final_reasoning = None
    current_perm_types = None
    final_tables = []

    for attempt in range(max_retries):
        print(f"\n步骤3：第{attempt + 1}轮推理...")
        # 调用LLM进行推理（若有历史错误，会自动纳入上下文）
        llm_output = query_llm(
            prompt=user_query,
            system_prompt=system_prompt,
            history=history
        )

        # 解析LLM输出（处理JSON格式）
        try:
            json_match = re.search(r'\{[\s\S]*\}', llm_output)
            if not json_match:
                raise ValueError("LLM未输出JSON格式")
            llm_json = json.loads(json_match.group())
            current_sql = llm_json.get("sql", "")
            current_reasoning = llm_json.get("reasoning", "")
            current_perm_types = llm_json.get("permission_types_defined", "")
        except Exception as e:
            error_msg = f"LLM输出解析失败：{str(e)}，原始输出：{llm_output[:200]}"
            history.append({"role": "assistant", "content": llm_output})
            history.append({"role": "user", "content": f"请修正输出格式，必须返回完整JSON。错误：{error_msg}"})
            print(error_msg)
            continue

        # 校验SQL表名
        table_valid, table_msg = validate_sql_table(current_sql, real_tables)
        if not table_valid:
            error_msg = f"SQL表名校验失败：{table_msg}"
            history.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
            history.append({"role": "user", "content": f"请根据真实表名修复SQL。错误：{error_msg}"})
            print(error_msg)
            continue

        # 执行SQL并检查结果（核心修改：无数据视为错误）
        exec_result = execute_sql(current_sql)
        if isinstance(exec_result, dict):
            # 情况1：SQL执行错误（如语法错、表不存在）
            if "error" in exec_result:
                error_details = exec_result["error"]
                history.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
                history.append({"role": "user",
                                "content": f"SQL执行失败，请根据错误修复：{error_details}。修复时需：1.移除无效表/字段；2.修正语法；3.确保仅使用有权限的表。"})
                print(f"SQL执行失败：{error_details}，准备重试修复...")
                continue

            # 情况2：SQL执行成功但无数据（新增：视为错误触发重试）
            data_count = len(exec_result.get("data", []))
            if data_count == 0:
                error_details = f"SQL执行成功但返回数据量为0（当前筛选条件无匹配数据）"
                history.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
                history.append({"role": "user",
                                "content": f"{error_details}，请修复筛选条件：1.确保LIKE前后都加%；2.扩大日期匹配范围；3.省份用模糊匹配（如%安徽%）；4.检查字段名是否正确。"})
                print(f"{error_details}，准备重试修复...")
                continue

        # 推理+执行成功（有数据），保存结果
        final_sql = current_sql
        final_reasoning = current_reasoning
        final_result = exec_result
        final_tables = extract_tables_from_sql(final_sql, real_tables)
        print(f"步骤3：第{attempt + 1}轮推理+执行成功！涉及表：{final_tables}，返回数据量：{len(final_result.get('data', []))}")
        break

    # 步骤4：返回完整结果（所有重试失败则返回失败信息）
    if not final_result or len(final_result.get("data", [])) == 0:
        return {
            "user_query": user_query,
            "database_structure_summary": f"{schema}下{len(real_tables)}张表，外键图谱{len(db_struct['foreign_key_graph'])}条关系",
            "reasoning_process": final_reasoning or "多次重试后仍未获取到有效数据",
            "generated_sql": final_sql or "未生成有效SQL",
            "permission_types_defined": current_perm_types or "无",
            "result": {"status": "error", "error": "多次重试后仍未查询到匹配数据"},
            "tables": final_tables
        }

    return {
        "user_query": user_query,
        "database_structure_summary": f"{schema}下{len(real_tables)}张表，外键图谱{len(db_struct['foreign_key_graph'])}条关系",
        "reasoning_process": final_reasoning,
        "generated_sql": final_sql,
        "permission_types_defined": current_perm_types,
        "result": final_result,
        "tables": final_tables
    }


def extract_tables_from_sql(sql: str, real_tables: List[str]) -> List[str]:
    """
    从SQL中提取真实存在的表名（去重）
    :param sql: 生成的SQL语句
    :param real_tables: 数据库中真实存在的表名列表
    :return: 提取到的表名列表
    """
    if not sql:
        return []
    # 正则匹配SQL中的表名（支持带schema的格式，如"schema.table"）
    table_pattern = re.compile(r'FROM\s+([a-zA-Z0-9_.]+)', re.IGNORECASE)
    join_pattern = re.compile(r'JOIN\s+([a-zA-Z0-9_.]+)', re.IGNORECASE)

    # 提取所有匹配的表名
    from_tables = table_pattern.findall(sql)
    join_tables = join_pattern.findall(sql)
    all_matches = from_tables + join_tables

    # 处理表名（去掉schema前缀，只保留表名本身）
    tables = []
    for full_table in all_matches:
        # 分割schema和表名（如"public.portrait" → "portrait"）
        table_name = full_table.split('.')[-1].strip()
        # 只保留真实存在的表名
        if table_name in real_tables and table_name not in tables:
            tables.append(table_name)
    return tables


if __name__ == "__main__":
    user_query = "帮我查询2025年安徽的网络画像数据"
    print(f"=== 启动完全动态权限智能体 ===")
    print(f"用户需求：{user_query}\n")

    agent_result = dynamic_permission_agent(user_query, schema=schema)

    # 格式化输出结果
    print(f"\n=== 智能体最终结果 ===")
    print(f"1. 需求：{agent_result['user_query']}")
    print(f"2. 数据库结构概览：{agent_result['database_structure_summary']}")
    print(f"3. 全流程推理过程（含修复）：\n{agent_result['reasoning_process']}")
    print(f"4. 生成的SQL：\n{agent_result['generated_sql']}")
    print(f"5. 定义的权限类型：\n{agent_result['permission_types_defined']}")

    # 关键：处理查询结果，转为DataFrame
    print(f"\n6. 查询结果（DataFrame格式）：")
    result = agent_result["result"]
    if isinstance(result, dict) and result.get("status") == "success":
        query_data = result.get("data", [])
        result_df = pd.DataFrame(query_data)
        print(f"   - 数据总行数：{len(result_df)}")
        print(f"   - 字段列表：{list(result_df.columns)}")
        print(f"   - 前5行数据预览：")
        print(result_df.head())
    else:
        error_msg = result.get("error") if isinstance(result, dict) else result
        print(f"   - 查询失败：{error_msg}")