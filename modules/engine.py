import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.1-8b-instant"


def _call_llm(system_prompt: str, user_message: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)



def generate_query(
    natural_language: str,
    db_type: str,
    tables: str = "",
    columns: str = "",
    extra_context: str = "",
    api_key: str = "",
) -> dict:

    system_prompt = """You are a senior database engineer.
Your job is to convert natural language requests into optimized database queries.

Rules:
- Always return valid JSON only. No extra text, no markdown.
- Choose the best query pattern: Window Function, CTE, Recursive CTE, Aggregation, Simple SELECT, or NoSQL Pipeline.
- Use correct syntax for the target database.
- Never use SELECT *.

Return exactly this JSON structure:
{
    "intent": {
        "type": "WINDOW or CTE or AGGREGATE or SELECT or RECURSIVE or NOSQL",
        "confidence": 90,
        "description": "one line explanation of what the query does"
    },
    "query_type": {
        "selected": "Window Function",
        "reason": "why this pattern was chosen",
        "alternatives": ["Simple SELECT", "Subquery"]
    },
    "query": {
        "sql": "the full executable query here",
        "language": "SQL"
    },
    "validation": {
        "syntax_ok": true,
        "explain_hint": "EXPLAIN ANALYZE ...",
        "index_suggestions": ["CREATE INDEX ..."],
        "optimizations": ["tip 1", "tip 2"],
        "warnings": []
    }
}"""

    user_message = f"""Database: {db_type}
Tables: {tables or "auto-detect from context"}
Columns: {columns or "auto-detect from context"}
Extra context: {extra_context or "none"}
Request: {natural_language}"""

    return _call_llm(system_prompt, user_message)


def suggest_schema(
    description: str,
    db_type: str,
    num_tables_hint: int = 0,
    api_key: str = "",
) -> list:

    system_prompt = """You are a database architect.
Your job is to design normalized database schemas from plain English descriptions.

Rules:
- Always return valid JSON only. No extra text, no markdown.
- Design tables in 3NF normalization.
- Use correct data types for the target database.
- Always include primary keys and foreign keys.
- Always suggest indexes.

Return exactly this JSON structure:
[
    {
        "name": "table_name",
        "purpose": "what this table stores",
        "columns": [
            {
                "name": "column_name",
                "type": "data type",
                "constraints": "PRIMARY KEY or NOT NULL or REFERENCES table(col)",
                "description": "what this column stores"
            }
        ],
        "indexes": ["CREATE INDEX ..."],
        "sample_insert": "INSERT INTO table_name (...) VALUES (...);"
    },
    {
        "__meta__": true,
        "relationships": [
            {
                "from": "table.column",
                "to": "table.column",
                "type": "ONE_TO_MANY"
            }
        ],
        "normalization_level": "3NF",
        "notes": "any design decisions"
    }
]"""

    user_message = f"""Database: {db_type}
Approximate number of tables: {num_tables_hint or "auto-determine"}
System description: {description}"""

    return _call_llm(system_prompt, user_message)


def convert_query(
    query: str,
    from_db: str,
    to_db: str,
    api_key: str = "",
) -> dict:

    system_prompt = """You are a senior database engineer who specializes in query translation.
Your job is to convert database queries from one database type to another.

Rules:
- Always return valid JSON only. No extra text, no markdown.
- Preserve the exact logic and intent of the original query.
- Use correct syntax for the target database.
- Handle differences in functions, data types, and keywords.
- If converting to MongoDB, use aggregation pipeline format.
- Note any features that don't have a direct equivalent.

Key syntax differences to handle:
- LIMIT (MySQL/PostgreSQL) → TOP (SQL Server) → FETCH FIRST N ROWS ONLY (Oracle) → $limit (MongoDB)
- AUTO_INCREMENT (MySQL) → SERIAL (PostgreSQL) → IDENTITY (SQL Server) → SEQUENCE (Oracle)
- ILIKE (PostgreSQL) → LIKE with COLLATE (MySQL) → LIKE (SQL Server/Oracle)
- NOW() (MySQL/PostgreSQL) → GETDATE() (SQL Server) → SYSDATE (Oracle) → $$NOW (MongoDB)
- GROUP_CONCAT (MySQL) → STRING_AGG (PostgreSQL/SQL Server) → LISTAGG (Oracle)

Return exactly this JSON structure:
{
    "original": {
        "query": "the original query as-is",
        "db_type": "source database name"
    },
    "converted": {
        "query": "the fully converted query",
        "db_type": "target database name",
        "language": "SQL or MongoDB"
    },
    "changes": [
        "change 1 — what was changed and why",
        "change 2 — what was changed and why"
    ],
    "warnings": [
        "any features that behave differently in target DB",
        "any manual adjustments needed"
    ],
    "equivalent_functions": {
        "original_function": "target_function"
    }
}"""

    user_message = f"""Convert this query from {from_db} to {to_db}:

{query}"""

    return _call_llm(system_prompt, user_message)

