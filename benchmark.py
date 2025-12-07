#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, argparse, sqlite3
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------  SQLite schema utils  -----------------------------

def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]

def pragma_table_info(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    return conn.execute(f"PRAGMA table_info({table})").fetchall()

def pragma_foreign_keys(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    return conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()

def sample_rows(conn: sqlite3.Connection, table: str, k: int) -> List[Dict[str, Any]]:
    if k <= 0: return []
    cols = [r[1] for r in pragma_table_info(conn, table)]
    try:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {int(k)}").fetchall()
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []

# -----------------------------  Prompt builder  -----------------------------

def build_schema_context(conn: sqlite3.Connection,
                         max_tables: int = 200,
                         rows_per_table: int = 3,
                         include_samples: bool = True,
                         dialect_label: str = "SQLite") -> str:
    tables = list_tables(conn)[:max_tables]
    parts: List[str] = []
    parts.append(
        f"You are an expert SQL engineer. Target dialect: {dialect_label}. Return ONLY runnable SQL (no backticks, no prose).\n"
    )
    for t in tables:
        cols = pragma_table_info(conn, t)
        col_defs = []
        for c in cols:
            name, ctype = c[1], c[2] or "TEXT"
            if "INT" in ctype.upper(): ctype = "INTEGER"
            col_defs.append(f"{name} {ctype}")
        parts.append(f"-- Table: {t}\nCREATE TABLE {t} ({', '.join(col_defs)});")

        for (_id, _seq, ref_table, from_col, to_col, *_rest) in pragma_foreign_keys(conn, t):
            parts.append(f"-- FK: {t}.{from_col} -> {ref_table}.{to_col}")

        if include_samples:
            rows = sample_rows(conn, t, rows_per_table)
            if rows:
                parts.append(f"-- Sample rows from {t}:")
                for r in rows:
                    parts.append("-- " + json.dumps(r, ensure_ascii=False))
        parts.append("")
    parts.append(
        "Rules:\n"
        "1) Use only the provided tables/columns.\n"
        "2) Prefer explicit JOIN ... ON ...\n"
        "3) If aggregation is used, include GROUP BY as needed.\n"
        "4) Limit output to 5 rows unless asked otherwise.\n"
        "5) Do NOT use backticks or markdown. Return SQL only. Your output should be directly executable.\n"
    )
    return "\n".join(parts)

# -----------------------------  Spreadsheet handling  -----------------------------

def read_questions(path: str,
                   question_col: str,
                   id_col: Optional[str],
                   sheet: Optional[str],
                   row_start: Optional[int],
                   row_end: Optional[int]) -> List[Tuple[str, str]]:
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path)

    if question_col not in df.columns:
        raise ValueError(f"Column '{question_col}' not found in {path}. Columns: {list(df.columns)}")

    # Use your current indexing logic exactly as in your last snippet:
    if row_start is not None or row_end is not None:
        start_idx = (row_start - 2) if row_start is not None else 0  # your current 0-based tweak
        end_idx_excl = (row_end - 1) if row_end is not None else len(df)
        df = df.iloc[start_idx:end_idx_excl]

    ids = df[id_col].astype(str).tolist() if id_col and id_col in df.columns else [str(i) for i in df.index]
    qs = df[question_col].astype(str).fillna("").tolist()
    return [(i, q.strip()) for i, q in zip(ids, qs) if q.strip()]

# -----------------------------  HF InferenceProvider (chat)  -----------------------------

def call_hf_chat(model: str,
                 system: str,
                 user: str,
                 *,
                 max_tokens: int = 512,
                 temperature: float = 0.0,
                 retries: int = 3,
                 retry_delay: float = 2.0) -> str:
    """
    Calls Hugging Face Inference Provider via InferenceClient.chat.completions.create
    using your HF_TOKEN (env var).
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HF_API_TOKEN) is not set.")

    client = InferenceClient(model=model, token=token)
    last_err: Optional[Exception] = None

    for _ in range(retries):
        try:
            res = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = res.choices[0].message.content.strip()
            # Strip common code fences if present
            if text.startswith("```"):
                text = text.strip("`\n")
                if "\n" in text:
                    first, rest = text.split("\n", 1)
                    text = rest if len(first) < 16 else text
            return text
        except Exception as e:
            last_err = e
            time.sleep(retry_delay)
    return f"ERROR: {last_err}"

# -----------------------------  CLI  -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build NL2SQL prompts; optionally run via Hugging Face Inference Provider (chat)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--question_col", default="Natural Language Query")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--row_start", type=int, default=None)
    ap.add_argument("--row_end", type=int, default=None)
    ap.add_argument("--out", required=True, help="Base output filename (e.g. 11_20.jsonl)")
    ap.add_argument("--rows_per_table", type=int, default=3)
    ap.add_argument("--max_tables", type=int, default=200)
    ap.add_argument("--include_samples", action="store_true")
    ap.add_argument("--dialect", default="SQLite")
    # Run flags
    ap.add_argument("--run", action="store_true", help="Call HF for each question")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model id")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N questions (to save credits)")
    args = ap.parse_args()

    # 1) Build schema context
    conn = sqlite3.connect(args.db)
    schema_ctx = build_schema_context(conn, args.max_tables, args.rows_per_table, args.include_samples, args.dialect)
    conn.close()

    # 2) Read questions
    pairs = read_questions(args.questions, args.question_col, args.id_col, args.sheet, args.row_start, args.row_end)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    # 3) Write prompts_<out>.jsonl (unchanged)
    base_dir = os.path.dirname(args.out) or "."
    base_name = os.path.basename(args.out)
    prompts_path = os.path.join(base_dir, f"prompts_{base_name}")
    os.makedirs(base_dir, exist_ok=True)

    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump({"system": schema_ctx}, f, ensure_ascii=False); f.write("\n")
        for qid, qtext in pairs:
            json.dump({"id": qid, "user": f"Question: {qtext}"}, f, ensure_ascii=False); f.write("\n")
    print(f"Wrote {prompts_path}")

    # 4) Optionally run HF per question
    if args.run:
        completions_path = os.path.join(base_dir, f"completions_{base_name}")
        with open(completions_path, "w", encoding="utf-8") as fout:
            for qid, qtext in pairs:
                user_msg = f"Question: {qtext}\nWrite a single {args.dialect} query that answers it. Return SQL only."
                sql = call_hf_chat(
                    model=args.model,
                    system=schema_ctx,
                    user=user_msg,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                json.dump({"id": qid, "SQL": sql}, fout, ensure_ascii=False); fout.write("\n")
        print(f"Wrote {completions_path}")

if __name__ == "__main__":
    main()
