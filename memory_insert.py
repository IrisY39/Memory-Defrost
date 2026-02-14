import os
import argparse
import requests
import json
import psycopg2
#批量写入：python Memory-Defrost-main/memory_insert.py --json Memory-Defrost-main/memories.json

GEMINI_API_KEY = "AIzaSyDbYueX2fzh0Wz5PDbal60YF1Izv94m09M"
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
DATABASE_URL = "postgresql://postgres:VGWCbznVRMjFYUkLUaLLbKiQFkULTekc@yamabiko.proxy.rlwy.net:59693/railway"


def get_embedding(text: str) -> list[float]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required")

    url = f"{GEMINI_EMBEDDING_URL}?key={GEMINI_API_KEY}"
    payload = {"content": {"parts": [{"text": text}]}}
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"embedding api error: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    emb = data.get("embedding", {}).get("values", [])
    if not emb:
        raise RuntimeError("empty embedding")
    return emb


def parse_tags(value: str) -> list[str]:
    if not value:
        return []
    return [t.strip() for t in value.split(",") if t.strip()]

def main():
    parser = argparse.ArgumentParser(description="Insert memories with embeddings into PostgreSQL")
    parser.add_argument("--content", help="memory content text")
    parser.add_argument("--tags", default="", help="comma separated tags")
    parser.add_argument("--priority", type=int, default=3, help="1-5 (1 highest)")
    # category removed
    parser.add_argument("--json", help="path to JSON array for batch insert")
    args = parser.parse_args()

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required")

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise RuntimeError("JSON must be a list")

        rows = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            priority = int(item.get("priority", 3))
            if priority < 1 or priority > 5:
                priority = 3
            tags = item.get("tags", [])
            if isinstance(tags, str):
                tags = parse_tags(tags)
            elif not isinstance(tags, list):
                tags = []

            embedding = get_embedding(content)
            rows.append((content, tags, embedding, priority))

        if not rows:
            raise RuntimeError("no valid items in JSON")

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO memories (content, tags, embedding, priority) VALUES (%s, %s, %s, %s)",
            rows
        )
        conn.commit()
        cur.close()
        conn.close()
        print("inserted memories:", len(rows))
        return

    if not args.content:
        raise RuntimeError("--content is required when --json is not provided")

    content = args.content.strip()
    if not content:
        raise RuntimeError("content is empty")

    if args.priority < 1 or args.priority > 5:
        raise RuntimeError("priority must be 1-5")

    tags = parse_tags(args.tags)

    embedding = get_embedding(content)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memories (content, tags, embedding, priority) VALUES (%s, %s, %s, %s)",
        (content, tags, embedding, args.priority)
    )
    conn.commit()
    cur.close()
    conn.close()
    print("inserted memory, embedding dim:", len(embedding))


if __name__ == "__main__":
    main()
