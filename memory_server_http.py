# memory_server_http.py
# 记忆服务 - 云端版本 (HTTP 传输)
# 使用 PostgreSQL + Gemini Embedding 语义搜索
# HTTP memory service (no MCP)

import os
import requests
import numpy as np
from datetime import datetime
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, Response
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor

# 配置
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# 使用最新的 gemini-embedding-001（3072维，100+语言支持）
# 注意：如果从 text-embedding-004 切换，需要重新生成所有 embedding
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

# Gateway upstream config (for /v1/chat/completions)
UPSTREAM_API_KEY = os.environ.get("OPENAI_API_KEY")
UPSTREAM_BASE_URL = os.environ.get("BASE_URL")
UPSTREAM_MODEL_NAME = os.environ.get("MODEL_NAME")

# Memory injection config
MEMORY_PREFIX = os.environ.get(
    "MEMORY_PREFIX",
    "Below are relevant memories. Use them if helpful."
)
MEMORY_FAIL_OPEN = os.environ.get("MEMORY_FAIL_OPEN", "1") not in ("0", "false", "False")


# 工具名称前缀（用于区分多个实例，避免重复声明错误）

# Embedding 缓存（减少 API 调用，加速响应）
EMBEDDING_CACHE = {}
EMBEDDING_CACHE_MAX_SIZE = 100  # 最多缓存 100 条

# 搜索模式：semantic（语义搜索，智能但慢）或 keyword（关键词搜索，快但需精确匹配）
# 设置环境变量 SEARCH_MODE 来切换，默认为 semantic
SEARCH_MODE = os.environ.get("SEARCH_MODE", "semantic").lower()

# 返回结果数量（默认 3 条，减少传输和处理时间）
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "3"))

# 渐进式注入：追踪 recall_memory 调用次数
# 简单实现：基于时间间隔判断是否为新会话
RECALL_COUNTER = {"count": 0, "last_call": None}
RECALL_SESSION_TIMEOUT = 300  # 5 分钟无调用视为新会话

# ========== 记忆缓存 ==========
# 缓存所有记忆到内存，避免每次 recall 都查数据库
_memory_cache: list[dict] = []
_cache_initialized = False


def init_memory_cache():
    """初始化记忆缓存（从数据库加载到内存）"""
    global _memory_cache, _cache_initialized
    if not DATABASE_URL:
        _cache_initialized = True
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, content, tags, embedding, priority, category, created_at, updated_at FROM memories ORDER BY id")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        _memory_cache = []
        for row in rows:
            _memory_cache.append({
                "id": row["id"],
                "content": row["content"],
                "tags": row["tags"] or [],
                "embedding": row["embedding"] or [],
                "priority": row.get("priority", 3) or 3,
                "category": row.get("category", "general") or "general",
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None
            })
        _cache_initialized = True
        print(f"[CACHE] 已加载 {len(_memory_cache)} 条记忆到内存", flush=True)
    except Exception as e:
        print(f"[CACHE ERROR] {e}", flush=True)
        _cache_initialized = True


def get_cached_memories() -> list[dict]:
    """获取缓存的记忆（如果未初始化则先初始化）"""
    global _cache_initialized
    if not _cache_initialized:
        init_memory_cache()
    return _memory_cache


def add_to_cache(memory: dict):
    """添加记忆到缓存"""
    global _memory_cache
    _memory_cache.append(memory)


def update_cache(memory_id: int, **updates):
    """更新缓存中的记忆"""
    global _memory_cache
    for m in _memory_cache:
        if m["id"] == memory_id:
            m.update(updates)
            break


def remove_from_cache(memory_id: int):
    """从缓存中删除记忆"""
    global _memory_cache
    _memory_cache = [m for m in _memory_cache if m["id"] != memory_id]

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT[] DEFAULT '{}',
            embedding FLOAT8[],
            priority INTEGER DEFAULT 3,
            category VARCHAR(50) DEFAULT 'general',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'embedding'
            ) THEN
                ALTER TABLE memories ADD COLUMN embedding FLOAT8[];
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'priority'
            ) THEN
                ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3;
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'category'
            ) THEN
                ALTER TABLE memories ADD COLUMN category VARCHAR(50) DEFAULT 'general';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'updated_at'
            ) THEN
                ALTER TABLE memories ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            END IF;
        END $$;
    """)
    conn.commit()
    cur.close()
    conn.close()


def get_embedding(text: str, use_cache: bool = True) -> list[float]:
    global EMBEDDING_CACHE

    cache_key = text[:200].strip().lower()
    if use_cache and cache_key in EMBEDDING_CACHE:
        print("[EMBEDDING] cache hit", flush=True)
        return EMBEDDING_CACHE[cache_key]

    if not GEMINI_API_KEY:
        print("[EMBEDDING] GEMINI_API_KEY missing", flush=True)
        return []

    try:
        url = f"{GEMINI_EMBEDDING_URL}?key={GEMINI_API_KEY}"
        payload = {"content": {"parts": [{"text": text}]}}
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", {}).get("values", [])
            if use_cache and embedding:
                if len(EMBEDDING_CACHE) >= EMBEDDING_CACHE_MAX_SIZE:
                    oldest_key = next(iter(EMBEDDING_CACHE))
                    del EMBEDDING_CACHE[oldest_key]
                EMBEDDING_CACHE[cache_key] = embedding
            return embedding
        else:
            print("[EMBEDDING] API error:", response.status_code, flush=True)
    except Exception as e:
        print("[EMBEDDING] error:", e, flush=True)
    return []


def translate_query(query: str) -> list[str]:
    if not GEMINI_API_KEY:
        return []

    is_ascii = all(ord(c) < 128 for c in query.replace(" ", ""))

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        if is_ascii:
            prompt = f"Translate '{query}' to Chinese and Japanese. Return ONLY the translations, one per line, no explanations."
        else:
            prompt = f"Translate '{query}' to English. Return ONLY the translation, no explanations."

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 50, "temperature": 0}
        }
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            translations = [t.strip() for t in text.strip().split("\n") if t.strip() and t.strip() != query]
            return translations[:3]
    except Exception as e:
        print("[TRANSLATE] error:", e, flush=True)

    return []


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_memories(query: str, memories: list[dict], category: str = None) -> list[tuple[float, dict]]:
    if category:
        memories = [m for m in memories if m.get("category", "general") == category]

    if SEARCH_MODE == "keyword":
        return search_memories_keyword(query, memories, MAX_RESULTS, category=None)

    all_queries = [query] + translate_query(query)

    scores_by_id = {}
    for q in all_queries:
        q_embedding = get_embedding(q)
        q_lower = q.lower()

        for m in memories:
            memory_id = m["id"]
            semantic_score = 0
            keyword_score = 0

            if q_embedding and m.get("embedding"):
                semantic_score = cosine_similarity(q_embedding, m["embedding"])

            content_lower = m["content"].lower()
            if q_lower in content_lower:
                keyword_score += 0.3

            for tag in m.get("tags", []):
                if q_lower in tag.lower() or tag.lower() in q_lower:
                    keyword_score += 0.25

            for word in q_lower.split():
                if len(word) >= 2 and word in content_lower:
                    keyword_score += 0.1

            priority_boost = (6 - m.get("priority", 3)) * 0.05

            base_score = max(semantic_score, keyword_score)
            if semantic_score > 0.3 and keyword_score > 0:
                base_score += 0.1

            final_score = base_score + priority_boost

            if final_score > 0.25:
                if memory_id not in scores_by_id or final_score > scores_by_id[memory_id][0]:
                    scores_by_id[memory_id] = (final_score, m)

    results = list(scores_by_id.values())
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:MAX_RESULTS]


def search_memories_keyword(query: str, memories: list[dict], top_k: int = None, category: str = None) -> list[tuple[float, dict]]:
    if category:
        memories = [m for m in memories if m.get("category", "general") == category]

    query_lower = query.lower()
    scored = []

    for m in memories:
        score = 0
        content_lower = m["content"].lower()

        if query_lower in content_lower:
            score += 10

        for tag in m.get("tags", []):
            if query_lower in tag.lower():
                score += 5

        for word in query_lower.split():
            if word in content_lower:
                score += 2

        priority_boost = (6 - m.get("priority", 3))
        score += priority_boost

        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: top_k or MAX_RESULTS]



async def index(request):
    return Response("Memory Server is running!", media_type="text/plain")


async def list_models(request):
    if not UPSTREAM_MODEL_NAME:
        return JSONResponse({"error": "MODEL_NAME is required"}, status_code=500)
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": UPSTREAM_MODEL_NAME,
            "object": "model",
            "created": 1677858242,
            "owned_by": "memory-gateway"
        }]
    })



def extract_query_from_payload(payload: dict) -> str:
    messages = payload.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                return "\n".join([p for p in parts if p]).strip()
    return ""


def recall_memory_text(query: str, category: str | None = None, top_k: int | None = None) -> str:
    try:
        memories = get_cached_memories()
        results = search_memories(query, memories, category=category)
        results = results[: (top_k or MAX_RESULTS)]
        lines = [m.get("content", "") for _, m in results if m.get("content")]
        return "\n".join(lines).strip()
    except Exception as e:
        print("memory recall error:", e)
        if not MEMORY_FAIL_OPEN:
            raise
        return ""


def inject_memory_into_messages(payload: dict, memory_text: str) -> None:
    if not memory_text:
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return

    memory_block = f"{MEMORY_PREFIX}\n{memory_text}"

    for msg in messages:
        if msg.get("role") == "system":
            existing = msg.get("content", "")
            if isinstance(existing, str) and existing.strip():
                msg["content"] = existing.rstrip() + "\n\n" + memory_block
            else:
                msg["content"] = memory_block
            print("injected memories:\n" + memory_text[:1000])
            return

    messages.insert(0, {"role": "system", "content": memory_block})
    print("injected memories:\n" + memory_text[:1000])


async def chat_completions(request):
    try:
        payload = await request.json()

        if not UPSTREAM_API_KEY or not UPSTREAM_BASE_URL or not UPSTREAM_MODEL_NAME:
            return JSONResponse({"error": "OPENAI_API_KEY/BASE_URL/MODEL_NAME required"}, status_code=500)

        payload["model"] = payload.get("model", UPSTREAM_MODEL_NAME)

        query = extract_query_from_payload(payload)
        if query:
            memory_text = recall_memory_text(query)
            if memory_text:
                inject_memory_into_messages(payload, memory_text)

        headers = {
            "Authorization": f"Bearer {UPSTREAM_API_KEY}",
            "Content-Type": "application/json"
        }

        upstream_resp = requests.post(
            f"{UPSTREAM_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        print("upstream status:", upstream_resp.status_code)
        print("upstream body:", upstream_resp.text[:400])

        return Response(
            upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type=upstream_resp.headers.get("Content-Type", "application/json")
        )

    except Exception as e:
        print("gateway error:", e)
        if not MEMORY_FAIL_OPEN:
            raise
        return JSONResponse({"error": str(e)}, status_code=500)


async def health_check(request):
    """健康检查端点"""
    embedding_status = "enabled" if GEMINI_API_KEY else "disabled"
    return JSONResponse({
        "status": "ok",
        "service": "memory-server",
        "storage": "postgresql",
        "semantic_search": embedding_status
    })



async def recall_http(request):
    """REST: recall memories by query."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    category = body.get("category")
    try:
        top_k = int(body.get("top_k", MAX_RESULTS))
    except Exception:
        top_k = MAX_RESULTS
    if top_k <= 0:
        top_k = MAX_RESULTS

    memories = get_cached_memories()
    results = search_memories(query, memories, category=category)
    results = results[:top_k]

    items = []
    for score, m in results:
        items.append({
            "id": m.get("id"),
            "content": m.get("content"),
            "tags": m.get("tags", []),
            "priority": m.get("priority", 3),
            "category": m.get("category", "general"),
            "score": score
        })

    return JSONResponse({
        "query": query,
        "count": len(items),
        "memories": items
    })


# 创建 Starlette 应用
app = Starlette(
    routes=[
        Route("/", index),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/health", health_check),
        Route("/recall", recall_http, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    # 初始化数据库
    if DATABASE_URL:
        print("初始化数据库...")
        init_db()
        print("数据库初始化完成!")

        # 初始化记忆缓存
        print("加载记忆缓存...")
        init_memory_cache()

        # 检测 embedding 维度，如果是旧版（768维）则自动重新生成
        if GEMINI_API_KEY and _memory_cache:
            sample = _memory_cache[0].get("embedding", [])
            if sample and len(sample) == 768:
                print(f"[AUTO-REGEN] 检测到旧版 embedding (768维)，正在自动升级到 3072 维...")
                updated = 0
                for m in _memory_cache:
                    try:
                        new_embedding = get_embedding(m["content"], use_cache=False)
                        if new_embedding:
                            conn = get_db_connection()
                            cur = conn.cursor()
                            cur.execute("UPDATE memories SET embedding = %s WHERE id = %s", (new_embedding, m["id"]))
                            conn.commit()
                            cur.close()
                            conn.close()
                            m["embedding"] = new_embedding
                            updated += 1
                    except Exception as e:
                        print(f"[AUTO-REGEN ERROR] 记忆 #{m['id']}: {e}", flush=True)
                print(f"[AUTO-REGEN] 完成！已更新 {updated} 条记忆的 embedding")
            elif sample:
                print(f"[EMBEDDING] 当前维度: {len(sample)} (已是最新)")
    else:
        print("警告: 未设置 DATABASE_URL，将无法保存数据")

    if GEMINI_API_KEY:
        print(f"Gemini Embedding: 已启用 (缓存上限: {EMBEDDING_CACHE_MAX_SIZE})")
    else:
        print("Gemini Embedding: 未启用（将使用关键词搜索）")

    print(f"搜索模式: {SEARCH_MODE} ({'语义搜索' if SEARCH_MODE == 'semantic' else '关键词搜索'})")
    print(f"返回结果数: {MAX_RESULTS}")

    # Railway 使用 PORT 环境变量
    port = int(os.environ.get("PORT", 8000))

    print("=" * 50)
    print("Memory Server (PostgreSQL + Embedding)")
    print("=" * 50)
    print(f"服务端口: {port}")
    print("健康检查: /health")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=port)
