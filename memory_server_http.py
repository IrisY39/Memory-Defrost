# memory_server_http.py
# 璁板繂鏈嶅姟 - 浜戠鐗堟湰 (HTTP 浼犺緭)
# 浣跨敤 PostgreSQL + Gemini Embedding 璇箟鎼滅储
# HTTP memory service (no MCP)

import os
import json
import hashlib
import requests
import numpy as np
from datetime import datetime
from requests.adapters import HTTPAdapter
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor

# 閰嶇疆
DATABASE_URL = os.environ.get("DATABASE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# 浣跨敤鏈€鏂扮殑 gemini-embedding-001锛?072缁达紝100+璇█鏀寔锛?
# 娉ㄦ剰锛氬鏋滀粠 text-embedding-004 鍒囨崲锛岄渶瑕侀噸鏂扮敓鎴愭墍鏈?embedding
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

# Gateway upstream config (for /v1/chat/completions)
UPSTREAM_API_KEY = os.environ.get("OPENAI_API_KEY")
UPSTREAM_BASE_URL = os.environ.get("BASE_URL")
UPSTREAM_MODEL_NAME = os.environ.get("MODEL_NAME")
MODELS_JSON = os.environ.get("MODELS_JSON")
UPSTREAM_TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "120"))
UPSTREAM_CONNECT_TIMEOUT = int(os.environ.get("UPSTREAM_CONNECT_TIMEOUT", "15"))

# Memory injection config
MEMORY_PREFIX = os.environ.get(
    "MEMORY_PREFIX",
    "Below are relevant memories. Use them if helpful."
)
MEMORY_FAIL_OPEN = os.environ.get("MEMORY_FAIL_OPEN", "1") not in ("0", "false", "False")


# 宸ュ叿鍚嶇О鍓嶇紑锛堢敤浜庡尯鍒嗗涓疄渚嬶紝閬垮厤閲嶅澹版槑閿欒锛?

# Embedding 缂撳瓨锛堝噺灏?API 璋冪敤锛屽姞閫熷搷搴旓級
EMBEDDING_CACHE = {}
EMBEDDING_CACHE_MAX_SIZE = 100  # 鏈€澶氱紦瀛?100 鏉?

# 鎼滅储妯″紡锛歴emantic锛堣涔夋悳绱紝鏅鸿兘浣嗘參锛夋垨 keyword锛堝叧閿瘝鎼滅储锛屽揩浣嗛渶绮剧‘鍖归厤锛?
# 璁剧疆鐜鍙橀噺 SEARCH_MODE 鏉ュ垏鎹紝榛樿涓?semantic
SEARCH_MODE = os.environ.get("SEARCH_MODE", "semantic").lower()
DEBUG_RECALL_SCORES = os.environ.get("DEBUG_RECALL_SCORES", "0") in ("1", "true", "True")

# 杩斿洖缁撴灉鏁伴噺锛堥粯璁?3 鏉★紝鍑忓皯浼犺緭鍜屽鐞嗘椂闂达級
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "3"))

# 娓愯繘寮忔敞鍏ワ細杩借釜 recall_memory 璋冪敤娆℃暟
# 绠€鍗曞疄鐜帮細鍩轰簬鏃堕棿闂撮殧鍒ゆ柇鏄惁涓烘柊浼氳瘽
RECALL_COUNTER = {"count": 0, "last_call": None}
RECALL_SESSION_TIMEOUT = 300  # 5 鍒嗛挓鏃犺皟鐢ㄨ涓烘柊浼氳瘽

# ========== 璁板繂缂撳瓨 ==========
# 缂撳瓨鎵€鏈夎蹇嗗埌鍐呭瓨锛岄伩鍏嶆瘡娆?recall 閮芥煡鏁版嵁搴?
_memory_cache: list[dict] = []
_cache_initialized = False

# Reuse upstream TCP/TLS connections to reduce handshake latency/failures.
UPSTREAM_SESSION = requests.Session()
UPSTREAM_SESSION.mount("https://", HTTPAdapter(pool_connections=50, pool_maxsize=50))
UPSTREAM_SESSION.mount("http://", HTTPAdapter(pool_connections=50, pool_maxsize=50))

# Cache parsed model registry to avoid reparsing MODELS_JSON on every request.
_MODEL_REGISTRY_CACHE = None
_RECENT_REQUESTS = {}
REQUEST_DEDUP_WINDOW_SECONDS = float(os.environ.get("REQUEST_DEDUP_WINDOW_SECONDS", "2.0"))


def init_memory_cache():
    """Function docstring."""
    global _memory_cache, _cache_initialized
    if not DATABASE_URL:
        _cache_initialized = True
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, content, tags, embedding, priority, created_at, updated_at FROM memories ORDER BY id")
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
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None
            })
        _cache_initialized = True
        print(f"[CACHE] loaded {len(_memory_cache)} memories into cache", flush=True)
    except Exception as e:
        print(f"[CACHE ERROR] {e}", flush=True)
        _cache_initialized = True


def get_cached_memories() -> list[dict]:
    """Function docstring."""
    global _cache_initialized
    if not _cache_initialized:
        init_memory_cache()
    return _memory_cache


def add_to_cache(memory: dict):
    """Function docstring."""
    global _memory_cache
    for m in _memory_cache:
        if m["id"] == memory_id:
            m.update(updates)
            break


def remove_from_cache(memory_id: int):
    """Function docstring."""
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
        print(f"[EMBEDDING] calling api for text: {text[:50]}", flush=True)
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
            if embedding:
                print(f"[EMBEDDING] ok (dim={len(embedding)})", flush=True)
            else:
                print("[EMBEDDING] empty embedding", flush=True)
            return embedding
        else:
            print("[EMBEDDING] API error:", response.status_code, flush=True)
    except Exception as e:
        print("[EMBEDDING] error:", e, flush=True)
    return []



def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_memories(query: str, memories: list[dict]) -> list[tuple[float, dict]]:

    if SEARCH_MODE == "keyword":
        return search_memories_keyword(query, memories, MAX_RESULTS)

    print(f"[SEARCH] mode=semantic query='{query[:80]}'", flush=True)
    all_queries = [query]

    scores_by_id = {}
    score_breakdown = {}
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

            if DEBUG_RECALL_SCORES:
                score_breakdown[memory_id] = {
                    "semantic": round(float(semantic_score), 4),
                    "keyword": round(float(keyword_score), 4),
                    "priority_boost": round(float(priority_boost), 4),
                    "final": round(float(final_score), 4),
                    "priority": m.get("priority", 3),
                    "preview": m["content"][:60].replace("\n", " ")
                }

            if final_score > 0.25:
                if memory_id not in scores_by_id or final_score > scores_by_id[memory_id][0]:
                    scores_by_id[memory_id] = (final_score, m)

    results = list(scores_by_id.values())
    results.sort(key=lambda x: x[0], reverse=True)

    final_results = results[:MAX_RESULTS]

    if DEBUG_RECALL_SCORES:
        print(
            f"[RECALL SCORES] query='{query[:80]}' "
            f"returned={len(final_results)} threshold_matched={len(results)}",
            flush=True
        )
        for score, mem in final_results:
            memory_id = mem["id"]
            item = score_breakdown.get(memory_id, {})
            semantic = item.get("semantic", 0.0)
            keyword = item.get("keyword", 0.0)
            priority_boost = item.get("priority_boost", 0.0)
            priority = item.get("priority", mem.get("priority", 3))
            preview = item.get("preview", mem.get("content", "")[:60].replace("\n", " "))
            print(
                f"[RECALL SCORE] id={memory_id} "
                f"semantic={semantic} keyword={keyword} "
                f"priority_boost={priority_boost} final={round(float(score), 4)} "
                f"priority={priority} preview={preview}",
                flush=True
            )

    return final_results


def search_memories_keyword(query: str, memories: list[dict], top_k: int = None) -> list[tuple[float, dict]]:

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
    models = _get_model_registry()
    if not models["data"]:
        return JSONResponse({"error": "MODEL_NAME or MODELS_JSON is required"}, status_code=500)
    return JSONResponse({
        "object": "list",
        "data": models["data"]
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


def recall_memory_text(query: str, top_k: int | None = None) -> str:
    try:
        memories = get_cached_memories()
        results = search_memories(query, memories)
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
        now_ts = datetime.now().timestamp()
        payload_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        client_host = getattr(getattr(request, "client", None), "host", "unknown")

        # Simple in-memory dedupe for accidental client retries.
        last_ts = _RECENT_REQUESTS.get(payload_fingerprint)
        if last_ts is not None and (now_ts - last_ts) < REQUEST_DEDUP_WINDOW_SECONDS:
            print(
                f"deduped request fp={payload_fingerprint[:10]} "
                f"client={client_host} window={REQUEST_DEDUP_WINDOW_SECONDS}s",
                flush=True
            )
            return JSONResponse(
                {"error": "duplicate request dropped"},
                status_code=409
            )
        _RECENT_REQUESTS[payload_fingerprint] = now_ts
        if len(_RECENT_REQUESTS) > 2000:
            cutoff = now_ts - max(REQUEST_DEDUP_WINDOW_SECONDS * 5, 10.0)
            old_keys = [k for k, ts in _RECENT_REQUESTS.items() if ts < cutoff]
            for k in old_keys:
                _RECENT_REQUESTS.pop(k, None)
        print(
            f"incoming request fp={payload_fingerprint[:10]} client={client_host} "
            f"stream={bool(payload.get('stream'))}",
            flush=True
        )

        model_registry = _get_model_registry()
        if not model_registry["data"]:
            return JSONResponse({"error": "MODEL_NAME or MODELS_JSON is required"}, status_code=500)

        requested_model = payload.get("model") or model_registry["default"]
        if not requested_model:
            return JSONResponse({"error": "model is required"}, status_code=400)

        model_cfg = model_registry["by_id"].get(requested_model)
        if not model_cfg:
            return JSONResponse({"error": f"unknown model: {requested_model}"}, status_code=400)

        payload["model"] = requested_model

        query = extract_query_from_payload(payload)
        if query:
            memory_text = recall_memory_text(query)
            if memory_text:
                inject_memory_into_messages(payload, memory_text)

        headers = {
            "Authorization": f"Bearer {model_cfg['api_key']}",
            "Content-Type": "application/json"
        }

        is_stream = bool(payload.get("stream"))

        if is_stream:
            upstream_resp = requests.post(
                f"{model_cfg['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_TIMEOUT),
                stream=True
            )
            print("upstream status:", upstream_resp.status_code)

            if upstream_resp.status_code >= 400:
                return Response(
                    upstream_resp.content,
                    status_code=upstream_resp.status_code,
                    media_type=upstream_resp.headers.get("Content-Type", "application/json")
                )

            def generate():
                try:
                    for chunk in upstream_resp.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk
                except (requests.exceptions.RequestException, OSError, RuntimeError) as e:
                    # Upstream/client disconnects are expected during streaming; end stream quietly.
                    print("stream aborted:", e)
                finally:
                    upstream_resp.close()

            return StreamingResponse(
                generate(),
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("Content-Type", "text/event-stream")
            )

        upstream_resp = requests.post(
            f"{model_cfg['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=(UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_TIMEOUT)
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


def _get_model_registry() -> dict:
    global _MODEL_REGISTRY_CACHE
    if _MODEL_REGISTRY_CACHE is not None:
        return _MODEL_REGISTRY_CACHE

    # MODELS_JSON format:
    # {
    #   "default": "modelA",
    #   "models": {
    #     "modelA": {"base_url": "...", "api_key": "..."},
    #     "modelB": {"base_url": "...", "api_key": "..."}
    #   }
    # }
    if MODELS_JSON:
        try:
            data = json.loads(MODELS_JSON)
            models = data.get("models", {})
            items = []
            by_id = {}
            for model_id, cfg in models.items():
                base_url = cfg.get("base_url")
                api_key = cfg.get("api_key")
                if not model_id or not base_url or not api_key:
                    continue
                by_id[model_id] = {"base_url": base_url, "api_key": api_key}
                items.append({
                    "id": model_id,
                    "object": "model",
                    "created": 1677858242,
                    "owned_by": "memory-gateway"
                })
            _MODEL_REGISTRY_CACHE = {
                "default": data.get("default") if data.get("default") in by_id else (items[0]["id"] if items else None),
                "data": items,
                "by_id": by_id
            }
            return _MODEL_REGISTRY_CACHE
        except Exception as e:
            print("model registry error:", e)

    if UPSTREAM_API_KEY and UPSTREAM_BASE_URL and UPSTREAM_MODEL_NAME:
        _MODEL_REGISTRY_CACHE = {
            "default": UPSTREAM_MODEL_NAME,
            "data": [{
                "id": UPSTREAM_MODEL_NAME,
                "object": "model",
                "created": 1677858242,
                "owned_by": "memory-gateway"
            }],
            "by_id": {
                UPSTREAM_MODEL_NAME: {
                    "base_url": UPSTREAM_BASE_URL,
                    "api_key": UPSTREAM_API_KEY
                }
            }
        }
        return _MODEL_REGISTRY_CACHE

    _MODEL_REGISTRY_CACHE = {"default": None, "data": [], "by_id": {}}
    return _MODEL_REGISTRY_CACHE


async def health_check(request):
    """Function docstring."""
    embedding_status = "enabled" if GEMINI_API_KEY else "disabled"
    return JSONResponse({
        "status": "ok",
        "service": "memory-server",
        "storage": "postgresql",
        "semantic_search": embedding_status
    })


async def sse_compat(request):
    # Compatibility endpoint for clients still probing legacy MCP SSE.
    return JSONResponse({
        "status": "deprecated",
        "message": "This gateway uses OpenAI-compatible HTTP routes: /v1/models and /v1/chat/completions."
    }, status_code=410)



async def recall_http(request):
    """Function docstring."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    try:
        top_k = int(body.get("top_k", MAX_RESULTS))
    except Exception:
        top_k = MAX_RESULTS
    if top_k <= 0:
        top_k = MAX_RESULTS

    memories = get_cached_memories()
    results = search_memories(query, memories)
    results = results[:top_k]

    items = []
    for score, m in results:
        items.append({
            "id": m.get("id"),
            "content": m.get("content"),
            "tags": m.get("tags", []),
            "priority": m.get("priority", 3),
            "score": score
        })

    return JSONResponse({
        "query": query,
        "count": len(items),
        "memories": items
    })


# 鍒涘缓 Starlette 搴旂敤
app = Starlette(
    routes=[
        Route("/", index),
        Route("/sse", sse_compat, methods=["GET"]),
        Route("/v1/models", list_models, methods=["GET"]),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/health", health_check),
        Route("/recall", recall_http, methods=["POST"]),
    ]
)


if __name__ == "__main__":
    if DATABASE_URL:
        print("Initializing database...")
        init_db()
        print("Database initialized.")
        print("Loading memory cache...")
        init_memory_cache()
    else:
        print("Warning: DATABASE_URL is not set.")

    if GEMINI_API_KEY:
        print(f"Gemini Embedding enabled (cache max: {EMBEDDING_CACHE_MAX_SIZE})")
    else:
        print("Gemini Embedding disabled (keyword fallback mode).")

    print(f"Search mode: {SEARCH_MODE}")
    print(f"Max results: {MAX_RESULTS}")

    port = int(os.environ.get("PORT", 8000))
    print("=" * 50)
    print("Memory Server (PostgreSQL + Embedding)")
    print("=" * 50)
    print(f"Port: {port}")
    print("Health: /health")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
