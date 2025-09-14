#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zendesk Ticket Summariser — LLM-only one-liners with batching, external rules (exact + fuzzy),
resolution detection, role-aware TeamViewer issue detection, animated HTML UI, caching,
resilient rate limiting, and speed controls.

This build:
  • ALWAYS uses the LLM for per-email one-liners (NO local fallback).
  • Supports batched one-liner calls: --ol-batch N
  • Supports per-email skip/length controls: --ol-min-len, --ol-max-chars
  • Keeps ticket-level LLM summary (Responses API first, then Chat fallback) unless --fast or --smart changes scope.

Usage (example):
  python zendesk_ticket_summarizer_llm_only_batched.py \
    --provider openai --openai "$OPENAI_API_KEY" --openai-model gpt-4o-mini \
    --view-id 123456 --llm-one-liners --ol-batch 40 --ol-min-len 0 --ol-max-chars 400 \
    --one-liner-tokens 22 --llm-rpm 60 --llm-burst 15 --max-comments 12 --concurrency 10 --debug
"""

import os, re, csv, sys, time, json, html, math, argparse, threading, random, hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import difflib

# ---- .env loader (optional) ----
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(raise_error_if_not_found=False), encoding="utf-8")
except Exception:
    pass

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Zendesk summariser with external rules, animated HTML and LLM-only batched one-liners."
    )

    # Zendesk creds
    ap.add_argument("--subdomain", default=os.getenv("ZD_SUBDOMAIN",""), help="Zendesk subdomain (e.g. inventry)")
    ap.add_argument("--email",     default=os.getenv("ZD_EMAIL",""), help="Zendesk email")
    ap.add_argument("--password",  default=os.getenv("ZD_PASSWORD",""), help="Zendesk password (avoid; token preferred)")
    ap.add_argument("--api-token", default=os.getenv("ZD_API_TOKEN",""), help="Zendesk API token (preferred)")

    # Provider selection
    ap.add_argument("--provider", choices=["auto","openai","openrouter","none"], default=os.getenv("LLM_PROVIDER","auto"),
                    help="LLM provider. 'auto' picks OpenAI if OPENAI_API_KEY is set, else OpenRouter. 'none' disables ticket-level LLM.")
    ap.add_argument("--fast", action="store_true", help="No LLM for ticket-level summary; fast extractive only (one-liners stay LLM in this build if --llm-one-liners is set)")
    ap.add_argument("--smart", action="store_true", help="Hybrid: LLM only for top-N complex tickets (see --llm-quota)")
    ap.add_argument("--llm-quota", type=int, default=50, help="Tickets to send to LLM in --smart mode")

    # OpenAI
    ap.add_argument("--openai",        default=os.getenv("OPENAI_API_KEY",""), help="OpenAI API key")
    ap.add_argument("--openai-model",  default=os.getenv("OPENAI_MODEL","gpt-4o-mini"), help='OpenAI model id (e.g. "gpt-4o-mini")')
    ap.add_argument("--no-responses", action="store_true", help="Skip the OpenAI Responses API and use Chat Completions only")

    # OpenRouter
    ap.add_argument("--openrouter",    default=os.getenv("OPENROUTER_API_KEY",""), help="OpenRouter API key")
    ap.add_argument("--model",         default=os.getenv("OPENROUTER_MODEL","openai/gpt-4o-mini"), help="OpenRouter model id")

    # Scope
    ap.add_argument("--view-id", type=int, default=0, help="Zendesk View ID (overrides assignee/statuses/days)")
    ap.add_argument("--assignee", default=os.getenv("ASSIGNEE_EMAIL",""), help="Assignee email; blank = ALL")
    ap.add_argument("--statuses", default="open,pending,hold", help="Statuses (comma list) if no --view-id")
    ap.add_argument("--days", type=int, default=int(os.getenv("DAYS","60")), help="Updated within last N days if no --view-id")
    ap.add_argument("--max", type=int, default=int(os.getenv("MAX_TICKETS","1000")), help="Max tickets")

    # Speed/quality
    ap.add_argument("--max-tokens", type=int, default=240, help="Max tokens per ticket summary (JSON)")
    ap.add_argument("--concurrency", type=int, default=8, help="Worker threads for fetching + summarising")
    ap.add_argument("--max-comments", type=int, default=12, help="How many most-recent comments to analyse per ticket")

    # LLM for dropdown one-liners (LLM-ONLY + batching)
    ap.add_argument("--llm-one-liners", action="store_true", help="Summarise each email into a one-liner via LLM ONLY (no local fallback).")
    ap.add_argument("--one-liner-tokens", type=int, default=24, help="Max tokens for each one-liner summary")

    # LLM throttling
    ap.add_argument("--llm-rpm", type=int, default=int(os.getenv("LLM_RPM","60")), help="LLM rate-limit: requests per minute (soft)")
    ap.add_argument("--llm-burst", type=int, default=int(os.getenv("LLM_BURST","15")), help="LLM token bucket burst size")

    # One-liner batching & size controls
    ap.add_argument("--ol-batch", type=int, default=40, help="Batch size for one-liner calls (list-of-items per chat request)")
    ap.add_argument("--ol-min-len", type=int, default=0, help="Minimum cleaned body length before sending to LLM (set 0 to never skip)")
    ap.add_argument("--ol-max-chars", type=int, default=600, help="Truncate email body before LLM (speed/cost control)")

    # Rules
    ap.add_argument("--rules-dir", default=os.getenv("RULES_DIR","rules"), help="Folder with per-category rule files and optional config.json")
    ap.add_argument("--rules", default=os.getenv("RULES_PATH",""), help="Single rules.json file (fallback). If both provided, --rules-dir is used.")

    # Logging / outputs
    ap.add_argument("--log-file", default=os.getenv("LOG_FILE",""), help="Optional path to write a combined console log.")
    ap.add_argument("--jsonl",    default=os.getenv("JSONL_PATH",""), help="Optional JSONL file path to append structured results per ticket.")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")
    return ap.parse_args()

# =============================================================================
# Logger helpers
# =============================================================================
log_fp = None
def _log_setup(path: str):
    global log_fp
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        try:
            log_fp = open(path, "a", encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not open --log-file: {e}")

def _log(msg: str):
    print(msg)
    if log_fp:
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_fp.write(f"{ts} {msg}\n")
            log_fp.flush()
        except Exception:
            pass

# =============================================================================
# Helpers
# =============================================================================
def iso_now_utc(): return datetime.now(timezone.utc)
def parse_iso(ts: str) -> datetime:
    if not ts: return iso_now_utc()
    try: return datetime.fromisoformat(ts.replace("Z","+00:00"))
    except: return iso_now_utc()

def elide(s: str, n: int) -> str:
    s=(s or "").strip()
    return s if len(s)<=n else s[:max(0,n-1)]+"…"

def strip_html_basic(text: str) -> str:
    if not text: return ""
    text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return text

def strip_email_trails_and_signatures(text: str) -> str:
    if not text: return ""
    t = text
    t = "\n".join([ln for ln in t.splitlines() if not ln.strip().startswith(">")])
    t = re.split(r"\nOn .* wrote:\n", t, maxsplit=1)[0]
    t = re.split(r"\nFrom: .*?\n(?:To:|Subject:)", t, maxsplit=1)[0]
    t = re.sub(r"^(hi|hello|hey)\b[^,\n]*,?\s*", "", t, flags=re.I)
    t = re.sub(r"\b(Kind regards|Best regards|Regards|Thanks|Thank you)[^\n]*", "", t, flags=re.I)
    t = re.sub(r"\n--\s*\n.*", "", t, flags=re.S)
    t = re.split(r"\nSent from my (iPhone|Android)\b", t, maxsplit=1)[0]
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

BOILERPLATE_PATTERNS = [
    r"(?i)take a look at our updated support portal.*",
    r"(?i)quick[- ]start guide.*",
    r"(?i)video tutorial.*",
    r"(?i)how to use the .* support portal.*",
    r"(?i)click here.*",
    r"(?i)register for .*",
    r"(?i)this ticket has now been assigned .*?",
    r"(?i)auto-?reply.*",
    r"(?i)do not reply.*",
    r"(?i)privacy policy.*",
    r"(?i)unsubscrib(e|ing).*",
    r"(?i)\b(confidentiality notice|disclaimer)\b.*",
]
def remove_boilerplate(t: str) -> str:
    if not t: return ""
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def clean_message_body(raw_html_or_text: str) -> str:
    t = strip_html_basic(raw_html_or_text or "")
    t = strip_email_trails_and_signatures(t)
    t = remove_boilerplate(t)
    t = re.sub(r"^(customer|agent)\s*:\s*", "", t, flags=re.I|re.M)
    lines = [ln.strip() for ln in t.splitlines() if len(ln.strip())>1]
    t = "\n".join(lines).strip()
    return t

def is_meaningful_text(t: str) -> bool:
    t = (t or "").strip()
    if not t: return False
    if len(t) < 6: return False
    return True

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# =============================================================================
# Rate limiter (token bucket)
# =============================================================================
class RateLimiter:
    def __init__(self, rate_per_minute: int = 60, burst: int = 15):
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.refill_rate = max(1, rate_per_minute) / 60.0  # tokens per second
        self.lock = threading.Lock()
        self.timestamp = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens < 1.0:
                needed = 1.0 - self.tokens
                sleep_for = needed / self.refill_rate
                time.sleep(max(0.01, sleep_for))
                self.tokens = 0.0
                self.timestamp = time.monotonic()
            else:
                self.tokens -= 1.0

# =============================================================================
# Zendesk API
# =============================================================================
class Zendesk:
    def __init__(self, subdomain: str, email: str, password: str, api_token: str, debug=False):
        if not subdomain or not email:
            sys.exit("ERROR: missing Zendesk creds (--subdomain/--email). Also provide --api-token or --password (token preferred).")
        if not (api_token or password):
            sys.exit("ERROR: provide --api-token (recommended) or --password.")
        self.base = f"https://{subdomain}.zendesk.com/api/v2"
        self.s = requests.Session()
        self.debug = debug

        if api_token:
            self.s.auth = (f"{email}/token", api_token)
        else:
            self.s.auth = (email, password)

        self.s.headers.update({"Content-Type": "application/json"})

    def _get(self, url, params=None):
        for _ in range(6):
            try:
                r = self.s.get(url, params=params or {}, timeout=45)
            except requests.RequestException as e:
                if self.debug:
                    _log(f"[DEBUG] Zendesk GET network error: {e}. Retrying shortly.")
                time.sleep(1.0)
                continue
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", "2"))
                if self.debug: _log(f"[DEBUG] 429; sleeping {wait}s")
                time.sleep(wait); continue
            if r.status_code == 401:
                sys.exit("401 Unauthorized. Use an API token (recommended) or enable password access to API.")
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Too many retries talking to Zendesk")

    def tickets_in_view(self, view_id: int, limit: int) -> List[Dict[str, Any]]:
        url = f"{self.base}/views/{view_id}/tickets.json"
        out, seen = [], set()
        while url:
            data = self._get(url)
            items = data.get("tickets", []) or []
            for t in items:
                tid = t.get("id")
                if tid and tid not in seen:
                    seen.add(tid); out.append(t)
                    if len(out) >= limit: return out
            url = data.get("next_page")
            time.sleep(0.05)
        return out

    def search_tickets(self, statuses: List[str], assignee_email: Optional[str],
                       updated_within_days: int, limit: int) -> List[Dict[str, Any]]:
        base_url = f"{self.base}/search.json"
        queries = []
        if statuses:
            for st in statuses:
                q = f"type:ticket status:{st}"
                if assignee_email:
                    q += f' assignee:"{assignee_email}"'
                queries.append(q)
        else:
            q = "type:ticket status<solved"
            if assignee_email:
                q += f' assignee:"{assignee_email}"'
            queries.append(q)

        results_by_id: Dict[int, Dict[str, Any]] = {}
        cutoff = iso_now_utc() - timedelta(days=updated_within_days)

        def run_query(q: str):
            page = 1
            while True:
                try:
                    data = self._get(base_url, params={"query": q, "per_page": 100, "page": page,
                                                       "sort_by":"updated_at","sort_order":"desc"})
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 422:
                        if self.debug: _log(f"[DEBUG] 422 for query '{q}', stopping this branch.")
                        break
                    raise
                items = data.get("results", []) or {}
                if not items: break
                for r in items:
                    ts = parse_iso(r.get("updated_at", ""))
                    if ts >= cutoff:
                        results_by_id[r["id"]] = r
                if not data.get("next_page"): break
                page += 1
                if len(results_by_id) >= limit * 2: break
                time.sleep(0.1)

        for q in queries:
            run_query(q)

        if not results_by_id:
            q = "type:ticket status<solved"
            if assignee_email:
                q += f' assignee:"{assignee_email}"'
            if self.debug: _log("[DEBUG] Per-status queries empty; trying status<solved> fallback")
            run_query(q)

        out = list(results_by_id.values())
        out.sort(key=lambda r: r.get("updated_at",""), reverse=True)
        if self.debug: _log(f"[DEBUG] Search merged results: {len(out)} (before limit)")
        return out[:limit]

    def users_me(self) -> Dict[str, Any]:
        return self._get(f"{self.base}/users/me.json").get("user", {})

    def user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        data = self._get(f"{self.base}/users/search.json", params={"query": email})
        users = data.get("users", [])
        return users[0] if users else None

    def tickets_assigned_to(self, email: str, limit: int = 5000) -> List[Dict[str, Any]]:
        me = self.users_me(); uid = me.get("id")
        if email and email.lower() != (me.get("email","").lower()):
            u = self.user_by_email(email); uid = u.get("id") if u else uid
        if not uid: return []
        url = f"{self.base}/users/{uid}/tickets/assigned.json"
        out, page = [], 1
        while True:
            data = self._get(url, params={"page": page, "sort_by": "updated_at", "sort_order": "desc"})
            items = data.get("tickets", [])
            out.extend(items)
            if not data.get("next_page") or len(out) >= limit: break
            page += 1; time.sleep(0.05)
        out = [t for t in out if t.get("status") not in ("solved","closed")]
        return out[:limit]

    def ticket_comments(self, ticket_id: int) -> List[Dict[str, Any]]:
        url = f"{self.base}/tickets/{ticket_id}/comments.json"
        comms, page = [], 1
        while True:
            data = self._get(url, params={"page": page, "include": "users"})
            comms.extend(data.get("comments", []))
            if not data.get("next_page"): break
            page += 1; time.sleep(0.05)
        return comms

    def users_show_many(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not user_ids: return {}
        out = {}
        uniq = list(dict.fromkeys(user_ids))
        for chunk in chunked(uniq, 100):
            ids = ",".join(map(str, chunk))
            data = self._get(f"{self.base}/users/show_many.json?ids={ids}")
            for u in data.get("users", []):
                out[u["id"]] = u
            time.sleep(0.02)
        return out

# =============================================================================
# LLM Clients (OpenAI / OpenRouter)
# =============================================================================
class OpenRouterClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 240, debug=False, rate_limiter: Optional[RateLimiter]=None):
        if not api_key: sys.exit("ERROR: provide OPENROUTER_API_KEY or use --fast.")
        self.api_key = api_key; self.model = model; self.max_tokens = max_tokens; self.debug = debug
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local-tool",
            "X-Title": "Zendesk JSON Summariser",
        }
        self.rl = rate_limiter or RateLimiter()

    @staticmethod
    def _strip_code_fences(s: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s or "", re.I)
        return m.group(1) if m else s

    def _post(self, payload: dict) -> Optional[dict]:
        for i in range(5):
            try:
                self.rl.acquire()
                r = requests.post(self.endpoint, headers=self.headers, data=json.dumps(payload), timeout=50)
            except requests.RequestException as e:
                if self.debug:
                    _log(f"[DEBUG] OpenRouter network error: {e}; attempt {i+1}/5")
                time.sleep(0.7 * (i + 1))
                continue
            if r.status_code == 429:
                if self.debug:
                    _log(f"[DEBUG] 429 from OpenRouter; sleeping {(i+1)*1.2:.1f}s")
                time.sleep(1.2 * (i + 1))
                continue
            if 500 <= r.status_code < 600:
                if self.debug:
                    _log(f"[DEBUG] OpenRouter HTTP {r.status_code}; retrying")
                time.sleep(1.2 * (i + 1) + random.random() * 0.4)
                continue
            try:
                return r.json()
            except Exception:
                return None
        return None

    def summarise_json(self, convo_text: str, subject: str, tid: int) -> Dict[str, str]:
        system = (
            "You are a senior support analyst. Write ABSTRACT, DECISION-ORIENTED summaries in your own words.\n"
            "Rules:\n"
            "1) NEVER copy sentences or boilerplate; no greetings, names, links, or signatures.\n"
            "2) Prefer concrete details: exact errors, versions, dates/times, who-did-what.\n"
            "3) If there is no actionable content, state that plainly.\n"
            "4) Output STRICT JSON with keys: summary, resolution_status, evidence, topic, next_step.\n"
            "5) resolution_status ∈ {resolved, waiting_on_customer, waiting_on_us, needs_more_info, unknown}.\n"
            "6) evidence: ≤ 2 short paraphrases or quotes (≤8 words) that justify the status.\n"
            "7) topic ∈ {Agreement Screen, Login, Billing, Printing, Install, Network, Data, Access, UI, Hardware, Bug, Other}.\n"
            "8) next_step: single, concrete sentence that an agent could take next.\n"
            "Tickets may be in any language; translate internally but output JSON in English. Return ONLY a JSON object."
        )
        user = (
            f"Ticket #{tid} — Subject: {subject}\n\n"
            "Conversation (oldest→newest). Each line is 'Customer:' or 'Agent:' with CLEANED content:\n"
            f"{convo_text}\n\n"
            'Return ONLY JSON like: {"summary":"","resolution_status":"","evidence":"","topic":"","next_step":""}'
        )
        payload = {
            "model": self.model,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        data = self._post(payload)
        if not data:
            return {"summary":"Summary unavailable.","resolution_status":"unknown","evidence":"","topic":"Other","next_step":"Review the ticket directly."}
        content = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
        raw = self._strip_code_fences(content)
        obj = self._safe_json_parse(raw, self.debug)
        if obj is not None:
            return self._normalise(obj)
        if self.debug:
            _log("[DEBUG] OpenRouter JSON parse fail; content head: " + content[:400])
        return {"summary":"Summary unavailable.","resolution_status":"unknown","evidence":"","topic":"Other","next_step":"Review the ticket directly."}

    def summarise_one_liners_batch(self, items: List[Tuple[str,str]], max_tokens_each: int = 24) -> List[str]:
        """
        items: [(role, text), ...]
        Returns a list of strings, same length, one concise line per input.
        """
        sys_prompt = (
            "You will receive a JSON array of items. Each item has {role, text}.\n"
            "For EACH item, output ONE concise line (<= 15 words), neutral voice: use verbs like 'reports', 'asks', 'confirms'.\n"
            "Do NOT include greetings, names, signatures, links, phone numbers, ticket IDs, or email addresses.\n"
            "Return ONLY a JSON array of strings. The array length MUST equal the input array length, preserving order."
        )
        user_payload = json.dumps([{"role": r, "text": t} for (r,t) in items], ensure_ascii=False)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Items:\n{user_payload}"},
            ],
            "temperature": 0.1,
            "max_tokens": max(64, max_tokens_each * len(items) + 32),
        }
        data = self._post(payload)
        if not data:
            return ["" for _ in items]

        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        raw = self._strip_code_fences(content)
        arr = self._safe_json_parse(raw, self.debug)
        trimmed = (raw or "").strip()

        if isinstance(arr, list) and len(arr) == len(items):
            return [(s or "").strip() if isinstance(s, str) else "" for s in arr]

        incomplete = trimmed.startswith("[") and not trimmed.endswith("]")
        length_mismatch = isinstance(arr, list) and len(arr) != len(items)

        if (incomplete or length_mismatch) and len(items) > 1:
            mid = len(items) // 2
            left = self.summarise_one_liners_batch(items[:mid], max_tokens_each)
            right = self.summarise_one_liners_batch(items[mid:], max_tokens_each)
            return left + right

        if self.debug:
            _log("[DEBUG] OpenRouter batch one-liner JSON parse failed. Head: " + content[:400])
        return ["" for _ in items]

    @staticmethod
    def _safe_json_parse(content: str, debug=False) -> Optional[Any]:
        try:
            return json.loads(content)
        except Exception:
            pass
        for pat in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
            m = re.search(pat, content or "")
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    if debug:
                        _log("[DEBUG] JSON extraction failed.")
        return None

    @staticmethod
    def _normalise(obj: dict) -> Dict[str, str]:
        out = {
            "summary": (obj.get("summary","") or "").strip(),
            "resolution_status": (obj.get("resolution_status","unknown") or "unknown").strip(),
            "evidence": (obj.get("evidence","") or "").strip(),
            "topic": (obj.get("topic","Other") or "Other").strip(),
            "next_step": (obj.get("next_step","") or "").strip(),
        }
        if out["resolution_status"] not in {"resolved","waiting_on_customer","waiting_on_us","needs_more_info","unknown"}:
            out["resolution_status"] = "unknown"
        return out

# --------------------- OpenAI Client ---------------------
class OpenAIClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 240, debug: bool = False, use_responses: bool = True, rate_limiter: Optional[RateLimiter]=None):
        if not api_key:
            sys.exit("ERROR: provide OPENAI_API_KEY or use --fast.")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.debug = debug
        self.use_responses = use_responses

        self.responses_url = "https://api.openai.com/v1/responses"
        self.chat_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.rl = rate_limiter or RateLimiter()

    # -------------- Public --------------
    def summarise_json(self, convo_text: str, subject: str, tid: int) -> Dict[str, str]:
        system = (
            "You are a senior support analyst. Write ABSTRACT, DECISION-ORIENTED summaries in your own words.\n"
            "Rules:\n"
            "1) NEVER copy sentences or boilerplate; no greetings, names, links, or signatures.\n"
            "2) Prefer concrete details: exact errors, versions, dates/times, who-did-what.\n"
            "3) If there is no actionable content, state that plainly.\n"
            "4) Output STRICT JSON with keys: summary, resolution_status, evidence, topic, next_step.\n"
            "5) resolution_status ∈ {resolved, waiting_on_customer, waiting_on_us, needs_more_info, unknown}.\n"
            "6) evidence: ≤ 2 short paraphrases or quotes (≤8 words) that justify the status.\n"
            "7) topic ∈ {Agreement Screen, Login, Billing, Printing, Install, Network, Data, Access, UI, Hardware, Bug, Other}.\n"
            "8) next_step: single, concrete sentence that an agent could take next.\n"
            "Tickets may be in any language; translate internally but output JSON in English. Return ONLY a JSON object."
        )
        user = (
            f"Ticket #{tid} — Subject: {subject}\n\n"
            "Conversation (oldest→newest). Each line is 'Customer:' or 'Agent:' with content:\n"
            f"{convo_text}\n\n"
            'Return ONLY JSON like: {"summary":"","resolution_status":"","evidence":"","topic":"","next_step":""}'
        )

        # Try Responses (structured JSON) first, unless disabled
        content = ""
        if self.use_responses:
            content = self._responses_chat(system, user, tid)

        # Fallback to Chat Completions if Responses failed/empty
        if not content:
            content = self._chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=0.2
            )

        raw = self._strip_code_fences(content) if content else ""
        obj = self._safe_json_parse(raw, self.debug)
        if obj is None:
            if self.debug:
                _log("[DEBUG] OpenAI JSON parse fail; content head: " + (content[:400] if content else ""))
            return {"summary": "", "resolution_status": "unknown", "evidence": "", "topic": "Other", "next_step": ""}

        return self._normalise(obj)

    def summarise_one_liners_batch(self, items: List[Tuple[str,str]], max_tokens_each: int = 24) -> List[str]:
        """
        Batch one-liners with Chat Completions.
        """
        sys_prompt = (
            "You will receive a JSON array of items. Each item has {role, text}.\n"
            "For EACH item, output ONE concise line (<= 15 words), neutral voice: 'reports', 'asks', 'confirms'.\n"
            "Do NOT include greetings, names, signatures, links, phone numbers, ticket IDs, or email addresses.\n"
            "Return ONLY a JSON array of strings, same length and order as input."
        )
        user_payload = json.dumps([{"role": r, "text": t} for (r,t) in items], ensure_ascii=False)
        payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content": sys_prompt},
                {"role":"user","content": f"Items:\n{user_payload}"}
            ],
            "temperature": 0.1,
            "max_tokens": max(64, max_tokens_each * len(items) + 32),
        }
        r = self._post_with_backoff(self.chat_url, payload, label="ChatBatch")
        if r is None or r.status_code != 200:
            return ["" for _ in items]
        try:
            data = r.json()
        except Exception:
            return ["" for _ in items]

        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        raw = self._strip_code_fences(content)
        arr = self._safe_json_parse(raw, self.debug)
        trimmed = (raw or "").strip()

        if isinstance(arr, list) and len(arr) == len(items):
            return [(s or "").strip() if isinstance(s, str) else "" for s in arr]

        incomplete = trimmed.startswith("[") and not trimmed.endswith("]")
        length_mismatch = isinstance(arr, list) and len(arr) != len(items)

        if (incomplete or length_mismatch) and len(items) > 1:
            mid = len(items) // 2
            left = self.summarise_one_liners_batch(items[:mid], max_tokens_each)
            right = self.summarise_one_liners_batch(items[mid:], max_tokens_each)
            return left + right

        if self.debug:
            _log("[DEBUG] OpenAI batch one-liner JSON parse failed. Head: " + content[:400])
        return ["" for _ in items]

    # -------------- HTTP helpers --------------
    def _post_with_backoff(self, url: str, payload: dict, label: str, attempts: int = 5) -> Optional[requests.Response]:
        body = json.dumps(payload)
        for i in range(attempts):
            try:
                self.rl.acquire()
                r = requests.post(url, headers=self.headers, data=body, timeout=60)
            except requests.RequestException as e:
                if self.debug:
                    _log(f"[DEBUG] {label} network error: {e}; attempt {i+1}/{attempts}")
                time.sleep(0.7 * (i + 1))
                continue

            rid = r.headers.get("x-request-id")
            if r.status_code == 429:
                if self.debug:
                    _log(f"[DEBUG] 429 from {label}; request-id={rid}; sleeping {(i+1)*1.2:.1f}s")
                time.sleep(1.2 * (i + 1))
                continue

            if 500 <= r.status_code < 600:
                if self.debug:
                    _log(f"[DEBUG] {label} HTTP {r.status_code}; request-id={rid}; retrying")
                time.sleep(1.2 * (i + 1) + random.random() * 0.4)
                continue

            return r
        return None

    # -------------- API callers --------------
    def _responses_chat(self, system_text: str, user_text: str, tid: int) -> str:
        json_schema_obj = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "resolution_status": {
                    "type": "string",
                    "enum": ["resolved", "waiting_on_customer", "waiting_on_us", "needs_more_info", "unknown"]
                },
                "evidence": {"type": "string"},
                "topic": {
                    "type": "string",
                    "enum": ["Agreement Screen", "Login", "Billing", "Printing", "Install", "Network", "Data", "Access", "UI", "Hardware", "Bug", "Other"]
                },
                "next_step": {"type": "string"}
            },
            "required": ["summary", "resolution_status", "evidence", "topic", "next_step"]
        }

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_output_tokens": max(128, min(512, self.max_tokens)),
            "instructions": system_text,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}]
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "TicketSummary",
                    "schema": json_schema_obj
                }
            }
        }

        if self.debug:
            _log(f"[DEBUG] ▶ LLM.summarise #{tid} via Responses")

        r = self._post_with_backoff(self.responses_url, payload, label="Responses")
        if r is None:
            if self.debug:
                _log("[DEBUG] Responses request failed after retries (no response object)")
            return ""

        rid = r.headers.get("x-request-id")
        if r.status_code != 200:
            if self.debug:
                _log(f"[DEBUG] Responses HTTP {r.status_code}: {r.text[:600]}")
            return ""

        try:
            data = r.json()
        except Exception:
            if self.debug:
                _log(f"[DEBUG] Responses non-JSON body; request-id={rid}; head={r.text[:600]}")
            return ""

        return (self._extract_responses_text(data) or "").strip()

    def _chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        base_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        variants = [("chat_plain", dict(base_payload))]

        for name, payload in variants:
            if self.debug:
                _log(f"[DEBUG] ▶ Chat call ({name})")
            r = self._post_with_backoff(self.chat_url, payload, label="Chat")
            if r is None:
                if self.debug:
                    _log("[DEBUG] Chat request failed after retries (no response object)")
                continue

            rid = r.headers.get("x-request-id")
            if r.status_code != 200:
                if self.debug:
                    _log(f"[DEBUG] Chat HTTP {r.status_code}; request-id={rid}; head={r.text[:600]}")
                continue

            try:
                data = r.json()
            except Exception:
                if self.debug:
                    _log(f"[DEBUG] Chat non-JSON body; request-id={rid}; head={r.text[:600]}")
                continue

            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            if content:
                return content.strip()

            if self.debug:
                _log(f"[DEBUG] Chat empty content; request-id={rid}; body={json.dumps(data)[:500]}")
        return ""

    # -------------- Parse helpers --------------
    @staticmethod
    def _strip_code_fences(s: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s or "", re.I)
        return m.group(1) if m else s

    @staticmethod
    def _extract_responses_text(data: dict) -> str:
        if not isinstance(data, dict):
            return ""
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        out = []
        for item in data.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                    out.append(c["text"])
        return "\n".join(out).strip()

    @staticmethod
    def _safe_json_parse(content: str, debug: bool = False) -> Optional[Any]:
        try:
            return json.loads(content)
        except Exception:
            pass
        for pat in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
            m = re.search(pat, content or "")
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    if debug:
                        _log("[DEBUG] JSON extraction failed.")
        return None

    @staticmethod
    def _normalise(obj: dict) -> Dict[str, str]:
        out = {
            "summary": (obj.get("summary", "") or "").strip(),
            "resolution_status": (obj.get("resolution_status", "unknown") or "unknown").strip(),
            "evidence": (obj.get("evidence", "") or "").strip(),
            "topic": (obj.get("topic", "Other") or "Other").strip(),
            "next_step": (obj.get("next_step", "") or "").strip(),
        }
        if out["resolution_status"] not in {"resolved", "waiting_on_customer", "waiting_on_us", "needs_more_info", "unknown"}:
            out["resolution_status"] = "unknown"
        for k in ("summary", "evidence", "next_step"):
            if len(out[k]) > 1800:
                out[k] = out[k][:1800] + "…"
        return out

# =============================================================================
# Category taxonomy & heuristics
# =============================================================================
DEFAULT_CATEGORIES = [
    "Agreement Screen","Login","Billing","Printing","Install","Network","Data","Access","UI","Hardware","Bug","Other"
]

TAX = {
    "Login":   ["password","reset","signin","sign in","login","log in","otp","2fa","mfa","locked"],
    "Billing": ["invoice","refund","charge","billing","payment","credit","subscription"],
    "Install": ["install","setup","configure","deploy","installer","prerequisite"],
    "Network": ["network","proxy","firewall","vpn","dns","timeout","connection"],
    "Data":    ["export","import","sync","csv","api","webhook","integration","dataset","sims"],
    "Printing":["print","printer","label","labels","zebra","queue"],
    "Agreement Screen": ["agreement","welcome screen","agreement page","visitor agreement"],
}

HARD = {"escalated","bug","outage","sev1","sev2","problem","rca","security","data_loss"}

def categorize(text: str) -> str:
    low = (text or "").lower()
    scores = {k:0 for k in TAX}
    for cat, kws in TAX.items():
        for kw in kws:
            if kw in low: scores[cat]+=1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best]>0 else "Other"

def is_question(t: str) -> bool:
    if not t: return False
    t = t.strip().lower()
    return t.endswith("?") or bool(re.search(r"\b(could you|can you|how|where|please advise|help me|what|why|when)\b", t))

def easy_score(n_msgs: int, last_q: bool, has_att: bool, cat: str, tags: List[str]) -> Tuple[int, List[str]]:
    score, why = 0, []
    if n_msgs<=3: score+=3; why.append("short_thread")
    elif n_msgs<=5: score+=2; why.append("medium_thread")
    if last_q: score+=2; why.append("customer_question")
    if not has_att: score+=1; why.append("no_attachments")
    if cat in ("Login","Billing","Install","Printing","Agreement Screen"): score+=2; why.append(f"clear_{cat.lower().replace(' ','_')}")
    if tags and any(t in HARD for t in tags): score-=3; why.append("hard_tag_present")
    return max(score,0), why

def suggested_next_step(cat: str) -> str:
    return {
        "Login":"Send reset steps; verify account; check lockout.",
        "Billing":"Ask invoice #; link billing portal; verify charges.",
        "Install":"Share installer + prereqs; ask OS/version; step guide.",
        "Network":"Ask firewall/proxy; traceroute/ping; try alt network.",
        "Data":"Check integration logs (e.g., SIMS/API); confirm credentials & connectivity.",
        "Printing":"Check printer status, driver/port, clear queue; test sample label.",
        "Agreement Screen":"Confirm PDF; apply and publish; share before/after screenshot.",
        "Other":"Ask for goal + exact error + screenshot.",
    }.get(cat, "Ask for goal + exact error + screenshot.")

# =============================================================================
# Smart Rule Engine (exact + fuzzy + concepts)
# =============================================================================
try:
    from rapidfuzz import fuzz as _fz
    _HAS_RF = True
except Exception:
    _HAS_RF = False

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _fuzzy_hit(needle: str, hay: str, ratio: int = 82) -> bool:
    n = _norm_text(needle); h = _norm_text(hay)
    if not n or not h: return False
    if n in h: return True
    if _HAS_RF:
        return max(_fz.partial_ratio(n, h), _fz.token_set_ratio(n, h)) >= ratio
    return difflib.SequenceMatcher(None, n, h).ratio() * 100 >= ratio

class SmartRuleEngine:
    """In-memory matcher for rules merged from files."""
    def __init__(self, rules: List[dict], defaults: dict = None, debug: bool=False):
        self.debug = debug
        self.defaults = defaults or {}
        self.rules = []
        self._prepare(rules)

    def _prepare(self, rules: List[dict]):
        self.rules = []
        for raw in (rules or []):
            rule = {**self.defaults, **raw}
            m = rule.get("match", {}) or {}
            # compile regex
            rx = []
            for pat in m.get("regex", []):
                try:
                    rx.append(re.compile(pat, re.I))
                except re.error as ex:
                    if self.debug: _log(f"[DEBUG] bad regex in rule '{rule.get('name','?')}': {ex}")
            rule["_all"]  = [x.lower() for x in m.get("all", [])]
            rule["_any"]  = [x.lower() for x in m.get("any", [])]
            rule["_none"] = [x.lower() for x in m.get("none", [])]
            rule["_rx"]   = rx
            # concepts
            c = rule.get("concepts") or {}
            rule["_c"] = {k: [x.lower() for x in (c.get(k) or [])] for k in c.keys()}
            logic = rule.get("logic") or {}
            rule["_require"] = logic.get("require", [])
            rule["_anyof"]   = logic.get("any_of", [])
            rule["_ratio"]   = int((rule.get("fuzzy") or {}).get("ratio", 82))
            self.rules.append(rule)
        if self.debug: _log(f"[DEBUG] Prepared {len(self.rules)} rule(s)")

    def match(self, subject: str, excerpts: List[str]) -> Optional[dict]:
        raw_text = f"{subject or ''}\n" + "\n".join(excerpts or [])
        text = _norm_text(raw_text)

        for r in self.rules:
            ratio = r["_ratio"]

            # none/exclude
            if r["_none"] and any(_fuzzy_hit(t, text, ratio) for t in r["_none"]):
                continue
            # all
            if r["_all"] and not all(_fuzzy_hit(t, text, ratio) for t in r["_all"]):
                continue
            # any
            any_hits = [t for t in r["_any"] if _fuzzy_hit(t, text, ratio)] if r["_any"] else []
            # regex
            rx_hits = [rx.pattern for rx in r["_rx"] if rx.search(raw_text)] if r["_rx"] else []
            # concepts
            concept_ok = True; concept_details=[]
            if r["_c"]:
                req_ok = all(any(_fuzzy_hit(t, text, ratio) for t in r["_c"].get(g, [])) for g in r["_require"]) if r["_require"] else True
                anyof_ok = any(any(_fuzzy_hit(t, text, ratio) for t in r["_c"].get(g, [])) for g in r["_anyof"]) if r["_anyof"] else True
                concept_ok = req_ok and anyof_ok
                if concept_ok:
                    for g in set((r["_require"] or []) + (r["_anyof"] or [])):
                        for t in r["_c"].get(g, []):
                            if _fuzzy_hit(t, text, ratio):
                                concept_details.append(f"{g}:{t}"); break

            literal_ok = (not r["_any"]) or bool(any_hits) or bool(rx_hits)
            if r["_c"]:
                matched = concept_ok and (literal_ok or (not r["_any"] and not r["_rx"]))
            else:
                matched = literal_ok

            if not matched:
                continue

            # confidence
            conf = 0.5
            conf += 0.15 * min(2, len(any_hits))
            conf += 0.15 if rx_hits else 0
            conf += 0.2 if r["_all"] else 0
            conf += 0.2 if r["_c"] and concept_ok else 0
            conf = max(0.0, min(1.0, conf))

            why = []
            if any_hits: why.append(f"any={any_hits[:3]}")
            if rx_hits:  why.append(f"regex={rx_hits[:1]}")
            if concept_details: why.append("concepts=" + ",".join(concept_details[:3]))

            return {
                "name": r.get("name","?"),
                "category": r.get("category"),
                "priority": r.get("priority"),
                "route": r.get("route"),
                "ease": r.get("ease"),
                "next_step": r.get("next_step"),
                "confidence": round(conf, 2),
                "why": "; ".join(why) if why else "rule-hit"
            }
        return None

# =============================================================================
# Rule loading
# =============================================================================
def load_rules_from_dir(rules_dir: str, debug=False) -> Tuple[List[dict], dict]:
    p = Path(rules_dir)
    if not p.exists() or not p.is_dir():
        if debug: _log(f"[DEBUG] rules dir '{rules_dir}' not found")
        return [], {}
    cfg = {}
    cfg_path = p / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
        except Exception as e:
            _log(f"[WARN] Failed to parse {cfg_path}: {e}")
    defaults = cfg.get("defaults", {})
    files = cfg.get("order")
    if not files:
        files = [f.name for f in sorted(p.glob("*.json")) if f.name != "config.json"]
    merged_rules = []
    for fname in files:
        fpath = p / fname
        if not fpath.exists(): 
            if debug: _log(f"[DEBUG] rules file listed but missing: {fpath}")
            continue
        try:
            data = json.load(open(fpath,"r",encoding="utf-8"))
        except Exception as e:
            _log(f"[WARN] Failed to parse rules file {fpath}: {e}")
            continue
        rules = data.get("rules") or []
        merged_rules.extend(rules)
        if debug: _log(f"[DEBUG] Loaded {len(rules)} rule(s) from {fpath}")
    return merged_rules, defaults

def load_rules_single_file(path: str, debug=False) -> Tuple[List[dict], dict]:
    if not path: return [], {}
    p = Path(path)
    if not p.exists():
        if debug: _log(f"[DEBUG] single rules file '{path}' not found")
        return [], {}
    try:
        data = json.load(open(p,"r",encoding="utf-8"))
        rules = data.get("rules") or []
        defaults = data.get("defaults") or {}
        if debug: _log(f"[DEBUG] Loaded {len(rules)} rule(s) from {p}")
        return rules, defaults
    except Exception as e:
        _log(f"[WARN] Failed to parse {p}: {e}")
        return [], {}

# =============================================================================
# Fast extractive summary
# =============================================================================
def extractive_summary(excerpts: List[str], max_lines: int = 5) -> str:
    tokens = [re.findall(r"[a-z0-9']+", s.lower()) for s in excerpts]
    df = {}
    for tks in tokens:
        for w in set(tks):
            if len(w)>2: df[w]=df.get(w,0)+1
    N = max(1, len(tokens))
    scores = []
    for i, tks in enumerate(tokens):
        tf = {}
        for w in tks:
            if len(w)>2: tf[w]=tf.get(w,0)+1
        sc = 0.0
        for w, c in tf.items():
            idf = math.log(1 + N/(1 + df.get(w,1)))
            sc += c * idf
        scores.append((sc, i))
    scores.sort(reverse=True)
    keep_idx = sorted([i for _,i in scores[:max_lines]])
    return " ".join([excerpts[i] for i in keep_idx]) if excerpts else "No content."

# =============================================================================
# Caching
# =============================================================================
CACHE_PATH = "summ_cache.json"
_cache_lock = threading.Lock()
def load_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_PATH): return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except:
        return {}
def save_cache(cache: Dict[str, Any]):
    with _cache_lock:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

# =============================================================================
# TeamViewer / Remote-session detection (role-aware)
# =============================================================================
AGENT_TV_FAILURE_PATTERNS = [
    r"\b(unable|could(?:\s*not|n't)|failed|can't|cannot)\s+(to\s+)?(connect|take|establish)\b.*\b(remote|team\s*viewer|teamviewer|tv)\b",
    r"\b(team\s*viewer|teamviewer|tv)\b.*\b(password|pass|cred(entia)?ls?|id)\b.*\b(invalid|wrong|not\s*work(ing)?|failed)\b",
    r"\b(could(?:\s*not|n't)|unable)\s+(get|take)\s+(the\s+)?remote\b",
    r"\b(session|connection)\b.*\b(timed\s*out|expired|dropped|failed)\b.*\b(team\s*viewer|teamviewer|tv|remote)\b",
]

REMOTE_AVAILABILITY_PHRASES = [
    r"\b(when|what time|which time|availability|available|book|schedule|arrange|slot|suitable time)\b",
    r"\b(let us know|share a (suitable|convenient) time|please advise a time)\b",
    r"\b(can we|shall we|could we)\s+(connect|do a remote|jump on (a )?remote)\b",
    r"\b(provide|share)\s+(your\s+)?(team\s*viewer|teamviewer|tv)\s+(id|password|credentials)\b",
]

def is_remote_availability_request(text: str) -> bool:
    t = (text or "").lower()
    for pat in REMOTE_AVAILABILITY_PHRASES:
        if re.search(pat, t, flags=re.I):
            return True
    return False

def detect_teamviewer_issue_role_aware(text: str, role: str) -> bool:
    """Flag only when an AGENT reports an actual failure, not when arranging a time."""
    if (role or "").lower() != "agent":
        return False
    t = (text or "").lower()
    if is_remote_availability_request(t):
        return False
    for pat in AGENT_TV_FAILURE_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return True
    return False

# =============================================================================
# One-liner condenser (local heuristic) — DISABLED in this build
# =============================================================================
def summarize_one_liner_local(text: str, role: str) -> str:
    # Not used. LLM-only build keeps this stub for reference/testing.
    return ""

# =============================================================================
# HTML/CSV
# =============================================================================
def write_csv(rows: List[Dict[str,Any]], path: str):
    if not rows:
        with open(path,"w",newline="",encoding="utf-8") as f:
            f.write("id,subject,requester,org,priority,triage_priority_hint,triage_route,triage_ease,status,updated_at,category,easy_score,quick_reasons,summary,resolution_status,evidence,next_step,url,tags,origin,rule_name,rule_confidence,rule_why\n")
        return
    fields = list(rows[0].keys())
    with open(path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            r_out = r.copy()
            if isinstance(r_out.get("tags"), list):
                r_out["tags"] = ",".join(r_out["tags"])
            w.writerow(r_out)

def write_html(rows: List[Dict[str,Any]], path: str, title="Zendesk Ticket Summaries", categories: List[str] = None):
    cats = categories or DEFAULT_CATEGORIES
    head = """<!doctype html><html><head><meta charset="utf-8"><title>%%TITLE%%</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root { --bg: #0b1020; --panel: #121a2b; --muted: #9aa3b2; --text: #e5ecf3; --accent: #64d3ff; --accent2: #7c5cff; --chip: #1e2a44; --border: #25314a; }
* { box-sizing: border-box }
html,body { margin:0; padding:0; background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Arial; }
.header { position: sticky; top: 0; z-index: 5; background: linear-gradient(120deg, rgba(100,211,255,.2), rgba(124,92,255,.15) 50%, rgba(18,26,43,.8)); backdrop-filter: blur(6px); border-bottom: 1px solid var(--border); padding: 14px 18px; display:flex; align-items:center; gap:14px; }
.h-title { font-weight:700; letter-spacing:.3px; }
.h-badge { padding:3px 8px; border-radius:999px; background:rgba(100,211,255,.2); border:1px solid rgba(100,211,255,.35); font-size:12px; }
.toolbar { display:flex; gap:10px; align-items:center; margin-left:auto; }
.input, .select { background: var(--panel); color: var(--text); border:1px solid var(--border); padding: 8px 10px; border-radius:10px; outline:none; min-width: 180px; }
.input::placeholder { color: #94a3b8 }
.container { padding: 16px 18px 28px; }
.card { background: var(--panel); border:1px solid var(--border); border-radius: 16px; padding: 14px; margin: 10px 0; animation: fadeIn .4s ease both; }
@keyframes fadeIn { from { opacity:0; transform: translateY(6px) } to { opacity:1; transform:none } }
.row { display:grid; grid-template-columns: 92px 1fr 130px 120px 170px; gap:12px; align-items:start; }
.row + .row { margin-top:12px }
.id a { color: var(--accent); text-decoration:none }
.subject { font-weight:600 }
.meta { color: var(--muted); font-size: 12px }
.badges { display:flex; gap:6px; flex-wrap:wrap; }
.badge { background: var(--chip); border:1px solid var(--border); color: #cbd5e1; font-size: 12px; padding:3px 8px; border-radius:999px }
.reso { display:inline-flex; align-items:center; gap:6px; font-weight:600; }
.reso-dot { width:8px; height:8px; border-radius:50%; background:#666; box-shadow:0 0 0 3px rgba(255,255,255,.05) inset }
.summary { white-space: pre-wrap; line-height:1.5; }
.hl { color:#cbd5e1 }
.next-step { margin-top:6px; color:#e2e8f0 }
.footer { color: var(--muted); margin-top: 14px; font-size: 12px }
.controls { display:flex; gap:8px; flex-wrap:wrap; }
hr.sep { border:0; border-top:1px dashed var(--border); margin:8px 0 }
.small { font-size: 12px; color: #a5b4fc }
.details { margin-top:8px; }
details > summary { cursor: pointer; user-select:none; color:#a5b4fc }
.perline { display:flex; align-items:center; gap:8px; margin:6px 0; }
.chip-tv { background:#3b1d27; border:1px solid #ef4444; color:#fecaca; font-size:11px; padding:2px 6px; border-radius:999px }
.badge-src { background:#1e293b; border-color:#64748b; }
</style>
</head><body>

<div class="header">
  <div class="h-title">Zendesk Ticket Summaries</div>
  <span class="h-badge">%%STAMP%%</span>
  <div class="toolbar">
    <input id="q" class="input" placeholder="Search subject / summary…" />
    <select id="filterStatus" class="select">
      <option value="">Resolution: All</option>
      <option value="resolved">Resolved</option>
      <option value="waiting_on_customer">Waiting on Customer</option>
      <option value="waiting_on_us">Waiting on Us</option>
      <option value="needs_more_info">Needs More Info</option>
      <option value="unknown">Unknown</option>
    </select>
    <select id="filterCat" class="select"></select>
    <select id="sortBy" class="select">
      <option value="easy">Sort by Easy Score</option>
      <option value="updated">Sort by Updated</option>
      <option value="priority">Sort by Priority</option>
    </select>
  </div>
</div>

<div class="container" id="app"></div>

<script>
const DATA = /*__DATA__*/ [];
const CATS = /*__CATS__*/ [];
</script>
<script>
const rows = DATA;

const filterCat = document.getElementById('filterCat');
(function buildCats(){
  const cats = Array.from(new Set([].concat(CATS, rows.map(r=>r.category).filter(Boolean)))).sort();
  const blank = document.createElement('option'); blank.value=""; blank.textContent = "Category: All";
  filterCat.appendChild(blank);
  cats.forEach(c => { const o=document.createElement('option'); o.value=c; o.textContent=c; filterCat.appendChild(o); });
})();

const q = document.getElementById('q');
const filterStatus = document.getElementById('filterStatus');
const sortBy = document.getElementById('sortBy');
const app = document.getElementById('app');

function dot(color){ return `<span class="reso-dot" style="background:${color}"></span>` }
function resoBadge(st){
  const map = {resolved:"#16a34a", waiting_on_customer:"#b45309", waiting_on_us:"#dc2626", needs_more_info:"#4f46e5", unknown:"#6b7280"};
  const label = {resolved:"Resolved", waiting_on_customer:"Waiting on Customer", waiting_on_us:"Waiting on Us", needs_more_info:"Needs More Info", unknown:"Unknown"}[st] || st;
  return `<span class="reso">${dot(map[st]||"#6b7280")} ${label}</span>`;
}
function priorityBadge(p){
  const map = { high:"#ef4444", urgent:"#dc2626", normal:"#64748b", low:"#475569", "-":"#374151" };
  const label = (p||"-").toUpperCase();
  const color = map[(p||"-").toLowerCase()] || "#374151";
  return `<span class="badge" style="border-color:${color}; color:${color}">${label}</span>`;
}
function categoryBadge(c){ return `<span class="badge">${c||"Other"}</span>`; }
function triageBadge(name, val){ if(!val||val==='-') return ""; return `<span class="badge">${name}: ${val}</span>`; }
function escapeHtml(s){ return (s||"").replace(/[&<>\"']/g, m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;' }[m])); }

function perLineHTML(line){
  const tv = line.tv ? `<span class="chip-tv">Remote/TV issue</span>` : "";
  const src = `<span class="badge badge-src">LLM</span>`;
  return `<div class="perline"><div class="small" style="min-width:80px">${escapeHtml(line.role)}:</div><div>${escapeHtml(line.text||"")}</div>${tv} ${src}</div>`;
}

function render(){
  const term = (q.value||"").toLowerCase();
  const fs = filterStatus.value;
  const fc = filterCat.value;

  let items = rows.filter(r => {
    const hitsTerm = !term || (r.subject||"").toLowerCase().includes(term) || (r.summary||"").toLowerCase().includes(term);
    const hitsStatus = !fs || r.resolution_status === fs;
    const hitsCat = !fc || r.category === fc;
    return hitsTerm && hitsStatus && hitsCat;
  });

  const sortMode = sortBy.value;
  if (sortMode === "easy"){
    items.sort((a,b)=> (b.easy_score-a.easy_score) || (a.updated_at.localeCompare(b.updated_at)));
  } else if (sortMode === "updated"){
    items.sort((a,b)=> b.updated_at.localeCompare(a.updated_at));
  } else {
    const rank = {urgent:4, high:3, normal:2, low:1, "-":0};
    items.sort((a,b)=> (rank[(b.priority||"-")] - rank[(a.priority||"-")]) || b.updated_at.localeCompare(a.updated_at));
  }

  app.innerHTML = items.map(r => `
    <div class="card">
      <div class="row">
        <div class="id"><a href="${r.url}" target="_blank">#${r.id}</a><div class="meta">${(r.updated_at||"").replace('T',' ').replace('Z','')}</div></div>
        <div>
          <div class="subject">${escapeHtml(r.subject||"")}</div>
          <div class="summary">${escapeHtml(r.summary||"")}${r.evidence? `<br><span class="hl"><b>Evidence:</b> ${escapeHtml(r.evidence)}</span>`:""}</div>
          <div class="next-step"><b>Next step:</b> ${escapeHtml(r.next_step||"")}</div>
          <div class="small">origin=${escapeHtml(r.origin||"-")}${r.rule_name? ` • rule=${escapeHtml(r.rule_name)} (${(r.rule_confidence??"").toString()})`:""}${r.rule_why? ` • why=${escapeHtml(r.rule_why)}`:""}</div>

          ${r.per_lines && r.per_lines.length ? `
            <details class="details">
              <summary>Conversation (last ${r.per_lines.length}) — per-email one-liners</summary>
              <div style="margin-top:8px">
                ${r.per_lines.map(perLineHTML).join("")}
              </div>
            </details>` : ""}

          <div class="footer">${escapeHtml(r.requester||"-")} • ${escapeHtml(r.status||"-")}</div>
        </div>
        <div class="badges">
          ${priorityBadge(r.priority||"-")}
          ${r.triage_priority_hint? priorityBadge(r.triage_priority_hint):""}
        </div>
        <div class="badges">
          ${categoryBadge(r.category||"Other")}
          ${triageBadge("Route", r.triage_route)}
          ${triageBadge("Ease", r.triage_ease)}
          ${r.any_tv_issue? `<span class="badge" style="border-color:#ef4444;color:#ef9a9a">Remote/TV: issue noted</span>` : ""}
        </div>
        <div>${resoBadge(r.resolution_status||"unknown")}</div>
      </div>
    </div>
  `).join("");
}

[q, filterStatus, filterCat, sortBy].forEach(el=> el.addEventListener('input', render));
render();
</script>
</body></html>"""
    esc = lambda x: html.escape(str(x), quote=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    data_json = json.dumps(rows, ensure_ascii=False)
    cats_json = json.dumps(cats, ensure_ascii=False)
    html_out = head.replace("%%TITLE%%", esc(title)).replace("%%STAMP%%", esc(stamp))
    html_out = html_out.replace("/*__DATA__*/ []", data_json).replace("/*__CATS__*/ []", cats_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)

# ========================== PART SPLIT MARKER — CONTINUES IN PART 2 ==========================
# The rest (main() orchestration: fetch comments, build batched LLM one-liners, apply rules,
# ticket-level LLM summaries, write CSV/HTML, etc.) will be in Part 2.
# ========================== PART 2 (paste directly after Part 1) ==========================

# ---- Optional hotfix: patch a tiny typo in Part 1 (extra brace in tickets_assigned_to URL)
def _patch_zendesk_class():
    def _tickets_assigned_to(self, email: str, limit: int = 5000) -> List[Dict[str, Any]]:
        me = self.users_me(); uid = me.get("id")
        if email and email.lower() != (me.get("email","").lower()):
            u = self.user_by_email(email); uid = u.get("id") if u else uid
        if not uid: return []
        url = f"{self.base}/users/{uid}/tickets/assigned.json"
        out, page = [], 1
        while True:
            data = self._get(url, params={"page": page, "sort_by": "updated_at", "sort_order": "desc"})
            items = data.get("tickets", [])
            out.extend(items)
            if not data.get("next_page") or len(out) >= limit: break
            page += 1; time.sleep(0.05)
        out = [t for t in out if t.get("status") not in ("solved","closed")]
        return out[:limit]
    Zendesk.tickets_assigned_to = _tickets_assigned_to
_patch_zendesk_class()

# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    _log_setup(args.log_file)

    # --- Validate Zendesk creds
    if not args.subdomain or not args.email:
        sys.exit("ERROR: Missing Zendesk base creds. Set .env or pass --subdomain --email.")
    if not (args.api_token or args.password):
        sys.exit("ERROR: Missing Zendesk auth. Provide --api-token (preferred) or --password.")

    # --- Decide ticket-level LLM provider (one-liners are LLM-only in this build if enabled)
    provider = args.provider
    if provider == "auto":
        if args.openai:
            provider = "openai"
        elif args.openrouter:
            provider = "openrouter"
        else:
            provider = "none"

    # One-liners are LLM-only if enabled
    if args.llm_one_liners:
        if provider == "openai" and not args.openai:
            sys.exit("ERROR: --llm-one-liners requires OpenAI key (--openai) in this build.")
        if provider == "openrouter" and not args.openrouter:
            sys.exit("ERROR: --llm-one-liners requires OpenRouter key (--openrouter).")
        if provider == "none":
            sys.exit("ERROR: --llm-one-liners requires an LLM provider (OpenAI/OpenRouter).")

    # Load rules (dir > single file)
    rules, rule_defaults = load_rules_from_dir(args.rules_dir, debug=args.debug)
    if not rules:
        srules, sdefaults = load_rules_single_file(args.rules, debug=args.debug)
        if srules:
            rules, rule_defaults = srules, sdefaults
    if args.debug:
        _log(f"[DEBUG] Total rules loaded: {len(rules)} (defaults={rule_defaults})")

    rule_engine = SmartRuleEngine(rules, defaults=rule_defaults, debug=args.debug)

    # Fetch tickets
    zd = Zendesk(args.subdomain, args.email, args.password, args.api_token, debug=args.debug)

    _log("Fetching tickets…")
    if args.view_id:
        tickets = zd.tickets_in_view(args.view_id, limit=args.max)
        if args.debug: _log(f"[DEBUG] View #{args.view_id} returned {len(tickets)} tickets")
    else:
        statuses = [s.strip() for s in args.statuses.split(",") if s.strip()]
        tickets = zd.search_tickets(statuses, assignee_email=(args.assignee or None),
                                    updated_within_days=args.days, limit=args.max)
        if (not tickets) and args.assignee:
            if args.debug: _log("[DEBUG] Search empty; fallback to assigned endpoint")
            tickets = zd.tickets_assigned_to(args.assignee, limit=args.max)
        if (not tickets) and not args.assignee:
            if args.debug: _log("[DEBUG] Still empty; try 'me' via assigned endpoint")
            tickets = zd.tickets_assigned_to(args.email, limit=args.max)

    _log(f"Found {len(tickets)} ticket(s) to process.")
    if args.debug:
        for t in tickets:
            _log(
                f"[FETCH] id={t.get('id')} "
                f"status={t.get('status','-')} "
                f"priority={t.get('priority','-')} "
                f"updated={t.get('updated_at','-')} "
                f"subject={elide((t.get('subject') or '').strip(), 100)}"
            )

    # Early save if no tickets
    if not tickets:
        write_csv([], "zendesk_ticket_summaries.csv")
        write_html([], "zendesk_ticket_summaries.html", categories=DEFAULT_CATEGORIES)
        _log("\nSaved: zendesk_ticket_summaries.csv\nSaved: zendesk_ticket_summaries.html")
        if log_fp: log_fp.close()
        return

    # Parallel fetch comments + users
    user_cache: Dict[int,Dict[str,Any]] = {}
    user_cache_lock = threading.Lock()
    comments_map: Dict[int,List[Dict[str,Any]]] = {}

    def fetch_comments(t):
        tid = t["id"]
        try:
            comms = zd.ticket_comments(tid)
            comments_map[tid] = comms
            ids = list({c["author_id"] for c in comms})
            if ids:
                users = zd.users_show_many(ids)
                with user_cache_lock:
                    user_cache.update(users)
        except Exception as e:
            comments_map[tid] = []
            if args.debug: _log(f"[DEBUG] fetch_comments error #{tid}: {e}")

    _log("Fetching conversations in parallel…")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(as_completed([ex.submit(fetch_comments, t) for t in tickets]))

    def complexity_score(comms: List[Dict[str,Any]]) -> int:
        n = len(comms)
        tot = sum(len(clean_message_body(c.get("html_body") or c.get("plain_body") or "")) for c in comms)
        avg = tot // max(n,1)
        return n*2 + (avg>600)*2 + (avg>1200)*3

    cache = load_cache()

    # Instantiate LLM client for ticket-level JSON and one-liners
    llm_rate = RateLimiter(rate_per_minute=args.llm_rpm, burst=args.llm_burst)
    orc = None
    if provider == "openai" and args.openai:
        if args.debug: _log(f"[DEBUG] Using OpenAI: {args.openai_model}")
        orc = OpenAIClient(args.openai, args.openai_model, max_tokens=args.max_tokens,
                           debug=args.debug, use_responses=(not args.no_responses), rate_limiter=llm_rate)
    elif provider == "openrouter" and args.openrouter:
        if args.debug: _log(f"[DEBUG] Using OpenRouter: {args.model}")
        orc = OpenRouterClient(args.openrouter, args.model, max_tokens=args.max_tokens,
                               debug=args.debug, rate_limiter=llm_rate)
    else:
        if args.debug: _log("[DEBUG] No ticket-level LLM provider; ticket summaries may be fast-mode.")
        # still okay: one-liners may require LLM; but we guarded above when --llm-one-liners is set.

    # Prepare per-ticket work items
    work = []
    for t in tickets:
        tid = t["id"]; subject = (t.get("subject") or "").strip()
        priority = t.get("priority") or "-"
        status = t.get("status") or "-"
        updated_at = t.get("updated_at") or ""
        tags = t.get("tags", []); org_id = t.get("organization_id", "")
        comms = comments_map.get(tid, [])

        if not comms:
            work.append({
                "id": tid, "subject": subject, "requester": "-", "org": org_id, "priority": priority,
                "status": status, "updated_at": updated_at, "tags": tags,
                "excerpts": [], "last_customer_q": False, "has_attachments": False,
                "fast_summary": "No conversation yet.", "llm_decision": False, "per_lines": [], "any_tv_issue": False
            })
            continue

        requester_id = comms[0]["author_id"]
        requester_name = user_cache.get(requester_id, {}).get("name", "-")

        last_customer_q = False; has_att = False; excerpts = []
        per_lines_placeholders = []  # will fill after LLM batch
        any_tv_issue = False

        trimmed = comms[-args.max_comments:]
        for c in trimmed:
            role = "Customer" if c["author_id"] == requester_id else "Agent"
            body_raw = c.get("html_body") or c.get("plain_body") or ""
            body = clean_message_body(body_raw)
            body_one = re.sub(r"\s+"," ", body).strip()
            if not is_meaningful_text(body_one):
                continue
            if role == "Customer" and is_question(body_one): last_customer_q = True
            if c.get("attachments"): has_att = True
            excerpts.append(f"{role}: {elide(body_one, 420)}")

            tv_hit = detect_teamviewer_issue_role_aware(body_one, role)
            if tv_hit:
                any_tv_issue = True

            per_lines_placeholders.append({
                "comment_id": c.get("id") or f"{tid}:{hashlib.md5(body_one.encode('utf-8')).hexdigest()}",
                "role": role,
                "clean_body": body_one,
                "tv": tv_hit,
                "one_liner": ""  # to be filled by batched LLM
            })

        if not excerpts:
            excerpts = ["Agent: Conversation contained only boilerplate content."]
        fast_sum = extractive_summary(excerpts, max_lines=5) if excerpts else "No content."
        work.append({
            "id": tid, "subject": subject, "requester": requester_name, "org": org_id, "priority": priority,
            "status": status, "updated_at": updated_at, "tags": tags,
            "excerpts": excerpts, "last_customer_q": last_customer_q, "has_attachments": has_att,
            "fast_summary": fast_sum, "llm_decision": None, "placeholders": per_lines_placeholders,
            "any_tv_issue": any_tv_issue
        })

    # Decide ticket-level LLM usage
    if args.fast:
        for w in work: w["llm_decision"] = False
    elif args.smart:
        ranked = sorted(work, key=lambda w: complexity_score(comments_map.get(w["id"], [])), reverse=True)
        llm_ids = set([w["id"] for w in ranked[:args.llm_quota]])
        for w in work: w["llm_decision"] = (w["id"] in llm_ids)
    else:
        for w in work: w["llm_decision"] = True

    # ---- Build global batch for one-liners (LLM-only)
    if args.llm_one_liners:
        if orc is None:
            sys.exit("ERROR: --llm-one-liners requires a working LLM client.")
        to_send: List[Tuple[str,str,str,int]] = []  # (role, text, cache_key, idx_in_global)
        slots: List[Tuple[int,int]] = []            # (work_index, placeholder_index)

        for wi, w in enumerate(work):
            placeholders = w.get("placeholders", [])
            for pi, pl in enumerate(placeholders):
                txt = pl["clean_body"]
                if len(txt) < args.ol_min_len:
                    continue
                clipped = txt[:args.ol_max_chars]
                # cache key specific to comment, model, tokens, trunc
                cache_key = f"ol:{pl['comment_id']}:{getattr(orc,'model','na')}:{args.one_liner_tokens}:{args.ol_max_chars}"
                cached = cache.get(cache_key)
                if isinstance(cached, str) and cached:
                    pl["one_liner"] = cached
                    continue
                to_send.append((pl["role"], clipped, cache_key, len(slots)))
                slots.append((wi, pi))

        # Batch call
        if to_send:
            _log(f"Summarising one-liners via LLM in batches of {args.ol_batch} (items={len(to_send)})…")
            for start in range(0, len(to_send), args.ol_batch):
                chunk = to_send[start:start+args.ol_batch]
                items = [(r, text) for (r, text, _, _) in chunk]
                try:
                    if isinstance(orc, OpenAIClient):
                        outs = orc.summarise_one_liners_batch(items, max_tokens_each=args.one_liner_tokens)
                    else:
                        outs = orc.summarise_one_liners_batch(items, max_tokens_each=args.one_liner_tokens)
                except Exception as e:
                    if args.debug: _log(f"[DEBUG] one-liner batch error: {e}")
                    outs = ["" for _ in items]

                # assign + cache
                for (out, (role, text, cache_key, global_idx)) in zip(outs, chunk):
                    wi, pi = slots[global_idx]
                    try:
                        work[wi]["placeholders"][pi]["one_liner"] = out
                        cache[cache_key] = out
                    except Exception:
                        pass
                save_cache(cache)

    # ---- Finalise per ticket (apply ticket-level LLM, rules, scores, and pack UI per_lines)
    def finalise(w):
        tid = w["id"]; subject = w["subject"]; updated_at = w["updated_at"]
        last_q = w["last_customer_q"]; has_att = w["has_attachments"]; tags = w["tags"]

        summary = w["fast_summary"]; resolution_status="unknown"; evidence=""; topic="Other"; next_step=""
        origin = "fast"

        # Ticket-level LLM JSON if selected
        if w["llm_decision"] and orc is not None:
            key = f"json:{tid}:{updated_at}:{getattr(orc, 'model', 'na')}:{args.max_tokens}"
            obj = cache.get(key)
            if not obj:
                if args.debug: _log(f"[DEBUG] ▶ LLM.summarise #{tid}")
                try:
                    obj = orc.summarise_json("\n".join(w["excerpts"]), subject, tid)
                except Exception as e:
                    if args.debug:
                        _log(f"[DEBUG] ticket LLM error #{tid}: {e}")
                    obj = {"summary": summary, "resolution_status":"unknown","evidence":"","topic":"Other","next_step":""}
                cache[key] = obj; save_cache(cache)
            summary = obj.get("summary", summary) or summary
            resolution_status = obj.get("resolution_status","unknown") or "unknown"
            evidence = obj.get("evidence","") or ""
            topic = obj.get("topic","Other") or "Other"
            next_step = obj.get("next_step","") or suggested_next_step(topic)
            origin = "llm"
        else:
            # Fast inference for resolution only
            txt = " ".join(w["excerpts"]).lower()
            if any(k in txt for k in ["thanks it works","that fixed it","all good now","resolved now","issue fixed","sorted now"]):
                resolution_status = "resolved"
            elif any(k in txt for k in ["could you","can you","please provide","need the following","awaiting your","when you get a chance","please confirm","with your reply","awaiting response","waiting for your reply","need more info"]):
                resolution_status = "waiting_on_customer"
            elif any(k in txt for k in ["we will update","we are investigating","engineering looking","we’re working on","i have raised a bug","passed to engineering","internal ticket raised"]):
                resolution_status = "waiting_on_us"
            else:
                resolution_status = "unknown"
            topic = categorize(subject + " " + summary)
            next_step = suggested_next_step(topic)
            origin = "fast"

        # Build per_lines from placeholders (LLM origin badge handled in HTML)
        per_lines = []
        for pl in w.get("placeholders", []):
            per_lines.append({
                "role": pl["role"],
                "text": (pl["one_liner"] or "").strip(),
                "tv": pl["tv"]
            })

        # Apply external rules
        rule_hit = rule_engine.match(subject, w.get("excerpts", []))
        triage_route = "-"
        triage_ease = "-"
        triage_priority_hint = "-"
        rule_name = ""
        rule_confidence = ""
        rule_why = ""

        if rule_hit:
            if args.debug: _log(f"[DEBUG] Rule hit for #{tid}: {rule_hit['name']} ({rule_hit['confidence']}) — {rule_hit['why']}")
            topic     = rule_hit.get("category") or topic
            next_step = rule_hit.get("next_step") or next_step
            triage_route = rule_hit.get("route") or "-"
            triage_ease  = rule_hit.get("ease") or "-"
            triage_priority_hint = rule_hit.get("priority") or "-"
            rule_name = rule_hit.get("name") or ""
            rule_confidence = rule_hit.get("confidence")
            rule_why = rule_hit.get("why") or ""
            # Optional: if confidence low, keep LLM topic (already applied); nothing else needed here.

        # Final category & score
        cat = categorize(subject + " " + summary) if topic == "Other" else topic
        score, why = easy_score(n_msgs=len(comments_map.get(tid, [])),
                                last_q=last_q, has_att=has_att, cat=cat, tags=tags)
        if rule_hit:
            score = max(score, 6)
            why.append("rule_hit")

        return {
            "id": tid, "subject": subject, "requester": w.get("requester","-"), "org": w.get("org",""),
            "priority": w["priority"],
            "triage_priority_hint": triage_priority_hint,
            "triage_route": triage_route,
            "triage_ease": triage_ease,
            "status": w["status"], "updated_at": updated_at, "category": cat, "easy_score": score,
            "quick_reasons": ",".join(why), "summary": summary, "resolution_status": resolution_status,
            "evidence": evidence, "next_step": next_step or suggested_next_step(cat),
            "url": f"https://{args.subdomain}.zendesk.com/agent/tickets/{tid}",
            "tags": w["tags"],
            "origin": origin,
            "rule_name": rule_name,
            "rule_confidence": rule_confidence,
            "rule_why": rule_why,
            # UI extras
            "per_lines": per_lines,
            "any_tv_issue": w.get("any_tv_issue", False),
        }

    results = []
    jsonl_path = args.jsonl
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(finalise, w) for w in work]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result(); results.append(r)
                _log(f"[{i}/{len(work)}] #{r['id']} {r['category']} {r['resolution_status']} score={r['easy_score']} {elide(r['subject'],70)} origin={r['origin']} rule={r.get('rule_name','')}")
                if jsonl_path:
                    try:
                        with open(jsonl_path, "a", encoding="utf-8") as jf:
                            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
                    except Exception as e:
                        if args.debug: _log(f"[DEBUG] JSONL write error: {e}")
            except Exception as e:
                _log(f"[{i}/{len(work)}] error: {e}")

    # Sort: easy first then updated
    results.sort(key=lambda r: (-r["easy_score"], r["updated_at"]))

    write_csv(results, "zendesk_ticket_summaries.csv")
    # Category list for UI = defaults ∪ categories from rules
    rules_cats = sorted(set([d.get("category") for d in rules if isinstance(d,dict) and d.get("category")]))
    ui_cats = sorted(set(DEFAULT_CATEGORIES + rules_cats))
    write_html(results, "zendesk_ticket_summaries.html", categories=ui_cats)

    _log("\nTop easy candidates:")
    for r in results[:10]:
        _log(f"- #{r['id']} [{r['easy_score']}] {r['category']} / {r['resolution_status']}: {elide(r['subject'],80)} → {r['url']}")
    _log("\nSaved: zendesk_ticket_summaries.csv\nSaved: zendesk_ticket_summaries.html")
    if log_fp: log_fp.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
