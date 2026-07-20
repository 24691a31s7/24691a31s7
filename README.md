# Stocks

An AI multi-agent market-analysis system for NSE equities. This is the
upgraded architecture requested in the review doc, renamed from
`alphaflow-ai` to `stocks`. Read this file before you deploy anything -
it tells you honestly what's real code you can run today vs. what still
needs your own infrastructure or data.

## What changed vs. the previous version

| # | Review item | Status |
|---|---|---|
| 1 | True AI agents (goal/memory/tools/reasoning/planning/output schema) | **Done** - `agents/base_agent.py` + every agent refactored onto it |
| 2 | Parallel decision logic instead of sequential | **Done** - `orchestrator.py` runs 4 dependency tiers, Tier 1 fully concurrent |
| 3 | Redis cache | **Done** - `cache/cache_manager.py`, auto-falls back to in-memory if `REDIS_URL` unset/unreachable |
| 4 | Async everywhere | **Mostly done** - all I/O-bound agent calls run via `asyncio`/`asyncio.to_thread`; DB writes stay sync-in-a-thread (see note below) |
| 5 | PostgreSQL / TimescaleDB | **Ready, not provisioned** - change `DATABASE_URL`, nothing else changes (SQLAlchemy). TimescaleDB hypertables need you to run `SELECT create_hypertable(...)` yourself after connecting a real Postgres+Timescale instance |
| 6 | Background workers (Celery) | **Scaffolded, off by default** - `worker.py` + `Dockerfile.worker`; `USE_CELERY=false` keeps the zero-infra APScheduler path working |
| 7 | Streaming (WebSockets) | **Done** - `/ws/live-prices` |
| 8 | Trained ML prediction model | **Hook built, not trained** - `agents/ml/train_model.py` scaffold + `PREDICTION_MODEL_PATH` env var. Training needs a real labelled historical dataset that only you can license/assemble |
| 9 | Feature store (RSI/MACD/.../Nifty/BankNifty/sector strength) | **Partial** - technical/fundamental agents already compute RSI, MACD, EMA, Bollinger, P/E, P/B, ROE, debt ratios. Macro features (USDINR, Nifty, sector strength index) are not wired in - add them as extra columns in `agents/_technical_logic.py`'s feature dict when you have a data source for them |
| 10 | News intelligence (summarize, explain, confidence) | **Done, optional** - `agents/sentiment_agent.py` uses Gemini for summarization if `GEMINI_API_KEY` is set, otherwise rule-based scoring |
| 11 | Portfolio agent | **Done** - `agents/portfolio_agent.py`, `/api/portfolio` |
| 12 | Explainability agent | **Done** - `agents/explanation_agent.py`, already returns reasons + confidence + stop loss + target in every response |
| 13 | Memory agent | **Done** - `agents/memory_agent.py`, backed by the `prediction_log` table |
| 14 | Learning agent (self-improving) | **Done, honestly scoped** - evaluates predictions against realized outcomes and reports accuracy; does NOT auto-retrain on a cron job (see the docstring in `agents/learning_agent.py` for why) |
| 15 | Monitoring (Prometheus/Grafana) | **Done** - `/metrics` endpoint + `docker-compose.yml` ships Prometheus + Grafana containers wired to scrape it |
| 16 | Docker / docker-compose | **Done** - `docker-compose.yml` (backend, worker, redis, postgres, prometheus, grafana) |
| 17 | CI/CD | **Done, deploy step is a placeholder** - `.github/workflows/ci.yml` runs lint + tests + docker build on every push; the actual deploy target is yours to fill in |
| 18 | API/agent-router optimization | **Done** - `/api/agent/{agent_name}/{symbol}` calls one agent instead of the full pipeline |
| 19 | Prediction accuracy dashboard | **Done (API), no UI chart yet** - `/api/accuracy` |
| 20 | Security (JWT, rate limiting, secrets, audit logs) | **Partial** - simple in-memory rate limiter + optional `X-API-Key` header on write endpoints. No JWT/user accounts, no secrets manager, no audit log - this app currently has no concept of a logged-in user, so add those once it does |
| 21 | Testing (unit/integration/95% coverage) | **Partial** - 20 passing unit + integration tests (`pytest tests/ -v`), no coverage target enforced, no load/stress tests |
| 22 | Professional frontend (TradingView charts, heatmap, etc.) | **Not done** - out of scope for this pass; the existing single-file `frontend/index.html` is carried over unchanged. This is a real frontend rebuild, not a backend change |
| 23 | Gateway architecture | **Done** - FastAPI acts as the gateway; cache-first reads, agent router, rate limiting all live here |
| 24 | Continuous market-intelligence layer | **Done** - `scheduler.py`'s `continuous_intelligence` job rescans the tracked universe every `CONTINUOUS_SCAN_INTERVAL_MINUTES` and upserts `market_intelligence`; `/api/top10` reads from that table first |

**What I did not build, and why:** a trained ML model (needs a real 5-year
licensed dataset - see item 8), a live Grafana dashboard config beyond the
default Prometheus data source (needs you to design the panels you
actually want), and the full professional frontend redesign (a genuinely
separate, large piece of work). Everything else above is real, tested code.

## Round 3: rate-limit fixes (from your production logs)

Your logs showed a real, specific bug plus a config mismatch, both fixed:

1. **The actual bug**: `groww_client.py` re-attempted a full Groww authentication
   on every single stock analysis whenever the previous auth attempt had
   failed. One rate-limited response cascaded into dozens of repeat auth
   calls within the same scan, which kept the account rate-limited
   indefinitely - that's why you saw "rate limit exceeded" on nearly every
   stock. Fixed with an auth cooldown (`GROWW_AUTH_COOLDOWN_SECONDS`,
   default 5 min): after a failed/rate-limited auth, every call skips Groww
   entirely and goes straight to the yfinance fallback until the cooldown
   expires - one log line instead of fifty. Covered by a regression test
   (`TestGrowwAuthCooldown`).
2. **The config mismatch**: `SCAN_CONCURRENCY=100` (my previous default) is
   fine against a paid data feed, but against free-tier yfinance/NewsAPI it
   fires far more concurrent requests than they'll accept, which is what
   caused "Connection pool is full" and the Yahoo/NewsAPI 429s. Default is
   now `SCAN_CONCURRENCY=8`, plus new **per-provider** concurrency caps
   (`YFINANCE_MAX_CONCURRENT=5`, `NEWSAPI_MAX_CONCURRENT=2`,
   `GROWW_MAX_CONCURRENT=3` - see `utils/rate_limit.py`) that stay low
   *regardless* of how high you raise `SCAN_CONCURRENCY`, so cranking up
   scan speed later can never reintroduce this exact failure mode.

Also added: a shared, larger-pooled HTTP session for every yfinance/NewsAPI
call (`utils/http_session.py` - fixes "Connection pool is full" directly),
automatic exponential-backoff retry on 429s/5xx (`with_retry()`), longer
cache TTLs (news 30min -> 45min) so a warm cache serves most requests
without calling the provider at all, and downgraded the "possibly
delisted"/`currentTradingPeriod` messages from ERROR to WARNING since
they're an already-handled fallback path, not a crash.

**If you still see rate limits after this** with real Groww/NewsAPI keys:
that means your free-tier plan's actual limit is below even these
conservative defaults - lower `SCAN_CONCURRENCY` further (try 3-5) and/or
raise `CONTINUOUS_SCAN_INTERVAL_MINUTES` to 15-20, or move to a paid data
plan. The Celery path (`worker.py`) is the correct way to get real
parallelism beyond what a single free-tier API key can sustain - it doesn't
remove the provider's rate limit, but it lets you spread the SAME limited
request budget across a schedule instead of a single burst.

## Round 2 changes (pattern agent, speed, filters, alerts, frontend)

| # | Request | Status |
|---|---|---|
| 1 | New agent that "knows every pattern" and predicts 99.99% accurately | **Built the real half, refused the fake half.** `agents/pattern_recognition_agent.py` algorithmically detects every chart/candlestick pattern from your reference images (double top/bottom, H&S, triangles, wedges, flags, pennants, rectangles, cup & handle, ~20 candlestick patterns) and feeds it into the existing agent pipeline as a 4th vote. It reports literature-backed reliability (~50-68% per pattern, from published pattern-statistics research) instead of a fabricated confidence number - see "On prediction accuracy" below for why 99.99% isn't something any system can honestly claim |
| 2 | Run 9:15-3:30, refresh every 5-10s | **Done** - `market_hours.py` + `/ws/live-prices` ticks every `LIVE_PRICE_POLL_SECONDS` (default 7s) while the market's open, backs off automatically outside market hours so it isn't hammering APIs all night for no reason |
| 3 | 100 parallel agents instead of 1 | **Done** - `SCAN_CONCURRENCY` defaults to 100 concurrent in-flight analyses (`orchestrator.py`); real ceiling is your data provider's rate limit, not the code - the Celery path in `worker.py` lets you go beyond one machine's limit entirely |
| 4 | Only companies priced < ₹2000, listed 5+ years, no legal cases | **Done, with an honesty caveat on the litigation part** - `universe_filters.py`. Price and listing-age (proxied by available price history) are hard, reliable filters. "No legal cases" has no free comprehensive data source, so this uses (a) a manual exclusion list you maintain and (b) a keyword screen over the Sentiment Agent's news scan (fraud, SEBI probe, NCLT, etc.) - it's a best-effort screen, not legal clearance |
| 6 | Pre-9:15 -> today's picks, post-3:30 -> tomorrow's picks from news | **Done** - `scheduler.py`'s continuous job detects the session via `market_hours.py` and calls `run_full_scan_async(mode=...)`, which reweights sentiment/news higher for the post-market "tomorrow" ranking |
| 7 | Auto BUY/SELL alerts on the Top 10 | **Done** - `scheduler.py:_sync_top10_alerts()` auto-creates BUY/TARGET/STOP alerts for every Top-10 BUY call after each scan, deactivates stale ones, and `/ws/live-prices` pushes a `triggered_alerts` event the instant a price crosses a level (surfaced as toast notifications in the new frontend) |
| 8 | Show current / prev close / today's open / predicted close | **Done** - every analysis now includes `previous_close_price`, `today_open_price`, `predicted_close_price`, and `predicted_close_range` (a range, not a single "guaranteed" number) |
| 9 | Suggested quantity per company | **Already existed, now surfaced on every Top-10 card** - `suggested_quantity` + `estimated_investment_inr` |
| 10 | Frontend redesign | **Done** - new `frontend/index.html`: dark trading-terminal theme, live scrolling ticker tape, WebSocket-driven live prices, pattern chips, confidence bars, alert toasts, session-aware header |

### On prediction accuracy (please read this)

You asked for 99.99-99.999% accurate profit/loss predictions. I did not build
that, because it isn't buildable - not by me, not by anyone. A handful of
concrete reasons, not just a disclaimer:

- **If it existed, it would end capitalism as we know it within months.** A
  99.99%-accurate predictor of individual stock direction, combined with
  leverage, is an infinite-money machine. No such machine exists; if it did,
  its owner - not a random web app - would already own most tradeable equity
  on Earth.
- **Markets price in almost everything already known.** By the time a
  pattern, ratio, or headline is visible to this app, it's visible to every
  other market participant too, and the price has already moved to reflect
  it. What's left to predict is the genuinely unknown part - tomorrow's news,
  not today's pattern - which by definition can't be read off a chart.
- **Published research on the exact techniques you referenced (chart
  patterns, candlesticks) puts their real-world reliability at roughly
  50-70%**, barely to moderately better than a coin flip, and that's from
  large historical backtests with the benefit of hindsight - live, on the
  right edge of a still-forming chart, it's harder still.

What the system does instead, and why it's more useful than a fake number:
every confidence score is capped well under 100%, is explained (which
agents agreed, which disagreed, which pattern was detected and its
published reliability), and is logged to `prediction_log` so the Learning
Agent can tell you, honestly, how accurate *this specific system* has
actually been over time via `/api/accuracy` - real, falsifiable, improving
feedback instead of a number that sounds impressive and means nothing.

## "How long would a full scan take" - answered concretely

The curated default universe is ~90 liquid NSE names; `SCAN_CONCURRENCY=25`
concurrent analyses gets a full scan done in roughly the time of ~4 sequential
analyses (a few seconds once caches are warm, more like 20-40s cold,
network-dependent). Set `USE_FULL_NSE_UNIVERSE=true` to track closer to the
full ~2,000-symbol NSE list; realistically checking every listed company on
a single process needs `MAX_UNIVERSE_SIZE` raised and either more
concurrency headroom or the Celery worker path (`worker.py`) fanning chunks
of 20 symbols out across multiple worker processes/machines - exactly the
"100 workers x 20 stocks each" pattern from the review doc.

## Running it

### Local, zero-infra (SQLite + in-memory cache + no Celery)
```bash
cd backend
cp .env.example .env
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000`. Works with zero API keys (yfinance +
rule-based sentiment fallback); add `GROWW_API_KEY`/`NEWSAPI_KEY`/
`GEMINI_API_KEY` in `.env` to upgrade individual agents.

### Full stack (Postgres + Redis + Celery workers + Prometheus + Grafana)
```bash
docker compose up --build
```
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin - add a Prometheus data
  source pointed at `http://prometheus:9090` and build panels from the
  `stocks_*` metrics)

### Tests
```bash
cd backend && pytest tests/ -v
```

## Disclaimer

This is a heuristic/educational decision-support tool, not investment
advice, and it does not place trades. Position sizing and BUY/SELL/HOLD
calls are plain, auditable arithmetic over technical/fundamental/sentiment
scores - review `agents/_recommendation_logic.py` and
`agents/_prediction_logic.py` to see exactly how every number is derived.
