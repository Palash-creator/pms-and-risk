# Portfolio Analytics Workbench

Executive AI Copilot + Real-Time Data + Risk Intelligence.

## Overview
The Portfolio Analytics Workbench is a four-page Streamlit application that blends institutional-grade analytics with an embedded AI copilot. It guides portfolio teams from data ingestion through risk diagnostics while keeping conversations grounded in the live portfolio context.

## App Architecture
- **UI Layer (Streamlit tabs & components)** → orchestrates navigation, KPI cards, and context-aware chat.
- **Data Providers (Polygon → yfinance → Synthetic)** → supply price history with resilient fallbacks and cached ingestion.
- **Portfolio Engine (modules/portfolio.py)** → manages portfolio state, dry-run generator, and concentration math.
- **Metrics Engine (modules/metrics.py)** → computes return series, KPIs, risk statistics, and chart data.
- **Chat Adapter (modules/chat.py)** → builds the system prompt and routes queries to Groq or Gemini endpoints.

| Module | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application wiring all four pages, logging, and session state. |
| `modules/ui.py` | Reusable UI helpers for KPI cards, layout styling, and small badges. |
| `modules/data.py` | Data ingestion orchestration with Polygon, yfinance, and synthetic fallbacks plus caching. |
| `modules/portfolio.py` | `PortfolioState` management, dry-run portfolio builder, and concentration metrics. |
| `modules/metrics.py` | Return calculations, KPI summaries, equity curves, and risk/diagnostic computations. |
| `modules/chat.py` | System prompt builder and provider-specific chat adapters (Groq/Gemini). |

## Installation
```bash
git clone <repo_url>
cd four_page_app
pip install -r requirements.txt
```

## Configuration
1. Create `.streamlit/secrets.toml` (or export environment variables) with:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   GEMINI_API_KEY = "your_gemini_key"
   POLYGON_API_KEY = "your_polygon_key"
   ```
2. (Optional) Add `POLYGON_API_KEY` to access live Polygon market data. Without it, the app falls back to yfinance or synthetic data.
3. Ensure outbound HTTPS access to Polygon, Groq, and Gemini endpoints.

## Running the App
```bash
streamlit run app.py
```
The app launches with a dark theme and four tabs: **Summary & Copilot**, **Data Ingestion & Builder**, **Portfolio Outlook**, and **Risk & Diagnostics**.

## Verification & Testing Workflow
Follow this phased plan to validate the full experience:

### Phase 1 — Setup Validation
- **Purpose:** Ensure environment prerequisites and connectivity.
- **Commands & Actions:**
  - `python --version` (expect ≥ 3.11).
  - `pip install -r requirements.txt` (dependencies install without error).
  - Configure secrets or environment variables for `GROQ_API_KEY`, `GEMINI_API_KEY`, `POLYGON_API_KEY`.
  - `curl https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-02?apiKey=$POLYGON_API_KEY` (expect HTTP 200 when key valid).
  - `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"` (expect HTTP 200).
  - `curl https://generativelanguage.googleapis.com/v1/models -H "x-goog-api-key: $GEMINI_API_KEY"` (expect HTTP 200).
  - `streamlit run app.py` (expect four tabs rendered with dark theme) while tailing logs via `tail -f logs/app.log` (log file auto-created with rolling entries).

### Phase 2 — Data Ingestion Tests (Page 2)
- **Purpose:** Confirm provider fallbacks, caching, and parallel ingestion.
- **Commands & Actions:**
  - With Polygon key set, ingest S&P 100 / Daily / 3Y; expect OHLC data with provider logged as Polygon.
  - Temporarily unset `POLYGON_API_KEY` and rerun ingestion; expect warning toast and logs indicating yfinance fallback.
  - Disable network (e.g., `sudo iptables -A OUTPUT -p tcp --dport 443 -j REJECT` locally or use offline mode) to trigger synthetic generator; verify synthetic price patterns and sector metadata.
  - Observe progress bar advancing per ticker and logs capturing each fetch.
  - Ingest ≥10 tickers and confirm completion under ~5s on local hardware (parallel threads ≤8).
  - Validate `st.session_state["universe"]` and `st.session_state["prices"]` populated via Streamlit inspection (`st.write`).

### Phase 3 — Portfolio Builder Tests (Page 2)
- **Purpose:** Verify portfolio construction utilities.
- **Commands & Actions:**
  - Click **Dry Run** to generate ~10 assets (2×5 sectors) and confirm weight sum equals 1.000 (via displayed table and log entry).
  - Use manual multiselect and weight inputs to craft a custom allocation; inspect resulting JSON preview.
  - Adjust weights and confirm concentration metrics (HHI, ENH, Top3/5/10%) update immediately.
  - Navigate to other tabs and back to ensure `PortfolioState` persists within session state.

### Phase 4 — Portfolio Outlook & Fund Summary (Page 3)
- **Purpose:** Validate return analytics and visualization.
- **Commands & Actions:**
  - Toggle period filters (1Y/3Y/5Y/YTD) and ensure KPIs recalculate.
  - Confirm KPI cards display percentage formatting and tooltips.
  - Select a benchmark series to overlay on the Plotly equity curve; verify legend entries for portfolio vs. benchmark.
  - Export summary JSON (if provided) and cross-check values with displayed KPIs.
  - Ensure period switching refreshes chart and KPI cards without delay.

### Phase 5 — Risk Metrics & Diagnostics (Page 4)
- **Purpose:** Exercise risk analytics and plots.
- **Commands & Actions:**
  - Review Sharpe, Sortino, Max Drawdown, Historical VaR(95%), Parametric VaR(95%), and Expected Shortfall(95%); confirm no calculation errors.
  - Inspect correlation matrix DataFrame and Plotly heatmap.
  - Verify returns histogram with VaR marker and normal PDF overlay render.
  - Confirm concentration table (HHI, ENH, Top3/5/10%) aligns with portfolio weights.
  - Use CSV export option and confirm downloaded file contents.

### Phase 6 — AI Copilot (Page 1)
- **Purpose:** Validate context-aware chat behavior across providers.
- **Commands & Actions:**
  - Inspect system prompt via debug print (if enabled) or logs to confirm summary/portfolio JSON embedded.
  - Query with Groq provider: “What is my portfolio’s 3Y CAGR?” and verify response matches displayed metric.
  - Ask an out-of-context question (“Who is the CEO of Apple?”) and confirm refusal referencing instructions.
  - Switch to Gemini provider and repeat tests; observe response latency (<3s for Groq ideal).
  - Trigger rate-limit scenario by sending ≥8 queries quickly; ensure warning banner displayed.

### Phase 7 — Logging & Error Recovery
- **Purpose:** Confirm observability and resilience.
- **Commands & Actions:**
  - Review `logs/app.log` for INFO entries with timestamps, durations, provider notes.
  - Induce an invalid ticker or simulate API timeout (e.g., revoke network mid-ingestion); verify fallback provider engages and UI warning surfaces.
  - Ensure no secrets are printed in logs or UI.

### Phase 8 — Performance & Cache
- **Purpose:** Measure caching efficiency and parallel stability.
- **Commands & Actions:**
  - Re-run identical ingestion parameters; observe cache hit via logs and reduced load time.
  - Compare initial ingestion time vs. cached retrieval (e.g., using Streamlit status output or manual timing).
  - Confirm ThreadPoolExecutor respects `max_workers=8` and no thread exhaustion occurs.

## Maintenance Tips
- Update `requirements.txt` when upstream APIs change.
- Monitor API quotas and adjust rate limiting in `modules/data.py` as needed.
- Extend chat adapters cautiously; never log API keys or raw prompts containing secrets.

---
© 2024 Portfolio Analytics Workbench. All rights reserved.
