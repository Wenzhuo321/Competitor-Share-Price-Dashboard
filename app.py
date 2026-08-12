import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import

import asyncio
import base64
import io
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = Path(__file__).parent

# ── Constants (verbatim from shareprice_Analysis_final.ipynb) ─────────────────

TICKERS = {
    "Oracle":     "ORCL",
    "Nasdaq":     "^IXIC",
    "ServiceNow": "NOW",
    "DAX":        "^GDAXI",
    "SAP":        "SAP",
    "Salesforce": "CRM",
    "Workday":    "WDAY",
}

COLORS = {
    "Oracle":     "#FB0512",
    "Nasdaq":     "#0C5F75",
    "ServiceNow": "#00A8A8",
    "DAX":        "#70AD47",
    "SAP":        "#0070F2",
    "Salesforce": "#C815AC",
    "Workday":    "#F0AB00",
}

STYLES = {
    "Oracle":     {"lw": 1.8, "ls": "-"},
    "Nasdaq":     {"lw": 1.8, "ls": "--"},
    "ServiceNow": {"lw": 1.5, "ls": "-"},
    "DAX":        {"lw": 1.5, "ls": "--"},
    "SAP":        {"lw": 1.5, "ls": "-"},
    "Salesforce": {"lw": 1.5, "ls": "-"},
    "Workday":    {"lw": 1.5, "ls": "-"},
}

LOGO_FILES = {
    "Oracle":     "oracle_logo.png",
    "Nasdaq":     "nasdaq_logo.png",
    "ServiceNow": "servicenow_logo.png",
    "DAX":        "dax_logo.png",
    "SAP":        "sap_logo.png",
    "Salesforce": "salesforce_logo.png",
    "Workday":    "workday_logo.png",
}

LOGO_SCALE_OVERRIDES = {
    "DAX":        0.45,
    "SAP":        0.55,
    "Salesforce": 0.69,
    "Workday":    0.67,
}

VERT_MARGIN_PX   = 1
TEXT_FONTSIZE_PT = 9.5
LOGO_REL_HEIGHT  = 0.08
LOGO_MAX_SCALE_UP = 1.0

# Chart 1: 2020 to today
C1_PARAMS = {
    "subplot_left":    0.10,
    "subplot_right":   0.88,
    "subplot_top":     0.89,
    "subplot_bottom":  0.09,
    "right_pad_frac":  0.15,
    "pct_offset_frac": 0.020,
    "logo_offset_frac": 0.070,
    "figsize":         (17, 3.1),
    "start_xlim":      "2020-01-01",
    "tick_freq":       "YS",
    "tick_fmt":        "Jan %Y",
    "tick_start":      "2020-01-01",
    "tick_end":        None,  # computed at build time
}

# Chart 2: YTD (Jan 1 current year to today)
C2_PARAMS = {
    "subplot_left":    0.08,
    "subplot_right":   0.875,
    "subplot_top":     0.89,
    "subplot_bottom":  0.09,
    "right_pad_frac":  0.15,
    "pct_offset_frac": 0.020,
    "logo_offset_frac": 0.070,
    "figsize":         (17, 3.1),
    "start_xlim":      None,  # computed at build time (Jan 1 current year)
    "tick_freq":       "MS",
    "tick_fmt":        "%b %Y",
    "tick_start":      None,  # computed at build time
    "tick_end":        None,
}

CACHE_TTL = 3600  # seconds

# ── In-memory cache ───────────────────────────────────────────────────────────

_cache: dict = {
    "perf_2020":  None,
    "prices_2020": None,
    "perf_ytd":   None,
    "prices_ytd": None,
    "updated_at": None,
}
_cache_lock = threading.Lock()
_chart_lock = threading.Lock()  # matplotlib is not thread-safe


# ── Data functions ────────────────────────────────────────────────────────────

def fetch_all(tickers_dict: dict, start: str, end: str) -> dict:
    """Download all tickers in one batch request to minimise rate-limit exposure."""
    import time
    symbols = list(tickers_dict.values())
    labels  = list(tickers_dict.keys())
    for attempt in range(3):
        try:
            df = yf.download(
                symbols, start=start, end=end,
                interval="1d", auto_adjust=False, progress=False,
                group_by="ticker"
            )
            raw = {}
            for label, sym in zip(labels, symbols):
                try:
                    if len(symbols) == 1:
                        s = df.get("Close")
                    else:
                        s = df[sym]["Close"] if sym in df.columns.get_level_values(0) else None
                    if s is None or (hasattr(s, "empty") and s.empty):
                        raw[label] = pd.Series(dtype=float)
                        continue
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    raw[label] = s.resample("W-FRI").last().ffill()
                except Exception:
                    raw[label] = pd.Series(dtype=float)
            return raw
        except Exception:
            if attempt < 2:
                time.sleep(10 + attempt * 10)
    return {label: pd.Series(dtype=float) for label in labels}


def _build_prices_perf(start: str, end: str):
    raw = fetch_all(TICKERS, start, end)

    # Check all tickers succeeded — if any missing, abort and keep stale cache
    missing = [label for label, s in raw.items() if s.empty]
    if missing:
        raise RuntimeError(f"Missing data for: {missing} — keeping previous cache")

    # Convert DAX from EUR to USD using weekly EURUSD=X rate
    eurusd_raw = fetch_all({"EURUSD": "EURUSD=X"}, start, end)
    eurusd = eurusd_raw.get("EURUSD", pd.Series(dtype=float))
    if not eurusd.empty:
        fx = eurusd.reindex(raw["DAX"].index).ffill().bfill()
        raw["DAX"] = raw["DAX"] * fx

    prices = pd.concat(list(raw.values()), axis=1, keys=list(raw.keys())).sort_index()
    prices.columns = prices.columns.get_level_values(0)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    valid = prices.dropna(thresh=1)
    if valid.empty:
        raise RuntimeError("No valid price data after cleaning")
    base_date = valid.index[0]
    perf = (prices / prices.loc[base_date] - 1) * 100
    return prices, perf


def _cache_is_stale() -> bool:
    if _cache["updated_at"] is None:
        return True
    age = (datetime.now(timezone.utc) - _cache["updated_at"]).total_seconds()
    return age > CACHE_TTL


def refresh_cache() -> None:
    with _cache_lock:
        if not _cache_is_stale():
            return
        today      = datetime.now().strftime("%Y-%m-%d")
        year_start = datetime.now().strftime("%Y-01-01")
        try:
            prices_2020, perf_2020 = _build_prices_perf("2020-01-01", today)
            prices_ytd,  perf_ytd  = _build_prices_perf(year_start, today)
            _cache["prices_2020"] = prices_2020
            _cache["perf_2020"]   = perf_2020
            _cache["prices_ytd"]  = prices_ytd
            _cache["perf_ytd"]    = perf_ytd
            _cache["updated_at"]  = datetime.now(timezone.utc)
        except Exception as e:
            # Keep serving stale cache rather than crashing
            print(f"Cache refresh failed (will retry on next request): {e}")


# ── Chart building ────────────────────────────────────────────────────────────

def load_logo_best(path: Path, target_h_pix: int, max_scale_up: float = 1.0):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    allowed_h = int(round(min(h * max_scale_up, target_h_pix)))
    final_h = max(1, allowed_h)
    if final_h != h:
        new_w = int(round(w * final_h / h))
        img = img.resize((new_w, final_h), Image.LANCZOS)
        w, h = new_w, final_h
    return np.array(img), h, w


def collision_avoid(adj_pix: list, item_heights: list, y_min_pix: float, y_max_pix: float) -> list:
    adj = list(adj_pix)
    for _ in range(200):
        moved = False
        for i in range(1, len(adj)):
            prev_h = item_heights[i - 1]
            curr_h = item_heights[i]
            min_gap = (prev_h / 2) + (curr_h / 2) + VERT_MARGIN_PX
            overlap = min_gap - (adj[i - 1] - adj[i])
            if overlap > 0.5:
                adj[i - 1] += overlap / 2
                adj[i]     -= overlap / 2
                moved = True
        if adj[0] + item_heights[0] / 2 > y_max_pix:
            adj[0] = y_max_pix - item_heights[0] / 2
            moved = True
        if adj[-1] - item_heights[-1] / 2 < y_min_pix:
            shift = y_min_pix - (adj[-1] - item_heights[-1] / 2)
            adj = [p + shift for p in adj]
            moved = True
        if not moved:
            break
    return adj


def build_chart(perf: pd.DataFrame, prices: pd.DataFrame, params: dict) -> str:
    with _chart_lock:
        today_str = datetime.now().strftime("%B %d, %Y")
        start_dt  = pd.Timestamp(params["start_xlim"] or prices.index[0])
        start_str = params.get("start_str") or start_dt.strftime("%B %d, %Y")

        fig, ax = plt.subplots(figsize=params["figsize"], dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        plt.subplots_adjust(
            left=params["subplot_left"],   right=params["subplot_right"],
            top=params["subplot_top"],     bottom=params["subplot_bottom"],
        )

        # Plot lines
        for label in TICKERS:
            if label in perf:
                ax.plot(
                    perf[label].index, perf[label].values,
                    color=COLORS[label],
                    linewidth=STYLES[label]["lw"],
                    linestyle=STYLES[label]["ls"],
                    zorder=3,
                )

        # X-axis ticks
        tick_start = pd.Timestamp(params["tick_start"] or prices.index[0])
        tick_end   = pd.Timestamp(params["tick_end"] or prices.index[-1])
        tick_dates = pd.date_range(tick_start, tick_end, freq=params["tick_freq"])
        ax.set_xticks(tick_dates)
        ax.set_xticklabels([d.strftime(params["tick_fmt"]) for d in tick_dates])

        # Measure pixel width for fraction-based offsets
        ax.set_xlim(left=start_dt, right=prices.index[-1])
        fig.canvas.draw()
        axis_width_px = ax.get_window_extent().width
        total_days    = (prices.index[-1] - start_dt).days or 1
        px_per_day    = axis_width_px / total_days

        def frac_to_td(frac):
            return pd.Timedelta(days=(axis_width_px * frac) / px_per_day)

        RIGHT_PAD   = frac_to_td(params["right_pad_frac"])
        PCT_OFFSET  = frac_to_td(params["pct_offset_frac"])
        LOGO_OFFSET = frac_to_td(params["logo_offset_frac"])

        ax.set_xlim(left=start_dt, right=prices.index[-1] + RIGHT_PAD)
        ax.plot(
            [ax.get_xlim()[0], mdates.date2num(prices.index[-1])], [0, 0],
            color="#cccccc", lw=0.8, zorder=1,
        )

        ax.tick_params(axis="x", labelsize=9, length=4)
        ax.yaxis.set_visible(False)
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["bottom"].set_bounds(ax.get_xlim()[0], mdates.date2num(prices.index[-1]))
        ax.grid(axis="x", color="#eeeeee", lw=0.5)
        ax.margins(y=0.01)

        # Load logos
        fig.canvas.draw()
        fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi
        target_logo_h = int(round(fig_h_px * LOGO_REL_HEIGHT))

        logos, logo_heights, logo_widths = {}, {}, {}
        for label, fname in LOGO_FILES.items():
            path = BASE_DIR / fname
            if path.exists():
                scale = LOGO_SCALE_OVERRIDES.get(label, 1.0)
                arr, fh, fw = load_logo_best(path, int(target_logo_h * scale), LOGO_MAX_SCALE_UP)
                logos[label]        = arr
                logo_heights[label] = fh
                logo_widths[label]  = fw

        # Collision avoidance
        final_vals    = {lbl: perf[lbl].dropna().iloc[-1] for lbl in perf.columns}
        sorted_labels = sorted(final_vals, key=final_vals.get, reverse=True)
        logo_x        = prices.index[-1] + LOGO_OFFSET
        PCT_X         = prices.index[-1] + PCT_OFFSET

        last_valid_x = {}
        for lbl in sorted_labels:
            s = perf[lbl].dropna()
            last_valid_x[lbl] = s.index[-1] if len(s) else prices.index[-1]

        def data_y_to_pix(y):
            return ax.transData.transform((mdates.date2num(logo_x), y))[1]

        def pix_to_data_y(pix):
            return ax.transData.inverted().transform((mdates.date2num(logo_x), pix))[1]

        text_px = int(round(fig.dpi * TEXT_FONTSIZE_PT / 72.0))
        ax_bbox = ax.get_window_extent()
        item_heights = [logo_heights.get(lbl, text_px + 4) for lbl in sorted_labels]
        adj_pix = [data_y_to_pix(final_vals[lbl]) for lbl in sorted_labels]
        adj_pix = collision_avoid(adj_pix, item_heights, ax_bbox.y0, ax_bbox.y1)
        positions = {lbl: pix_to_data_y(adj_pix[i]) for i, lbl in enumerate(sorted_labels)}

        # Percentage labels + leader lines
        for label in sorted_labels:
            val   = final_vals[label]
            y_adj = positions[label]
            y_raw = val
            sign  = "+" if val >= 0 else ""

            if abs(y_adj - y_raw) > 1.0:
                ax.annotate(
                    "", xy=(last_valid_x[label], y_raw), xytext=(PCT_X, y_adj),
                    arrowprops=dict(
                        arrowstyle="-", color=COLORS[label],
                        lw=0.6, linestyle=(0, (3, 3)),
                        connectionstyle="arc3,rad=0.0",
                    ),
                    annotation_clip=False, zorder=2,
                )

            ax.annotate(
                f"{sign}{val:.0f}%", xy=(PCT_X, y_adj),
                fontsize=TEXT_FONTSIZE_PT, color=COLORS[label],
                va="center", ha="left", fontweight="bold",
                annotation_clip=False, zorder=9,
            )

        # Logo placement
        fig.canvas.draw()
        for label in sorted_labels:
            if label not in logos:
                continue
            img_arr = logos[label]
            y_data  = positions[label]
            x_disp, y_disp = ax.transData.transform((mdates.date2num(logo_x), y_data))
            h_px, w_px = img_arr.shape[0], img_arr.shape[1]
            fig_w_px2, fig_h_px2 = fig.get_size_inches() * fig.dpi
            logo_ax = fig.add_axes(
                [x_disp / fig_w_px2, (y_disp - h_px / 2) / fig_h_px2,
                 w_px / fig_w_px2, h_px / fig_h_px2],
                zorder=10,
            )
            logo_ax.imshow(img_arr, aspect="auto", interpolation="none")
            logo_ax.axis("off")

        # Title — matches notebook exactly (0.88 / 0.82, within subplot_top=0.89)
        fig.text(0.10, 0.88, "Share Price Performance",
                 fontsize=14, fontweight="bold", va="top", ha="left")
        fig.text(0.10, 0.82, f"{start_str} – {today_str}",
                 fontsize=10, color="#555555", va="top", ha="left")

        # Encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches=None)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# ── Helpers for /api/data ─────────────────────────────────────────────────────

def serialize_prices(prices: pd.DataFrame) -> dict:
    result = {}
    for col in prices.columns:
        s = prices[col].dropna()
        result[col] = [
            {"date": idx.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for idx, v in s.items()
        ]
    return result


def serialize_perf(perf: pd.DataFrame) -> dict:
    result = {}
    for col in perf.columns:
        s = perf[col].dropna()
        result[col] = [
            {"date": idx.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for idx, v in s.items()
        ]
    return result


def get_final_perf(perf_2020: pd.DataFrame, perf_ytd: pd.DataFrame) -> dict:
    result = {}
    for company in TICKERS:
        s_2020 = perf_2020[company].dropna() if company in perf_2020 else pd.Series()
        s_ytd  = perf_ytd[company].dropna()  if company in perf_ytd  else pd.Series()
        result[company] = {
            "since_2020": round(float(s_2020.iloc[-1]), 1) if len(s_2020) else None,
            "ytd":        round(float(s_ytd.iloc[-1]),  1) if len(s_ytd)  else None,
        }
    return result


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm cache in background so the server responds immediately on first request
    asyncio.create_task(asyncio.to_thread(refresh_cache))
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/charts")
async def get_charts():
    await asyncio.to_thread(refresh_cache)

    today = datetime.now()
    year_start_str = today.strftime("%Y-01-01")

    c1 = dict(C1_PARAMS)
    c1["tick_end"]   = f"{today.year}-01-01"   # last Jan-1 tick (e.g. "2026-01-01")
    c1["start_str"]  = "January 01, 2020"      # fixed title start date

    c2 = dict(C2_PARAMS)
    c2["start_xlim"] = year_start_str
    c2["tick_start"] = year_start_str
    c2["tick_end"]   = today.replace(day=1).strftime("%Y-%m-%d")  # first of current month

    chart1, chart2 = await asyncio.gather(
        asyncio.to_thread(build_chart, _cache["perf_2020"], _cache["prices_2020"], c1),
        asyncio.to_thread(build_chart, _cache["perf_ytd"],  _cache["prices_ytd"],  c2),
    )

    return JSONResponse({
        "chart_2020": chart1,
        "chart_ytd":  chart2,
        "updated_at": _cache["updated_at"].isoformat(),
    })


@app.get("/api/data")
async def get_data():
    await asyncio.to_thread(refresh_cache)

    # Encode logos as base64 for the frontend legend
    logos_b64 = {}
    for label, fname in LOGO_FILES.items():
        path = BASE_DIR / fname
        if path.exists():
            logos_b64[label] = base64.b64encode(path.read_bytes()).decode("utf-8")

    return JSONResponse({
        "perf_2020":    serialize_perf(_cache["perf_2020"]),
        "perf_ytd":     serialize_perf(_cache["perf_ytd"]),
        "prices_2020":  serialize_prices(_cache["prices_2020"]),
        "prices_ytd":   serialize_prices(_cache["prices_ytd"]),
        "final_perf":   get_final_perf(_cache["perf_2020"], _cache["perf_ytd"]),
        "colors":       COLORS,
        "logos":        logos_b64,
        "updated_at":   _cache["updated_at"].isoformat(),
    })
