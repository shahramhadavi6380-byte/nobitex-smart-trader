import os
import json
import time
import math
import logging
import html

import requests
import numpy as np
import pandas as pd


# ============================================================
# NOBITEX SMART TRADER V6
# Trend + Support/Resistance + Volume + EMA/RSI/MACD
# Multi-timeframe analysis
# No automatic trading
# ============================================================

BASE = "https://apiv2.nobitex.ir"

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

STATE_FILE = "bot_state.json"
STATE_VERSION = 8

# ------------------------------------------------------------
# SCANNER SETTINGS
# ------------------------------------------------------------

TOP_PER_QUOTE = 15
MAX_WATCH = 5

MIN_WATCH_SCORE = 60
BUY_SETUP_SCORE = 80

SCAN_EVERY_SECONDS = 4 * 60 * 60

TIMEFRAMES = (15, 60, 240)

CANDLE_COUNT = 350

REQUEST_PAUSE = 0.45
HTTP_TIMEOUT = 25

TELEGRAM_CHUNK_SIZE = 3800

PRICE_MISMATCH_THRESHOLD = 0.15

# ------------------------------------------------------------
# STRATEGY SETTINGS
# ------------------------------------------------------------

VOLUME_LOOKBACK = 20

MIN_VOLUME_RATIO = 1.5

MAX_SPREAD_PCT = 1.2

MIN_RISK_REWARD = 2.0

MAX_STOP_DISTANCE_PCT = 8.0

# تخمین محافظه‌کارانه هزینه رفت و برگشت معامله
# این عدد باید در صورت مشخص بودن کارمزد واقعی تنظیم شود.
ESTIMATED_ROUND_TRIP_COST_PCT = 0.50

SUPPORT_RESISTANCE_LOOKBACK = 100

PIVOT_LEFT = 3
PIVOT_RIGHT = 3

SUPPORT_RESISTANCE_TOLERANCE = 0.008

# ------------------------------------------------------------
# SCORING WEIGHTS
# ------------------------------------------------------------

# مجموع امتیاز خام = 85

WEIGHT_TREND = 25
WEIGHT_SR = 25
WEIGHT_VOLUME = 20

WEIGHT_EMA = 5
WEIGHT_RSI = 5
WEIGHT_MACD = 5

RAW_MAX_SCORE = (
    WEIGHT_TREND
    + WEIGHT_SR
    + WEIGHT_VOLUME
    + WEIGHT_EMA
    + WEIGHT_RSI
    + WEIGHT_MACD
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("NobitexSmartTrader")

S = requests.Session()

S.headers.update({
    "User-Agent": "NobitexSmartTrader/6.0",
    "Accept": "application/json"
})


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def now_ts():
    return int(time.time())


def num(value, default=float("nan")):

    try:

        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except Exception:

        return default


def clamp(value, low, high):

    return max(low, min(high, value))


def fmt(value):

    value = num(value)

    if not math.isfinite(value):
        return "نامشخص"

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    if abs(value) >= 1:
        return f"{value:,.5f}"

    return f"{value:.8f}"


def esc(value):

    return html.escape(
        str(value),
        quote=True
    )


def line_separator():

    return "━━━━━━━━━━━━━━━━━━━━"


def score_bar(score):

    score = clamp(
        num(score, 0),
        0,
        100
    )

    filled = int(round(score / 10))

    return (
        "🟩" * filled
        + "⬜" * (10 - filled)
    )


def score_label(score):

    if score >= 85:
        return "بسیار قوی"

    if score >= 75:
        return "قوی"

    if score >= 65:
        return "نسبتاً مناسب"

    if score >= 60:
        return "متوسط"

    return "ضعیف"


def quote_label(quote):

    if quote == "rls":
        return "بازار تومانی"

    if quote == "usdt":
        return "بازار تتری"

    return quote


# ============================================================
# TELEGRAM
# ============================================================

def telegram(message):

    if not BOT_TOKEN or not CHAT_ID:

        raise RuntimeError(
            "BOT_TOKEN or CHAT_ID is missing"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    lines = message.split("\n")

    chunks = []
    current = ""

    for line in lines:

        candidate = (
            current + "\n" + line
            if current
            else line
        )

        if len(candidate) <= TELEGRAM_CHUNK_SIZE:

            current = candidate

        else:

            if current:
                chunks.append(current)

            current = line

    if current:
        chunks.append(current)

    for chunk in chunks:

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=HTTP_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):

            raise RuntimeError(
                "Telegram error: " + str(result)
            )

        time.sleep(1)


# ============================================================
# API
# ============================================================

def api_get(path, params=None, attempts=4):

    for attempt in range(attempts):

        try:

            response = S.get(
                BASE + path,
                params=params,
                timeout=HTTP_TIMEOUT
            )

            if response.status_code == 429:

                wait = min(
                    30,
                    3 * (attempt + 1)
                )

                log.warning(
                    "Rate limited. Waiting %s seconds",
                    wait
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

            payload = response.json()

            if (
                isinstance(payload, dict)
                and payload.get("status") == "failed"
            ):

                time.sleep(
                    2 * (attempt + 1)
                )

                continue

            return payload

        except Exception as error:

            log.warning(
                "API error %s: %s",
                path,
                error
            )

            time.sleep(
                2 * (attempt + 1)
            )

    return None


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():

    try:

        with open(
            STATE_FILE,
            encoding="utf-8"
        ) as file:

            state = json.load(file)

        if isinstance(state, dict):

            state.setdefault("last_scan", 0)
            state.setdefault("watchlist", [])
            state.setdefault("last_monitor", 0)

            return state

    except Exception as error:

        log.warning(
            "State load error: %s",
            error
        )

    return {

        "version": STATE_VERSION,

        "last_scan": 0,

        "watchlist": [],

        "last_monitor": 0

    }


def save_state(state):

    state["version"] = STATE_VERSION

    temporary_file = STATE_FILE + ".tmp"

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temporary_file,
        STATE_FILE
    )


# ============================================================
# MARKET SYMBOL
# ============================================================

def symbol_to_market_key(symbol):

    symbol = symbol.upper()

    if symbol.endswith("IRT"):

        return (
            symbol[:-3].lower()
            + "-rls"
        )

    if symbol.endswith("USDT"):

        return (
            symbol[:-4].lower()
            + "-usdt"
        )

    return ""


def symbol_base(symbol):

    symbol = symbol.upper()

    if symbol.endswith("IRT"):
        return symbol[:-3]

    if symbol.endswith("USDT"):
        return symbol[:-4]

    return symbol


# ============================================================
# GET MARKETS
# ============================================================

def get_markets():

    payload = api_get("/market/stats")

    if (
        not payload
        or not isinstance(
            payload.get("stats"),
            dict
        )
    ):

        raise RuntimeError(
            "دریافت بازارهای نوبیتکس ناموفق بود"
        )

    markets = []

    for key, item in payload["stats"].items():

        if not isinstance(item, dict):
            continue

        parts = key.lower().split("-")

        if len(parts) != 2:
            continue

        base, quote = parts

        if quote not in ("rls", "usdt"):
            continue

        if str(
            item.get("isClosed", False)
        ).lower() == "true":

            continue

        price = num(
            item.get("latest"),
            0
        )

        if price <= 0:
            continue

        symbol = (
            base.upper()
            + (
                "IRT"
                if quote == "rls"
                else "USDT"
            )
        )

        markets.append({

            "symbol": symbol,

            "market_key": key.lower(),

            "quote": quote,

            "price": price,

            "volume": num(
                item.get("volumeDst"),
                0
            ),

            "change24": num(
                item.get("dayChange"),
                0
            )

        })

    return markets


# ============================================================
# GET ALL PRICES
# ============================================================

def get_all_prices():

    payload = api_get("/market/stats")

    if (
        not payload
        or not isinstance(
            payload.get("stats"),
            dict
        )
    ):

        return {}

    prices = {}

    for key, item in payload["stats"].items():

        if not isinstance(item, dict):
            continue

        if str(
            item.get("isClosed", False)
        ).lower() == "true":

            continue

        price = num(
            item.get("latest"),
            0
        )

        if price > 0:

            prices[key.lower()] = price

    return prices


# ============================================================
# CANDLE DATA
# ============================================================

def candles(symbol, resolution):

    current_time = now_ts()

    payload = api_get(
        "/market/udf/history",
        {
            "symbol": symbol,
            "resolution": str(resolution),
            "to": current_time,
            "countback": CANDLE_COUNT
        }
    )

    if (
        not payload
        or payload.get("s") != "ok"
    ):

        return None

    try:

        df = pd.DataFrame({

            "time": payload["t"],

            "open": payload["o"],

            "high": payload["h"],

            "low": payload["l"],

            "close": payload["c"],

            "volume": payload.get(
                "v",
                [0] * len(payload["t"])
            )

        })

        for column in (

            "time",
            "open",
            "high",
            "low",
            "close",
            "volume"

        ):

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = (
            df.dropna()
            .sort_values("time")
            .drop_duplicates("time")
        )

        # حذف کندل باز

        df = df[
            df.time + resolution * 60
            <= current_time
        ]

        if len(df) < 220:
            return None

        last_closed = int(
            df.time.iloc[-1]
            + resolution * 60
        )

        if (
            current_time - last_closed
            > resolution * 60 * 2
        ):

            return None

        return df.reset_index(drop=True)

    except Exception as error:

        log.warning(
            "Candle error %s: %s",
            symbol,
            error
        )

        return None


# ============================================================
# INDICATORS
# ONLY EMA20 / EMA50 / RSI14 / MACD
# ============================================================

def indicators(df):

    d = df.copy()

    close = d["close"]

    # EMA

    d["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    d["ema50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI 14

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(0, np.nan)
    )

    d["rsi"] = (
        100 - 100 / (1 + rs)
    )

    d.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    d.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "rsi"
    ] = 50

    # MACD

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = ema12 - ema26

    d["macd_signal"] = d["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d["macd"] - d["macd_signal"]
    )

    return d


# ============================================================
# BASIC ANALYSIS
# ============================================================

def analyze(df):

    d = indicators(df)

    row = d.iloc[-1]

    previous = d.iloc[-2]

    keys = (

        "close",
        "open",
        "high",
        "low",
        "volume",
        "ema20",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist"

    )

    values = {

        key: num(row[key])

        for key in keys

    }

    values["previous_close"] = num(
        previous["close"]
    )

    values["previous_ema20"] = num(
        previous["ema20"]
    )

    values["previous_rsi"] = num(
        previous["rsi"]
    )

    values["previous_macd_hist"] = num(
        previous["macd_hist"]
    )

    required = (

        "close",
        "open",
        "high",
        "low",
        "volume",
        "ema20",
        "ema50",
        "rsi",
        "macd",
        "macd_signal"

    )

    if not all(
        math.isfinite(values[key])
        for key in required
    ):

        return None

    # Relative volume
    # Current candle compared with previous 20 candles

    previous_volumes = (
        d["volume"]
        .iloc[-VOLUME_LOOKBACK - 1:-1]
    )

    avg_volume = num(
        previous_volumes.mean(),
        0
    )

    volume_ratio = 0

    if avg_volume > 0:

        volume_ratio = (
            values["volume"] / avg_volume
        )

    values["avg_volume"] = avg_volume

    values["volume_ratio"] = round(
        volume_ratio,
        2
    )

    values["bullish_candle"] = (
        values["close"] > values["open"]
    )

    values["ema_bullish"] = (
        values["ema20"] > values["ema50"]
    )

    values["macd_bullish"] = (
        values["macd"] > values["macd_signal"]
    )

    values["rsi_bullish"] = (
        50 <= values["rsi"] <= 68
    )

    values["last_candle_time"] = int(
        df.time.iloc[-1]
    )

    return values


# ============================================================
# SUPPORT / RESISTANCE
# LOCAL PIVOT DETECTION
# ============================================================

def find_support_resistance(df, price):

    recent = df.tail(
        SUPPORT_RESISTANCE_LOOKBACK
    ).reset_index(drop=True)

    highs = recent["high"].values

    lows = recent["low"].values

    pivot_highs = []

    pivot_lows = []

    for i in range(
        PIVOT_LEFT,
        len(recent) - PIVOT_RIGHT
    ):

        current_high = highs[i]

        left_highs = highs[
            i - PIVOT_LEFT:i
        ]

        right_highs = highs[
            i + 1:i + 1 + PIVOT_RIGHT
        ]

        if (
            current_high >= max(left_highs)
            and current_high > max(right_highs)
        ):

            pivot_highs.append(
                float(current_high)
            )

        current_low = lows[i]

        left_lows = lows[
            i - PIVOT_LEFT:i
        ]

        right_lows = lows[
            i + 1:i + 1 + PIVOT_RIGHT
        ]

        if (
            current_low <= min(left_lows)
            and current_low < min(right_lows)
        ):

            pivot_lows.append(
                float(current_low)
            )

    # اگر پیوت کافی نبود، از سقف و کف اخیر استفاده می‌کنیم.

    if not pivot_highs:

        pivot_highs = [
            float(recent.high.max())
        ]

    if not pivot_lows:

        pivot_lows = [
            float(recent.low.min())
        ]

    supports = sorted(
        [
            level
            for level in pivot_lows
            if level < price
        ],
        reverse=True
    )

    resistances = sorted(
        [
            level
            for level in pivot_highs
            if level > price
        ]
    )

    # نزدیک‌ترین حمایت و مقاومت

    support = (
        supports[0]
        if supports
        else float(recent.low.min())
    )

    resistance = (
        resistances[0]
        if resistances
        else float(recent.high.max())
    )

    # فاصله قیمت از سطوح

    support_distance = (
        (price - support) / price * 100
        if price > 0
        else float("nan")
    )

    resistance_distance = (
        (resistance - price) / price * 100
        if price > 0
        else float("nan")
    )

    return {

        "support": support,

        "resistance": resistance,

        "support_distance": support_distance,

        "resistance_distance": resistance_distance,

        "pivot_high_count": len(pivot_highs),

        "pivot_low_count": len(pivot_lows)

    }


# ============================================================
# TREND ANALYSIS
# ============================================================

def trend_analysis(a4, a1):

    score = 0

    reasons = []

    # 4H primary trend

    if a4["close"] > a4["ema20"]:

        score += 8

        reasons.append(
            "قیمت ۴ساعته بالای EMA20"
        )

    if a4["ema20"] > a4["ema50"]:

        score += 9

        reasons.append(
            "EMA20 بالای EMA50 در ۴ساعته"
        )

    # 1H trend confirmation

    if a1["close"] > a1["ema20"]:

        score += 4

        reasons.append(
            "قیمت یک‌ساعته بالای EMA20"
        )

    if a1["ema20"] > a1["ema50"]:

        score += 4

        reasons.append(
            "EMA20 بالای EMA50 در یک‌ساعته"
        )

    return {

        "score": min(
            WEIGHT_TREND,
            score
        ),

        "reasons": reasons,

        "bullish_4h": (
            a4["close"] > a4["ema20"]
            and a4["ema20"] > a4["ema50"]
        ),

        "bullish_1h": (
            a1["close"] > a1["ema20"]
            and a1["ema20"] > a1["ema50"]
        )

    }


# ============================================================
# VOLUME SCORE
# ============================================================

def volume_score(a15):

    ratio = a15["volume_ratio"]

    if ratio >= 2.5:
        score = 20

    elif ratio >= 2.0:
        score = 18

    elif ratio >= 1.5:
        score = 15

    elif ratio >= 1.2:
        score = 10

    elif ratio >= 0.8:
        score = 6

    elif ratio >= 0.5:
        score = 3

    else:
        score = 0

    return score


# ============================================================
# INDICATOR SCORE
# EMA / RSI / MACD
# ============================================================

def indicator_score(a15):

    score = 0

    details = {}

    # EMA

    if a15["ema20"] > a15["ema50"]:

        score += WEIGHT_EMA

        details["ema"] = True

    else:

        details["ema"] = False

    # RSI

    if 50 <= a15["rsi"] <= 68:

        score += WEIGHT_RSI

        details["rsi"] = True

    else:

        details["rsi"] = False

    # MACD

    if a15["macd"] > a15["macd_signal"]:

        score += WEIGHT_MACD

        details["macd"] = True

    else:

        details["macd"] = False

    return score, details


# ============================================================
# SUPPORT / RESISTANCE SCORE
# ============================================================

def sr_score(a15, sr):

    price = a15["close"]

    support_distance = sr["support_distance"]

    resistance_distance = sr["resistance_distance"]

    score = 0

    reasons = []

    # Confirmed breakout:
    # Previous closed candle below resistance
    # Current closed candle above resistance

    breakout = (

        a15["previous_close"] <= sr["resistance"]

        and price > sr["resistance"]

        and a15["bullish_candle"]

    )

    if breakout:

        score += 25

        reasons.append(
            "شکست مقاومت با بسته‌شدن کندل"
        )

    elif (
        math.isfinite(support_distance)
        and support_distance <= 1.5
        and support_distance >= 0
    ):

        score += 20

        reasons.append(
            "قیمت نزدیک حمایت"
        )

    elif (
        math.isfinite(resistance_distance)
        and resistance_distance <= 1.5
        and resistance_distance >= 0
    ):

        score += 12

        reasons.append(
            "قیمت نزدیک مقاومت"
        )

    elif (
        math.isfinite(support_distance)
        and support_distance <= 4
        and support_distance >= 0
    ):

        score += 10

        reasons.append(
            "حمایت در فاصله قابل بررسی"
        )

    else:

        score += 4

    return {

        "score": min(
            WEIGHT_SR,
            score
        ),

        "breakout": breakout,

        "reasons": reasons

    }


# ============================================================
# SPREAD
# ============================================================

def spread_pct(symbol):

    payload = api_get(
        f"/v3/orderbook/{symbol}",
        attempts=2
    )

    try:

        if not isinstance(payload, dict):
            return None

        data = payload.get(
            "data",
            payload
        )

        bids = data.get("bids")

        asks = data.get("asks")

        if not bids or not asks:
            return None

        first_bid = bids[0]

        first_ask = asks[0]

        if isinstance(
            first_bid,
            (list, tuple)
        ):

            bid = num(first_bid[0])

        else:

            bid = num(
                first_bid.get("price")
            )

        if isinstance(
            first_ask,
            (list, tuple)
        ):

            ask = num(first_ask[0])

        else:

            ask = num(
                first_ask.get("price")
            )

        if bid <= 0 or ask <= bid:
            return None

        return (
            100
            * (ask - bid)
            / ((ask + bid) / 2)
        )

    except Exception:

        return None


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_trade_levels(
    entry,
    support
):

    if entry <= 0:
        return None

    # Stop slightly below support

    if support > 0 and support < entry:

        stop = support * 0.995

    else:

        stop = entry * 0.97

    risk_distance = entry - stop

    risk_pct = (
        risk_distance / entry * 100
    )

    if risk_distance <= 0:
        return None

    if risk_pct > MAX_STOP_DISTANCE_PCT:

        return None

    estimated_cost = (
        entry
        * ESTIMATED_ROUND_TRIP_COST_PCT
        / 100
    )

    # اهداف با لحاظ هزینه تخمینی معامله

    target1 = (
        entry
        + 2.2 * risk_distance
        + estimated_cost
    )

    target2 = (
        entry
        + 3.2 * risk_distance
        + estimated_cost
    )

    net_reward = (
        target1 - entry - estimated_cost
    )

    effective_risk = (
        risk_distance + estimated_cost
    )

    rr = (
        net_reward / effective_risk
        if effective_risk > 0
        else 0
    )

    return {

        "entry": entry,

        "sl": stop,

        "tp1": target1,

        "tp2": target2,

        "risk_pct": risk_pct,

        "rr": rr,

        "estimated_cost_pct": (
            ESTIMATED_ROUND_TRIP_COST_PCT
        )

    }


# ============================================================
# BUILD CANDIDATE
# ============================================================

def build_candidate(
    market,
    analyses,
    spread
):

    a15 = analyses[15]

    a1 = analyses[60]

    a4 = analyses[240]

    price = a15["close"]

    # --------------------------------------------------------
    # 1. TREND
    # --------------------------------------------------------

    trend = trend_analysis(
        a4,
        a1
    )

    # --------------------------------------------------------
    # 2. SUPPORT / RESISTANCE
    # --------------------------------------------------------

    sr = find_support_resistance(
        analyses["df15"],
        price
    )

    sr_result = sr_score(
        a15,
        sr
    )

    # --------------------------------------------------------
    # 3. VOLUME
    # --------------------------------------------------------

    v_score = volume_score(a15)

    # --------------------------------------------------------
    # 4. EMA / RSI / MACD
    # --------------------------------------------------------

    i_score, i_details = indicator_score(
        a15
    )

    # --------------------------------------------------------
    # RAW SCORE: 85
    # NORMALIZED SCORE: 100
    # --------------------------------------------------------

    raw_score = (

        trend["score"]

        + sr_result["score"]

        + v_score

        + i_score

    )

    final_score = (
        raw_score / RAW_MAX_SCORE * 100
    )

    final_score = round(
        clamp(final_score, 0, 100),
        1
    )

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if sr_result["breakout"]:

        entry_low = sr["resistance"] * 0.998

        entry_high = sr["resistance"] * 1.005

    elif (
        sr["support"] < price
        and sr["support_distance"] <= 1.5
    ):

        entry_low = sr["support"]

        entry_high = sr["support"] * 1.01

    else:

        entry_low = price * 0.995

        entry_high = price * 1.005

    # --------------------------------------------------------
    # RISK CALCULATION
    # --------------------------------------------------------

    entry_reference = price

    risk = calculate_trade_levels(
        entry_reference,
        sr["support"]
    )

    if risk is None:

        risk = {

            "entry": entry_reference,

            "sl": entry_reference * 0.97,

            "tp1": entry_reference * 1.066,

            "tp2": entry_reference * 1.096,

            "risk_pct": 3.0,

            "rr": 2.0,

            "estimated_cost_pct": (
                ESTIMATED_ROUND_TRIP_COST_PCT
            )

        }

    # --------------------------------------------------------
    # BUY CONDITIONS
    # --------------------------------------------------------

    trend_ok = (

        trend["bullish_4h"]

        and trend["bullish_1h"]

    )

    volume_ok = (
        a15["volume_ratio"] >= MIN_VOLUME_RATIO
    )

    indicators_ok = (

        a15["ema20"] > a15["ema50"]

        and 50 <= a15["rsi"] <= 68

        and a15["macd"] > a15["macd_signal"]

    )

    breakout_ok = (
        sr_result["breakout"]
    )

    spread_ok = (

        spread is not None

        and spread <= MAX_SPREAD_PCT

    )

    rr_ok = (
        risk["rr"] >= MIN_RISK_REWARD
    )

    not_overextended = (

        price <= a15["ema20"] * 1.035

    )

    buy_conditions = all([

        final_score >= BUY_SETUP_SCORE,

        trend_ok,

        volume_ok,

        indicators_ok,

        breakout_ok,

        spread_ok,

        rr_ok,

        not_overextended

    ])

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if buy_conditions:

        classification = "BUY"

        status = "BUY_SETUP"

    elif final_score >= MIN_WATCH_SCORE:

        classification = "WATCH"

        status = "WAITING"

    else:

        classification = "NO_TRADE"

        status = "NO_TRADE"

    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    reasons = []

    reasons.extend(
        trend["reasons"]
    )

    reasons.extend(
        sr_result["reasons"]
    )

    if volume_ok:

        reasons.append(
            "حجم معاملات بالاتر از میانگین"
        )

    if a15["ema20"] > a15["ema50"]:

        reasons.append(
            "EMA20 بالای EMA50"
        )

    if 50 <= a15["rsi"] <= 68:

        reasons.append(
            "RSI در محدوده مناسب"
        )

    if a15["macd"] > a15["macd_signal"]:

        reasons.append(
            "MACD صعودی"
        )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "symbol": market["symbol"],

        "quote": market["quote"],

        "volume": market["volume"],

        "change24": market["change24"],

        "score": final_score,

        "raw_score": raw_score,

        "rank": final_score,

        "classification": classification,

        "status": status,

        "spread": spread,

        "price_at_scan": price,

        "entry_low": entry_low,

        "entry_high": entry_high,

        "entry": risk["entry"],

        "sl": risk["sl"],

        "tp1": risk["tp1"],

        "tp2": risk["tp2"],

        "risk_pct": risk["risk_pct"],

        "rr": risk["rr"],

        "estimated_cost_pct": (
            risk["estimated_cost_pct"]
        ),

        "support": sr["support"],

        "resistance": sr["resistance"],

        "support_distance": (
            sr["support_distance"]
        ),

        "resistance_distance": (
            sr["resistance_distance"]
        ),

        "breakout": breakout_ok,

        "volume_ratio": (
            a15["volume_ratio"]
        ),

        "scores": {

            "trend": trend["score"],

            "sr": sr_result["score"],

            "volume": v_score,

            "indicators": i_score,

            "15m": 0,

            "1h": 0,

            "4h": 0

        },

        "rsi": {

            "15m": a15["rsi"],

            "1h": a1["rsi"],

            "4h": a4["rsi"]

        },

        "ema": {

            "15m": (
                a15["ema20"] > a15["ema50"]
            ),

            "1h": (
                a1["ema20"] > a1["ema50"]
            ),

            "4h": (
                a4["ema20"] > a4["ema50"]
            )

        },

        "macd": {

            "15m": (
                a15["macd"] > a15["macd_signal"]
            ),

            "1h": (
                a1["macd"] > a1["macd_signal"]
            ),

            "4h": (
                a4["macd"] > a4["macd_signal"]
            )

        },

        "trend_ok": trend_ok,

        "volume_ok": volume_ok,

        "indicators_ok": indicators_ok,

        "spread_ok": spread_ok,

        "rr_ok": rr_ok,

        "reasons": reasons,

        "created": now_ts(),

        "alerts": [],

        "price_mismatch": False

    }


# ============================================================
# ANALYZE MARKET
# ============================================================

def analyze_market(market):

    analyses = {}

    dataframes = {}

    for timeframe in TIMEFRAMES:

        df = candles(
            market["symbol"],
            timeframe
        )

        if df is None:

            return (
                None,
                f"داده کندل {timeframe} دقیقه‌ای موجود نبود"
            )

        result = analyze(df)

        if result is None:

            return (
                None,
                f"تحلیل {timeframe} دقیقه‌ای ناقص بود"
            )

        analyses[timeframe] = result

        dataframes[timeframe] = df

        time.sleep(REQUEST_PAUSE)

    # Dataframe needed for support/resistance

    analyses["df15"] = dataframes[15]

    spread = spread_pct(
        market["symbol"]
    )

    candidate = build_candidate(
        market,
        analyses,
        spread
    )

    return candidate, None


# ============================================================
# REMOVE DUPLICATE ASSETS
# PREFER IRT
# ============================================================

def deduplicate_candidates(candidates):

    grouped = {}

    for item in candidates:

        base = symbol_base(
            item["symbol"]
        )

        if base not in grouped:

            grouped[base] = item

            continue

        current = grouped[base]

        if (
            item["quote"] == "rls"
            and current["quote"] != "rls"
        ):

            grouped[base] = item

        elif (
            item["quote"] == current["quote"]
            and item["rank"] > current["rank"]
        ):

            grouped[base] = item

    result = list(
        grouped.values()
    )

    result.sort(
        key=lambda item: item["rank"],
        reverse=True
    )

    return result


# ============================================================
# CANDIDATE CARD
# ============================================================

def candidate_card(index, item):

    symbol = esc(
        item["symbol"]
    )

    score = item["score"]

    spread = item["spread"]

    spread_text = (

        f"{spread:.3f}%"

        if spread is not None

        else "نامشخص"

    )

    if item["classification"] == "BUY":

        heading = "🟢 سیگنال خرید تحلیلی"

    elif item["classification"] == "WATCH":

        heading = "🟡 واچ‌لیست"

    else:

        heading = "🔴 بدون موقعیت مناسب"

    reasons = item["reasons"][:5]

    reasons_text = "\n".join(

        "• " + esc(reason)

        for reason in reasons

    )

    lines = [

        line_separator(),

        f"🏅 <b>رتبه {index} | {symbol}</b>",

        f"💱 {quote_label(item['quote'])}",

        "",

        heading,

        "",

        f"⭐ امتیاز: <b>{score:.1f}/100</b>",

        f"<code>{score_bar(score)}</code>",

        f"📌 وضعیت: <b>{item['classification']}</b>",

        "",

        "📊 <b>تجزیه امتیاز:</b>",

        f"• روند: {item['scores']['trend']}/25",

        f"• حمایت و مقاومت: {item['scores']['sr']}/25",

        f"• حجم: {item['scores']['volume']}/20",

        f"• EMA/RSI/MACD: {item['scores']['indicators']}/15",

        "",

        "💰 <b>اطلاعات بازار:</b>",

        f"قیمت آخرین کندل: <code>{fmt(item['price_at_scan'])}</code>",

        f"حجم نسبی ۱۵ دقیقه: <b>{item['volume_ratio']:.2f}x</b>",

        f"اسپرد: <b>{spread_text}</b>",

        "",

        "📈 <b>وضعیت تایم‌فریم‌ها:</b>",

        (
            "4H: "
            + (
                "🟢 صعودی"
                if item["ema"]["4h"]
                else "🔴 غیرصعودی"
            )
        ),

        (
            "1H: "
            + (
                "🟢 صعودی"
                if item["ema"]["1h"]
                else "🔴 غیرصعودی"
            )
        ),

        (
            "15M: "
            + (
                "🟢 صعودی"
                if item["ema"]["15m"]
                else "🔴 غیرصعودی"
            )
        ),

        "",

        (
            "RSI در 15M / 1H / 4H: "
            f"{item['rsi']['15m']:.1f} / "
            f"{item['rsi']['1h']:.1f} / "
            f"{item['rsi']['4h']:.1f}"
        ),

        "",

        "📍 <b>حمایت و مقاومت:</b>",

        f"حمایت: <code>{fmt(item['support'])}</code>",

        f"مقاومت: <code>{fmt(item['resistance'])}</code>",

        (
            "شکست مقاومت: "
            + (
                "🟢 تأییدشده"
                if item["breakout"]
                else "🟡 تأییدنشده"
            )
        ),

        "",

        "🎯 <b>نقشه معاملاتی محاسباتی:</b>",

        (
            f"محدوده ورود: "
            f"<code>{fmt(item['entry_low'])}</code>"
            f" تا "
            f"<code>{fmt(item['entry_high'])}</code>"
        ),

        f"قیمت مرجع: <code>{fmt(item['entry'])}</code>",

        f"حد ضرر: <code>{fmt(item['sl'])}</code>",

        f"هدف اول: <code>{fmt(item['tp1'])}</code>",

        f"هدف دوم: <code>{fmt(item['tp2'])}</code>",

        f"ریسک تا حد ضرر: <b>{item['risk_pct']:.2f}%</b>",

        f"نسبت سود به زیان هدف اول: <b>{item['rr']:.2f}</b>",

        "",

        "💡 <b>دلایل مثبت:</b>",

        reasons_text if reasons_text else "مورد برجسته‌ای ثبت نشد",

        line_separator()

    ]

    return lines


# ============================================================
# SCAN REPORT
# ============================================================

def scan_report(
    markets,
    screened,
    selected,
    candidates,
    watchlist,
    failures_screen,
    failures_deep
):

    lines = [

        "🚀 <b>NOBITEX SMART TRADER | V6</b>",

        "🧠 سیستم تحلیل چندتایم‌فریمی",

        line_separator(),

        "",

        "📊 <b>گزارش اسکن بازار:</b>",

        f"🌐 بازارهای فعال: <b>{markets}</b>",

        f"🔎 بازارهای دارای داده یک‌ساعته: <b>{screened}</b>",

        f"🎯 بازارهای انتخاب‌شده برای تحلیل عمیق: <b>{selected}</b>",

        f"✅ تحلیل عمیق موفق: <b>{candidates}</b>",

        f"⚠️ خطای غربالگری: <b>{failures_screen}</b>",

        f"⚠️ خطای تحلیل عمیق: <b>{failures_deep}</b>",

        f"👀 ارزهای واردشده به واچ‌لیست: <b>{watchlist}</b>",

        "",

        "🧭 <b>روش تحلیل:</b>",

        "• 4H: تشخیص روند اصلی",

        "• 1H: بررسی موقعیت بازار",

        "• 15M: بررسی نقطه ورود",

        "• حمایت و مقاومت: تشخیص سطوح مهم",

        "• حجم: مقایسه با میانگین ۲۰ کندل",

        "• اندیکاتورها: EMA20/50، RSI14 و MACD",

        "",

        line_separator(),

        "🏆 <b>پنج ارز برتر:</b>",

        ""

    ]

    return lines


# ============================================================
# FULL SCAN
# ============================================================

def run_full_scan(state):

    log.info(
        "Starting Nobitex Smart Trader V6"
    )

    markets = get_markets()

    log.info(
        "Active markets: %s",
        len(markets)
    )

    if not markets:

        raise RuntimeError(
            "هیچ بازار فعالی دریافت نشد"
        )

    # --------------------------------------------------------
    # STEP 1: 1H SCREENING
    # --------------------------------------------------------

    screened = []

    failures_screen = 0

    for index, market in enumerate(
        markets,
        1
    ):

        log.info(
            "Screen %s/%s: %s",
            index,
            len(markets),
            market["symbol"]
        )

        try:

            df = candles(
                market["symbol"],
                60
            )

            if df is None:

                failures_screen += 1

                continue

            result = analyze(df)

            if result is None:

                failures_screen += 1

                continue

            market["screen_score"] = 0

            # غربالگری اولیه فقط برای انتخاب بازارهای
            # دارای داده معتبر و حجم مناسب است.

            market["_screen_analysis"] = result

            screened.append(market)

        except Exception as error:

            failures_screen += 1

            log.warning(
                "Screen error %s: %s",
                market["symbol"],
                error
            )

        time.sleep(REQUEST_PAUSE)

    if not screened:

        raise RuntimeError(
            "هیچ بازاری داده معتبر برای تحلیل نداشت"
        )

    # --------------------------------------------------------
    # STEP 2: SELECT TOP VOLUME MARKETS
    # --------------------------------------------------------

    selected = []

    for quote in ("rls", "usdt"):

        group = [

            market

            for market in screened

            if market["quote"] == quote

        ]

        group.sort(
            key=lambda item: item["volume"],
            reverse=True
        )

        selected.extend(
            group[:TOP_PER_QUOTE]
        )

    # --------------------------------------------------------
    # STEP 3: DEEP ANALYSIS
    # --------------------------------------------------------

    candidates = []

    failures_deep = 0

    for index, market in enumerate(
        selected,
        1
    ):

        log.info(
            "Deep analysis %s/%s: %s",
            index,
            len(selected),
            market["symbol"]
        )

        try:

            candidate, reason = analyze_market(
                market
            )

            if candidate is not None:

                candidates.append(candidate)

            else:

                failures_deep += 1

                log.warning(
                    "Skipped %s: %s",
                    market["symbol"],
                    reason
                )

        except Exception as error:

            failures_deep += 1

            log.warning(
                "Deep analysis error %s: %s",
                market["symbol"],
                error
            )

    if not candidates:

        raise RuntimeError(
            "تحلیل عمیق هیچ نماد معتبری تولید نکرد"
        )

    # --------------------------------------------------------
    # STEP 4: DEDUPLICATION
    # --------------------------------------------------------

    candidates = deduplicate_candidates(
        candidates
    )

    candidates.sort(
        key=lambda item: item["rank"],
        reverse=True
    )

    top_five = candidates[:MAX_WATCH]

    watchlist = [

        item

        for item in top_five

        if item["score"] >= MIN_WATCH_SCORE

    ]

    # --------------------------------------------------------
    # STEP 5: STATE UPDATE
    # --------------------------------------------------------

    state["watchlist"] = watchlist

    state["last_scan"] = now_ts()

    state["last_scan_markets"] = len(markets)

    state["last_scan_screened"] = len(screened)

    state["last_scan_selected"] = len(selected)

    state["last_scan_analyzed"] = len(candidates)

    state["last_scan_failures"] = (
        failures_screen + failures_deep
    )

    state["top_results"] = top_five

    # --------------------------------------------------------
    # STEP 6: TELEGRAM REPORT
    # --------------------------------------------------------

    message = scan_report(

        len(markets),

        len(screened),

        len(selected),

        len(candidates),

        len(watchlist),

        failures_screen,

        failures_deep

    )

    if not top_five:

        message.append(
            "هیچ نماد مناسبی پیدا نشد."
        )

    for index, item in enumerate(
        top_five,
        1
    ):

        message.extend(
            candidate_card(index, item)
        )

    buys = [

        item

        for item in top_five

        if item["classification"] == "BUY"

    ]

    watches = [

        item

        for item in top_five

        if item["classification"] == "WATCH"

    ]

    message.extend([

        "",

        line_separator(),

        "📌 <b>خلاصه تصمیم:</b>",

        f"🟢 موقعیت‌های دارای تأیید اولیه: <b>{len(buys)}</b>",

        f"🟡 ارزهای قابل پایش: <b>{len(watches)}</b>",

        "",

        (
            "⛔ هیچ موقعیت خریدی با تمام شروط استراتژی "
            "پیدا نشد."
            if not buys
            else
            "🟢 موقعیت‌های BUY نیازمند بررسی نهایی معامله‌گر هستند."
        ),

        "",

        "⚠️ <b>نکات مهم:</b>",

        "• امتیاز بالا به معنی احتمال قطعی سود نیست.",

        "• شکست مقاومت باید با بسته‌شدن کندل تأیید شود.",

        "• حجم بالا به‌تنهایی سیگنال خرید نیست.",

        "• نسبت سود به زیان با هزینه تخمینی معامله محاسبه شده است.",

        "• کارمزد واقعی ممکن است با مقدار تخمینی متفاوت باشد.",

        "• این ربات هیچ سفارش خرید یا فروشی ثبت نمی‌کند.",

        "• استراتژی هنوز نیازمند بک‌تست و معاملات آزمایشی است.",

        line_separator()

    ])

    telegram(
        "\n".join(message)
    )

    log.info(
        "Full scan completed successfully"
    )


# ============================================================
# PRICE CONSISTENCY
# ============================================================

def check_price_consistency(
    item,
    current_price
):

    scan_price = num(
        item.get("price_at_scan"),
        0
    )

    if scan_price <= 0 or current_price <= 0:

        return False, None

    difference = (
        abs(current_price - scan_price)
        / scan_price
    )

    if difference > PRICE_MISMATCH_THRESHOLD:

        return True, difference

    return False, difference


# ============================================================
# MONITOR
# ============================================================

def monitor(state):

    watchlist = state.get(
        "watchlist",
        []
    )

    if not watchlist:

        log.info(
            "No watchlist to monitor"
        )

        return

    prices = get_all_prices()

    if not prices:

        log.warning(
            "Could not refresh market prices"
        )

        return

    for item in watchlist:

        symbol = item["symbol"]

        key = symbol_to_market_key(
            symbol
        )

        price = num(
            prices.get(key),
            0
        )

        if price <= 0:
            continue

        # ----------------------------------------------------
        # PRICE MISMATCH
        # ----------------------------------------------------

        mismatch, difference = check_price_consistency(
            item,
            price
        )

        if mismatch:

            if not item.get(
                "price_mismatch",
                False
            ):

                item["price_mismatch"] = True

                item["status"] = "DATA_MISMATCH"

                percent = difference * 100

                alert = (

                    f"🚨 <b>هشدار ناهماهنگی قیمت | {esc(symbol)}</b>\n\n"

                    f"قیمت زمان اسکن: <code>{fmt(item['price_at_scan'])}</code>\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"اختلاف: <b>{percent:.2f}%</b>\n\n"

                    "⛔ پایش این نماد متوقف شد.\n"

                    "ابتدا واحد قیمت و داده‌های بازار بررسی شود."

                )

                try:

                    telegram(alert)

                except Exception as error:

                    log.error(
                        "Mismatch Telegram error: %s",
                        error
                    )

            item["last_price"] = price

            item["last_checked"] = now_ts()

            continue

        if item.get(
            "price_mismatch",
            False
        ):

            continue

        status = item.get(
            "status",
            "WAITING"
        )

        low = num(
            item.get("entry_low")
        )

        high = num(
            item.get("entry_high")
        )

        stop = num(
            item.get("sl")
        )

        target1 = num(
            item.get("tp1")
        )

        target2 = num(
            item.get("tp2")
        )

        alert = None

        # ----------------------------------------------------
        # WAITING
        # ----------------------------------------------------

        if status == "WAITING":

            if price <= stop:

                item["status"] = "INVALIDATED"

                alert = (

                    f"⛔ <b>موقعیت {esc(symbol)} نامعتبر شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"حد ابطال: <code>{fmt(stop)}</code>\n\n"

                    "این موقعیت دیگر شرایط اولیه را ندارد."

                )

            elif low <= price <= high:

                item["status"] = "ENTRY_ZONE_ALERTED"

                item["entry_alert_price"] = price

                alert = (

                    f"🟡 <b>قیمت {esc(symbol)} وارد محدوده پایش شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"محدوده: <code>{fmt(low)}</code> تا <code>{fmt(high)}</code>\n\n"

                    f"حد ضرر محاسباتی: <code>{fmt(stop)}</code>\n"

                    f"هدف اول: <code>{fmt(target1)}</code>\n"

                    f"هدف دوم: <code>{fmt(target2)}</code>\n\n"

                    "⚠️ ورود نیازمند بررسی مجدد شرایط بازار است."

                )

            elif price > high * 1.015:

                item["status"] = "MISSED"

                alert = (

                    f"⚠️ <b>محدوده ورود {esc(symbol)} از دسترس خارج شد.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n\n"

                    "از تعقیب قیمت خودداری کن."

                )

        # ----------------------------------------------------
        # BUY SETUP
        # ----------------------------------------------------

        elif status == "BUY_SETUP":

            if price <= stop:

                item["status"] = "INVALIDATED"

                alert = (

                    f"⛔ <b>شرایط اولیه {esc(symbol)} از بین رفت.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n"

                    f"حد ابطال: <code>{fmt(stop)}</code>"

                )

            elif price <= high and price >= low:

                item["status"] = "ENTRY_ZONE_ALERTED"

                item["entry_alert_price"] = price

                alert = (

                    f"🟢 <b>قیمت {esc(symbol)} در محدوده ورود تحلیلی است.</b>\n\n"

                    f"قیمت فعلی: <code>{fmt(price)}</code>\n"

                    f"حد ضرر: <code>{fmt(stop)}</code>\n"

                    f"هدف اول: <code>{fmt(target1)}</code>\n"

                    f"هدف دوم: <code>{fmt(target2)}</code>\n\n"

                    "قبل از تصمیم، کندل و حجم را مجدداً بررسی کن."

                )

        # ----------------------------------------------------
        # AFTER ENTRY ALERT
        # ----------------------------------------------------

        elif status == "ENTRY_ZONE_ALERTED":

            alerts = item.setdefault(
                "alerts",
                []
            )

            if (
                price <= stop
                and "SL_TOUCHED" not in alerts
            ):

                alerts.append(
                    "SL_TOUCHED"
                )

                alert = (

                    f"🔴 <b>قیمت {esc(symbol)} به حد ضرر محاسباتی رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این گزارش به معنی معامله واقعی نیست."

                )

            elif (
                price >= target2
                and "TP2_TOUCHED" not in alerts
            ):

                alerts.append(
                    "TP2_TOUCHED"
                )

                alert = (

                    f"🏆 <b>قیمت {esc(symbol)} به هدف دوم رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این صرفاً گزارش حرکت قیمت است."

                )

            elif (
                price >= target1
                and "TP1_TOUCHED" not in alerts
            ):

                alerts.append(
                    "TP1_TOUCHED"
                )

                alert = (

                    f"🎯 <b>قیمت {esc(symbol)} به هدف اول رسید.</b>\n\n"

                    f"قیمت: <code>{fmt(price)}</code>\n\n"

                    "این صرفاً گزارش حرکت قیمت است."

                )

        item["last_price"] = price

        item["last_checked"] = now_ts()

        if alert:

            try:

                telegram(alert)

                item["last_alert_time"] = now_ts()

            except Exception as error:

                log.error(
                    "Telegram alert error: %s",
                    error
                )

    state["last_monitor"] = now_ts()

    save_state(state)


# ============================================================
# MAIN
# ============================================================

def main():

    if not BOT_TOKEN or not CHAT_ID:

        raise RuntimeError(
            "BOT_TOKEN و CHAT_ID باید در GitHub Secrets تعریف شوند."
        )

    state = load_state()

    old_version = int(
        state.get("version", 0)
    )

    # Reset old strategy state

    if old_version < STATE_VERSION:

        log.info(
            "State version changed. Resetting old watchlist."
        )

        state["last_scan"] = 0

        state["watchlist"] = []

        state["last_monitor"] = 0

    event_name = (

        os.getenv("EVENT_NAME", "")

        or os.getenv("GITHUB_EVENT_NAME", "")

    )

    force = (
        event_name == "workflow_dispatch"
    )

    last_scan = int(
        state.get("last_scan", 0)
    )

    due = (

        now_ts() - last_scan

        >= SCAN_EVERY_SECONDS

    )

    log.info(
        "================================"
    )

    log.info(
        "Nobitex Smart Trader V6"
    )

    log.info(
        "Force scan: %s",
        force
    )

    log.info(
        "Scan due: %s",
        due
    )

    log.info(
        "================================"
    )

    # --------------------------------------------------------
    # FULL SCAN
    # --------------------------------------------------------

    if force or due:

        try:

            run_full_scan(state)

            save_state(state)

        except Exception as error:

            log.exception(
                "FULL SCAN FAILED"
            )

            try:

                telegram(

                    "⚠️ <b>خطا در اسکن کامل نوبیتکس:</b>\n"

                    + esc(str(error)[:2500])

                )

            except Exception as telegram_error:

                log.error(
                    "Telegram error: %s",
                    telegram_error
                )

    # --------------------------------------------------------
    # PRICE MONITOR
    # --------------------------------------------------------

    try:

        monitor(state)

    except Exception as error:

        log.exception(
            "MONITOR FAILED"
        )

        try:

            telegram(

                "⚠️ <b>خطا در پایش قیمت‌ها:</b>\n"

                + esc(str(error)[:2500])

            )

        except Exception:

            pass

    save_state(state)

    log.info(
        "Bot cycle completed"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
