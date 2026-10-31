import requests
import time

API = "https://apiv2.nobitex.ir"
BOT_TOKEN = ""
CHAT_ID = ""

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": message
    }, timeout=20)

def get_markets():
    data = requests.get(
        API + "/market/stats",
        timeout=20
    ).json()

    stats = data.get("stats", {})
    markets = []

    for symbol, info in stats.items():
        if symbol.endswith("-usdt"):
            markets.append((symbol, info))

    markets.sort(
        key=lambda x: float(x[1].get("volumeDst", 0)),
        reverse=True
    )

    return markets[:12]

def get_candles(symbol, resolution):
    now = int(time.time())

    url = API + "/market/udf/history"

    params = {
        "symbol": symbol.upper().replace("-", ""),
        "resolution": resolution,
        "from": now - 500 * int(resolution) * 60,
        "to": now,
        "countback": 200
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    ).json()

    if response.get("s") != "ok":
        return None

    return response

def ema(values, period):
    result = []
    alpha = 2 / (period + 1)

    for i, value in enumerate(values):
        if i == 0:
            result.append(value)
        else:
            result.append(
                alpha * value +
                (1 - alpha) * result[-1]
            )

    return result

def rsi(values, period=14):
    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze(symbol):
    c15 = get_candles(symbol, 15)
    c60 = get_candles(symbol, 60)
    c240 = get_candles(symbol, 240)

    if not all([c15, c60, c240]):
        return None

    close15 = c15["c"]
    close60 = c60["c"]
    close240 = c240["c"]
    volume15 = c15["v"]

    if min(len(close15), len(close60), len(close240)) < 200:
        return None

    e20 = ema(close15, 20)
    e50_60 = ema(close60, 50)
    e200_240 = ema(close240, 200)

    current = close15[-1]
    previous = close15[-2]

    score = 0

    if close240[-1] > e200_240[-1]:
        score += 1

    if close60[-1] > e50_60[-1]:
        score += 1

    if current > e20[-1] and previous <= e20[-2]:
        score += 1

    current_rsi = rsi(close15)

    if 48 <= current_rsi <= 68:
        score += 1

    if len(volume15) >= 21:
        avg_volume = sum(volume15[-21:-1]) / 20
        if volume15[-1] >= avg_volume * 1.25:
            score += 1

    if score < 4:
        return None

    return {
        "price": current,
        "score": score,
        "rsi": current_rsi
    }

def main():
    for symbol, info in get_markets():
        try:
            result = analyze(symbol)

            if result:
                message = (
                    f"🚀 سیگنال احتمالی خرید\n\n"
                    f"ارز: {symbol.upper()}\n"
                    f"قیمت: {result['price']}\n"
                    f"امتیاز: {result['score']}/5\n"
                    f"RSI: {result['rsi']:.1f}\n\n"
                    f"⚠️ قبل از معامله بررسی شود."
                )

                send_telegram(message)

            time.sleep(2)

        except Exception as error:
            print(symbol, error)

if __name__ == "__main__":
    main()
