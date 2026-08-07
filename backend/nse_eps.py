"""
Filed diluted EPS from NSE's structured corporate-filings data.

Every listed company files its quarterly results with NSE as XBRL (machine-
readable XML). This is the most reliable source of filed diluted EPS — it works
even for companies whose result PDFs have a corrupt text layer (e.g. Reliance),
and the XML files are ~60 KB versus multi-MB PDFs.

Flow: one API call lists all quarterly filings for a symbol (with consolidated/
standalone flags and per-filing XBRL links); each relevant quarter's XBRL is
downloaded and the diluted-EPS fact whose context matches the filing's exact
reporting period is extracted.
"""

import requests
import re
import time
import logging
from datetime import datetime

logger = logging.getLogger("eps-nse")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

_MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

# Filed results never change once published; cache per (symbol, label, basis, eps_type)
# for the process lifetime. None records a definitive miss (filing list fetched
# fine but no usable XBRL for that quarter); transient failures aren't cached.
_NSE_CACHE: dict[tuple[str, str, bool, str], dict | None] = {}

# Diluted-EPS XBRL tags in order of preference (namespace prefixes vary across
# taxonomy versions, so tags are matched with any prefix).
_DILUTED_TAG_PATTERNS = [
    r"DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    r"DilutedEarningsLossPerShareFromContinuingOperations",
    r"DilutedEarningsPerShareAfterExtraordinaryItems",
    r"DilutedEarningsPerShare",
]

_BASIC_TAG_PATTERNS = [
    r"BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    r"BasicEarningsLossPerShareFromContinuingOperations",
    r"BasicEarningsPerShareAfterExtraordinaryItems",
    r"BasicEarningsPerShare",
]


def _seconds_left(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _parse_xbrl_eps(xml: str, from_date: datetime, to_date: datetime, eps_type: str = "diluted") -> float | None:
    """
    Extracts the basic or diluted EPS fact whose XBRL context period exactly matches the
    filing's reporting period (so quarter figures are never confused with the
    YTD or previous-period columns that share the same document).
    """
    contexts: dict[str, tuple[str, str]] = {}
    for m in re.finditer(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', xml, re.DOTALL):
        cid, body = m.group(1), m.group(2)
        sm = re.search(r"<xbrli:startDate>([^<]+)</xbrli:startDate>", body)
        em = re.search(r"<xbrli:endDate>([^<]+)</xbrli:endDate>", body)
        if sm and em:
            contexts[cid] = (sm.group(1).strip(), em.group(1).strip())

    target = (from_date.strftime("%Y-%m-%d"), to_date.strftime("%Y-%m-%d"))

    tag_patterns = _BASIC_TAG_PATTERNS if eps_type == "basic" else _DILUTED_TAG_PATTERNS

    for tag_pattern in tag_patterns:
        fact_re = re.compile(
            r"<[\w.-]+:" + tag_pattern + r'\b[^>]*contextRef="([^"]+)"[^>]*>([^<]*)<'
        )
        matches: list[tuple[str, float]] = []
        for m in fact_re.finditer(xml):
            ctx_id, raw = m.group(1), m.group(2).strip()
            if contexts.get(ctx_id) != target or not raw:
                continue
            try:
                matches.append((ctx_id, float(raw)))
            except ValueError:
                continue
        if matches:
            # Banking-taxonomy filings mark YTD/other columns with the *same*
            # period dates as the quarter; the standardized context id "OneD"
            # (current-quarter column) is the only reliable discriminator.
            for ctx_id, value in matches:
                if ctx_id == "OneD":
                    return value
            return matches[0][1]
    return None


def fetch_diluted_eps_nse(symbol: str, from_q: tuple[int, int], to_q: tuple[int, int],
                          consolidated: bool = True, deadline: float = None,
                          eps_type: str = "diluted") -> dict:
    """
    Returns {quarter_label: {"value": float, "kind": kind, "xbrl": True}}
    for every quarter in [from_q, to_q] that has a filed XBRL result on NSE.
    Quarters are processed newest-first and the lookup stops when `deadline`
    (time.monotonic() timestamp) runs out; whatever was resolved is returned.
    """
    eps_type = (eps_type or "diluted").lower()
    symbol = (symbol or "").strip().upper()
    results: dict[str, dict] = {}
    if not symbol:
        return results

    want_flag = "Consolidated" if consolidated else "Non-Consolidated"

    remaining = _seconds_left(deadline)
    if remaining is not None and remaining <= 0:
        return results

    try:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        # Cookie warm-up; NSE rejects API calls from cookieless clients.
        session.get("https://www.nseindia.com", timeout=10)
        resp = session.get(
            f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly",
            headers={"Accept": "application/json",
                     "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"NSE filings list returned {resp.status_code} for {symbol}.")
            return results
        items = resp.json()
        if not isinstance(items, list):
            return results
    except Exception as e:
        logger.warning(f"NSE filings list fetch failed for {symbol}: {e}")
        return results

    # One filing per quarter: keep the latest broadcast for re-filed results.
    filings: dict[str, tuple[tuple[int, int], datetime, datetime, str]] = {}
    for it in items:
        try:
            if it.get("consolidated") != want_flag:
                continue
            xbrl_url = (it.get("xbrl") or "").strip()
            if not xbrl_url:
                continue
            from_dt = datetime.strptime(it["fromDate"], "%d-%b-%Y")
            to_dt = datetime.strptime(it["toDate"], "%d-%b-%Y")
            q = (to_dt.year, to_dt.month)
            if not (from_q <= q <= to_q):
                continue
            label = f"{_MONTHS[to_dt.month]} {to_dt.year}"
            if label not in filings:  # list is newest-first; first wins
                filings[label] = (q, from_dt, to_dt, xbrl_url)
        except Exception:
            continue

    # Newest quarters first so the most relevant data survives a tight budget.
    for label, (q, from_dt, to_dt, xbrl_url) in sorted(filings.items(), key=lambda kv: kv[1][0], reverse=True):
        cache_key = (symbol, label, consolidated, eps_type)
        if cache_key in _NSE_CACHE:
            cached = _NSE_CACHE[cache_key]
            if cached is not None:
                results[label] = cached
            continue

        remaining = _seconds_left(deadline)
        if remaining is not None and remaining <= 0:
            logger.info(f"Deadline reached; stopping NSE XBRL lookup for {symbol} at {label}.")
            break

        try:
            xr = requests.get(xbrl_url, headers=BROWSER_HEADERS,
                              timeout=15 if remaining is None else min(15, remaining))
            if xr.status_code != 200:
                continue
            value = _parse_xbrl_eps(xr.text, from_dt, to_dt, eps_type)
            kind = "Basic" if eps_type == "basic" else "Diluted"
            if value is not None:
                info = {"value": value, "kind": kind, "xbrl": True, "source": f"NSE XBRL ({kind})"}
                _NSE_CACHE[cache_key] = info
                results[label] = info
            else:
                # XBRL downloaded and scanned completely: definitive miss.
                _NSE_CACHE[cache_key] = None
        except Exception as e:
            logger.warning(f"NSE XBRL fetch/parse failed for {symbol} {label}: {e}")

    return results

fetch_eps_nse = fetch_diluted_eps_nse

