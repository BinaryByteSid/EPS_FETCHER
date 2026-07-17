"""
Filed diluted EPS from BSE's structured results API.

This is the most reliable source for *recent* quarters (2025-26): NSE's XBRL feed
lags ~2 quarters, and many recent result PDFs are scanned images, but BSE exposes
every filed quarterly result as structured JSON through the same endpoints its
website's Financial-Results tab uses.

Flow:
  1. TabResults_PAR gives the company's latest quarter's internal "Qtr" id.
  2. Walk that id backward (quarterly ids are consecutive whole numbers; .50 ids
     are annual roll-ups and are skipped), calling the detailed-result endpoint
     for each — standalone via Corp_detailedResult_Transpose_ng, consolidated via
     Corp_BSEDnBResults_SEBI_Consolidated_Res_ng.
  3. Each response carries its own Date Begin/End, so quarters are mapped by the
     actual reported period, not by trusting the id arithmetic.
"""

import re
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("eps-bse-api")

API_BASE = "https://api.bseindia.com/BseIndiaAPI/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
}

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MON_LABEL = {v: k.title() for k, v in _MONTHS.items()}

# Filed results never change; cache per (scripcode, label, consolidated).
# None records a definitive miss; transient failures are not cached.
_CACHE: dict[tuple[str, str, bool], dict | None] = {}

# Give up after this many consecutive empty/failed ids while walking back, so a
# delisted gap or numbering quirk cannot loop forever.
_MAX_EMPTY_STREAK = 3


def _seconds_left(deadline):
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _get_json(session, url, deadline):
    remaining = _seconds_left(deadline)
    if remaining is not None and remaining <= 0:
        return None
    timeout = 20 if remaining is None else min(20, max(3, remaining))
    for attempt in range(2):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code != 200:
                return None
            text = r.text.lstrip()
            if not text or text[0] not in "[{\"":
                return None  # HTML error shell
            data = r.json()
            # Some BSE endpoints (e.g. TabResults_PAR) double-encode: the body
            # is a JSON string that itself contains JSON.
            if isinstance(data, str):
                import json
                data = json.loads(data)
            return data
        except Exception:
            if attempt == 0 and (_seconds_left(deadline) or 99) > 3:
                time.sleep(1.5)
                continue
            return None
    return None


def _quarters_in_range(from_q: tuple[int, int], to_q: tuple[int, int]) -> list[str]:
    """All 'Mon YYYY' quarter labels (Mar/Jun/Sep/Dec) within [from_q, to_q]."""
    labels = []
    for year in range(from_q[0], to_q[0] + 1):
        for mon in (3, 6, 9, 12):
            q = (year, mon)
            if from_q <= q <= to_q:
                labels.append(f"{_MON_LABEL[mon]} {year}")
    return labels


def _label_from_dates(begin: str, end: str) -> str | None:
    """Map a filing's Date End (e.g. '31-Mar-26') to a 'Mon YYYY' quarter label."""
    m = re.match(r"\d{1,2}-([A-Za-z]{3})-(\d{2,4})", (end or "").strip())
    if not m:
        return None
    mon = _MONTHS.get(m.group(1).lower())
    if not mon:
        return None
    yr = int(m.group(2))
    if yr < 100:
        yr += 2000
    return f"{_MON_LABEL[mon]} {yr}"


def _latest_qtr_id(session, scripcode: str, deadline) -> float | None:
    """Read the latest quarterly Qtr id from TabResults_PAR."""
    data = _get_json(session, f"{API_BASE}/TabResults_PAR/w?scripcode={scripcode}&tabtype=RESULTS", deadline)
    if not isinstance(data, dict):
        return None
    snap = data.get("resultinS") or []
    if not snap:
        return None
    # LLQ = latest quarter link, carries qtr=<id> in its URL.
    for key in ("LLQ", "LSQ", "LFY"):
        link = snap[0].get(key, "")
        m = re.search(r"qtr=([\d.]+)", link)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _diluted_from_standalone(session, scripcode: str, qtr_id: str, deadline):
    data = _get_json(session, f"{API_BASE}/Corp_detailedResult_Transpose_ng/w?Scrip_cd={scripcode}&Qtr={qtr_id}", deadline)
    if not isinstance(data, dict):
        return None
    rows = data.get("table1") or []
    fields = {str(r.get("fld_desc", "")).strip().lower(): r.get("Value") for r in rows}
    return _pick(fields, "date begin"), _pick(fields, "date end"), _pick_diluted(fields)


def _diluted_from_consolidated(session, scripcode: str, qtr_id: str, name: str, deadline):
    url = (f"{API_BASE}/Corp_BSEDnBResults_SEBI_Consolidated_Res_ng/w"
           "?usp1=usp_BSEINDIA_CONSILDATERESULT_UAT&usp2=USP_GetResult_Type_consolidated"
           "&usp3=usp_GET_BSEDnBResults_SplitUP_consoldated&type1=C"
           f"&strtype={qtr_id}&strscripcd={scripcode}&strscripname={requests.utils.quote(name)}"
           "&strresultType=&action=show")
    data = _get_json(session, url, deadline)
    if not isinstance(data, list) or not data:
        return None
    fields = {str(r.get("Description", "")).strip().lower(): r.get("Amount") for r in data}
    return _pick(fields, "date begin"), _pick(fields, "date end"), _pick_diluted(fields)


def _pick(fields: dict, key: str):
    return fields.get(key)


def _pick_diluted(fields: dict) -> float | None:
    """
    Return the filed diluted EPS. Prefers explicit diluted rows; falls back to a
    generically-labeled EPS row (some companies file only a combined figure).
    Blank placeholders ('-', '') parse to None and are skipped.
    """
    def _num(raw):
        try:
            return float(str(raw).replace(",", "").strip())
        except (ValueError, AttributeError):
            return None

    # 1. Explicit diluted rows (banks, most large caps).
    for key in ("diluted eps after extraordinary items",
                "diluted eps before extraordinary items"):
        val = _num(fields.get(key))
        if val is not None:
            return val

    # 2. Any other row whose label mentions diluted EPS.
    for key, raw in fields.items():
        if "diluted" in key and ("eps" in key or "per share" in key or "earning" in key):
            val = _num(raw)
            if val is not None:
                return val

    # 3. Generic EPS row (combined basic/diluted) as a last resort.
    for key, raw in fields.items():
        if key.startswith("eps") or "earnings per share" in key or "eps after" in key or "eps before" in key:
            val = _num(raw)
            if val is not None:
                return val

    return None


def fetch_diluted_eps_bse_api(scripcode: str, company_name: str,
                              from_q: tuple[int, int], to_q: tuple[int, int],
                              consolidated: bool = True, deadline: float = None) -> dict:
    """
    Returns {quarter_label: {"value": float, "kind": "Diluted", "xbrl": False,
    "source": "BSE API"}} for filed quarters in [from_q, to_q]. Walks BSE's Qtr
    ids newest-first and stops when the range is covered or the deadline is hit.
    """
    scripcode = str(scripcode or "").split(".")[0].strip()
    results: dict[str, dict] = {}
    if not scripcode or not scripcode.isdigit():
        return results

    # Serve everything already cached from earlier requests up-front, and only
    # go to the (slow, rate-limited) API for quarters still missing. Cached
    # values survive later throttling instead of flipping back to Screener.
    requested = _quarters_in_range(from_q, to_q)
    for label in requested:
        cached = _CACHE.get((scripcode, label, consolidated))
        if cached is not None:
            results[label] = cached
    if all((scripcode, l, consolidated) in _CACHE for l in requested):
        return results

    session = requests.Session()
    session.headers.update(HEADERS)

    latest = _latest_qtr_id(session, scripcode, deadline)
    if latest is None:
        logger.info(f"No BSE Qtr id found for scripcode {scripcode}.")
        return results

    # The latest whole-number id is the most recent quarter; each id one lower is
    # the quarter before it. We need at most (quarters between to_q and from_q)
    # + a little slack, capped so we never scan a company's whole history.
    span = (to_q[0] - from_q[0]) * 4 + (to_q[1] - from_q[1]) // 3
    n_ids = min(max(span + 3, 4), 20)
    latest_int = int(latest)
    candidate_ids = [f"{latest_int - k}.00" for k in range(n_ids)]

    def _fetch(qtr_id: str):
        try:
            if consolidated:
                return qtr_id, _diluted_from_consolidated(session, scripcode, qtr_id, company_name, deadline)
            return qtr_id, _diluted_from_standalone(session, scripcode, qtr_id, deadline)
        except Exception as e:
            logger.warning(f"BSE API error for scrip {scripcode} qtr {qtr_id}: {e}")
            return qtr_id, None

    # BSE's API is slow per call, so fetch the candidate quarters concurrently.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_fetch, qid) for qid in candidate_ids]
        for fut in as_completed(futures):
            _qid, got = fut.result()
            if not got:
                continue
            begin, end, value = got
            label = _label_from_dates(begin, end)
            if not label:
                continue
            try:
                q = (int(label.split()[1]), _MONTHS[label.split()[0].lower()])
            except (ValueError, KeyError, IndexError):
                continue
            if not (from_q <= q <= to_q):
                continue
            cache_key = (scripcode, label, consolidated)
            if value is not None:
                info = {"value": value, "kind": "Diluted", "xbrl": False, "source": "BSE API"}
                _CACHE[cache_key] = info
                results[label] = info
            else:
                _CACHE[cache_key] = None

    return results
