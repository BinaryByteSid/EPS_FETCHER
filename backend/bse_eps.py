import requests
from bs4 import BeautifulSoup
import pdfplumber
import re
import io
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

logger = logging.getLogger("eps-bse")

SCREENER_BASE = "https://www.screener.in"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Result PDFs are targeted per quarter, but Q4 filings bundle the annual
# results and investor material (Infosys' runs to ~43 MB / 379 pages); the
# statutory EPS statements sit within the first ~40 pages.
MAX_PDF_BYTES = 60 * 1024 * 1024
MAX_PAGES_PER_PDF = 40

# Filed results never change once published, so extracted values are cached for
# the lifetime of the process to avoid re-downloading PDFs on repeat requests.
# A value of None records a *definitive* miss (PDF fully scanned, no usable
# figure) so hopeless PDFs are not re-parsed on every request; transient
# failures (network errors, deadline cut-offs) are never cached.
_EPS_CACHE: dict[tuple[str, str, bool], dict | None] = {}

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _seconds_left(deadline: float | None) -> float | None:
    """Remaining budget in seconds, or None when no deadline was given."""
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _parse_eps_token(token: str) -> float | None:
    """
    Strictly parses one whitespace-separated token as an EPS figure.
    Accepts thousands commas and accountant-style parentheses for negatives.
    Returns None for anything else: OCR-corrupted tokens like '13J8', and bare
    integers — filings always print EPS with decimals, so an integer right
    after 'diluted' is a corrupt decimal split ('20.98' extracted as '2 0.98')
    or a footnote number, never the EPS itself.
    """
    token = token.strip().rstrip("*")
    negative = token.startswith("(") and token.endswith(")")
    core = token.strip("()").replace(",", "")
    if "." not in core or not _NUM_RE.fullmatch(core):
        return None
    value = float(core)
    if negative and value > 0:
        value = -value
    return value


_SHARE_COUNT_RE = re.compile(r"in\s*shares|number of|no\.?\s*of\s*shares")


def _keyword_line_values(words: list[dict], keyword: str) -> list[dict]:
    """
    For every line containing `keyword`, returns the numeric token nearest to
    the keyword: {'x0': ..., 'value': ..., 'line': lowercased line text}.

    The nearest-token rule is parse-based, not positional: if the nearest
    digit-bearing token does not parse as a clean decimal (OCR-corrupted
    '13J8', or a bare integer from a split decimal), the whole line is dropped
    rather than sliding right to a token that belongs to a later period column.
    """
    hits = []
    key_words = [w for w in words if keyword in w['text'].lower()]
    for k_word in key_words:
        k_y_center = (k_word['top'] + k_word['bottom']) / 2

        line_words = [
            w for w in words
            if w['x0'] > k_word['x1'] and abs((w['top'] + w['bottom'])/2 - k_y_center) < 4
        ]
        line_words.sort(key=lambda x: x['x0'])

        line_text = " ".join(w['text'] for w in line_words).lower()
        # Skip foreign-currency EPS and share-count lines outright.
        if "$" in line_text or "usd" in line_text or re.search(r"in\s*shares", line_text):
            continue

        for w in line_words:
            if not any(ch.isdigit() for ch in w['text']):
                continue  # unit annotations like (₹) carry no digits
            val = _parse_eps_token(w['text'])
            if val is None or abs(val) > 10000:
                break  # nearest numeric token is corrupt / a share count
            hits.append({'x0': w['x0'], 'value': val,
                         'line': (k_word['text'] + " " + line_text).lower()})
            break
    return hits


def _extract_diluted_value(page, expected_basic: float | None = None) -> float | None:
    """
    Extracts the reported quarter's diluted EPS from a page.

    When `expected_basic` (Screener's basic EPS for the same quarter) is known,
    the extraction is anchored on the table's Basic-EPS row: the basic row's
    nearest value must match the expected figure, and the diluted candidate
    must sit in the same x-column as that anchor. This rejects pages where
    Column 1 is unreadable (the nearest parseable number would silently belong
    to the preceding-quarter column), pages from the wrong statement basis,
    and PDFs served for the wrong period — without any hardcoded x-thresholds.
    """
    words = page.extract_words()
    diluted_hits = _keyword_line_values(words, "diluted")
    if not diluted_hits:
        return None

    if expected_basic is None:
        return diluted_hits[0]['value']

    # Tight tolerance: Screener publishes the filed basic EPS nearly verbatim,
    # while the *preceding quarter's* figure — the classic wrong-column grab —
    # is typically only a few percent away, so a loose band would let it pass.
    tol = max(abs(expected_basic) * 0.02, 0.05)

    basic_anchors = [
        h for h in _keyword_line_values(words, "basic")
        if abs(h['value'] - expected_basic) <= tol
    ]

    for hit in diluted_hits:
        if "basic" in hit['line']:
            # Combined "Basic and diluted" row: the single figure is both.
            if abs(hit['value'] - expected_basic) <= tol:
                return hit['value']
            continue
        # Separate rows: the diluted figure must sit in the same column as a
        # validated basic anchor.
        if any(abs(hit['x0'] - a['x0']) <= 25 for a in basic_anchors):
            return hit['value']

    return None


def _page_mode(text_lower: str) -> str | None:
    """'consolidated' / 'standalone' when the page is unambiguous, else None."""
    has_cons = "consolidated" in text_lower
    has_stand = "standalone" in text_lower
    if has_cons and not has_stand:
        return "consolidated"
    if has_stand and not has_cons:
        return "standalone"
    return None


def _extract_diluted_from_pdf(content: bytes, consolidated: bool, quarter_label: str, deadline: float | None,
                              expected_basic: float | None = None) -> tuple[float | None, bool, bool]:
    """
    Scans a result PDF for the diluted EPS of its reported quarter, preferring
    pages that match the requested consolidated/standalone basis. Pages that
    mention neither (or both) are kept as a fallback, and the opposite basis is
    used only as a last resort (covers companies that file just one statement).

    Candidates from matching pages are majority-voted rather than trusting the
    first hit: Q4 bundles open with press-release summary tables whose 'diluted'
    line is the full-year figure, while the statutory quarter tables that follow
    repeat the true quarter value.

    Returns (value, completed, rejected): `completed` is False when the scan was cut
    short by the deadline. `rejected` is True when the PDF was rejected due to a year
    mismatch (BSE silent redirect) — callers should NOT cache these misses so the
    quarter can be retried on the next request.
    """
    wanted = "consolidated" if consolidated else "standalone"
    candidates: list[float] = []  # values from wanted-mode pages, in page order
    fallback: tuple[int, float] | None = None  # (priority, value)
    completed = True
    current_mode = None

    # Parse target year from quarter label (e.g. "2025" from "Mar 2025")
    target_year = None
    parts = quarter_label.strip().split()
    if len(parts) == 2:
        try:
            target_year = int(parts[1])
        except ValueError:
            pass

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        # Validate that the target year (e.g. "2025" or "25") is present in the first 5 pages
        if target_year is not None:
            year_found = False
            year_str = str(target_year)
            short_year = year_str[-2:]
            pattern = re.compile(r'\b(' + year_str + r'|' + short_year + r')\b')
            
            for p_idx in range(min(5, len(pdf.pages))):
                p_text = pdf.pages[p_idx].extract_text() or ""
                if pattern.search(p_text):
                    year_found = True
                    break
            if not year_found:
                logger.warning(f"Rejecting PDF for {quarter_label}: target year {target_year} not found in first 5 pages.")
                return None, True, True  # rejected=True → do NOT cache, allow retry

            # Reject if the PDF is actually a *newer* filing served via a BSE silent redirect.
            # We match patterns like "quarter ended December 31, 2025" on a SINGLE LINE and
            # extract the year.  Processing line-by-line prevents the greedy `[\s\w,./]*`
            # from crossing newlines and accidentally consuming a next-line date (e.g.
            # "January 12, 2024" on the letter's continuation line after a Dec 2023 header).
            _PERIOD_RE = re.compile(
                r'(?:quarter|year|nine\s+months|half[- ]?year)\s+(?:and\s+\S+\s+)?ended'
                r'[^\n]*(20\d{2})',
                re.IGNORECASE,
            )
            for p_idx in range(min(5, len(pdf.pages))):
                p_text = pdf.pages[p_idx].extract_text() or ""
                for line in p_text.splitlines():
                    m = _PERIOD_RE.search(line)
                    if m:
                        period_year = int(m.group(1))
                        if period_year > target_year:
                            logger.warning(
                                f"Rejecting PDF for {quarter_label}: reporting period year "
                                f"{period_year} > target year {target_year} on page {p_idx} "
                                f"(line: {line.strip()!r}). Likely BSE silent redirect."
                            )
                            return None, True, True  # rejected=True → do NOT cache

        for page_num, page in enumerate(pdf.pages):
            if page_num >= MAX_PAGES_PER_PDF:
                break
            remaining = _seconds_left(deadline)
            if remaining is not None and remaining <= 0:
                completed = False
                break
            text = page.extract_text() or ""
            
            # Stateful mode tracking: carry forward classification to continuation pages
            mode = _page_mode(text.lower())
            if mode is not None:
                current_mode = mode
            else:
                mode = current_mode
                
            if "diluted" not in text.lower():
                continue
            value = _extract_diluted_value(page, expected_basic)
            if value is None:
                continue
                
            if mode == wanted:
                # Two agreeing pages settle it early.
                if any(abs(value - prev) < 0.01 for prev in candidates):
                    return value, True, False
                candidates.append(value)
            else:
                priority = 1 if mode is None else 0
                if fallback is None or priority > fallback[0]:
                    fallback = (priority, value)

    if candidates:
        # Most frequent value wins; on a tie prefer the later occurrence, since
        # statutory tables follow the press-release summaries.
        rounded = [round(v, 2) for v in candidates]
        counts = {v: rounded.count(v) for v in rounded}
        best_idx = max(range(len(candidates)), key=lambda i: (counts[rounded[i]], i))
        return candidates[best_idx], completed, False

    return (fallback[1] if fallback else None), completed, False


def _quarter_pdf_links(symbol: str, consolidated: bool, timeout: float) -> list[tuple[str, str]]:
    """
    Reads the 'Raw PDF' row of Screener's quarters table and returns
    [(quarter_label, pdf_url), ...] — one result-filing link per quarter column.
    """
    url = f"{SCREENER_BASE}/company/{symbol}/consolidated/" if consolidated \
        else f"{SCREENER_BASE}/company/{symbol}/"
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    if consolidated and resp.status_code == 404:
        resp = requests.get(f"{SCREENER_BASE}/company/{symbol}/", headers=HEADERS, timeout=timeout)
    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    quarters_section = soup.find("section", id="quarters")
    if not quarters_section:
        return []
    table = quarters_section.find("table")
    if not table:
        return []

    header_row = table.find("thead").find("tr") if table.find("thead") else table.find("tr")
    if not header_row:
        return []
    quarter_labels = [c.get_text(strip=True) for c in header_row.find_all(["th", "td"])][1:]

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        if "raw pdf" not in cells[0].get_text(strip=True).lower():
            continue
        links = []
        for i, cell in enumerate(cells[1:]):
            if i >= len(quarter_labels):
                break
            anchor = cell.find("a", href=True)
            if not anchor:
                continue
            href = anchor["href"]
            if href.startswith("/"):
                href = SCREENER_BASE + href
            links.append((quarter_labels[i], href))
        return links

    return []


def fetch_diluted_eps_bse(company: dict, from_q: tuple[int, int], to_q: tuple[int, int],
                          consolidated: bool = True, deadline: float = None,
                          expected_basic: dict | None = None) -> dict:
    """
    Extracts filed diluted EPS per quarter from the result PDFs that Screener
    links in the 'Raw PDF' row of the quarters table (each link 302-redirects to
    the official BSE filing for exactly that quarter). Respects `deadline`
    (time.monotonic() timestamp): quarters are processed newest-first and the
    lookup stops as soon as the budget runs out, returning what it has.

    `expected_basic` maps quarter labels to Screener's basic EPS; when present
    for a quarter, extraction is anchored on the PDF's Basic-EPS row (see
    _extract_diluted_value), which rejects wrong-column, wrong-basis, and
    wrong-period figures.
    """
    from backend.scraper import parse_quarter

    symbol = company.get("symbol", "").upper()
    results = {}
    if not symbol:
        return results

    try:
        remaining = _seconds_left(deadline)
        if remaining is not None and remaining <= 0:
            return results
        links = _quarter_pdf_links(symbol, consolidated,
                                   timeout=10 if remaining is None else min(10, remaining))
    except Exception as e:
        logger.error(f"Error fetching Screener quarters page for {symbol}: {e}")
        return results

    wanted = []
    for label, pdf_url in links:
        try:
            if from_q <= parse_quarter(label) <= to_q:
                wanted.append((label, pdf_url))
        except ValueError:
            continue

    # Newest quarters first, so the most relevant data survives a tight budget.
    to_fetch = []
    for label, pdf_url in reversed(wanted):
        cache_key = (symbol, label, consolidated)
        if cache_key in _EPS_CACHE:
            cached = _EPS_CACHE[cache_key]
            if cached is not None:
                results[label] = cached
            continue
        to_fetch.append((label, pdf_url, cache_key))

    if not to_fetch:
        return results

    def _download(pdf_url: str) -> bytes | None:
        remaining = _seconds_left(deadline)
        if remaining is not None and remaining <= 0:
            return None
        resp = requests.get(pdf_url, headers=HEADERS,
                            timeout=15 if remaining is None else min(15, remaining))
        if resp.status_code != 200:
            return None
        return resp.content

    # Downloads are pure I/O against BSE's slow CDN, so run them concurrently
    # but keep workers low (2) to avoid triggering BSE rate-limiting, which
    # causes silent redirects to wrong PDFs.
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        futures = {executor.submit(_download, url): (label, key)
                   for label, url, key in to_fetch}
        while futures:
            remaining = _seconds_left(deadline)
            if remaining is not None and remaining <= 0:
                logger.info(f"Deadline reached; stopping BSE PDF lookup for {symbol} "
                            f"with {len(futures)} quarter(s) unresolved.")
                break
            try:
                done = next(as_completed(futures, timeout=remaining))
            except FuturesTimeout:
                logger.info(f"Deadline reached awaiting PDF downloads for {symbol}; "
                            f"{len(futures)} quarter(s) unresolved.")
                break
            label, cache_key = futures.pop(done)
            try:
                content = done.result()
            except Exception as e:
                logger.warning(f"Error downloading result PDF for {symbol} {label}: {e}")
                continue
            if content is None:
                continue
            if len(content) > MAX_PDF_BYTES:
                logger.info(f"Skipping oversized result PDF for {symbol} {label} "
                            f"({len(content)} bytes).")
                _EPS_CACHE[cache_key] = None
                continue
            try:
                value, completed, rejected = _extract_diluted_from_pdf(
                    content, consolidated, label, deadline,
                    expected_basic=(expected_basic or {}).get(label))
            except Exception as e:
                logger.warning(f"Error parsing result PDF for {symbol} {label}: {e}")
                continue
            if value is not None:
                info = {"value": value, "kind": "Diluted", "xbrl": False}
                _EPS_CACHE[cache_key] = info
                results[label] = info
            elif completed and not rejected:
                # Only cache a miss when the PDF was valid but genuinely had no EPS.
                # Do NOT cache when rejected due to year mismatch (BSE redirect) so
                # the quarter can be retried on the next request.
                _EPS_CACHE[cache_key] = None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results
