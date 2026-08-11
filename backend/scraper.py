import requests
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import date, timedelta
import calendar
from functools import lru_cache
import io

COMPANIES_DB = []
BSE_BHAVCOPY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bseindia.com/markets/MarketInfo/BhavCopy",
}

def load_companies_db():
    global COMPANIES_DB
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "ISINS.xlsx")
    if not os.path.exists(path):
        print(f"Warning: ISINS.xlsx mapping database not found at {path}")
        return []
    try:
        # Load excel file
        df = pd.read_excel(path)
        
        # Match columns case-insensitively
        cols = {str(c).strip().lower(): c for c in df.columns}
        
        # Match ISIN
        isin_col = None
        for k, original_col in cols.items():
            if "isin" in k:
                isin_col = original_col
                break
                
        # Match Symbol
        symbol_col = None
        for k, original_col in cols.items():
            if "symbol" in k or "nse" in k or "ticker" in k:
                symbol_col = original_col
                break
                
        # Match BSE Code
        bse_col = None
        for k, original_col in cols.items():
            if "bse" in k or "scrip" in k or "code" in k or "number" in k:
                bse_col = original_col
                break

        # Match Company Name
        name_col = None
        for k, original_col in cols.items():
            if "name" in k or "company" in k:
                name_col = original_col
                break
                
        if not isin_col or not symbol_col:
            # Fallback
            isin_col = df.columns[0]
            symbol_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            bse_col = df.columns[2] if len(df.columns) > 2 else None
            name_col = df.columns[3] if len(df.columns) > 3 else None
            
        companies = []
        for idx, row in df.iterrows():
            isin_val = str(row[isin_col]).strip() if pd.notna(row[isin_col]) else ""
            symbol_val = str(row[symbol_col]).strip() if pd.notna(row[symbol_col]) else ""
            name_val = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else ""
            bse_val = ""
            if bse_col and pd.notna(row[bse_col]):
                # Remove decimals (e.g. 500325.0 -> 500325)
                bse_val = str(row[bse_col]).split(".")[0].strip()
                
            if isin_val or symbol_val:
                companies.append({
                    "symbol": symbol_val.upper(),
                    "name": name_val or symbol_val.upper(),
                    "isin": isin_val.upper(),
                    "bse": bse_val
                })
        COMPANIES_DB = companies
        print(f"Successfully loaded {len(COMPANIES_DB)} companies from mapping database.")
        return COMPANIES_DB
    except Exception as e:
        print("Warning: Could not read ISIN mapping database:", e)
        COMPANIES_DB = []
        return COMPANIES_DB

# Run loader on startup
load_companies_db()

def get_company_by_symbol(symbol: str) -> dict:
    """
    Looks up BSE code and ISIN for a symbol from our loaded COMPANIES_DB.
    """
    symbol_clean = symbol.strip().upper()
    for c in COMPANIES_DB:
        if c["symbol"] == symbol_clean:
            return c
    return {
        "symbol": symbol_clean,
        "name": symbol_clean,
        "isin": "N/A",
        "bse": "N/A"
    }


def _empty_company_meta(query: str) -> dict:
    query_clean = query.strip().upper()
    isin_value = query_clean if len(query_clean) == 12 and query_clean.startswith("IN") else "N/A"
    return {
        "symbol": query_clean,
        "name": query_clean,
        "isin": isin_value,
        "bse": "N/A"
    }


def _normalize_name(s: str) -> str:
    """
    Normalises a company name for fuzzy matching: lowercase, punctuation to
    spaces, and drops noise words ('limited', 'ltd', 'the', '&', 'and').
    """
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(limited|ltd|the|and)\b", " ", s)
    return " ".join(s.split())


def _match_by_name(query: str) -> dict | None:
    """
    Resolves a company by (partial) name against COMPANIES_DB. Prefers an exact
    normalised match, then the shortest name that starts with the query, then the
    shortest name containing all query tokens. Returns None if nothing matches or
    the query is too short to be a meaningful name.
    """
    q = _normalize_name(query)
    if len(q) < 3:
        return None
    q_tokens = q.split()

    exact = None
    prefix_best = None      # (name_length, company)
    tokens_best = None      # (name_length, company)

    for company in COMPANIES_DB:
        n = _normalize_name(company.get("name", ""))
        if not n:
            continue
        if n == q:
            exact = company
            break
        if n.startswith(q):
            if prefix_best is None or len(n) < prefix_best[0]:
                prefix_best = (len(n), company)
        elif all(tok in n.split() for tok in q_tokens):
            if tokens_best is None or len(n) < tokens_best[0]:
                tokens_best = (len(n), company)

    if exact:
        return exact
    if prefix_best:
        return prefix_best[1]
    if tokens_best:
        return tokens_best[1]
    return None


def resolve_company(query: str) -> dict:
    """
    Resolves a user-entered symbol, ISIN, BSE code, or company name to a company
    metadata record. Prefers the local mapping database (exact code match, then
    fuzzy name match), then falls back to BSE's official search API.
    """
    query_clean = query.strip().upper()

    for company in COMPANIES_DB:
        if query_clean and query_clean in (company["symbol"], company["isin"], company["bse"]):
            return company

    # Local fuzzy name match (works offline; more reliable than the BSE search
    # API, which is often blocked from cloud/datacenter IPs).
    name_match = _match_by_name(query)
    if name_match:
        return name_match

    try:
        search_url = (
            "https://api.bseindia.com/BseIndiaAPI/api/GetQuoteAllSearchDatabeta/w"
            f"?searchString={quote_plus(query_clean)}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bseindia.com/",
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, list) and payload:
                equity_match = next(
                    (
                        item
                        for item in payload
                        if str(item.get("Type", "")).lower().startswith("in equity")
                    ),
                    payload[0],
                )
                symbol_value = str(equity_match.get("shortName") or "").strip().upper()
                bse_value = str(equity_match.get("strSricpCode") or "").split(".")[0].strip()
                isin_value = str(equity_match.get("Isin") or "").strip().upper()
                name_value = str(equity_match.get("scripName") or equity_match.get("shortName") or "").strip()

                if symbol_value:
                    return {
                        "symbol": symbol_value,
                        "name": name_value or symbol_value,
                        "isin": isin_value or (query_clean if len(query_clean) == 12 and query_clean.startswith("IN") else "N/A"),
                        "bse": bse_value or "N/A",
                    }
    except Exception as e:
        print(f"Warning: Failed to resolve '{query_clean}' via BSE official search: {e}")

    return _empty_company_meta(query_clean)


@lru_cache(maxsize=256)
def _download_bse_stock_history(company_code: str, from_date: date, to_date: date) -> dict | None:
    """
    Downloads the official BSE stock history payload for a script code and date range.
    Returns None when the API is unavailable or returns an unexpected payload.
    """
    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/StockpricesearchData/w"
        f"?MonthDate={from_date.strftime('%d/%m/%Y')}"
        f"&YearDate={to_date.strftime('%d/%m/%Y')}"
        f"&pageType=0&Scode={quote_plus(company_code)}&Seg=C&rbType=D"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori?expandable=7&scripcode=500325&flag=sp&Submit=G",
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code != 200:
            return None
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("StockData"), list):
            return payload
    except Exception:
        return None

    return None


def get_bse_month_end_price(company: dict, year: int, month: int, lookback_days: int = 10) -> dict:
    """
    Returns the end-of-month closing price for a company from the official BSE bhavcopy.
    If the calendar month-end is not a trading day, the search walks backward until a
    trading session is found.
    """
    company_symbol = company.get("symbol", "").strip().upper()
    company_isin = company.get("isin", "").strip().upper()
    company_bse = str(company.get("bse", "")).strip().split(".")[0]
    if not company_bse:
        return {
            "price": None,
            "trade_date": None,
        }

    from_date = date(year, month, 1)
    to_date = date(year, month, calendar.monthrange(year, month)[1])

    payload = _download_bse_stock_history(company_bse, from_date, to_date)
    if payload:
        stock_data = payload.get("StockData", [])
        if stock_data:
            first_row = stock_data[0]
            close_raw = first_row.get("qe_close")
            trade_date_raw = first_row.get("Dates")

            try:
                close_price = float(str(close_raw).replace(",", "").strip()) if close_raw not in (None, "") else None
            except Exception:
                close_price = None

            return {
                "price": close_price,
                "trade_date": trade_date_raw,
            }

    return {
        "price": None,
        "trade_date": None,
    }


def get_bse_month_end_prices(company: dict, quarter_labels: list[str]) -> dict:
    """
    Returns a quarter-label to month-end price mapping for a company.
    """
    prices = {}
    for quarter_label in quarter_labels:
        year, month = parse_quarter(quarter_label)
        prices[quarter_label] = get_bse_month_end_price(company, year, month)["price"]
    return prices


def parse_quarter(q_str: str) -> tuple[int, int]:
    """
    Parses a quarter string like 'Mar 2023' into a tuple (year, month_number)
    for easy comparison and sorting.
    """
    q_str = q_str.strip()
    parts = q_str.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid quarter format: '{q_str}'. Expected format like 'Mar 2023'.")
    
    month_str, year_str = parts
    months = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    
    month_key = month_str[:3].title()
    if month_key not in months:
        raise ValueError(f"Invalid month name: '{month_str}' in quarter '{q_str}'.")
    
    try:
        year = int(year_str)
    except ValueError:
        raise ValueError(f"Invalid year: '{year_str}' in quarter '{q_str}'.")
        
    return year, months[month_key]

def scrape_screener_quarters(symbol: str, consolidated: bool = True, period: str = "quarterly") -> list[dict]:
    """
    Scrapes financial data of a company from Screener.in.
    `period` selects the quarterly results table ("quarterly", <section id="quarters">)
    or the annual profit & loss table ("yearly", <section id="profit-loss">).
    Returns a list of dicts with keys: Symbol, Quarter (the period label, e.g.
    'Sep 2024' for a quarter or 'Mar 2024' for FY2024), Sales, Net Profit, EPS in Rs.
    """
    symbol = symbol.strip().upper()
    period = (period or "quarterly").lower()
    yearly = period in ("yearly", "annual", "year")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # If consolidated is selected, we request the consolidated page first.
    # Screener has pages like /company/RELIANCE/consolidated/ and fallback /company/RELIANCE/
    if consolidated:
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
    else:
        url = f"https://www.screener.in/company/{symbol}/"

    try:
        response = requests.get(url, headers=headers, timeout=10)

        # If we got a 404 on the consolidated URL, fall back to standalone
        if consolidated and response.status_code == 404:
            url = f"https://www.screener.in/company/{symbol}/"
            response = requests.get(url, headers=headers, timeout=10)

    except requests.RequestException as e:
        raise Exception(f"Network error while connecting to Screener.in: {str(e)}")

    if response.status_code == 404:
        raise Exception(f"Company symbol '{symbol}' was not found on Screener.in.")
    elif response.status_code != 200:
        raise Exception(f"Screener.in returned status code {response.status_code} for symbol '{symbol}'.")

    soup = BeautifulSoup(response.content, "html.parser")

    # Quarterly data lives in <section id="quarters">; annual (FY) data in
    # <section id="profit-loss">. Both tables share the same row/column layout.
    section_id = "profit-loss" if yearly else "quarters"
    data_section = soup.find("section", id=section_id)
    if not data_section:
        kind = "Annual (profit & loss)" if yearly else "Quarterly financials"
        raise Exception(f"{kind} section not found for '{symbol}'. The company might not report it.")

    table = data_section.find("table")
    if not table:
        raise Exception(f"Data table not found in {section_id} section for '{symbol}'.")
        
    # Parse headers (quarter names)
    header_row = None
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
    if not header_row:
        header_row = table.find("tr")
        
    if not header_row:
        raise Exception("Failed to parse header row from quarters table.")
        
    cols = header_row.find_all(["th", "td"])
    raw_headers = [col.get_text(strip=True) for col in cols]
    
    if len(raw_headers) <= 1:
        raise Exception("No quarterly columns found in quarters table.")
        
    # The first column is usually empty or contains row descriptions
    quarter_names = raw_headers[1:]
    
    # Parse data rows
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")
    
    data_by_metric = {}
    for row in rows:
        row_cols = row.find_all("td")
        if not row_cols:
            continue
        # Standardize the row name by removing '+' sign (used for expanding breakdown in Screener UI)
        label = row_cols[0].get_text(strip=True).rstrip("+").strip()
        values = [col.get_text(strip=True) for col in row_cols[1:]]
        data_by_metric[label.lower()] = values
        
    # Define keys for matching
    sales_keys = ["sales", "revenue"]
    net_profit_keys = ["net profit"]
    eps_keys = ["eps in rs", "eps in rs.", "eps"]
    
    sales_vals = None
    for k in sales_keys:
        if k in data_by_metric:
            sales_vals = data_by_metric[k]
            break
            
    net_profit_vals = None
    for k in net_profit_keys:
        if k in data_by_metric:
            net_profit_vals = data_by_metric[k]
            break
            
    eps_vals = None
    for k in eps_keys:
        if k in data_by_metric:
            eps_vals = data_by_metric[k]
            break
            
    if sales_vals is None or net_profit_vals is None or eps_vals is None:
        missing = []
        if sales_vals is None: missing.append("Sales")
        if net_profit_vals is None: missing.append("Net Profit")
        if eps_vals is None: missing.append("EPS in Rs")
        raise Exception(f"Required quarterly metrics {missing} not found on Screener.in for symbol '{symbol}'.")
        
    records = []
    for i, q in enumerate(quarter_names):
        # Skip non-period columns (the annual table ends with a "TTM" column, and
        # any header that isn't a 'Mon YYYY' label can't be filtered or priced).
        try:
            parse_quarter(q)
        except ValueError:
            continue

        # Safe access in case row data length is shorter than header length
        s_val = sales_vals[i] if i < len(sales_vals) else ""
        np_val = net_profit_vals[i] if i < len(net_profit_vals) else ""
        eps_val = eps_vals[i] if i < len(eps_vals) else ""
        
        def clean_value(v):
            v_clean = v.replace(",", "").replace("%", "").strip()
            if not v_clean or v_clean == "-" or v_clean == "":
                return None
            try:
                if "." in v_clean:
                    return float(v_clean)
                return int(v_clean)
            except ValueError:
                return v_clean
                
        records.append({
            "Symbol": symbol,
            "Quarter": q,
            "Sales": clean_value(s_val),
            "Net Profit": clean_value(np_val),
            "EPS in Rs": clean_value(eps_val)
        })
        
    return records

def filter_quarters(records: list[dict], from_quarter: str, to_quarter: str) -> list[dict]:
    """
    Filters the scraped records based on the user's selected From and To quarter range.
    Raises Exception if requested quarter(s) fall outside the available rolling quarters
    exposed by Screener's free tier.
    """
    if not records:
        return []
        
    # Get available quarter names
    available_quarters = [r["Quarter"] for r in records]
    
    # Parse available quarters to dates for boundary checking
    parsed_available = []
    for q in available_quarters:
        try:
            parsed_available.append(parse_quarter(q))
        except ValueError:
            # Skip unparseable quarters if any (unlikely in screener)
            pass
            
    if not parsed_available:
        raise Exception("Could not parse any quarter labels from Screener.in data.")
        
    # Available quarters are sorted chronologically
    sorted_pairs = sorted(zip(parsed_available, available_quarters), key=lambda x: x[0])
    parsed_sorted = [x[0] for x in sorted_pairs]
    labels_sorted = [x[1] for x in sorted_pairs]
    
    oldest_available_parsed = parsed_sorted[0]
    newest_available_parsed = parsed_sorted[-1]
    
    oldest_label = labels_sorted[0]
    newest_label = labels_sorted[-1]
    
    # Parse requested range
    try:
        req_from_parsed = parse_quarter(from_quarter)
    except ValueError as e:
        raise Exception(f"Invalid From Quarter selected: {str(e)}")
        
    try:
        req_to_parsed = parse_quarter(to_quarter)
    except ValueError as e:
        raise Exception(f"Invalid To Quarter selected: {str(e)}")
        
    if req_from_parsed > req_to_parsed:
        raise Exception(f"From Quarter ({from_quarter}) cannot be later than To Quarter ({to_quarter}).")
        
    # Check if the requested range falls outside the free tier availability
    if req_from_parsed < oldest_available_parsed:
        raise Exception(
            f"The requested start quarter '{from_quarter}' is not available in the free tier of Screener.in. "
            f"Screener's free tier only exposes the last ~13 rolling quarters (starting from '{oldest_label}' for this company). "
            f"Please select a 'From' quarter of '{oldest_label}' or later, or use Screener Premium."
        )
        
    if req_to_parsed > newest_available_parsed:
        raise Exception(
            f"The requested end quarter '{to_quarter}' is not available. "
            f"The latest quarter available for this company is '{newest_label}'. "
            f"Please select '{newest_label}' or an earlier quarter as the 'To' quarter."
        )
        
    # Filter the records
    filtered_records = []
    for r in records:
        try:
            pq = parse_quarter(r["Quarter"])
            if req_from_parsed <= pq <= req_to_parsed:
                filtered_records.append(r)
        except ValueError:
            # Skip any unparseable quarter labels
            pass
            
    # Sort records chronologically
    filtered_records = sorted(filtered_records, key=lambda r: parse_quarter(r["Quarter"]))
    
    return filtered_records

def generate_yoy_comparison(all_records: list[dict], filtered_records: list[dict]) -> list[dict]:
    """
    For each record in the filtered list, looks up the corresponding quarter from the 
    previous year in the full list of records and generates comparison fields + YoY % changes.
    """
    comparison_records = []
    
    for r in filtered_records:
        q_curr = r["Quarter"]
        # Find corresponding quarter from previous year (same month, year - 1)
        parts = q_curr.strip().split()
        if len(parts) != 2:
            continue
        month, year_str = parts
        try:
            year = int(year_str)
            q_prev_label = f"{month} {year - 1}"
        except ValueError:
            q_prev_label = None
            
        r_prev = None
        if q_prev_label:
            r_prev = next((x for x in all_records if x["Quarter"] == q_prev_label), None)
            
        # Format comparison label (e.g. "Mar 23 vs 24")
        comp_label = q_curr
        if q_prev_label:
            try:
                curr_yr_short = year_str[-2:]
                prev_yr_short = str(year - 1)[-2:]
                comp_label = f"{month} {prev_yr_short} vs {curr_yr_short}"
            except Exception:
                pass
                
        sales_curr = r.get("Sales")
        sales_prev = r_prev.get("Sales") if r_prev else None
        sales_yoy = None
        if sales_curr is not None and sales_prev is not None and sales_prev != 0:
            sales_yoy = round(((sales_curr - sales_prev) / sales_prev) * 100, 2)
            
        np_curr = r.get("Net Profit")
        np_prev = r_prev.get("Net Profit") if r_prev else None
        np_yoy = None
        if np_curr is not None and np_prev is not None and np_prev != 0:
            np_yoy = round(((np_curr - np_prev) / np_prev) * 100, 2)
            
        eps_curr = r.get("EPS in Rs")
        eps_prev = r_prev.get("EPS in Rs") if r_prev else None
        eps_yoy = None
        if eps_curr is not None and eps_prev is not None and eps_prev != 0:
            eps_yoy = round(((eps_curr - eps_prev) / eps_prev) * 100, 2)
            
        comparison_records.append({
            "Symbol": r["Symbol"],
            "Comparison": comp_label,
            "Quarter": q_curr,
            "Prev Quarter": q_prev_label or "N/A",
            
            "Sales (Current)": sales_curr,
            "Sales (Previous)": sales_prev,
            "Sales YoY (%)": sales_yoy,
            
            "Net Profit (Current)": np_curr,
            "Net Profit (Previous)": np_prev,
            "Net Profit YoY (%)": np_yoy,
            
            "EPS (Current)": eps_curr,
            "EPS (Previous)": eps_prev,
            "EPS YoY (%)": eps_yoy
        })
        
    return comparison_records

def resolve_symbol(query: str) -> str:
    """
    Attempts to resolve an input query (Symbol, BSE Code, or ISIN) to its clean symbol.
    First checks the local mapping database, then falls back to BSE's official search API.
    """
    return resolve_company(query)["symbol"]


