import yfinance as yf

candidates = {
    "SBI ETF Nifty 50": ["SETFNIF50.NS", "SBIETFIT.NS", "NIFTYBEES.NS"],
    "HDFC Sensex ETF": ["HDFCSENSEX.NS", "HDFCSENETF.NS", "SENSEXBEES.NS"],
    "SBI ETF Bank": ["SETFNIFBK.NS", "BANKBEES.NS"],
    "Nippon India ETF IT": ["NETFIT.NS", "ITBEES.NS"],
    "Nippon India ETF Pharma": ["NETFPHARMA.NS", "PHARMABEES.NS"],
    "Nippon India Junior BeES": ["JUNIORBEES.NS"],
    "Motilal Oswal Midcap ETF": ["MOM50.NS", "MID150BEES.NS", "MOM100.NS", "MIDCAP.NS"],
    "CPSE / PSU ETF": ["CPSEETF.NS", "PSUBNKBEES.NS"]
}

for name, tickers in candidates.items():
    print(f"\nTesting {name}:")
    for t in tickers:
        try:
            d = yf.download(t, period="5y", progress=False)
            if not d.empty and len(d) > 1000:
                print(f"  [SUCCESS] {t} ({len(d)} days)")
            else:
                print(f"  [WARN] {t} returned {len(d)} days")
        except Exception as e:
            print(f"  [FAIL] {t} - {type(e).__name__}")
