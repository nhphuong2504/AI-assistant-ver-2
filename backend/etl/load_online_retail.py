import pandas as pd
import sqlite3
from pathlib import Path

RAW_PATH = Path("data/raw/online_retail.csv")
DB_PATH = Path("data/retail.sqlite")

REQUIRED_COLS = {
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country"
}

def load_csv() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)

    # normalize column names: remove spaces, trim
    df.columns = [c.strip().replace(" ", "") for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    # Parse datetime 
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=False)

    return df

def clean_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    # Flags
    df["is_cancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
    df["is_missing_customer"] = df["CustomerID"].isna()
    df["is_return"] = df["Quantity"] <= 0
    df["is_invalid_price"] = df["UnitPrice"] <= 0
    df["is_bad_date"] = df["InvoiceDate"].isna()

    # Revenue
    df["revenue"] = df["Quantity"] * df["UnitPrice"]

    # "clean" inclusion for analytics
    df["include_clean"] = ~(
        df["is_cancelled"]
        | df["is_missing_customer"]
        | df["is_return"]
        | df["is_invalid_price"]
        | df["is_bad_date"]
    )

    # Types
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    # Rename to snake_case for DB
    df = df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    })

    return df

def write_sqlite(df: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    with sqlite3.connect(DB_PATH) as conn:
        # Audit table (everything + flags)
        df.to_sql("transactions_all", conn, index=False)

        # Clean table for analytics
        clean_cols = [
            "invoice_no", "stock_code", "description", "quantity",
            "invoice_date", "unit_price", "customer_id", "country", "revenue"
        ]
        df_clean = df.loc[df["include_clean"], clean_cols]
        df_clean.to_sql("transactions", conn, index=False)

        # Indexes for speed (Text-to-SQL and analytics)
        conn.executescript("""
        CREATE INDEX idx_tx_customer_date ON transactions(customer_id, invoice_date);
        CREATE INDEX idx_tx_invoice ON transactions(invoice_no);
        CREATE INDEX idx_tx_stock ON transactions(stock_code);
        """)

def main():
    df = load_csv()
    df = clean_and_flag(df)
    write_sqlite(df)

    print("ETL complete")
    print(f"Total rows: {len(df):,}")
    print(f"Clean rows: {int(df['include_clean'].sum()):,}")
    print("Date range (raw parsed):",
          df["invoice_date"].min(), "→", df["invoice_date"].max())

if __name__ == "__main__":
    main()
