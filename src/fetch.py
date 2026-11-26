"""National quarterly Medicaid drug units by product, 2018-2025, from the State Drug Utilization Data.

State 'XX' is the national total row set. Units are summed over utilization types (FFS and managed care).
"""
import json, urllib.request, csv, time

RES = {2018: "0c9dc15a-0894-5db6-b364-82e4b8c6354f", 2019: "384def39-ab0c-5e48-b311-1981cf9a38a7",
       2020: "0b248e6e-68ad-574b-afff-94f3b91ec77e", 2021: "99089534-5317-54ee-85a8-b11558e4c907",
       2022: "54f6377c-5819-50cb-8b11-fc9a64e60246", 2023: "df8055e5-3a20-5434-8a89-0e16073c8134",
       2024: "7c764387-5eaf-57da-bc5e-b9b7e0b86bad", 2025: "a613f4ae-0615-572c-812a-a381379914cd"}


def q(rid, offset):
    body = {"properties": ["product_name", "year", "quarter",
                           {"expression": {"operator": "sum", "operands": [{"expression": {"operator": "+", "operands": ["units_reimbursed", 0]}}]}, "alias": "units"},
                           {"expression": {"operator": "sum", "operands": [{"expression": {"operator": "+", "operands": ["number_of_prescriptions", 0]}}]}, "alias": "rx"}],
            "conditions": [{"property": "state", "value": "XX"}],
            "groupings": ["product_name", "year", "quarter"], "limit": 5000, "offset": offset, "schema": False}
    req = urllib.request.Request(f"https://data.medicaid.gov/api/1/datastore/query/{rid}",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for i in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["results"]
        except Exception:
            time.sleep(3)
    raise RuntimeError("failed")


if __name__ == "__main__":
    import sys
    years = [int(a) for a in sys.argv[1:]] or list(RES)
    for yr in years:
        rid = RES[yr]
        with open(f"data/units_{yr}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["product_name", "year", "quarter", "units", "rx"])
            off = 0
            while True:
                rows = q(rid, off)
                for r in rows:
                    w.writerow([r["product_name"].strip(), r["year"], r["quarter"], r["units"], r["rx"]])
                print(yr, off, len(rows), flush=True)
                if len(rows) < 5000: break
                off += 5000
