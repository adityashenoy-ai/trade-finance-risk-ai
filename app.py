# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from openai import OpenAI

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Trade Finance Risk Intelligence", layout="wide")
st.title("⚖️ AI Trade Finance Risk Intelligence System (SMEs)")
st.markdown("Ingest GST, Shipments, Bank statements, FX exposures → Outputs: default probability, EW signals, risk-adjusted loan, automated risk memo")

# ---------------- OpenAI ----------------
if "OPENAI_API_KEY" not in st.secrets:
    st.warning("Add OPENAI_API_KEY to Streamlit Secrets to enable LLM-generated memos. You can still use heuristic scoring without it.")
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------- Sidebar inputs ----------------

import os
import pandas as pd

st.sidebar.subheader("Data Source")

use_sample = st.sidebar.checkbox("Use Sample Data Instead of Uploading Files")

sample_path = "sample_data/"   # folder inside repo

if use_sample:
    gst_df = pd.read_csv(os.path.join(sample_path, "sample_gst.csv"))
    ship_df = pd.read_csv(os.path.join(sample_path, "sample_shipments.csv"))
    bank_df = pd.read_csv(os.path.join(sample_path, "sample_bank.csv"))
    fx_df = pd.read_csv(os.path.join(sample_path, "sample_fx.csv"))
    profile_df = pd.read_csv(os.path.join(sample_path, "sample_profile.csv"))

    st.success("Loaded sample datasets from the repository!")
else:
    gst_file = st.file_uploader("Upload GST CSV")
    ship_file = st.file_uploader("Upload Shipment CSV")
    bank_file = st.file_uploader("Upload Bank Statement CSV")
    fx_file = st.file_uploader("Upload FX Exposure CSV")
    profile_file = st.file_uploader("Upload SME Profile CSV")

    if gst_file and ship_file and bank_file and fx_file and profile_file:
        gst_df = pd.read_csv(gst_file)
        ship_df = pd.read_csv(ship_file)
        bank_df = pd.read_csv(bank_file)
        fx_df = pd.read_csv(fx_file)
        profile_df = pd.read_csv(profile_file)
    else:
        st.warning("Upload all 5 files OR enable 'Use Sample Data'.")
        st.stop()


with st.sidebar:
    st.header("Upload Data (CSV)")
    gst_file = st.file_uploader("GST / Sales CSV (date, taxable_value, gstin, gstin_buyer optional)", type=["csv"])
    shipments_file = st.file_uploader("Shipment history CSV (shipment_id, date, carrier, status, delay_days, value)", type=["csv"])
    bank_file = st.file_uploader("Bank statements CSV (date, balance, inflow, outflow)", type=["csv"])
    fx_file = st.file_uploader("FX exposure CSV (date, currency, amount_in_base, direction)", type=["csv"])
    profile_file = st.file_uploader("Company profile CSV (single row) (name, entity_type, annual_revenue, region)", type=["csv"])

    st.markdown("---")
    st.write("Or generate realistic sample datasets to try the app:")
    if st.button("Generate sample datasets (GST, Shipments, Bank, FX, Profile)"):
        # we'll create in-memory CSVs below in main area

        st.success("Sample datasets generated below. Use the download buttons to get CSVs or click 'Use sample data' in the main app.")
    st.markdown("---")
    st.write("Model & options")
    model = st.selectbox("LLM Model (for memos)", options=["gpt-4o-mini","gpt-4o"])
    use_llm = st.checkbox("Enable LLM memos (uses API key)", value=True if client else False)
    sample_mode = st.checkbox("LLM: short/sample mode (faster, cheaper)", value=True)

# ---------------- Helper functions ----------------
def generate_sample_gst(rows=36):
    rng = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="M")
    data = []
    for d in rng:
        taxable = int(np.random.normal(3_00_000, 1_00_000))
        if taxable < 50_000: taxable = max(50_000, taxable)
        data.append({"date": d.strftime("%Y-%m-%d"), "taxable_value": taxable, "gstin": "27AAAPL0001A1Z5"})
    return pd.DataFrame(data)

def generate_sample_shipments(rows=120):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="D")
    data = []
    for i, d in enumerate(dates):
        value = int(np.random.uniform(10_000, 2_00_000))
        delay = int(np.random.choice([0]*80 + [1]*10 + [3]*7 + [7]*3))  # mostly on-time
        status = "delivered" if delay < 7 else "issue"
        data.append({"shipment_id": f"SHP{i+1}", "date": d.strftime("%Y-%m-%d"), "carrier": "CarrierX", "status": status, "delay_days": delay, "value": value})
    return pd.DataFrame(data)

def generate_sample_bank(rows=180):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="D")
    balance = 500_000
    data = []
    for d in dates:
        inflow = int(np.random.normal(2_00_000, 70_000))
        outflow = int(np.random.normal(1_80_000, 60_000))
        balance = max(0, balance + inflow - outflow + np.random.randint(-20_000, 20_000))
        data.append({"date": d.strftime("%Y-%m-%d"), "balance": balance, "inflow": inflow, "outflow": outflow})
    return pd.DataFrame(data)

def generate_sample_fx(rows=12):
    months = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="M")
    data = []
    for m in months:
        amount = float(np.random.uniform(-50_000, 150_000))  # negative = net short, positive = net long
        currency = np.random.choice(["USD","EUR","AED"])
        data.append({"date": m.strftime("%Y-%m-%d"), "currency": currency, "amount_in_base": amount, "direction": "long" if amount>0 else "short"})
    return pd.DataFrame(data)

def generate_sample_profile():
    return pd.DataFrame([{
        "name":"MicroTrade Exports",
        "entity_type":"SME Exporter",
        "annual_revenue": 9_000_000,
        "region":"Karnataka",
        "main_customers":"Middle East",
        "gstin":"29AAAPL0001A1Z5"
    }])

# Heuristic risk scoring function
def heuristic_risk_score(profile, gst_df=None, ship_df=None, bank_df=None, fx_df=None, country_risk=40):
    # profile: dict with annual_revenue
    score = 0.0
    # base: smaller revenue -> higher risk
    rev = float(profile.get("annual_revenue", 1_000_000))
    rev_score = np.interp(rev, [0, 50_000_000, 200_000_000], [100, 40, 10])  # more revenue -> lower risk
    score += 0.4 * rev_score

    # shipment delays
    if ship_df is not None and len(ship_df):
        delays = ship_df["delay_days"].fillna(0)
        pct_delayed = (delays>0).mean() * 100
        delay_score = np.interp(pct_delayed, [0, 30, 80], [0, 30, 80])
    else:
        delay_score = 10
    score += 0.2 * delay_score

    # bank liquidity: low average balance relative to monthly revenue -> risk
    avg_balance = None
    if bank_df is not None and len(bank_df):
        avg_balance = bank_df["balance"].mean()
        monthly_revenue = gst_df["taxable_value"].resample("M", on="date").sum().mean() if (gst_df is not None and "date" in gst_df.columns) else (rev/12)
    else:
        avg_balance = 0
        monthly_revenue = rev/12
    # liquidity ratio
    liq_ratio = (avg_balance / (monthly_revenue+1)) * 100
    liq_score = np.interp(liq_ratio, [0, 50, 500], [80, 30, 5])
    score += 0.2 * liq_score

    # FX exposure adds volatility risk
    fx_exposure = 0
    if fx_df is not None and len(fx_df):
        fx_exposure = np.abs(fx_df["amount_in_base"]).mean()
        fx_pct = (fx_exposure / (rev+1)) * 1000
        fx_score = np.interp(fx_pct, [0, 0.5, 5], [0, 20, 60])
    else:
        fx_score = 5
    score += 0.1 * fx_score

    # country risk added
    cr = float(country_risk)
    cr_score = np.interp(cr, [0,50,100],[0,30,80])
    score += 0.2 * cr_score

    # normalize to 0-100
    score = min(max(score, 0), 100)
    return round(score,2), {"rev_score":round(rev_score,2),"delay_score":round(delay_score,2),"liq_score":round(liq_score,2),"fx_score":round(fx_score,2),"cr_score":round(cr_score,2)}

# LLM memo builder
def build_memo_prompt(company, metrics_summary, llm_sample_mode=True):
    prompt = f"""
You are a trade-finance risk analyst. Given the company profile and the computed metrics, produce:
1) A short risk memo (markdown) with headline risk level (Low/Medium/High).
2) Top 3 early-warning signals (bulleted).
3) Reasoned probability of default (0-100) and short rationale.
4) Suggested risk-adjusted loan amount and pragmatic covenants (3 bullets).

Company profile JSON:
{json.dumps(company, ensure_ascii=False)}

Computed metrics JSON:
{json.dumps(metrics_summary, ensure_ascii=False)}

Write concise, actionable output for an underwriter.
Return MARKDOWN only.
"""
    if llm_sample_mode:
        prompt += "\nUse concise language; sample mode on (short output)."
    return prompt

def call_llm_for_memo(prompt, model="gpt-4o-mini", temperature=0.0):
    if client is None:
        return "LLM not enabled — add OPENAI_API_KEY to Streamlit Secrets to enable memos."
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], temperature=0.0)
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM call failed: {e}"

# ---------------- Main app UI ----------------
st.markdown("## 1) Data — upload or use sample data")

use_samples = st.button("Use in-app sample data (recommended to try)")

if use_samples:
    gst_df = generate_sample_gst()
    shipments_df = generate_sample_shipments()
    bank_df = generate_sample_bank()
    fx_df = generate_sample_fx()
    profile_df = generate_sample_profile()
else:
    # read uploaded files if present (and coerce date columns)
    gst_df = pd.read_csv(gst_file) if gst_file else None
    shipments_df = pd.read_csv(shipments_file) if shipments_file else None
    bank_df = pd.read_csv(bank_file) if bank_file else None
    fx_df = pd.read_csv(fx_file) if fx_file else None
    profile_df = pd.read_csv(profile_file) if profile_file else None

# Show preview + download buttons
colA, colB = st.columns(2)
with colA:
    if gst_df is not None:
        st.write("### GST / Sales (preview)")
        if "date" in gst_df.columns:
            try:
                gst_df["date"] = pd.to_datetime(gst_df["date"])
            except:
                pass
        st.dataframe(gst_df.head())
        csv = gst_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download sample GST CSV", data=csv, file_name="sample_gst.csv", mime="text/csv")
with colB:
    if shipments_df is not None:
        st.write("### Shipments (preview)")
        if "date" in shipments_df.columns:
            try:
                shipments_df["date"] = pd.to_datetime(shipments_df["date"])
            except:
                pass
        st.dataframe(shipments_df.head())
        st.download_button("Download sample shipments CSV", data=shipments_df.to_csv(index=False).encode('utf-8'), file_name="sample_shipments.csv", mime="text/csv")

colC, colD = st.columns(2)
with colC:
    if bank_df is not None:
        st.write("### Bank statements (preview)")
        if "date" in bank_df.columns:
            try:
                bank_df["date"] = pd.to_datetime(bank_df["date"])
            except:
                pass
        st.dataframe(bank_df.head())
        st.download_button("Download sample bank CSV", data=bank_df.to_csv(index=False).encode('utf-8'), file_name="sample_bank.csv", mime="text/csv")
with colD:
    if fx_df is not None:
        st.write("### FX exposures (preview)")
        st.dataframe(fx_df.head())
        st.download_button("Download sample fx CSV", data=fx_df.to_csv(index=False).encode('utf-8'), file_name="sample_fx.csv", mime="text/csv")

if profile_df is not None:
    st.write("### Company profile (preview)")
    st.dataframe(profile_df.head())
    st.download_button("Download sample profile CSV", data=profile_df.to_csv(index=False).encode('utf-8'), file_name="sample_profile.csv", mime="text/csv")

# ---------------- Run analysis ----------------
st.markdown("## 2) Run Risk Analysis")
company = {}
if profile_df is not None:
    company = profile_df.iloc[0].to_dict()
else:
    st.info("No company profile uploaded — using sample/default profile for scoring.")
    company = {
        "name":"Sample SME Co",
        "entity_type":"Exporter",
        "annual_revenue": 8_000_000,
        "region":"Karnataka",
    }

# quick preprocessing for scoring
# ensure date columns are datetimes
if gst_df is not None and "date" in gst_df.columns:
    gst_df["date"] = pd.to_datetime(gst_df["date"], errors="coerce")
if shipments_df is not None and "date" in shipments_df.columns:
    shipments_df["date"] = pd.to_datetime(shipments_df["date"], errors="coerce")
if bank_df is not None and "date" in bank_df.columns:
    bank_df["date"] = pd.to_datetime(bank_df["date"], errors="coerce")
if fx_df is not None and "date" in fx_df.columns:
    fx_df["date"] = pd.to_datetime(fx_df["date"], errors="coerce")

# country risk input (slider)
country_risk = st.slider("External country risk index for main market (0 low — 100 high)", 0, 100, 30)

if st.button("Compute Risk Score & Generate Memo"):
    with st.spinner("Computing metrics and calling LLM..."):
        score, breakdown = heuristic_risk_score(company, 
                                               gst_df= (gst_df.assign(date=pd.to_datetime(gst_df["date"])) if gst_df is not None and "date" in gst_df.columns else gst_df),
                                               ship_df=shipments_df,
                                               bank_df=bank_df,
                                               fx_df=fx_df,
                                               country_risk=country_risk)
        # early warning rules
        warnings = []
        if breakdown["delay_score"] > 25:
            warnings.append("High shipment delay rate (>~20%) — operational bottlenecks or logistics risk.")
        if breakdown["liq_score"] > 50:
            warnings.append("Low liquidity relative to monthly revenue — potential cashflow stress.")
        if breakdown["fx_score"] > 25:
            warnings.append("Material FX exposure observed — FX volatility can affect margins.")
        if company.get("annual_revenue",0) < 5_00_000:
            warnings.append("Very small annual revenue — scale risk.")

        # risk adjusted loan: simple conservative formula
        monthly_rev = (gst_df.assign(date=pd.to_datetime(gst_df["date"])).set_index("date").resample("M")["taxable_value"].sum().mean()
                       if (gst_df is not None and "date" in gst_df.columns) else (company.get("annual_revenue",0)/12))
        # base lending multiple (2x monthly rev), penalize by score
        base_loan = monthly_rev * 2
        risk_multiplier = np.interp(score, [0,50,100], [1.0, 0.6, 0.25])
        recommended_loan = int(base_loan * risk_multiplier)

        # probability estimate mapping from score
        prob_default = round(score,2)  # treat heuristic score as approximate probability

        metrics_summary = {
            "heuristic_score": score,
            "breakdown": breakdown,
            "monthly_revenue_estimate": int(monthly_rev) if monthly_rev else None,
            "recommended_loan": recommended_loan,
            "probability_of_default_pct": prob_default,
            "early_warnings": warnings
        }

        st.success(f"Computed heuristic risk score: {score}/100")
        st.write("### Metrics Summary")
        st.json(metrics_summary)

        # prepare LLM prompt and call if enabled
        memo_markdown = "LLM memos disabled or API key missing."
        if use_llm and client is not None:
            prompt = build_memo_prompt(company, metrics_summary, llm_sample_mode=sample_mode)
            memo_markdown = call_llm_for_memo(prompt, model=model)
            st.markdown("### LLM Risk Memo")
            st.markdown(memo_markdown)
        else:
            st.info("LLM memo was not executed (enable 'Enable LLM memos' and provide API key in Secrets).")

        # Downloadable artifacts
        report = {
            "company": company,
            "metrics": metrics_summary,
            "llm_memo": memo_markdown
        }
        report_md = f"# Risk Report — {company.get('name')}\n\n"
        report_md += f"**Heuristic score:** {score}/100\n\n"
        report_md += "## Metrics\n\n"
        report_md += json.dumps(metrics_summary, indent=2, ensure_ascii=False)
        report_md += "\n\n## LLM Memo\n\n"
        report_md += memo_markdown if isinstance(memo_markdown,str) else str(memo_markdown)

        st.download_button("Download risk report (JSON)", data=json.dumps(report, indent=2, ensure_ascii=False), file_name="risk_report.json", mime="application/json")
        st.download_button("Download risk report (Markdown)", data=report_md, file_name="risk_report.md", mime="text/markdown")

# ---------------- End ----------------
st.markdown("---")
st.caption("Prototype for trade finance risk intelligence. For production use: validate with real models, use vector DB for historical regs/claims, and ensure data governance.")
