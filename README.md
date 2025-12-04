⚖️ AI Trade Finance Risk Intelligence System
Automated credit assessment for SMEs using GST, shipments, banking, FX and LLM-based underwriting memos
🚀 Overview

This project is a Streamlit-based AI system that performs automated trade-finance risk analysis for SMEs using real operational and financial data.

It ingests:

📄 GST returns / Sales data

🚢 Shipment history

🏦 Bank statements

💱 FX exposure data

🏭 Company profile (Sector, location, revenue)

And outputs:

📉 Probability of default

🚨 Early warning signals

💰 Risk-adjusted loan amount

📝 Automated underwriting memo (LLM)

📦 Downloadable JSON + Markdown risk report

This project reflects real-world work done in trade finance, MSME credit, and SME underwriting, scaled using AI.

🌟 Why This Project Is Powerful

This prototype demonstrates:

✔ Applied domain knowledge in credit, underwriting, and risk
✔ Ability to build end-to-end AI apps
✔ Usage of LLMs for intelligent memo generation
✔ Structured scoring logic + explainability
✔ Real fintech workflow replication (NBFCs, banks, neobanks, PSP lenders)

This is the type of project that gets attention from:

Razorpay Capital

OneCard

Cashfree

Jupiter

SBI Global Factors

Drip Capital

Niyo

SME neobanks

Working capital lending startups

AI credit-scoring companies

🧠 Features
1️⃣ Data Ingestion

Upload CSVs for:

GST / sales

Shipment timelines

Bank balances & flows

FX exposure

Company profile

Or click Use Sample Data to generate synthetic but realistic datasets.

2️⃣ Heuristic Risk Scoring

A transparent, explainable multi-factor scoring model combining:

Revenue scaling risk

Logistics delay risk

Liquidity risk

FX risk

Country risk

Operational anomalies

Outputs a 0–100 risk score.

3️⃣ Early Warning Signals

Rule-based anomalies such as:

Shipment delays

Low liquidity

High FX exposure

Thin monthly revenue

High volatility

4️⃣ Risk-Adjusted Loan Recommendation

Estimated using:

Base = 2 × monthly revenue  
Risk multiplier = f(score)  


Produces a recommended loan amount and rationale.

5️⃣ LLM Underwriting Memo

If an OpenAI API key is provided, the app generates:

Executive risk summary

Probability of default

Key underwriting concerns

Suggested covenants

All in clean markdown

6️⃣ Downloadable Reports

Export:

📄 risk_report.md

📦 risk_report.json
