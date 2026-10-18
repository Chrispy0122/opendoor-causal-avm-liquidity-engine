import os
import math
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import underwriting engine, datasets, and ML bundle
import opendoor_core_engine as engine

app = FastAPI(
    title="Opendoor Acquisition & Macro-Risk Underwriting Terminal API",
    description="Institutional AVM, KNN Comps & Causal DML Rate-Shock Valuation Engine",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-index properties for O(1) lookup
df_props = engine.df_props
props_by_id: Dict[str, pd.Series] = {
    row["property_id"]: row for _, row in df_props.iterrows()
}

# Curate 30 diverse properties across 5 Sunbelt cities and 4 wealth tiers
CITIES = ["Phoenix, AZ", "Austin, TX", "Atlanta, GA", "Raleigh, NC", "Charlotte, NC"]
TIERS = ["Entry-Level", "Mid-Market", "Move-Up", "Luxury"]

curated_records = []
for city in CITIES:
    for tier in TIERS:
        n = 2 if tier in ["Entry-Level", "Mid-Market"] else 1
        sub = df_props[(df_props["market_city"] == city) & (df_props["zip_wealth_tier"] == tier)].head(n)
        for _, row in sub.iterrows():
            curated_records.append({
                "property_id": str(row["property_id"]),
                "market_city": str(row["market_city"]),
                "zip_code": str(row["zip_code"]),
                "zip_wealth_tier": str(row["zip_wealth_tier"]),
                "living_area_sqft": int(row["living_area_sqft"]),
                "lot_size_sqft": int(row["lot_size_sqft"]),
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": float(row["bathrooms"]),
                "year_built": int(row["year_built"]),
                "property_age_years": int(row["property_age_years"]),
                "condition_score": float(row["condition_score"]),
                "has_pool": int(row["has_pool"]),
                "has_solar_or_ev": int(row["has_solar_or_ev"]),
                "estimated_rehab_cost_usd": round(float(row["estimated_rehab_cost_usd"]), 2),
                "daily_holding_cost_usd": round(float(row["daily_holding_cost_usd"]), 2),
                "mortgage_rate_30yr": round(float(row["mortgage_rate_30yr"]), 2),
                "cash_buyer_share_pct": round(float(row["cash_buyer_share_pct"]) * 100, 1),
                "school_district_score": float(row["school_district_score"]),
                "median_household_income_usd": int(row["median_household_income_usd"]),
                "local_inventory_months": round(float(row["local_inventory_months"]), 2),
                "baseline_sale_price_usd": round(float(row["sale_price_usd"]), 2),
                "baseline_dom_days": int(row["days_on_market"])
            })

class UnderwriteRequest(BaseModel):
    property_id: str = Field(..., example="OD_PROP_00104")
    simulated_mortgage_rate: float = Field(..., ge=2.0, le=12.0, example=6.5)

def build_macro_regime_insight(
    prop_row: pd.Series,
    res: Dict[str, Any],
    simulated_rate: float
) -> str:
    """
    Generates an institutional econometric risk summary contextualizing
    the property's wealth tier, zip code cash-buyer share, and simulated interest rate.
    """
    tier = prop_row["zip_wealth_tier"]
    zip_code = str(prop_row["zip_code"])
    city = str(prop_row["market_city"])
    recorded_rate = float(prop_row["mortgage_rate_30yr"])
    cash_share = float(prop_row["cash_buyer_share_pct"]) * 100.0
    delta_bps = int(round((simulated_rate - recorded_rate) * 100))
    added_dom = res["causal_dom_added_days"]
    price_shift = res["causal_price_shift_pct"]
    daily_burn = res["daily_holding_cost_usd"]
    stressed_dom = res["stressed_dom_days"]
    total_burn = res["total_holding_cost_usd"]
    buffer = res["liquidity_risk_buffer_usd"]

    # Tier-specific debt qualification context
    if tier == "Entry-Level":
        tier_narrative = (
            f"As an {tier} asset, prospective buyers in {city} (Zip {zip_code}) are severely debt-constrained "
            f"by agency DTI qualification ratios. A {delta_bps:+d} bps rate adjustment triggers sharp purchasing power "
            f"contraction, shifting baseline liquidity by {added_dom:+.1f} DOM days."
        )
    elif tier == "Mid-Market":
        tier_narrative = (
            f"Within the {tier} segment, conforming 30-year borrowers face substantial monthly payment sensitivity. "
            f"The {delta_bps:+d} bps rate deviation introduces lock-in friction across local trade-up inventory, "
            f"extending expected time on market by {added_dom:+.1f} days."
        )
    elif tier == "Move-Up":
        tier_narrative = (
            f"The {tier} segment benefits from accumulated equity from prior home sales, providing moderate macro "
            f"absorption against interest rate spikes (causal DOM impact dampened to {added_dom:+.1f} days)."
        )
    else:  # Luxury
        tier_narrative = (
            f"In the {tier} tier, buyers command significant non-mortgage net worth and jumbo asset backing. "
            f"This cohort demonstrates strong structural resilience against conforming rate shocks, limiting DOM expansion to {added_dom:+.1f} days."
        )

    # Cash buyer insulation context
    if cash_share < 18.0:
        cash_narrative = (
            f"Crucially, Zip {zip_code} exhibits low institutional/cash presence ({cash_share:.1f}% cash buyers), "
            f"resulting in minimal rate insulation (insulation factor: ~1.0x). Local clearing volume remains tethered "
            f"to conventional retail mortgage availability."
        )
    elif cash_share < 35.0:
        cash_narrative = (
            f"Zip {zip_code} carries a healthy {cash_share:.1f}% cash-buyer liquidity share, establishing an active "
            f"equity floor that partially cushions transactional velocity against Federal Reserve tightening cycles."
        )
    else:
        cash_narrative = (
            f"Remarkably, Zip {zip_code} is heavily anchored by institutional and private cash buyers ({cash_share:.1f}%), "
            f"acting as a robust econometric firewall that shields clearance pricing from mortgage market illiquidity."
        )

    # Capital allocation & balance sheet safety conclusion
    if delta_bps > 50:
        balance_sheet_impact = (
            f"Under this +{delta_bps} bps shock, resale valuation compresses by {price_shift:+.2f}%. "
            f"Opendoor's holding burn totals ${total_burn:,.2f} (${daily_burn:.2f}/day × {stressed_dom} days), "
            f"necessitating a ${buffer:,.2f} dynamic illiquidity buffer to maintain underwriting margins and prevent inventory write-downs."
        )
    elif delta_bps < -50:
        balance_sheet_impact = (
            f"With financing costs easing by {abs(delta_bps)} bps, transactional velocity accelerates ({added_dom:+.1f} DOM days). "
            f"Reduced carry burn allows Opendoor to submit an aggressively competitive cash offer while maintaining targeted unit margins."
        )
    else:
        balance_sheet_impact = (
            f"Operating near baseline neutral rates, the portfolio incurs standard holding burn of ${total_burn:,.2f} "
            f"over {stressed_dom} days, yielding a balanced risk-adjusted cash bid."
        )

    return f"{tier_narrative} {cash_narrative} {balance_sheet_impact}"

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Opendoor Quantitative Underwriting & Macro-Risk Terminal",
        "endpoints": ["/api/properties", "/api/underwrite"]
    }

@app.get("/api/properties")
def get_properties():
    """
    Returns curated sample of 30 diverse properties across all 5 Sunbelt cities
    and 4 wealth tiers, along with model telemetry metrics and causal parameters.
    """
    bundle = engine.bundle
    return {
        "properties": curated_records,
        "count": len(curated_records),
        "metrics": bundle["metrics"],
        "causal_tier_parameters": bundle["causal_tier_parameters"]
    }

@app.post("/api/underwrite")
def underwrite_property(payload: UnderwriteRequest):
    """
    Executes real-time iBuying underwriting:
    1. Hedonic XGBoost AVM + SHAP dollar attribution
    2. Cosine-similarity KNN Comps valuation
    3. Causal DML interest rate macro-shock stress test
    4. Unit economics fee and capital buffer waterfall
    """
    prop_id = payload.property_id
    if prop_id not in props_by_id:
        raise HTTPException(status_code=404, detail=f"Property '{prop_id}' not found in database.")

    prop_row = props_by_id[prop_id]
    simulated_rate = float(payload.simulated_mortgage_rate)

    # Run core underwriting engine
    underwrite_result = engine.underwrite_ibuying_offer(prop_row, simulated_mortgage_rate=simulated_rate)

    # Attach dynamic macroeconomic regime narrative
    macro_insight = build_macro_regime_insight(prop_row, underwrite_result, simulated_rate)
    underwrite_result["macro_regime_insight"] = macro_insight

    return underwrite_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
