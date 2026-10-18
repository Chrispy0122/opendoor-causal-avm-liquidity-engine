import pandas as pd
import numpy as np
import joblib
import shap

# ==========================================
# 1. LOAD DATA & HYBRID QUANT BUNDLE
# ==========================================
print("Loading Opendoor datasets and Hybrid AVM + KNN + DML bundle...")
df_props = pd.read_csv("opendoor_properties_10000.csv")
df_zips = pd.read_csv("opendoor_macro_zip_regimes.csv")
bundle = joblib.load("opendoor_quant_hybrid.pkl")

avm_pipeline = bundle["avm_pipeline"]
dom_pipeline = bundle["dom_pipeline"]
knn_preprocessor = bundle["knn_preprocessor"]
knn_comps_model = bundle["knn_comps_model"]
causal_tier_parameters = bundle["causal_tier_parameters"]
avm_features = bundle["avm_features"]
knn_features = bundle["knn_features"]
comps_reference_df = bundle["comps_reference_df"]

# Initialize SHAP Explainer for the XGBoost AVM
avm_preprocessor = avm_pipeline.named_steps["preprocessor"]
avm_regressor = avm_pipeline.named_steps["regressor"]
shap_explainer = shap.TreeExplainer(avm_regressor)

# ==========================================
# 2. ENGINE 1: XGBOOST AVM + SHAP DOLLAR ATTRIBUTION
# ==========================================
def evaluate_property_avm(prop_row):
    """Predicts baseline AVM valuation, baseline Days on Market, and top SHAP dollar drivers."""
    X_single = pd.DataFrame([prop_row[avm_features]])
    avm_price = float(avm_pipeline.predict(X_single)[0])
    baseline_dom = float(dom_pipeline.predict(X_single)[0])
    
    # Compute SHAP values in actual US Dollars ($)
    X_trans = avm_preprocessor.transform(X_single)
    shap_vals = shap_explainer.shap_values(X_trans)[0]
    feature_names = avm_preprocessor.get_feature_names_out()
    
    shap_df = pd.DataFrame({
        "feature": [f.replace("num__", "").replace("cat__", "") for f in feature_names],
        "dollar_impact": [round(float(v), 2) for v in shap_vals]
    })
    shap_df["abs_impact"] = shap_df["dollar_impact"].abs()
    top_drivers = shap_df.sort_values(by="abs_impact", ascending=False).head(5)
    
    return round(avm_price, 2), round(baseline_dom, 1), top_drivers

# ==========================================
# 3. ENGINE 2: KNN COMPARABLE SALES ("COMPS")
# ==========================================
def find_knn_comps(prop_row, top_k=5):
    """Finds the closest historical Comparable Sales ('Comps') using Cosine Similarity."""
    X_knn_query = knn_preprocessor.transform(pd.DataFrame([prop_row[knn_features]]))
    distances, indices = knn_comps_model.kneighbors(X_knn_query, n_neighbors=top_k + 1)
    
    comps_list = []
    for dist, idx in zip(distances[0], indices[0]):
        comp = comps_reference_df.iloc[idx]
        if comp["property_id"] == prop_row["property_id"]:
            continue  # Skip self
        similarity_pct = round((1.0 - float(dist)) * 100.0, 1)
        comps_list.append({
            "property_id": comp["property_id"],
            "market_city": comp["market_city"],
            "zip_code": str(comp["zip_code"]),
            "living_area_sqft": int(comp["living_area_sqft"]),
            "bedrooms": int(comp["bedrooms"]),
            "bathrooms": float(comp["bathrooms"]),
            "condition_score": float(comp["condition_score"]),
            "sale_price_usd": round(float(comp["sale_price_usd"]), 2),
            "days_on_market": int(comp["days_on_market"]),
            "similarity_pct": similarity_pct
        })
        if len(comps_list) == top_k:
            break
            
    comp_prices = [c["sale_price_usd"] for c in comps_list]
    comp_doms = [c["days_on_market"] for c in comps_list]
    
    return comps_list, round(float(np.median(comp_prices)), 2), round(float(np.median(comp_doms)), 1)

# ==========================================
# 4. ENGINE 3 & 4: CAUSAL DML MACRO-SHOCK & IBUYING BID OPTIMIZER
# ==========================================
def underwrite_ibuying_offer(prop_row, simulated_mortgage_rate=None):
    """
    Combines XGBoost AVM, KNN Comps, and Double Machine Learning (DML) Causal Elasticity
    to compute Opendoor's Optimal All-Cash Offer under any interest rate regime.
    """
    # Step A: Baseline AVM & KNN Comps
    avm_price, base_dom, shap_drivers = evaluate_property_avm(prop_row)
    comps_list, comp_median_price, comp_median_dom = find_knn_comps(prop_row, top_k=4)
    
    # Anchored Valuation (75% XGBoost AVM + 25% KNN Local Comps Median)
    anchored_valuation = round((0.75 * avm_price) + (0.25 * comp_median_price), 2)
    
    # Step B: Causal Inference Macro-Shock Adjustment (DML CATE)
    recorded_rate = float(prop_row["mortgage_rate_30yr"])
    target_rate = float(simulated_mortgage_rate) if simulated_mortgage_rate is not None else recorded_rate
    rate_delta = target_rate - recorded_rate
    
    tier = prop_row["zip_wealth_tier"]
    dml_params = causal_tier_parameters[tier]
    
    # Local cash-buyer insulation factor (more cash buyers = less rate shock)
    cash_share = float(prop_row["cash_buyer_share_pct"])
    insulation_factor = max(0.35, 1.15 - cash_share)
    
    # Apply DML Causal Derivatives
    causal_dom_shift = rate_delta * dml_params["causal_dom_per_1pct_rate"] * insulation_factor
    causal_price_pct_shift = (rate_delta * dml_params["causal_price_pct_per_1pct_rate"] * insulation_factor) / 100.0
    
    stressed_resale_price = round(anchored_valuation * (1.0 + causal_price_pct_shift), 2)
    stressed_dom = int(np.clip(round(base_dom + causal_dom_shift), 7, 210))
    
    # Step C: Opendoor Unit Economics & Spread Breakdown
    daily_burn = float(prop_row["daily_holding_cost_usd"])
    total_holding_cost = round(stressed_dom * daily_burn, 2)
    rehab_cost = round(float(prop_row["estimated_rehab_cost_usd"]), 2)
    
    # Dynamic Liquidity Risk Premium (Extra buffer if house takes > 60 days to sell)
    excess_dom = max(0, stressed_dom - 60)
    liquidity_risk_pct = round(min(0.035, (excess_dom / 100.0) * 0.025), 4)
    liquidity_risk_buffer_usd = round(stressed_resale_price * liquidity_risk_pct, 2)
    
    # Opendoor 5% Service & Target Margin
    opendoor_fee_usd = round(stressed_resale_price * 0.05, 2)
    
    # Final Maximum Safe Cash Offer to Homeowner
    optimal_cash_offer = round(
        stressed_resale_price - total_holding_cost - rehab_cost - liquidity_risk_buffer_usd - opendoor_fee_usd,
        2
    )
    
    return {
        "property_id": prop_row["property_id"],
        "market_city": prop_row["market_city"],
        "zip_code": str(prop_row["zip_code"]),
        "zip_wealth_tier": tier,
        "cash_buyer_share_pct": round(cash_share * 100, 1),
        "recorded_mortgage_rate": recorded_rate,
        "simulated_mortgage_rate": target_rate,
        "avm_baseline_price_usd": avm_price,
        "knn_comp_median_price_usd": comp_median_price,
        "anchored_valuation_usd": anchored_valuation,
        "stressed_resale_price_usd": stressed_resale_price,
        "baseline_dom_days": int(round(base_dom)),
        "stressed_dom_days": stressed_dom,
        "causal_dom_added_days": round(causal_dom_shift, 1),
        "causal_price_shift_pct": round(causal_price_pct_shift * 100, 2),
        "daily_holding_cost_usd": daily_burn,
        "total_holding_cost_usd": total_holding_cost,
        "estimated_rehab_cost_usd": rehab_cost,
        "liquidity_risk_buffer_usd": liquidity_risk_buffer_usd,
        "opendoor_service_fee_usd": opendoor_fee_usd,
        "optimal_cash_offer_usd": optimal_cash_offer,
        "shap_drivers": shap_drivers.to_dict(orient="records"),
        "knn_comps": comps_list
    }

# ==========================================
# 5. LIVE TERMINAL STRESS TEST (5.0% vs 7.5% FED RATE SHOCK)
# ==========================================
if __name__ == "__main__":
    # Pick a sample Entry-Level or Mid-Market property to see the macro shock clearly
    sample_prop = df_props[
        (df_props["zip_wealth_tier"] == "Entry-Level") & 
        (df_props["mortgage_rate_30yr"].between(4.8, 5.2))
    ].iloc[0]
    
    print("\n" + "="*75)
    print(f"🏠 OPENDOOR UNDERWRITING REPORT: {sample_prop['property_id']} ({sample_prop['market_city']} - Zip {sample_prop['zip_code']})")
    print("="*75)
    print(f"• Tier: {sample_prop['zip_wealth_tier']} | {sample_prop['living_area_sqft']} sqft | {sample_prop['bedrooms']} Bed / {sample_prop['bathrooms']} Bath | Condition: {sample_prop['condition_score']}/10")
    print(f"• Zip Cash-Buyer Share: {sample_prop['cash_buyer_share_pct']*100:.0f}% | Daily Holding Burn: ${sample_prop['daily_holding_cost_usd']}/day")
    
    # Scenario A: Normal 5.0% Mortgage Rate
    res_5pct = underwrite_ibuying_offer(sample_prop, simulated_mortgage_rate=5.00)
    # Scenario B: Federal Reserve Hike to 7.5% Mortgage Rate (+250 bps shock!)
    res_75pct = underwrite_ibuying_offer(sample_prop, simulated_mortgage_rate=7.50)
    
    print(f"\n📊 [1. HEDONIC AVM & KNN COMPS VALUATION]")
    print(f"• XGBoost AVM Estimate: ${res_5pct['avm_baseline_price_usd']:,.2f}")
    print(f"• KNN Top Comps Median: ${res_5pct['knn_comp_median_price_usd']:,.2f} -> Anchored FMV: ${res_5pct['anchored_valuation_usd']:,.2f}")
    print("• Top SHAP Property Value Drivers ($):")
    for d in res_5pct["shap_drivers"][:4]:
        print(f"    -> {d['feature']}: ${d['dollar_impact']:+,.2f}")
        
    print(f"\n⚡ [2. CAUSAL DML MACRO-SHOCK STRESS TEST: 5.00% vs. 7.50% MORTGAGE RATE]")
    print(f"  [At 5.00% Rate] ➔ Expected Resale: ${res_5pct['stressed_resale_price_usd']:,.2f} | Liquidity Risk: {res_5pct['stressed_dom_days']} Days on Market")
    print(f"                    Holding Cost: -${res_5pct['total_holding_cost_usd']:,.2f} | Rehab: -${res_5pct['estimated_rehab_cost_usd']:,.2f}")
    print(f"                    ✅ OPTIMAL CASH OFFER: ${res_5pct['optimal_cash_offer_usd']:,.2f}")
    
    print(f"\n  [At 7.50% Rate] ➔ Expected Resale: ${res_75pct['stressed_resale_price_usd']:,.2f} ({res_75pct['causal_price_shift_pct']:+.2f}% Causal Compression)")
    print(f"                    Liquidity Risk: {res_75pct['stressed_dom_days']} Days on Market (+{res_75pct['causal_dom_added_days']} Extra Days!)")
    print(f"                    Holding Cost: -${res_75pct['total_holding_cost_usd']:,.2f} | Illiquidity Buffer: -${res_75pct['liquidity_risk_buffer_usd']:,.2f}")
    print(f"                    🛡️ RISK-ADJUSTED CASH OFFER: ${res_75pct['optimal_cash_offer_usd']:,.2f}")
    
    offer_spread = round(res_5pct["optimal_cash_offer_usd"] - res_75pct["optimal_cash_offer_usd"], 2)
    print(f"\n💡 ECONOMETRIC INSIGHT: A +2.5% Fed Rate shock forces Opendoor to lower its cash bid by ${offer_spread:,.2f} to prevent balance-sheet losses!")