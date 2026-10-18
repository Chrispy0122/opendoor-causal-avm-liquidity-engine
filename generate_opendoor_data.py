import numpy as np
import pandas as pd
import random

# Set seeds for reproducibility (reproducibilidad)
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. DEFINE OPENDOOR SUNBELT MARKETS & TIERS
# ==========================================
MARKETS = {
    "Phoenix, AZ":   {"base_ppsqft": 245.0, "tax_rate": 0.0065, "zip_prefix": "850"},
    "Austin, TX":    {"base_ppsqft": 295.0, "tax_rate": 0.0180, "zip_prefix": "787"},
    "Atlanta, GA":   {"base_ppsqft": 215.0, "tax_rate": 0.0095, "zip_prefix": "303"},
    "Raleigh, NC":   {"base_ppsqft": 230.0, "tax_rate": 0.0085, "zip_prefix": "276"},
    "Charlotte, NC": {"base_ppsqft": 220.0, "tax_rate": 0.0088, "zip_prefix": "282"}
}

WEALTH_TIERS = {
    "Entry-Level": {
        "price_mult": 0.82,
        "income_mean": 64000,
        "cash_buyer_share": 0.11,       # Only 11% pay cash -> highly vulnerable to Fed rate hikes!
        "rate_dom_elasticity": 19.5,    # +19.5 extra Days on Market per +1% mortgage rate
        "rate_price_elasticity": -0.048 # -4.8% price drop per +1% mortgage rate
    },
    "Mid-Market": {
        "price_mult": 1.00,
        "income_mean": 96000,
        "cash_buyer_share": 0.22,
        "rate_dom_elasticity": 14.0,
        "rate_price_elasticity": -0.035
    },
    "Move-Up": {
        "price_mult": 1.25,
        "income_mean": 142000,
        "cash_buyer_share": 0.35,
        "rate_dom_elasticity": 9.5,
        "rate_price_elasticity": -0.024
    },
    "Luxury": {
        "price_mult": 1.65,
        "income_mean": 235000,
        "cash_buyer_share": 0.58,       # 58% pay cash -> insulated from mortgage rate shocks!
        "rate_dom_elasticity": 4.2,
        "rate_price_elasticity": -0.012
    }
}

SEASONS = ["Spring_Peak", "Summer_Active", "Autumn_Cool", "Winter_Slow"]

# ==========================================
# 2. GENERATE CSV #2: 40 ZIP CODE MACRO SUB-MARKETS
# ==========================================
def generate_zip_catalog():
    zip_records = []
    tier_names = list(WEALTH_TIERS.keys())
    
    for city, m_info in MARKETS.items():
        # Create 8 distinct zip codes per city (5 cities * 8 = 40 Zip Code Sub-Markets)
        for idx in range(8):
            tier = tier_names[idx % 4]
            t_info = WEALTH_TIERS[tier]
            zip_code = f"{m_info['zip_prefix']}{15 + idx * 9}"
            
            school_base = {"Entry-Level": 5.4, "Mid-Market": 6.9, "Move-Up": 8.1, "Luxury": 9.1}[tier]
            school_score = round(float(np.clip(np.random.normal(school_base, 0.6), 3.0, 10.0)), 1)
            
            median_income = int(np.random.normal(t_info["income_mean"], t_info["income_mean"] * 0.08))
            cash_share = round(float(np.clip(np.random.normal(t_info["cash_buyer_share"], 0.04), 0.05, 0.75)), 2)
            
            # Price volatility & current inventory baseline
            volatility_idx = round(float(np.random.uniform(0.08, 0.22)), 3)
            baseline_inventory_months = round(float(np.random.uniform(1.8, 4.2)), 2)
            
            zip_records.append({
                "zip_code": zip_code,
                "market_city": city,
                "zip_wealth_tier": tier,
                "school_district_score": school_score,
                "median_household_income_usd": median_income,
                "cash_buyer_share_pct": cash_share,
                "annual_property_tax_rate": m_info["tax_rate"],
                "historical_volatility_index": volatility_idx,
                "baseline_inventory_months": baseline_inventory_months,
                "true_causal_dom_elasticity": t_info["rate_dom_elasticity"],
                "true_causal_price_elasticity": t_info["rate_price_elasticity"]
            })
            
    return pd.DataFrame(zip_records)

# ==========================================
# 3. GENERATE CSV #1: 10,000 PROPERTY TRANSACTIONS
# ==========================================
def generate_opendoor_properties(df_zips, n_houses=10000):
    records = []
    # Weight sampling so Entry-Level and Mid-Market have more transactions than Luxury
    tier_weights = {"Entry-Level": 0.32, "Mid-Market": 0.38, "Move-Up": 0.20, "Luxury": 0.10}
    zip_probs = df_zips["zip_wealth_tier"].map(tier_weights).values
    zip_probs = zip_probs / zip_probs.sum()
    
    for i in range(n_houses):
        prop_id = f"OD_PROP_{i:05d}"
        
        # Sample a Zip Code sub-market from our 40-zip catalog
        zip_row = df_zips.sample(n=1, weights=zip_probs).iloc[0]
        city = zip_row["market_city"]
        zip_code = zip_row["zip_code"]
        tier = zip_row["zip_wealth_tier"]
        
        m_info = MARKETS[city]
        t_info = WEALTH_TIERS[tier]
        season = random.choice(SEASONS)
        
        # Slight property-level variation around the zip code's school score
        school_score = round(float(np.clip(zip_row["school_district_score"] + np.random.normal(0, 0.4), 2.0, 10.0)), 1)
        
        # Physical Hedonic Features
        tier_sqft_mean = {"Entry-Level": 1550, "Mid-Market": 2150, "Move-Up": 2850, "Luxury": 3850}[tier]
        living_sqft = int(np.clip(np.random.normal(tier_sqft_mean, 380), 850, 6200))
        lot_sqft = int(np.clip(living_sqft * np.random.uniform(2.2, 4.8), 2500, 28000))
        
        bedrooms = int(np.clip(round(living_sqft / 580) + random.choice([-1, 0, 1]), 2, 6))
        bathrooms = round(max(1.0, bedrooms * 0.75 + random.choice([-0.5, 0.0, 0.5])), 1)
        
        age_years = int(np.clip(np.random.exponential(scale=22), 1, 75))
        year_built = 2026 - age_years
        has_pool = 1 if (city in ["Phoenix, AZ", "Austin, TX"] and random.random() < 0.42) or (random.random() < 0.18) else 0
        has_solar_or_ev = 1 if random.random() < 0.25 else 0
        
        # Condition & Opendoor Renovation Cost
        condition_score = round(float(np.clip(np.random.normal(7.2 - (age_years * 0.03), 1.3), 3.0, 10.0)), 1)
        rehab_per_sqft = max(2.5, (10.0 - condition_score) * 4.8)
        estimated_rehab_cost_usd = round(max(1500.0, living_sqft * rehab_per_sqft + np.random.normal(0, 1200)), 2)
        
        # Macroeconomic Regime at Transaction Time (Confounded with Season & Supply)
        mortgage_rate_30yr = round(float(np.random.uniform(3.50, 7.80)), 2)
        rate_delta_from_neutral = mortgage_rate_30yr - 4.50
        yoy_rate_shock_bps = int(round(rate_delta_from_neutral * 45 + np.random.normal(0, 35)))
        
        season_inv_shift = {"Spring_Peak": -0.6, "Summer_Active": -0.3, "Autumn_Cool": 0.4, "Winter_Slow": 0.9}[season]
        local_inventory_months = round(float(np.clip(
            zip_row["baseline_inventory_months"] + 0.42 * rate_delta_from_neutral + season_inv_shift + np.random.normal(0, 0.45),
            0.8, 8.5
        )), 2)
        
        # TARGET 1: Market Sale Price ($)
        age_discount = max(0.72, 1.0 - (age_years * 0.0038))
        condition_multiplier = 0.84 + (condition_score * 0.022)
        school_multiplier = 0.90 + (school_score * 0.015)
        
        baseline_hedonic_value = (
            living_sqft * m_info["base_ppsqft"] * t_info["price_mult"]
            * age_discount * condition_multiplier * school_multiplier
            + (lot_sqft * 3.5)
            + (22000.0 if has_pool else 0.0)
            + (9500.0 if has_solar_or_ev else 0.0)
        )
        
        # Causal effect of interest rates moderated by the zip code's cash_buyer_share_pct!
        effective_price_elasticity = t_info["rate_price_elasticity"] * (1.15 - zip_row["cash_buyer_share_pct"])
        causal_rate_price_shift = 1.0 + (effective_price_elasticity * rate_delta_from_neutral)
        inventory_price_shift = 1.0 - ((local_inventory_months - 2.5) * 0.012)
        
        sale_price_usd = round(float(
            baseline_hedonic_value * causal_rate_price_shift * inventory_price_shift
            * np.random.normal(1.0, 0.030)
        ), 2)
        sale_price_usd = max(145000.0, sale_price_usd)
        
        # TARGET 2: Days on Market (DOM)
        season_dom_shift = {"Spring_Peak": -8.0, "Summer_Active": -4.0, "Autumn_Cool": 6.0, "Winter_Slow": 14.0}[season]
        effective_dom_elasticity = t_info["rate_dom_elasticity"] * (1.15 - zip_row["cash_buyer_share_pct"])
        
        structural_dom = (
            26.0
            + (effective_dom_elasticity * rate_delta_from_neutral)
            + (local_inventory_months * 5.0)
            + season_dom_shift
            - ((condition_score - 7.0) * 3.1)
            - ((school_score - 6.5) * 1.8)
            + np.random.normal(0, 5.0)
        )
        days_on_market = int(np.clip(round(structural_dom), 6, 185))
        
        # Daily Holding Cost ($) = Property Tax + Cost of Capital (7.5%) + Utilities ($18/day)
        daily_tax_cost = (sale_price_usd * m_info["tax_rate"]) / 365.0
        daily_capital_cost = (sale_price_usd * 0.075) / 365.0
        daily_holding_cost_usd = round(daily_tax_cost + daily_capital_cost + 18.0, 2)
        
        records.append({
            "property_id": prop_id,
            "market_city": city,
            "zip_code": zip_code,
            "zip_wealth_tier": tier,
            "listing_season": season,
            "school_district_score": school_score,
            "median_household_income_usd": zip_row["median_household_income_usd"],
            "cash_buyer_share_pct": zip_row["cash_buyer_share_pct"],
            "living_area_sqft": living_sqft,
            "lot_size_sqft": lot_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "property_age_years": age_years,
            "has_pool": has_pool,
            "has_solar_or_ev": has_solar_or_ev,
            "condition_score": condition_score,
            "estimated_rehab_cost_usd": estimated_rehab_cost_usd,
            "local_inventory_months": local_inventory_months,
            "mortgage_rate_30yr": mortgage_rate_30yr,
            "yoy_rate_shock_bps": yoy_rate_shock_bps,
            "daily_holding_cost_usd": daily_holding_cost_usd,
            "sale_price_usd": sale_price_usd,
            "days_on_market": days_on_market
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    print("1. Generating Opendoor Macro Zip-Code Catalog (40 Sub-Markets)...")
    df_zips = generate_zip_catalog()
    df_zips.to_csv("opendoor_macro_zip_regimes.csv", index=False)
    print(f"   -> Saved 'opendoor_macro_zip_regimes.csv' ({len(df_zips)} zip codes).")
    
    print("\n2. Generating Opendoor Property Transactions (10,000 Houses)...")
    df_props = generate_opendoor_properties(df_zips, 10000)
    df_props.to_csv("opendoor_properties_10000.csv", index=False)
    print(f"   -> Saved 'opendoor_properties_10000.csv' ({len(df_props):,} properties).")
    
    print("\nSummary by Wealth Tier (Notice Cash Buyer Share vs. Days on Market):")
    summary = df_props.groupby("zip_wealth_tier")[
        ["sale_price_usd", "cash_buyer_share_pct", "days_on_market", "daily_holding_cost_usd"]
    ].mean().round(2)
    print(summary)