import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

# ==========================================
# 1. LOAD OPENDOOR DATASETS
# ==========================================
print("1. Loading Opendoor PropTech datasets (10,000 Properties + 40 Zip Sub-Markets)...")
df_props = pd.read_csv("opendoor_properties_10000.csv")
df_zips = pd.read_csv("opendoor_macro_zip_regimes.csv")

# ==========================================
# 2. ENGINE 1: KNN "COMPARABLE SALES" ("COMPS") MATCHER
# ==========================================
print("\n2. Training Engine 1: Unsupervised KNN Comparable Sales ('Comps') Matcher...")

knn_num_features = [
    "living_area_sqft", "lot_size_sqft", "bedrooms", "bathrooms",
    "property_age_years", "condition_score", "school_district_score"
]
knn_cat_features = ["market_city", "zip_wealth_tier"]
knn_all_features = knn_num_features + knn_cat_features

knn_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), knn_num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), knn_cat_features)
    ]
)

X_knn_matrix = knn_preprocessor.fit_transform(df_props[knn_all_features])

# Fit KNN to retrieve Top 10 Comparable Properties via Cosine Distance
knn_comps_model = NearestNeighbors(n_neighbors=10, metric="cosine")
knn_comps_model.fit(X_knn_matrix)
print(f"   -> KNN Comps Engine indexed across {len(df_props):,} historical transactions.")

# ==========================================
# 3. ENGINE 2: XGBOOST HEDONIC AVM (SALE PRICE PREDICTION + SHAP)
# ==========================================
print("\n3. Training Engine 2: XGBoost Automated Valuation Model (AVM)...")

avm_num_features = [
    "living_area_sqft", "lot_size_sqft", "bedrooms", "bathrooms",
    "property_age_years", "has_pool", "has_solar_or_ev", "condition_score",
    "school_district_score", "median_household_income_usd", "cash_buyer_share_pct",
    "local_inventory_months", "mortgage_rate_30yr"
]
avm_cat_features = ["market_city", "zip_wealth_tier", "listing_season"]
avm_all_features = avm_num_features + avm_cat_features

X = df_props[avm_all_features]
y_price = df_props["sale_price_usd"]
y_dom = df_props["days_on_market"]

X_train, X_test, y_price_train, y_price_test, y_dom_train, y_dom_test = train_test_split(
    X, y_price, y_dom, test_size=0.20, random_state=42
)

avm_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), avm_num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), avm_cat_features)
    ]
)

avm_pipeline = Pipeline(steps=[
    ("preprocessor", avm_preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    ))
])

avm_pipeline.fit(X_train, y_price_train)
price_preds = avm_pipeline.predict(X_test)

avm_r2 = r2_score(y_price_test, price_preds)
avm_mae = mean_absolute_error(y_price_test, price_preds)
avm_mape = mean_absolute_percentage_error(y_price_test, price_preds) * 100

print(f"   -> AVM Test R² Score: {avm_r2:.4f}")
print(f"   -> AVM Mean Absolute Error (MAE): ${avm_mae:,.2f} (MAPE: {avm_mape:.2f}%)")

# ==========================================
# 4. ENGINE 3: XGBOOST DAYS-ON-MARKET (DOM) LIQUIDITY FORECASTER
# ==========================================
print("\n4. Training Engine 3: XGBoost Days-on-Market (Liquidity Risk) Forecaster...")

dom_Preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), avm_num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), avm_cat_features)
    ]
)

dom_pipeline = Pipeline(steps=[
    ("preprocessor", dom_Preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        random_state=42
    ))
])

dom_pipeline.fit(X_train, y_dom_train)
dom_preds = dom_pipeline.predict(X_test)

dom_r2 = r2_score(y_dom_test, dom_preds)
dom_mae = mean_absolute_error(y_dom_test, dom_preds)

print(f"   -> DOM Liquidity Test R² Score: {dom_r2:.4f}")
print(f"   -> DOM Mean Absolute Error: {dom_mae:.2f} Days")

# ==========================================
# 5. ENGINE 4: DOUBLE MACHINE LEARNING (DML) CAUSAL INFERENCE
# ==========================================
print("\n5. Training Engine 4: Double Machine Learning (DML) Causal Rate-Shock Estimator...")
print("   -> Isolating causal effect of +1.0% Mortgage Rate hike via Orthogonalization...")

# Confounding Controls W (Everything except the treatment 'mortgage_rate_30yr')
dml_control_num = [
    "living_area_sqft", "lot_size_sqft", "property_age_years", "condition_score",
    "school_district_score", "median_household_income_usd", "cash_buyer_share_pct",
    "local_inventory_months"
]
dml_control_cat = ["market_city", "zip_wealth_tier", "listing_season"]
dml_controls = dml_control_num + dml_control_cat

W_matrix = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), dml_control_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), dml_control_cat)
    ]
).fit_transform(df_props[dml_controls])

T_rate = df_props["mortgage_rate_30yr"].values
Y_dom_full = df_props["days_on_market"].values
Y_log_price = np.log(df_props["sale_price_usd"].values)

# Step A: Residualize Treatment T (Mortgage Rate) on Controls W
model_T = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
model_T.fit(W_matrix, T_rate)
T_residuals = T_rate - model_T.predict(W_matrix)

# Step B: Residualize Outcomes (DOM & Log-Price) on Controls W
model_Y_dom = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model_Y_dom.fit(W_matrix, Y_dom_full)
Y_dom_residuals = Y_dom_full - model_Y_dom.predict(W_matrix)

model_Y_price = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model_Y_price.fit(W_matrix, Y_log_price)
Y_price_residuals = Y_log_price - model_Y_price.predict(W_matrix)

# Step C: Estimate Heterogeneous Causal Effect (CATE) per Wealth Tier via Orthogonalized OLS
causal_tier_parameters = {}
print("\n   [DML Causal Discovery — Effect of +1.0% Mortgage Rate Hike]:")
for tier in ["Entry-Level", "Mid-Market", "Move-Up", "Luxury"]:
    mask = (df_props["zip_wealth_tier"] == tier).values
    t_res_tier = T_residuals[mask]
    dom_res_tier = Y_dom_residuals[mask]
    price_res_tier = Y_price_residuals[mask]
    
    # Causal Slope: Cov(Y_res, T_res) / Var(T_res)
    causal_dom_per_1pct = float(np.sum(t_res_tier * dom_res_tier) / np.sum(t_res_tier ** 2))
    causal_price_pct_per_1pct = float((np.sum(t_res_tier * price_res_tier) / np.sum(t_res_tier ** 2)) * 100.0)
    
    causal_tier_parameters[tier] = {
        "causal_dom_per_1pct_rate": round(causal_dom_per_1pct, 2),
        "causal_price_pct_per_1pct_rate": round(causal_price_pct_per_1pct, 2)
    }
    print(f"   • {tier:<12} ➔ +{causal_dom_per_1pct:>5.2f} Extra Days on Market | {causal_price_pct_per_1pct:>+6.2f}% Price Shift")

# ==========================================
# 6. VERIFY SHAP & SAVE HYBRID ARTIFACT (.PKL)
# ==========================================
print("\n6. Verifying SHAP TreeExplainer on AVM & saving artifact bundle...")
avm_explainer = shap.TreeExplainer(avm_pipeline.named_steps["regressor"])
sample_trans = avm_pipeline.named_steps["preprocessor"].transform(X_test.iloc[:3])
_ = avm_explainer.shap_values(sample_trans)
print("   -> SHAP TreeExplainer verified successfully!")

opendoor_bundle = {
    "avm_pipeline": avm_pipeline,
    "dom_pipeline": dom_pipeline,
    "knn_preprocessor": knn_preprocessor,
    "knn_comps_model": knn_comps_model,
    "causal_tier_parameters": causal_tier_parameters,
    "avm_features": avm_all_features,
    "knn_features": knn_all_features,
    "comps_reference_df": df_props[[
        "property_id", "market_city", "zip_code", "zip_wealth_tier",
        "living_area_sqft", "bedrooms", "bathrooms", "condition_score",
        "sale_price_usd", "days_on_market"
    ]],
    "metrics": {
        "avm_r2": round(avm_r2, 4),
        "avm_mae": round(avm_mae, 2),
        "avm_mape_pct": round(avm_mape, 2),
        "dom_r2": round(dom_r2, 4),
        "dom_mae": round(dom_mae, 2)
    }
}

joblib.dump(opendoor_bundle, "opendoor_quant_hybrid.pkl")
print("\nSUCCESS! Saved Opendoor Hybrid Bundle to 'opendoor_quant_hybrid.pkl'.")