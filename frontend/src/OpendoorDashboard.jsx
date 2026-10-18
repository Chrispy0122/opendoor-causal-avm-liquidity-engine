import React, { useState, useEffect, useMemo } from 'react';
import {
  Building2,
  TrendingUp,
  TrendingDown,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  DollarSign,
  Clock,
  ArrowRight,
  Search,
  Filter,
  CheckCircle2,
  Info,
  Sparkles,
  RefreshCw,
  Layers,
  Home,
  Flame,
  ChevronRight,
  BarChart3,
  Scale,
  Zap,
  Check
} from 'lucide-react';

// Formatters
const fmtUSD = (val) => {
  if (val === undefined || val === null) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(val);
};

const fmtIntUSD = (val) => {
  if (val === undefined || val === null) return '$0';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(val);
};

const fmtPct = (val, decimals = 1) => {
  if (val === undefined || val === null) return '0.0%';
  const sign = val > 0 ? '+' : '';
  return `${sign}${Number(val).toFixed(decimals)}%`;
};

// Wealth Tier styling definitions (Opendoor architectural palette)
const TIER_CONFIG = {
  'Entry-Level': {
    badgeBg: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    dot: 'bg-emerald-500',
    color: '#15803D'
  },
  'Mid-Market': {
    badgeBg: 'bg-blue-50 border-blue-200 text-blue-800',
    dot: 'bg-blue-500',
    color: '#0051FF'
  },
  'Move-Up': {
    badgeBg: 'bg-indigo-50 border-indigo-200 text-indigo-800',
    dot: 'bg-indigo-500',
    color: '#4F46E5'
  },
  'Luxury': {
    badgeBg: 'bg-amber-50 border-amber-200 text-amber-800',
    dot: 'bg-amber-500',
    color: '#B45309'
  }
};

// Clean feature name mapping for SHAP
const FEATURE_LABELS = {
  living_area_sqft: 'Living Area (SqFt)',
  lot_size_sqft: 'Lot Size (SqFt)',
  condition_score: 'Physical Condition (1-10)',
  property_age_years: 'Property Age (Years)',
  year_built: 'Year Built',
  school_district_score: 'School District Score',
  median_household_income_usd: 'Zip Median Income ($)',
  cash_buyer_share_pct: 'Cash Buyer Share (%)',
  local_inventory_months: 'Local Inventory (Months)',
  mortgage_rate_30yr: 'Market 30Y Mortgage Rate',
  bedrooms: 'Bedrooms',
  bathrooms: 'Bathrooms',
  has_pool: 'Private Pool',
  has_solar_or_ev: 'Solar / EV Charger',
  estimated_rehab_cost_usd: 'Estimated Rehab ($)',
  daily_holding_cost_usd: 'Daily Holding Cost ($/day)',
  yoy_rate_shock_bps: 'YoY Rate Shock (bps)',
  listing_season: 'Listing Season'
};

export default function OpendoorDashboard() {
  const [properties, setProperties] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [causalParams, setCausalParams] = useState(null);
  const [selectedPropId, setSelectedPropId] = useState(null);
  const [selectedTier, setSelectedTier] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [simulatedRate, setSimulatedRate] = useState(6.50);
  const [underwrite, setUnderwrite] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isUpdating, setIsUpdating] = useState(false);
  const [approvedBids, setApprovedBids] = useState({});
  const [approvalFeedback, setApprovalFeedback] = useState(null);

  // 1. Initial Load of 30 curated properties & ML metadata
  useEffect(() => {
    async function fetchProperties() {
      try {
        setLoading(true);
        const res = await fetch('/api/properties');
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();
        setProperties(data.properties || []);
        setMetrics(data.metrics || null);
        setCausalParams(data.causal_tier_parameters || null);

        if (data.properties && data.properties.length > 0) {
          const firstProp = data.properties[0];
          setSelectedPropId(firstProp.property_id);
          setSimulatedRate(firstProp.mortgage_rate_30yr || 6.50);
        }
      } catch (err) {
        console.error('Failed to load properties:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchProperties();
  }, []);

  // Selected property object
  const currentProperty = useMemo(() => {
    return properties.find((p) => p.property_id === selectedPropId) || null;
  }, [properties, selectedPropId]);

  // When selected property changes, sync simulated rate to its recorded rate if uninitialized
  const handleSelectProperty = (prop) => {
    setSelectedPropId(prop.property_id);
    setSimulatedRate(prop.mortgage_rate_30yr);
  };

  // 2. Real-time Underwrite call on property or simulated rate change
  useEffect(() => {
    if (!selectedPropId) return;

    let isMounted = true;
    async function runUnderwriting() {
      try {
        setIsUpdating(true);
        const res = await fetch('/api/underwrite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            property_id: selectedPropId,
            simulated_mortgage_rate: Number(simulatedRate)
          })
        });
        if (!res.ok) throw new Error('Underwriting failed');
        const data = await res.json();
        if (isMounted) {
          setUnderwrite(data);
        }
      } catch (err) {
        console.error('Underwriting error:', err);
      } finally {
        if (isMounted) setIsUpdating(false);
      }
    }

    const timer = setTimeout(() => {
      runUnderwriting();
    }, 60);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, [selectedPropId, simulatedRate]);

  // Filtered properties for sidebar
  const filteredProperties = useMemo(() => {
    return properties.filter((p) => {
      const matchTier = selectedTier === 'All' || p.zip_wealth_tier === selectedTier;
      const q = searchQuery.toLowerCase().trim();
      const matchSearch =
        !q ||
        p.property_id.toLowerCase().includes(q) ||
        p.market_city.toLowerCase().includes(q) ||
        p.zip_code.toLowerCase().includes(q);
      return matchTier && matchSearch;
    });
  }, [properties, selectedTier, searchQuery]);

  // Handle Bid Approval
  const handleApproveBid = () => {
    if (!underwrite) return;
    const bidInfo = {
      property_id: underwrite.property_id,
      amount: underwrite.optimal_cash_offer_usd,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      rate: simulatedRate
    };
    setApprovedBids((prev) => ({
      ...prev,
      [underwrite.property_id]: bidInfo
    }));
    setApprovalFeedback(bidInfo);
    setTimeout(() => {
      setApprovalFeedback(null);
    }, 4500);
  };

  const isCurrentBidApproved = Boolean(approvedBids[selectedPropId]);

  // Rate delta calculations
  const recordedRate = currentProperty?.mortgage_rate_30yr || 6.50;
  const rateDeltaBps = Math.round((simulatedRate - recordedRate) * 100);
  const currentTier = currentProperty?.zip_wealth_tier || 'Entry-Level';
  const tierCausalParam = causalParams?.[currentTier] || null;

  return (
    <div className="min-h-screen bg-[#F7F7F5] text-[#0F172A] flex flex-col font-sans antialiased selection:bg-[#0051FF] selection:text-white">
      {/* =========================================================================
          TOP INSTITUTIONAL HEADER
      ========================================================================= */}
      <header className="sticky top-0 z-30 bg-[#FFFFFF] border-b border-[#E4E4E0] px-6 py-3.5 shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
        <div className="max-w-[1720px] mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          {/* Brand & Wordmark */}
          <div className="flex items-center gap-3.5">
            <div className="w-8 h-8 rounded bg-[#0051FF] flex items-center justify-center text-white font-bold text-sm shadow-sm tracking-wider">
              OD
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base tracking-tight text-[#0F172A]">OPENDOOR</span>
                <span className="text-xs font-mono text-[#64748B] font-medium tracking-wider">// QUANT UNDERWRITING TERMINAL</span>
                <span className="inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-medium font-mono uppercase rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  LIVE ENGINE
                </span>
              </div>
              <p className="text-[11px] text-[#64748B]">
                Dual-Stage Hedonic XGBoost AVM + KNN Comps + Orthogonal Causal DML Rate-Shock Valuation
              </p>
            </div>
          </div>

          {/* Model Telemetry Badges */}
          {metrics && (
            <div className="flex items-center flex-wrap gap-2 text-xs font-mono">
              <div className="bg-[#F7F7F5] border border-[#E4E4E0] px-2.5 py-1 rounded flex items-center gap-1.5 text-[#334155]">
                <span className="text-[#64748B]">AVM R²:</span>
                <span className="font-bold text-[#0F172A]">{metrics.avm_r2}</span>
              </div>
              <div className="bg-[#F7F7F5] border border-[#E4E4E0] px-2.5 py-1 rounded flex items-center gap-1.5 text-[#334155]">
                <span className="text-[#64748B]">AVM MAE:</span>
                <span className="font-bold text-[#0F172A]">${metrics.avm_mae.toLocaleString()}</span>
                <span className="text-emerald-600 font-semibold text-[11px]">({metrics.avm_mape_pct}%)</span>
              </div>
              <div className="bg-[#F7F7F5] border border-[#E4E4E0] px-2.5 py-1 rounded flex items-center gap-1.5 text-[#334155]">
                <span className="text-[#64748B]">DOM R²:</span>
                <span className="font-bold text-[#0F172A]">{metrics.dom_r2}</span>
              </div>
              <div className="bg-[#F7F7F5] border border-[#E4E4E0] px-2.5 py-1 rounded flex items-center gap-1.5 text-[#334155]">
                <span className="text-[#64748B]">Comps Index:</span>
                <span className="font-bold text-[#0051FF]">10,000 Sunbelt</span>
              </div>
            </div>
          )}
        </div>
      </header>

      {/* =========================================================================
          CONFIRMATION TOAST NOTIFICATION
      ========================================================================= */}
      {approvalFeedback && (
        <div className="fixed bottom-6 right-6 z-50 bg-[#0F172A] text-white px-5 py-4 rounded-lg shadow-xl border border-slate-700 max-w-md flex items-start gap-3.5 animate-in fade-in slide-in-from-bottom-3 duration-200">
          <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="space-y-1">
            <div className="font-semibold text-sm flex items-center gap-2">
              <span>Acquisition Bid Approved</span>
              <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/40">
                {approvalFeedback.property_id}
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Institutional commitment of <strong className="font-mono text-white">{fmtUSD(approvalFeedback.amount)}</strong> confirmed at <span className="font-mono">{approvalFeedback.rate.toFixed(2)}%</span> mortgage regime. Dispatched to Capital Desk & Escrow.
            </p>
            <p className="text-[10px] text-slate-400 font-mono">Logged at {approvalFeedback.timestamp}</p>
          </div>
        </div>
      )}

      {/* =========================================================================
          MAIN TERMINAL WORKSPACE
      ========================================================================= */}
      <main className="max-w-[1720px] w-full mx-auto p-4 md:p-6 flex-1 flex flex-col gap-5">
        {/* =======================================================================
            MACRO STRESS BAR (PROMINENT AT TOP)
        ======================================================================= */}
        <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            {/* Slider & Label */}
            <div className="flex-1 space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <Sliders className="w-4 h-4 text-[#0051FF]" />
                  <span className="text-xs font-bold uppercase tracking-wider text-[#0F172A]">
                    Simulated 30-Year Fixed Mortgage Rate Stress
                  </span>
                  <span className="text-[11px] text-[#64748B] font-mono">
                    (Baseline recorded: {recordedRate.toFixed(2)}%)
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-mono font-bold text-[#0051FF]">
                    {Number(simulatedRate).toFixed(2)}%
                  </span>
                  <span
                    className={`text-xs font-mono font-semibold px-2 py-0.5 rounded border ${
                      rateDeltaBps > 0
                        ? 'bg-amber-50 text-amber-800 border-amber-200'
                        : rateDeltaBps < 0
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : 'bg-slate-50 text-slate-700 border-slate-200'
                    }`}
                  >
                    {rateDeltaBps > 0 ? `+${rateDeltaBps} bps shock` : rateDeltaBps < 0 ? `${rateDeltaBps} bps easing` : 'Par (Neutral)'}
                  </span>
                </div>
              </div>

              {/* Slider Track */}
              <div className="relative flex items-center">
                <input
                  type="range"
                  min="3.50"
                  max="8.50"
                  step="0.25"
                  value={simulatedRate}
                  onChange={(e) => setSimulatedRate(parseFloat(e.target.value))}
                  className="w-full h-2 bg-[#E4E4E0] rounded-lg appearance-none cursor-pointer focus:outline-none"
                />
              </div>

              {/* Slider Scale Ticks */}
              <div className="flex justify-between text-[10px] font-mono text-[#64748B] px-1">
                <span>3.50% (Ultra-Low)</span>
                <span>4.50%</span>
                <span>5.50%</span>
                <span>6.50% (Cycle Avg)</span>
                <span>7.50%</span>
                <span>8.50% (Extreme Shock)</span>
              </div>
            </div>

            {/* Quick Presets */}
            <div className="flex lg:flex-col justify-start lg:justify-center gap-2 shrink-0 border-t lg:border-t-0 lg:border-l border-[#E4E4E0] pt-3 lg:pt-0 lg:pl-6">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-[#64748B]">Macro Presets</span>
              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  onClick={() => setSimulatedRate(4.50)}
                  className={`px-3 py-1.5 text-xs font-mono font-medium rounded border transition-colors ${
                    Math.abs(simulatedRate - 4.50) < 0.01
                      ? 'bg-[#0051FF] text-white border-[#0051FF]'
                      : 'bg-[#F7F7F5] text-[#334155] border-[#E4E4E0] hover:bg-[#EFEFEA]'
                  }`}
                >
                  4.50% Neutral
                </button>
                <button
                  onClick={() => setSimulatedRate(6.25)}
                  className={`px-3 py-1.5 text-xs font-mono font-medium rounded border transition-colors ${
                    Math.abs(simulatedRate - 6.25) < 0.01
                      ? 'bg-[#0051FF] text-white border-[#0051FF]'
                      : 'bg-[#F7F7F5] text-[#334155] border-[#E4E4E0] hover:bg-[#EFEFEA]'
                  }`}
                >
                  6.25% Baseline
                </button>
                <button
                  onClick={() => setSimulatedRate(7.75)}
                  className={`px-3 py-1.5 text-xs font-mono font-medium rounded border transition-colors ${
                    Math.abs(simulatedRate - 7.75) < 0.01
                      ? 'bg-[#0051FF] text-white border-[#0051FF]'
                      : 'bg-[#F7F7F5] text-[#334155] border-[#E4E4E0] hover:bg-[#EFEFEA]'
                  }`}
                >
                  7.75% Fed Shock
                </button>
              </div>
            </div>
          </div>

          {/* Dynamic Macro Regime Insight Banner */}
          {underwrite && (
            <div className="mt-4 pt-4 border-t border-[#E4E4E0] bg-[#F7F7F5]/80 rounded-lg p-3.5 flex flex-col md:flex-row md:items-start gap-4">
              <div className="flex items-center md:flex-col gap-2 shrink-0 md:min-w-[170px]">
                <div className="flex items-center gap-1.5 text-xs font-bold text-[#0F172A]">
                  <Sparkles className="w-3.5 h-3.5 text-[#0051FF]" />
                  <span>MACRO REGIME</span>
                </div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className={`px-2 py-0.5 rounded text-[11px] font-medium border ${TIER_CONFIG[underwrite.zip_wealth_tier]?.badgeBg || 'bg-slate-100'}`}>
                    {underwrite.zip_wealth_tier}
                  </span>
                  <span className="text-[11px] font-mono text-[#475569] bg-white px-2 py-0.5 rounded border border-[#E4E4E0]">
                    Cash Share: {underwrite.cash_buyer_share_pct}%
                  </span>
                </div>
                {tierCausalParam && (
                  <span className="text-[10px] font-mono text-[#64748B]">
                    DML: {tierCausalParam.causal_dom_per_1pct_rate > 0 ? `+${tierCausalParam.causal_dom_per_1pct_rate}d` : `${tierCausalParam.causal_dom_per_1pct_rate}d`} / 100bps
                  </span>
                )}
              </div>

              <div className="flex-1 text-xs text-[#334155] leading-relaxed">
                {underwrite.macro_regime_insight}
              </div>
            </div>
          )}
        </div>

        {/* =======================================================================
            3-COLUMN TERMINAL GRID: PIPELINE | VALUATION & COMPS | OFFER WATERFALL
        ======================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* =====================================================================
              COLUMN 1: ACQUISITION PIPELINE (LEFT SIDEBAR - 3 COLS)
          ===================================================================== */}
          <div className="lg:col-span-3 bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl flex flex-col shadow-[0_1px_3px_rgba(0,0,0,0.02)] h-[840px]">
            {/* Pipeline Header */}
            <div className="p-4 border-b border-[#E4E4E0] space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Home className="w-4 h-4 text-[#0051FF]" />
                  <h2 className="text-xs font-bold uppercase tracking-wider text-[#0F172A]">Acquisition Pipeline</h2>
                </div>
                <span className="text-xs font-mono font-medium text-[#64748B] bg-[#F7F7F5] px-2 py-0.5 rounded border border-[#E4E4E0]">
                  {filteredProperties.length} Homes
                </span>
              </div>

              {/* Search Box */}
              <div className="relative">
                <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[#94A3B8]" />
                <input
                  type="text"
                  placeholder="Filter by ID, City, or Zip..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 text-xs bg-[#F7F7F5] border border-[#E4E4E0] rounded-lg focus:outline-none focus:border-[#0051FF] text-[#0F172A] placeholder-[#94A3B8]"
                />
              </div>

              {/* Wealth Tier Filter Tabs */}
              <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-medium pt-0.5">
                {['All', 'Entry-Level', 'Mid-Market', 'Move-Up', 'Luxury'].map((tier) => (
                  <button
                    key={tier}
                    onClick={() => setSelectedTier(tier)}
                    className={`px-2 py-1 rounded text-[11px] font-medium whitespace-nowrap transition-colors cursor-pointer ${
                      selectedTier === tier
                        ? 'bg-[#0051FF] text-white shadow-xs'
                        : 'bg-[#F7F7F5] text-[#64748B] hover:text-[#0F172A] border border-[#E4E4E0]'
                    }`}
                  >
                    {tier}
                  </button>
                ))}
              </div>
            </div>

            {/* Property Scrollable List */}
            <div className="flex-1 overflow-y-auto divide-y divide-[#E4E4E0] p-1.5">
              {filteredProperties.map((prop) => {
                const isSelected = prop.property_id === selectedPropId;
                const isApproved = Boolean(approvedBids[prop.property_id]);
                const tierStyle = TIER_CONFIG[prop.zip_wealth_tier] || TIER_CONFIG['Entry-Level'];

                return (
                  <button
                    key={prop.property_id}
                    onClick={() => handleSelectProperty(prop)}
                    className={`w-full text-left p-3 rounded-lg transition-all flex flex-col gap-1.5 ${
                      isSelected
                        ? 'bg-[#EFF4FF] border border-[#0051FF]/40 shadow-xs'
                        : 'hover:bg-[#F7F7F5] border border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-[#0F172A]">
                          {prop.property_id}
                        </span>
                        {isApproved && (
                          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.2 rounded">
                            <Check className="w-2.5 h-2.5" /> Bid OK
                          </span>
                        )}
                      </div>
                      <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${tierStyle.badgeBg}`}>
                        {prop.zip_wealth_tier}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-xs text-[#64748B]">
                      <span>{prop.market_city} (Zip {prop.zip_code})</span>
                      <span className="font-mono text-[#0F172A] font-semibold">
                        {fmtIntUSD(prop.baseline_sale_price_usd)}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] font-mono text-[#64748B]">
                      <span>{prop.living_area_sqft} sqft • {prop.bedrooms}b/{prop.bathrooms}ba</span>
                      <span>Burn: ${prop.daily_holding_cost_usd}/d</span>
                    </div>
                  </button>
                );
              })}

              {filteredProperties.length === 0 && (
                <div className="p-8 text-center text-xs text-[#94A3B8]">
                  No properties matching current filter criteria.
                </div>
              )}
            </div>
          </div>

          {/* =====================================================================
              COLUMN 2: VALUATION & COMPS (CENTER COLUMN - 5 COLS)
          ===================================================================== */}
          <div className="lg:col-span-5 flex flex-col gap-5">
            {/* Active Property Metadata Card */}
            {currentProperty && (
              <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E4E4E0] pb-3">
                  <div>
                    <div className="flex items-center gap-2.5">
                      <h1 className="text-base font-bold text-[#0F172A] tracking-tight m-0">
                        {currentProperty.property_id}
                      </h1>
                      <span className="text-xs text-[#64748B]">
                        • {currentProperty.market_city} (Zip {currentProperty.zip_code})
                      </span>
                      <span className={`text-[11px] font-medium px-2 py-0.5 rounded border ${TIER_CONFIG[currentProperty.zip_wealth_tier]?.badgeBg}`}>
                        {currentProperty.zip_wealth_tier}
                      </span>
                    </div>
                    <p className="text-xs text-[#64748B] mt-0.5">
                      Sunbelt Infill Portfolio • Year Built: {currentProperty.year_built} ({currentProperty.property_age_years} yrs old)
                    </p>
                  </div>

                  {/* Badges for pool, solar */}
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {currentProperty.has_pool === 1 && (
                      <span className="text-[10px] font-mono bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded">
                        Pool
                      </span>
                    )}
                    {currentProperty.has_solar_or_ev === 1 && (
                      <span className="text-[10px] font-mono bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 rounded">
                        Solar/EV
                      </span>
                    )}
                    <span className="text-[10px] font-mono bg-slate-100 text-slate-700 px-2 py-0.5 rounded">
                      School: {currentProperty.school_district_score}/10
                    </span>
                  </div>
                </div>

                {/* Valuation Anchoring Overview */}
                {underwrite && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-[#F7F7F5] border border-[#E4E4E0] p-3 rounded-lg">
                      <span className="text-[10px] uppercase tracking-wider font-semibold text-[#64748B] block">
                        XGBoost AVM Estimate
                      </span>
                      <span className="text-sm font-mono font-bold text-[#0F172A] block mt-1">
                        {fmtUSD(underwrite.avm_baseline_price_usd)}
                      </span>
                      <span className="text-[10px] text-[#64748B] font-mono">Weight: 75% Hedonic</span>
                    </div>

                    <div className="bg-[#F7F7F5] border border-[#E4E4E0] p-3 rounded-lg">
                      <span className="text-[10px] uppercase tracking-wider font-semibold text-[#64748B] block">
                        KNN Comps Median
                      </span>
                      <span className="text-sm font-mono font-bold text-[#0F172A] block mt-1">
                        {fmtUSD(underwrite.knn_comp_median_price_usd)}
                      </span>
                      <span className="text-[10px] text-[#64748B] font-mono">Weight: 25% Local Comps</span>
                    </div>

                    <div className="bg-[#EFF4FF] border border-[#0051FF]/30 p-3 rounded-lg">
                      <span className="text-[10px] uppercase tracking-wider font-semibold text-[#0051FF] block">
                        Anchored Valuation
                      </span>
                      <span className="text-sm font-mono font-bold text-[#0051FF] block mt-1">
                        {fmtUSD(underwrite.anchored_valuation_usd)}
                      </span>
                      <span className="text-[10px] text-[#0051FF]/80 font-mono">Baseline FMV Anchor</span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* SHAP Hedonic Dollar Attribution Card */}
            {underwrite && (
              <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-4">
                <div className="flex items-center justify-between border-b border-[#E4E4E0] pb-3">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-[#0051FF]" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#0F172A] m-0">
                      SHAP Hedonic Dollar Attribution
                    </h3>
                  </div>
                  <span className="text-[11px] text-[#64748B] font-mono">
                    Marginal Value Contribution ($)
                  </span>
                </div>

                <div className="space-y-3">
                  {underwrite.shap_drivers?.map((driver, idx) => {
                    const isPos = driver.dollar_impact >= 0;
                    const maxAbs = Math.max(...underwrite.shap_drivers.map((d) => Math.abs(d.dollar_impact)), 1);
                    const barWidthPct = Math.min(100, Math.round((Math.abs(driver.dollar_impact) / maxAbs) * 100));
                    const cleanLabel = FEATURE_LABELS[driver.feature] || driver.feature;

                    return (
                      <div key={idx} className="space-y-1 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-[#334155] font-medium">{cleanLabel}</span>
                          <span
                            className={`font-mono font-semibold ${
                              isPos ? 'text-emerald-700' : 'text-amber-800'
                            }`}
                          >
                            {isPos ? `+${fmtUSD(driver.dollar_impact)}` : fmtUSD(driver.dollar_impact)}
                          </span>
                        </div>

                        {/* Diverging Bar */}
                        <div className="w-full h-2 bg-[#F1F1ED] rounded-full overflow-hidden flex">
                          {isPos ? (
                            <div
                              className="h-full bg-emerald-600 rounded-full transition-all duration-300"
                              style={{ width: `${barWidthPct}%` }}
                            />
                          ) : (
                            <div
                              className="h-full bg-amber-600 rounded-full transition-all duration-300 ml-auto"
                              style={{ width: `${barWidthPct}%` }}
                            />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* KNN Comparable Sales ("Comps") Table */}
            {underwrite && (
              <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-3">
                <div className="flex items-center justify-between border-b border-[#E4E4E0] pb-3">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#0051FF]" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#0F172A] m-0">
                      KNN Nearest Twin Comparable Sales
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-[#64748B]">
                    Cosine Similarity Vector Match
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-[#E4E4E0] text-[10px] font-semibold text-[#64748B] uppercase tracking-wider">
                        <th className="py-2 pr-3">Property ID</th>
                        <th className="py-2 px-3">Similarity</th>
                        <th className="py-2 px-3">SqFt / Spec</th>
                        <th className="py-2 px-3">DOM</th>
                        <th className="py-2 pl-3 text-right">Sale Price</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#E4E4E0] font-mono">
                      {underwrite.knn_comps?.map((comp, idx) => (
                        <tr key={idx} className="hover:bg-[#F7F7F5] transition-colors">
                          <td className="py-2.5 pr-3 text-[#0F172A] font-semibold">
                            {comp.property_id}
                            <span className="block text-[10px] text-[#64748B] font-sans font-normal">
                              {comp.market_city} (Zip {comp.zip_code})
                            </span>
                          </td>
                          <td className="py-2.5 px-3">
                            <span className="inline-block bg-blue-50 text-[#0051FF] border border-blue-200 px-1.5 py-0.5 rounded text-[11px] font-semibold">
                              {comp.similarity_pct}% Match
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-[#334155] text-[11px]">
                            {comp.living_area_sqft} sqft
                            <span className="block text-[10px] text-[#64748B]">
                              {comp.bedrooms}b / {comp.bathrooms}ba • {comp.condition_score}/10
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-[#334155] text-[11px]">
                            {comp.days_on_market} days
                          </td>
                          <td className="py-2.5 pl-3 text-right font-bold text-[#0F172A]">
                            {fmtUSD(comp.sale_price_usd)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* =====================================================================
              COLUMN 3: CAUSAL LIQUIDITY RISK & BID WATERFALL (RIGHT - 4 COLS)
          ===================================================================== */}
          <div className="lg:col-span-4 flex flex-col gap-5">
            {/* Causal DOM & Liquidity Risk Gauge */}
            {underwrite && (
              <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-4">
                <div className="flex items-center justify-between border-b border-[#E4E4E0] pb-3">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-[#0051FF]" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#0F172A] m-0">
                      Causal Liquidity & DOM Stress
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-[#64748B]">
                    DML Orthogonalized
                  </span>
                </div>

                {/* DOM Visual Comparison */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-[#F7F7F5] border border-[#E4E4E0] p-3 rounded-lg">
                    <span className="text-[10px] font-semibold uppercase text-[#64748B] block">Baseline DOM</span>
                    <span className="text-2xl font-mono font-bold text-[#0F172A] block mt-1">
                      {underwrite.baseline_dom_days} <span className="text-xs font-normal text-[#64748B]">days</span>
                    </span>
                    <span className="text-[10px] text-[#64748B]">Pre-Shock Velocity</span>
                  </div>

                  <div className={`p-3 rounded-lg border ${
                    underwrite.causal_dom_added_days > 0
                      ? 'bg-amber-50 border-amber-200'
                      : 'bg-emerald-50 border-emerald-200'
                  }`}>
                    <span className="text-[10px] font-semibold uppercase text-[#64748B] block">Stressed DOM</span>
                    <span className={`text-2xl font-mono font-bold block mt-1 ${
                      underwrite.causal_dom_added_days > 0 ? 'text-amber-800' : 'text-emerald-800'
                    }`}>
                      {underwrite.stressed_dom_days} <span className="text-xs font-normal">days</span>
                    </span>
                    <span className={`text-[10px] font-mono font-semibold ${
                      underwrite.causal_dom_added_days > 0 ? 'text-amber-700' : 'text-emerald-700'
                    }`}>
                      {underwrite.causal_dom_added_days > 0
                        ? `+${underwrite.causal_dom_added_days} extra days`
                        : `${underwrite.causal_dom_added_days} days`}
                    </span>
                  </div>
                </div>

                {/* Holding Burn Metric */}
                <div className="bg-[#F7F7F5] border border-[#E4E4E0] p-3 rounded-lg flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Flame className="w-4 h-4 text-amber-600" />
                    <div>
                      <span className="text-xs font-semibold text-[#0F172A] block">Daily Balance Sheet Burn</span>
                      <span className="text-[10px] text-[#64748B]">Taxes, Insurance, Utilities & Capital Cost</span>
                    </div>
                  </div>
                  <span className="font-mono text-sm font-bold text-[#0F172A]">
                    ${underwrite.daily_holding_cost_usd.toFixed(2)}/day
                  </span>
                </div>
              </div>
            )}

            {/* Institutional Offer Waterfall Sheet */}
            {underwrite && (
              <div className="bg-[#FFFFFF] border border-[#E4E4E0] rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.02)] space-y-4">
                <div className="flex items-center justify-between border-b border-[#E4E4E0] pb-3">
                  <div className="flex items-center gap-2">
                    <Scale className="w-4 h-4 text-[#0051FF]" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-[#0F172A] m-0">
                      iBuying Unit Economics Waterfall
                    </h3>
                  </div>
                  <span className="text-[11px] font-mono text-[#64748B]">
                    US GAAP Net Cash Bid
                  </span>
                </div>

                {/* Accounting Breakdown Rows */}
                <div className="space-y-2 text-xs font-mono">
                  {/* Row 1: Anchored FMV */}
                  <div className="flex items-center justify-between text-[#334155] py-1">
                    <span className="font-sans font-medium text-[#0F172A]">Anchored Fair Market Value</span>
                    <span className="font-semibold">{fmtUSD(underwrite.anchored_valuation_usd)}</span>
                  </div>

                  {/* Row 2: Causal Price Shift */}
                  <div className="flex items-center justify-between text-amber-800 py-1 bg-amber-50/50 px-2 rounded">
                    <span className="font-sans font-medium">
                      Causal Price Shift ({underwrite.causal_price_shift_pct}%)
                    </span>
                    <span>
                      {fmtUSD(underwrite.stressed_resale_price_usd - underwrite.anchored_valuation_usd)}
                    </span>
                  </div>

                  {/* Row 3: Stressed Resale Value */}
                  <div className="flex items-center justify-between text-[#0F172A] font-bold py-1.5 border-t border-b border-[#E4E4E0]">
                    <span className="font-sans">Expected Stressed Resale</span>
                    <span>{fmtUSD(underwrite.stressed_resale_price_usd)}</span>
                  </div>

                  {/* Deductions header */}
                  <div className="text-[10px] uppercase font-sans font-semibold text-[#64748B] pt-2">
                    Less: Portfolio Carry & Execution Deductions
                  </div>

                  {/* Deduction 1: Holding Cost */}
                  <div className="flex items-center justify-between text-[#475569] py-1">
                    <span className="font-sans">
                      Total Holding Burn ({underwrite.stressed_dom_days}d × ${underwrite.daily_holding_cost_usd}/d)
                    </span>
                    <span className="text-red-700">-{fmtUSD(underwrite.total_holding_cost_usd)}</span>
                  </div>

                  {/* Deduction 2: Estimated Rehab */}
                  <div className="flex items-center justify-between text-[#475569] py-1">
                    <span className="font-sans">Estimated Renovation & Rehab</span>
                    <span className="text-red-700">-{fmtUSD(underwrite.estimated_rehab_cost_usd)}</span>
                  </div>

                  {/* Deduction 3: Liquidity Risk Buffer */}
                  <div className="flex items-center justify-between text-[#475569] py-1">
                    <span className="font-sans flex items-center gap-1">
                      Liquidity Risk Buffer
                      {underwrite.stressed_dom_days > 60 && (
                        <span className="text-[9px] bg-red-100 text-red-800 px-1 rounded">&gt;60d Burn</span>
                      )}
                    </span>
                    <span className="text-red-700">-{fmtUSD(underwrite.liquidity_risk_buffer_usd)}</span>
                  </div>

                  {/* Deduction 4: Opendoor Fee */}
                  <div className="flex items-center justify-between text-[#475569] py-1">
                    <span className="font-sans">Opendoor Margin & Service (5.0%)</span>
                    <span className="text-red-700">-{fmtUSD(underwrite.opendoor_service_fee_usd)}</span>
                  </div>

                  {/* Subtotal Deductions */}
                  <div className="flex items-center justify-between text-[11px] text-[#64748B] pt-1 border-t border-dashed border-[#E4E4E0]">
                    <span className="font-sans">Total Carry & Fee Deductions</span>
                    <span className="font-semibold text-red-700">
                      -{fmtUSD(
                        underwrite.total_holding_cost_usd +
                        underwrite.estimated_rehab_cost_usd +
                        underwrite.liquidity_risk_buffer_usd +
                        underwrite.opendoor_service_fee_usd
                      )}
                    </span>
                  </div>
                </div>

                {/* Final Cash Offer Hero Card */}
                <div className="bg-[#EFF4FF] border-2 border-[#0051FF] rounded-xl p-4.5 space-y-2 mt-4 text-center">
                  <span className="text-[11px] font-bold uppercase tracking-wider text-[#0051FF] block">
                    OPTIMAL ALL-CASH OFFER (NET TO SELLER)
                  </span>
                  <div className="text-3xl font-mono font-extrabold text-[#0051FF] tracking-tight">
                    {fmtUSD(underwrite.optimal_cash_offer_usd)}
                  </div>
                  <div className="flex items-center justify-center gap-2 text-[11px] font-mono text-[#475569]">
                    <span>
                      Spread from FMV: -{fmtUSD(underwrite.anchored_valuation_usd - underwrite.optimal_cash_offer_usd)}
                    </span>
                    <span>
                      ({(
                        ((underwrite.optimal_cash_offer_usd - underwrite.anchored_valuation_usd) /
                          underwrite.anchored_valuation_usd) *
                        100
                      ).toFixed(1)}%)
                    </span>
                  </div>
                </div>

                {/* Action Confirmation Button */}
                <button
                  onClick={handleApproveBid}
                  className={`w-full py-3 px-4 rounded-lg font-semibold text-xs tracking-wide transition-all shadow-sm flex items-center justify-center gap-2 ${
                    isCurrentBidApproved
                      ? 'bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-500/20'
                      : 'bg-[#0051FF] hover:bg-[#0043D6] active:bg-[#0038B3] text-white shadow-blue-500/25 cursor-pointer'
                  }`}
                >
                  {isCurrentBidApproved ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 text-white" />
                      <span>ACQUISITION BID APPROVED • LOCKED</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>APPROVE ACQUISITION BID</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* =========================================================================
          MINIMALIST INSTITUTIONAL FOOTER
      ========================================================================= */}
      <footer className="mt-auto border-t border-[#E4E4E0] bg-[#FFFFFF] py-3.5 px-6 text-xs text-[#64748B]">
        <div className="max-w-[1720px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 font-mono text-[11px]">
          <div>
            <span>Opendoor Technologies Inc. // Quantitative Investment & Portfolio Risk System</span>
          </div>
          <div className="flex items-center gap-4 text-[#475569]">
            <span>Phoenix • Austin • Atlanta • Raleigh • Charlotte</span>
            <span>Orthogonal DML v2.4</span>
            <span className="text-[#0051FF] font-semibold">Engine Latency: ~50ms</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
