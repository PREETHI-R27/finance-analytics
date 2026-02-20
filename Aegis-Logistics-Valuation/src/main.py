"""
Aegis Logistics Valuation Model Builder

This script generates a comprehensive valuation model for Aegis Logistics Ltd (NSE: AEGISLOG).
It creates an Excel file containing:
1. Discounted Cash Flow (DCF) Analysis
2. Comparable Company Analysis (Trading Comps)
3. Dividend Discount Model (DDM)

Author: Investment Analyst
Date: Feb 2026
"""

import pandas as pd
import xlsxwriter
import os

# --- Constants & Assumptions ---

# Market Data
CURRENT_PRICE = 775.0
SHARES_OUTSTANDING = 35.10 # Crores
MARKET_CAP = CURRENT_PRICE * SHARES_OUTSTANDING # INR Cr
NET_DEBT = 300.0 # Estimate INR Cr (Gross Debt - Cash)

# WACC Inputs
RISK_FREE_RATE = 0.068       # India 10Y Bond Yield
BETA = 1.0                   # Conservative estimate
MARKET_RISK_PREMIUM = 0.055  # Equity Risk Premium
COST_OF_DEBT_PRETAX = 0.072  # Approx borrowing cost
TAX_RATE = 0.25              # Corporate Tax Rate
EQUITY_WEIGHT = 0.90         # Based on Market Cap vs Net Debt
DEBT_WEIGHT = 0.10

# Forecast Assumptions
REV_GROWTH_INITIAL = 0.12    # Initial growth rate (12%)
REV_GROWTH_TERMINAL = 0.05   # Terminal period growth rate (5%)
EBITDA_MARGIN = 0.14         # Conservative margin estimate
CAPEX_PERCENT_REV = 0.04     # Maintenance Capex % of Revenue
WC_PERCENT_REV = 0.08        # Working Capital % of Revenue
TERMINAL_GROWTH_RATE = 0.04  # Perpetual growth rate

# File Paths
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'Aegis_Valuation_Model.xlsx')

def ensure_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def calculate_wacc():
    """Calculates Weighted Average Cost of Capital."""
    cost_of_equity = RISK_FREE_RATE + BETA * MARKET_RISK_PREMIUM
    cost_of_debt_post_tax = COST_OF_DEBT_PRETAX * (1 - TAX_RATE)
    wacc = (cost_of_equity * EQUITY_WEIGHT) + (cost_of_debt_post_tax * DEBT_WEIGHT)
    return wacc, cost_of_equity

def generate_model():
    """Generates the valuation model in Excel."""
    ensure_output_directory()
    
    print(f"Generating model at: {OUTPUT_FILE}")
    writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')
    workbook = writer.book

    # --- Formats ---
    # Blue for inputs, Black for formulas/output
    fmt_header = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'border': 1, 'align': 'center'})
    
    # Input Formats (Blue)
    fmt_input_num = workbook.add_format({'num_format': '0.00', 'font_color': 'blue', 'border': 1})
    fmt_input_pct = workbook.add_format({'num_format': '0.0%', 'font_color': 'blue', 'border': 1})
    fmt_input_curr = workbook.add_format({'num_format': '₹#,##0', 'font_color': 'blue', 'border': 1})

    # Calculation Formats (Black)
    fmt_calc_num = workbook.add_format({'num_format': '0.00', 'font_color': 'black', 'border': 1})
    fmt_calc_pct = workbook.add_format({'num_format': '0.0%', 'font_color': 'black', 'border': 1})
    fmt_calc_curr = workbook.add_format({'num_format': '₹#,##0', 'font_color': 'black', 'border': 1})

    fmt_bold = workbook.add_format({'bold': True, 'border': 1})
    fmt_border = workbook.add_format({'border': 1})
    fmt_title = workbook.add_format({'bold': True, 'font_size': 14, 'font_color': '#366092'})

    # --- Sheet 1: DCF ---
    ws_dcf = workbook.add_worksheet('DCF')
    ws_dcf.set_column('A:A', 30)
    ws_dcf.set_column('B:M', 12)
    ws_dcf.hide_gridlines(2)

    # Title
    ws_dcf.write('A1', 'Aegis Logistics Ltd - DCF Valuation', fmt_title)
    ws_dcf.write('A2', 'Amounts in INR Cr unless stated otherwise')

    # --- Assumptions Section (Blue Inputs) ---
    ws_dcf.write('A4', 'Key Assumptions', fmt_header)
    ws_dcf.write('B4', 'Value', fmt_header)

    assumptions = [
        ('Risk Free Rate', RISK_FREE_RATE, fmt_input_pct),
        ('Beta', BETA, fmt_input_num),
        ('Market Risk Premium', MARKET_RISK_PREMIUM, fmt_input_pct),
        ('Cost of Debt (Pre-tax)', COST_OF_DEBT_PRETAX, fmt_input_pct),
        ('Tax Rate', TAX_RATE, fmt_input_pct),
        ('Equity Weight', EQUITY_WEIGHT, fmt_input_pct),
        ('Debt Weight', DEBT_WEIGHT, fmt_input_pct),
        ('Terminal Growth Rate', TERMINAL_GROWTH_RATE, fmt_input_pct),
        ('Shares Outstanding (Cr)', SHARES_OUTSTANDING, fmt_input_num),
        ('Net Debt (Cr)', NET_DEBT, fmt_input_curr),
        ('Current Share Price (INR)', CURRENT_PRICE, fmt_input_curr)
    ]

    row_idx = 4
    for label, value, fmt in assumptions:
        ws_dcf.write(row_idx, 0, label, fmt_border)
        ws_dcf.write(row_idx, 1, value, fmt)
        row_idx += 1

    # Calculated WACC (Black)
    wacc, cost_of_equity = calculate_wacc()

    ws_dcf.write(row_idx, 0, 'WACC (Calculated)', fmt_bold)
    ws_dcf.write(row_idx, 1, wacc, fmt_calc_pct)
    row_idx += 2

    # --- DCF Projection Section ---
    years_hist = ['FY20A', 'FY21A', 'FY22A', 'FY23A', 'FY24A']
    years_proj = ['FY25E', 'FY26E', 'FY27E', 'FY28E', 'FY29E']
    headers = ['Metric'] + years_hist + years_proj

    ws_dcf.write_row(row_idx, 0, headers, fmt_header)
    start_proj_col = len(years_hist) + 1 # Column G (index 6)

    # Historical Data (Hardcoded Inputs)
    revenue_hist = [7183, 3843, 4630, 8630, 7150]
    ebitda_hist = [570, 450, 580, 750, 850]

    # Projections
    revenue_proj = []
    ebitda_proj = []
    ebit_proj = []
    nopat_proj = []
    capex_proj = []
    change_wc_proj = []
    fcf_proj = []
    discount_factors = []
    pv_fcf = []

    last_rev = revenue_hist[-1]
    last_wc = last_rev * WC_PERCENT_REV

    for i, year in enumerate(years_proj):
        # Tapering growth
        growth = REV_GROWTH_INITIAL - (i * 0.015)
        if growth < REV_GROWTH_TERMINAL: growth = REV_GROWTH_TERMINAL
        
        rev = last_rev * (1 + growth)
        ebitda = rev * EBITDA_MARGIN
        dep = rev * 0.02 # Approx D&A
        ebit = ebitda - dep
        tax = ebit * TAX_RATE
        nopat = ebit - tax
        capex = rev * CAPEX_PERCENT_REV
        
        wc_req = rev * WC_PERCENT_REV
        delta_wc = wc_req - last_wc
        
        fcf = nopat + dep - capex - delta_wc
        
        df = 1 / ((1 + wacc) ** (i + 1))
        pv = fcf * df
        
        # Store values
        revenue_proj.append(rev)
        ebitda_proj.append(ebitda)
        ebit_proj.append(ebit)
        nopat_proj.append(nopat)
        capex_proj.append(capex)
        change_wc_proj.append(delta_wc)
        fcf_proj.append(fcf)
        discount_factors.append(df)
        pv_fcf.append(pv)
        
        # Update trackers
        last_rev = rev
        last_wc = wc_req

    # Write Data Rows
    row_idx += 1
    
    # Revenue
    ws_dcf.write(row_idx, 0, 'Revenue', fmt_bold)
    ws_dcf.write_row(row_idx, 1, revenue_hist, fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, revenue_proj, fmt_calc_curr)
    row_idx += 1

    # Growth %
    ws_dcf.write(row_idx, 0, 'Growth %', fmt_border)
    ws_dcf.write(row_idx, 1, '', fmt_border) # Skip first
    for i in range(1, len(revenue_hist)):
        g = (revenue_hist[i] / revenue_hist[i-1]) - 1
        ws_dcf.write(row_idx, 1+i, g, fmt_calc_pct)
    for i in range(len(revenue_proj)):
        prev = revenue_hist[-1] if i == 0 else revenue_proj[i-1]
        g = (revenue_proj[i] / prev) - 1
        ws_dcf.write(row_idx, start_proj_col+i, g, fmt_calc_pct)
    row_idx += 1

    # EBITDA
    ws_dcf.write(row_idx, 0, 'EBITDA', fmt_bold)
    ws_dcf.write_row(row_idx, 1, ebitda_hist, fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, ebitda_proj, fmt_calc_curr)
    row_idx += 1

    # EBIT (Approx Hist)
    ws_dcf.write(row_idx, 0, 'EBIT', fmt_border)
    ws_dcf.write_row(row_idx, 1, [x * 0.85 for x in ebitda_hist], fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, ebit_proj, fmt_calc_curr)
    row_idx += 1

    # NOPAT
    ws_dcf.write(row_idx, 0, 'NOPAT', fmt_border)
    ws_dcf.write_row(row_idx, 1, [x * 0.85 * (1-TAX_RATE) for x in ebitda_hist], fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, nopat_proj, fmt_calc_curr)
    row_idx += 1

    # Adjustments
    ws_dcf.write(row_idx, 0, 'Add: D&A', fmt_border)
    ws_dcf.write_row(row_idx, 1, [x * 0.15 for x in ebitda_hist], fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, [x * 0.02 for x in revenue_proj], fmt_calc_curr)
    row_idx += 1

    ws_dcf.write(row_idx, 0, 'Less: Capex', fmt_border)
    ws_dcf.write_row(row_idx, 1, [x * 0.04 for x in revenue_hist], fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, capex_proj, fmt_calc_curr)
    row_idx += 1

    ws_dcf.write(row_idx, 0, 'Less: Change in WC', fmt_border)
    ws_dcf.write_row(row_idx, 1, [0]*5, fmt_input_curr)
    ws_dcf.write_row(row_idx, start_proj_col, change_wc_proj, fmt_calc_curr)
    row_idx += 1

    # FCF
    ws_dcf.write(row_idx, 0, 'Unlevered Free Cash Flow', fmt_header)
    ws_dcf.write_row(row_idx, 1, [0]*5, fmt_calc_curr)
    ws_dcf.write_row(row_idx, start_proj_col, fcf_proj, fmt_calc_curr)
    row_idx += 2

    # Discount Factors
    ws_dcf.write(row_idx, 0, 'Discount Factor', fmt_border)
    ws_dcf.write_row(row_idx, 1, ['']*5, fmt_border)
    ws_dcf.write_row(row_idx, start_proj_col, discount_factors, fmt_calc_num)
    row_idx += 1

    ws_dcf.write(row_idx, 0, 'PV of FCF', fmt_bold)
    ws_dcf.write_row(row_idx, 1, ['']*5, fmt_border)
    ws_dcf.write_row(row_idx, start_proj_col, pv_fcf, fmt_calc_curr)
    row_idx += 3

    # --- Valuation Output ---
    val_row = row_idx
    ws_dcf.write(val_row, 0, 'Valuation Summary', fmt_header)
    ws_dcf.write(val_row, 1, 'Amount (Cr)', fmt_header)

    ws_dcf.write(val_row+1, 0, 'Sum of PV (Forecast Period)', fmt_border)
    ws_dcf.write(val_row+1, 1, sum(pv_fcf), fmt_calc_curr)

    # Terminal Value
    last_fcf = fcf_proj[-1]
    tv = (last_fcf * (1 + TERMINAL_GROWTH_RATE)) / (wacc - TERMINAL_GROWTH_RATE)
    pv_tv = tv * discount_factors[-1]

    ws_dcf.write(val_row+2, 0, 'PV of Terminal Value', fmt_border)
    ws_dcf.write(val_row+2, 1, pv_tv, fmt_calc_curr)

    enterprise_value = sum(pv_fcf) + pv_tv
    ws_dcf.write(val_row+3, 0, 'Enterprise Value', fmt_bold)
    ws_dcf.write(val_row+3, 1, enterprise_value, fmt_calc_curr)

    ws_dcf.write(val_row+4, 0, 'Less: Net Debt', fmt_border)
    ws_dcf.write(val_row+4, 1, NET_DEBT, fmt_input_curr)

    equity_value = enterprise_value - NET_DEBT
    ws_dcf.write(val_row+5, 0, 'Equity Value', fmt_bold)
    ws_dcf.write(val_row+5, 1, equity_value, fmt_calc_curr)

    ws_dcf.write(val_row+6, 0, 'Implied Share Price', fmt_header)
    price = equity_value / SHARES_OUTSTANDING
    ws_dcf.write(val_row+6, 1, price, fmt_calc_curr)

    ws_dcf.write(val_row+7, 0, 'Upside / (Downside)', fmt_border)
    upside = (price / CURRENT_PRICE) - 1
    ws_dcf.write(val_row+7, 1, upside, fmt_calc_pct)

    # --- Sensitivity Table ---
    sens_row = val_row
    ws_dcf.write(sens_row, 4, 'Sensitivity Analysis (Share Price)', fmt_header)
    ws_dcf.merge_range(sens_row, 4, sens_row, 8, 'Sensitivity Analysis (Share Price)', fmt_header)

    ws_dcf.write(sens_row+1, 4, 'WACC \\ Growth', fmt_bold)
    growth_sens = [TERMINAL_GROWTH_RATE - 0.01, TERMINAL_GROWTH_RATE, TERMINAL_GROWTH_RATE + 0.01, TERMINAL_GROWTH_RATE + 0.02]
    wacc_sens = [wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]

    for c, g in enumerate(growth_sens):
        ws_dcf.write(sens_row+1, 5+c, g, fmt_input_pct)

    for r, w in enumerate(wacc_sens):
        ws_dcf.write(sens_row+2+r, 4, w, fmt_input_pct)
        for c, g in enumerate(growth_sens):
            tv_s = (last_fcf * (1 + g)) / (w - g)
            pv_tv_s = tv_s / ((1+w)**5)
            pv_fcf_s = sum([f / ((1+w)**(i+1)) for i, f in enumerate(fcf_proj)])
            ev_s = pv_fcf_s + pv_tv_s
            eq_s = ev_s - NET_DEBT
            p_s = eq_s / SHARES_OUTSTANDING
            ws_dcf.write(sens_row+2+r, 5+c, p_s, fmt_calc_curr)

    # --- Sheet 2: Comparable Companies ---
    ws_comps = workbook.add_worksheet('Comps')
    ws_comps.hide_gridlines(2)
    ws_comps.write('A1', 'Comparable Company Analysis', fmt_title)
    ws_comps.set_column('A:A', 20)
    ws_comps.set_column('B:F', 15)

    comps_headers = ['Ticker', 'Company', 'Price', 'Mkt Cap', 'EV/EBITDA', 'P/E']
    ws_comps.write_row('A3', comps_headers, fmt_header)

    # Estimated peers data
    comps_data = [
        ['ADANIPORTS', 'Adani Ports', 1300, 280000, 20.5, 35.0],
        ['CONCOR', 'Container Corp', 900, 55000, 18.2, 30.0],
        ['VRL', 'VRL Logistics', 600, 5000, 12.5, 25.0],
        ['AEGISLOG', 'Aegis Logistics', CURRENT_PRICE, MARKET_CAP, enterprise_value/ebitda_hist[-1], CURRENT_PRICE/(nopat_hist_proxy/SHARES_OUTSTANDING) if 'nopat_hist_proxy' in locals() else 30.0]
    ]

    for i, row in enumerate(comps_data):
        ws_comps.write(i+3, 0, row[0], fmt_border)
        ws_comps.write(i+3, 1, row[1], fmt_border)
        ws_comps.write(i+3, 2, row[2], fmt_input_curr if i < 3 else fmt_calc_curr)
        ws_comps.write(i+3, 3, row[3], fmt_input_curr if i < 3 else fmt_calc_curr)
        ws_comps.write(i+3, 4, row[4], fmt_input_num if i < 3 else fmt_calc_num)
        ws_comps.write(i+3, 5, row[5], fmt_input_num if i < 3 else fmt_calc_num)

    # --- Sheet 3: DDM ---
    ws_ddm = workbook.add_worksheet('DDM')
    ws_ddm.hide_gridlines(2)
    ws_ddm.write('A1', 'Dividend Discount Model', fmt_title)
    ws_ddm.set_column('A:B', 25)

    ws_ddm.write('A3', 'Assumptions', fmt_header)
    ws_ddm.write('A4', 'Payout Ratio', fmt_border)
    ws_ddm.write('B4', 0.30, fmt_input_pct)
    ws_ddm.write('A5', 'Cost of Equity', fmt_border)
    ws_ddm.write('B5', cost_of_equity, fmt_calc_pct)

    ws_ddm.write_row('A7', ['Year', 'Dividend (Cr)'], fmt_header)
    div_payout = 0.30
    dividends = [n * div_payout for n in nopat_proj] 

    for i, d in enumerate(dividends):
        ws_ddm.write(7+i, 0, years_proj[i], fmt_border)
        ws_ddm.write(7+i, 1, d, fmt_calc_curr)

    tv_ddm = (dividends[-1] * (1 + TERMINAL_GROWTH_RATE)) / (cost_of_equity - TERMINAL_GROWTH_RATE)
    pv_tv_ddm = tv_ddm * discount_factors[-1]
    pv_divs = sum([d * df for d, df in zip(dividends, discount_factors)])
    val_ddm = pv_divs + pv_tv_ddm
    price_ddm = val_ddm / SHARES_OUTSTANDING

    r = 7 + len(dividends) + 1
    ws_ddm.write(r, 0, 'Implied Equity Value', fmt_bold)
    ws_ddm.write(r, 1, val_ddm, fmt_calc_curr)
    ws_ddm.write(r+1, 0, 'Implied Share Price', fmt_header)
    ws_ddm.write(r+1, 1, price_ddm, fmt_calc_curr)

    workbook.close()
    print("Model generation complete.")

if __name__ == "__main__":
    generate_model()
