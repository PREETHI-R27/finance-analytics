# Aegis Logistics Valuation

## Project Overview
This project contains a financial valuation model for **Aegis Logistics Ltd (NSE: AEGISLOG)**, built using Python and Excel. It performs a fundamental analysis using three methodologies:
1.  **Discounted Cash Flow (DCF)** Analysis
2.  **Comparable Company Analysis** (Trading Comps)
3.  **Dividend Discount Model (DDM)**

## Folder Structure
*   `src/`: Contains the Python source code (`main.py`) to generate the model.
*   `output/`: Contains the generated Excel model (`Aegis_Valuation_Model.xlsx`) and the Investment Summary (`Summary.md`).

## How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the generation script:
    ```bash
    python src/main.py
    ```
3.  The Excel model will be generated in the `output/` directory.

## Key Assumptions
*   **WACC**: 11.0% (Risk-Free Rate: 6.8%, Beta: 1.0)
*   **Revenue Growth**: 12% initial, tapering to 5%
*   **EBITDA Margin**: 14% (Conservative estimate)

## Disclaimer
This model is for educational purposes only and does not constitute financial advice.
