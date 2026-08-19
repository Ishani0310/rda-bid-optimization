# RDA Bid Optimization System

A Streamlit web application using Integer Linear Programming (`PuLP`) to determine the cost-optimal contract allocation for Road Development Authority (RDA) batches under bidder capacity constraints.

## Features
- Dynamic detection of project codes (`P1`, `P39`, `P157`, etc.) across varying sheet layouts.
- Enforces contractor capacity limits based on available balance or gross annual quotas.
- Multi-sheet selection covering both visible and hidden batch tabs.
- Formatted Excel report export (`.xlsx`) with currency formatting.

## Setup & Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

Launch the web application:
streamlit run app.py