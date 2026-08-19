import io
import math
import re
import pandas as pd
import pulp
import streamlit as st

# --------------------------------------------------
# Page Settings
# --------------------------------------------------

st.set_page_config(
    page_title="RDA Bid Optimization System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RDA Bid Optimization System")

# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload RDA Batch Excel File (.xlsx, .xls, .xlsm)",
    type=["xls", "xlsx", "xlsm"],
    help="Upload Excel files containing bidder matrices and capacity columns.",
)

if uploaded_file is not None:
    try:
        st.success("Excel file uploaded successfully! ✅")

        # 1. Load ALL sheets without filtering out hidden sheets or master sheets
        uploaded_file.seek(0)
        xl = pd.ExcelFile(uploaded_file)
        all_sheets = xl.sheet_names

        # 2. Show all sheets in dropdown directly
        selected_sheet = st.selectbox(
            "Select Sheet / Tab to Optimize (Includes All Hidden & Visible Sheets)",
            options=all_sheets,
            index=0,
        )

        st.subheader(f"📌 Active Sheet: {selected_sheet}")

        df_raw = pd.read_excel(xl, sheet_name=selected_sheet, header=None)

        # 3. Auto-detect Header Row
        detected_header_idx = 0
        for idx in range(min(20, len(df_raw))):
            row_strs = [str(val).lower() for val in df_raw.iloc[idx].values]
            if any(
                "bider" in s
                or "bidder" in s
                or "annual capacity" in s
                or "cappacity" in s
                for s in row_strs
            ):
                detected_header_idx = idx
                break

        df = pd.read_excel(
            xl, sheet_name=selected_sheet, header=detected_header_idx
        )

        # 4. Clean & Deduplicate Column Names
        cols_seen = {}
        unique_cols = []
        for c in df.columns:
            c_str = str(c).strip()
            if c_str in cols_seen:
                cols_seen[c_str] += 1
                unique_cols.append(f"{c_str}_{cols_seen[c_str]}")
            else:
                cols_seen[c_str] = 0
                unique_cols.append(c_str)
        df.columns = unique_cols

        # 5. Locate Bidder Column
        bidder_cols = [
            c
            for c in df.columns
            if "bider" in c.lower() or "bidder" in c.lower()
        ]
        bidder_col = bidder_cols[0] if bidder_cols else df.columns[0]

        # 6. Locate Capacity Column (Prioritizes Remaining Balance Capacity)
        balance_cap_cols = [
            c
            for c in df.columns
            if "balance" in str(c).lower()
            and (
                "capacity" in str(c).lower()
                or "cappacity" in str(c).lower()
            )
        ]

        if balance_cap_cols:
            cap_col = balance_cap_cols[0]
        else:
            ann_cap_cols = [
                c
                for c in df.columns
                if "capacity" in str(c).lower()
                or "cappacity" in str(c).lower()
            ]
            cap_col = ann_cap_cols[0] if ann_cap_cols else df.columns[-1]

        # 7. Clean Bidder Rows
        df_clean = df[df[bidder_col].notna()].copy()
        df_clean = df_clean[
            ~df_clean[bidder_col]
            .astype(str)
            .str.lower()
            .str.contains("note|total|bider|bidder|crosscheck")
        ].copy()

        # 8. Locate Project Columns
        proj_cols = [
            c
            for c in df_clean.columns
            if str(c).startswith("P")
            and len(str(c)) <= 6
            and str(c)[1:].isdigit()
        ]
        if not proj_cols:
            exclude_cols = [bidder_col, cap_col]
            proj_cols = [
                c
                for c in df_clean.columns
                if c not in exclude_cols
                and "unnamed" not in c.lower()
                and not any(
                    k in c.lower()
                    for k in [
                        "grade",
                        "cum",
                        "amount",
                        "balance",
                        "capacity",
                        "cappacity",
                        "crosscheck",
                    ]
                )
            ]

        # 9. Extract Capacities and Valid Bids
        bidders = df_clean[bidder_col].astype(str).str.strip().tolist()
        df_clean["bidder_id"] = bidders
        df_clean = df_clean.set_index("bidder_id")

        capacity_dict = {}
        for b in bidders:
            raw_c = df_clean.loc[b, cap_col]
            if isinstance(raw_c, pd.Series):
                raw_c = (
                    raw_c.dropna().iloc[0]
                    if not raw_c.dropna().empty
                    else None
                )
            parsed_c = pd.to_numeric(raw_c, errors="coerce")
            capacity_dict[b] = (
                float(parsed_c)
                if (pd.notna(parsed_c) and not math.isnan(float(parsed_c)))
                else 0.0
            )

        # 10. Formulate & Solve Optimization Model
        model = pulp.LpProblem("RDA_Assignment", pulp.LpMinimize)
        x = {}
        valid_bids = {}

        for b in bidders:
            for p in proj_cols:
                val = df_clean.loc[b, p]
                if isinstance(val, pd.Series):
                    val = (
                        val.dropna().iloc[0]
                        if not val.dropna().empty
                        else None
                    )

                if pd.notna(val):
                    parsed_val = pd.to_numeric(val, errors="coerce")
                    if pd.notna(parsed_val) and not math.isnan(
                        float(parsed_val)
                    ):
                        amt = float(parsed_val)
                        if amt > 0:
                            safe_b = re.sub(r"[^a-zA-Z0-9_]", "_", b)
                            safe_p = re.sub(r"[^a-zA-Z0-9_]", "_", str(p))
                            var_name = f"x_{safe_b}_{safe_p}"
                            x[(b, p)] = pulp.LpVariable(var_name, cat="Binary")
                            valid_bids[(b, p)] = amt

        unassigned = {
            p: pulp.LpVariable(
                f"unassigned_{re.sub(r'[^a-zA-Z0-9_]', '_', str(p))}",
                cat="Binary",
            )
            for p in proj_cols
        }
        PENALTY = 1e12

        # Objective Function
        model += pulp.lpSum(
            valid_bids[(b, p)] * x[(b, p)] for (b, p) in valid_bids
        ) + pulp.lpSum(PENALTY * unassigned[p] for p in proj_cols)

        # Constraint 1: Assign each project to exactly 1 bidder (or unassigned penalty)
        for p in proj_cols:
            p_bidders = [b for (b, proj) in valid_bids.keys() if proj == p]
            if p_bidders:
                model += (
                    pulp.lpSum(x[(b, p)] for b in p_bidders) + unassigned[p]
                    == 1
                )
            else:
                model += unassigned[p] == 1

        # Constraint 2: Total awarded bid amount per bidder <= Capacity Limit
        for b in bidders:
            b_projects = [
                p for (bidder, p) in valid_bids.keys() if bidder == b
            ]
            if b_projects:
                model += (
                    pulp.lpSum(
                        valid_bids[(b, p)] * x[(b, p)] for p in b_projects
                    )
                    <= capacity_dict[b]
                )

        status = model.solve(pulp.PULP_CBC_CMD(msg=False))

        # 11. Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Bidders", len(bidders))
        with col2:
            st.metric("Number of Projects", len(proj_cols))

        if status == 1:
            st.success("Optimal solution found! 🎉")

        # 12. Compile Results
        assigned_rows = []
        total_cost = 0.0

        for p in proj_cols:
            assigned_b = "UNASSIGNED"
            bid_amt = 0.0
            for b in bidders:
                if (b, p) in x and pulp.value(x[(b, p)]) > 0.5:
                    assigned_b = b
                    bid_amt = valid_bids[(b, p)]
                    total_cost += bid_amt
                    break
            assigned_rows.append(
                {
                    "Project": p,
                    "Assigned Bidder": assigned_b,
                    "Bid Amount": bid_amt,
                }
            )

        st.metric("Minimum Total Cost", f"LKR {total_cost:,.2f}")
        st.subheader("Optimal Project Assignment")
        df_assignment = pd.DataFrame(assigned_rows)
        st.dataframe(df_assignment, use_container_width=True)

        # 13. Raw Sheet Data Viewer (Shows entire raw table of the selected sheet)
        with st.expander("🔍 View Full Extracted Sheet Data"):
            st.dataframe(df_clean, use_container_width=True)

        # 14. Download Button
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_assignment.to_excel(
                writer, index=False, sheet_name="Optimal Assignment"
            )
            workbook = writer.book
            worksheet = writer.sheets["Optimal Assignment"]
            money_format = workbook.add_format({"num_format": "#,##0.00"})
            worksheet.set_column("A:A", 15)
            worksheet.set_column("B:B", 30)
            worksheet.set_column("C:C", 25, money_format)

        output.seek(0)
        st.download_button(
            label="📥 Download Optimization Results",
            data=output,
            file_name=f"{selected_sheet}_Optimization_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Error processing the Excel file: {e}")
else:
    st.info("👆 Please upload an RDA Batch Excel file to start optimization.")