import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR = APP_DIR.parent


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def _load_css() -> None:
    css_path = APP_DIR / "assets" / "style.css"
    css = _read_text(css_path)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _data_paths() -> dict:
    return {
        "milestone1_csv": ROOT_DIR / "Milestone-01" / "milestone-04" / "Airline Dataset Updated - v2.csv",
        "milestone2_csv": ROOT_DIR / "Milestone-02" / "cleaned_flights.csv",
        "milestone3_csv": ROOT_DIR / "Milestone-03" / "FInal_Flights_cleaned_data.csv",
        "milestone1_app": ROOT_DIR / "Milestone-01" / "milestone-04" / "app1.py",
        "milestone2_app": ROOT_DIR / "Milestone-02" / "milestone-04" / "app2.py",
        "milestone3_app": ROOT_DIR / "Milestone-03" / "app3.py",
        "milestone1_readme": ROOT_DIR / "Milestone-01" / "milestone-04" / "README.md",
        "milestone2_readme": ROOT_DIR / "Milestone-02" / "milestone-04" / "README.md",
        "milestone3_readme": ROOT_DIR / "Milestone-03" / "readme.md",
    }


@st.cache_data
def load_m1() -> pd.DataFrame:
    p = _data_paths()["milestone1_csv"]
    df = pd.read_csv(p)
    df["Departure Date"] = pd.to_datetime(df["Departure Date"], format="mixed", errors="coerce")
    df["is_on_time"] = df["Flight Status"].eq("On Time")
    df["is_delayed"] = df["Flight Status"].eq("Delayed")
    df["is_cancelled"] = df["Flight Status"].eq("Cancelled")
    df["Month"] = df["Departure Date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_m2() -> pd.DataFrame:
    p = _data_paths()["milestone2_csv"]
    df = pd.read_csv(p)

    def get_flight_status(row):
        if row.get("CANCELLED", 0) == 1:
            return "Cancelled"
        if row.get("ARR_DELAY", 0) > 0:
            return "Delayed"
        return "On Time"

    df["Flight Status"] = df.apply(get_flight_status, axis=1)
    month_dict = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    df["Month"] = df["MONTH"].map(month_dict)
    day_dict = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }
    df["Day"] = df["DAY_OF_WEEK"].map(day_dict)
    df["Route"] = df["ORIGIN"].astype(str) + " → " + df["DEST"].astype(str)
    return df


@st.cache_data
def load_m3() -> pd.DataFrame:
    p = _data_paths()["milestone3_csv"]
    df = pd.read_csv(p)
    return df


def render_overview() -> None:
    st.markdown(
        """
        <div class="afi-card">
          <h3>Project Statement</h3>
          <div class="afi-muted">
            The objective of this project is to analyze large-scale airline flight data to uncover operational trends,
            delay patterns, and cancellation reasons using data visualization techniques. The goal is to help understand
            airline and airport-level performance and contribute to actionable insights using visual analysis.
          </div>
          <div class="afi-hr"></div>
          <h3>Expected Outcomes</h3>
          <ul class="afi-muted">
            <li>Understand and preprocess aviation datasets for analysis</li>
            <li>Explore trends in flight schedules, delays, cancellations, and routes</li>
            <li>Visualize key metrics using bar charts, time series, heatmaps, maps, and comparisons</li>
            <li>Provide insights for stakeholders including airline operators and analysts</li>
            <li>Summarize findings through a final visual report and presentation</li>
          </ul>
          <div class="afi-hr"></div>
          <h3>Dataset</h3>
          <div class="afi-muted">Source: Kaggle Airlines Flights Dataset</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    cols = st.columns(3)
    paths = _data_paths()
    with cols[0]:
        st.markdown('<div class="afi-badge">Milestone 1 Dataset</div>', unsafe_allow_html=True)
        st.code(str(paths["milestone1_csv"]))
    with cols[1]:
        st.markdown('<div class="afi-badge">Milestone 2 Dataset</div>', unsafe_allow_html=True)
        st.code(str(paths["milestone2_csv"]))
    with cols[2]:
        st.markdown('<div class="afi-badge">Milestone 3 Dataset</div>', unsafe_allow_html=True)
        st.code(str(paths["milestone3_csv"]))


def render_data_acquisition() -> None:
    st.subheader("Data Sources and Understanding")
    st.write("Quick schema + preview for each milestone dataset.")

    tabs = st.tabs(["Milestone 1 (Operations Status)", "Milestone 2 (Delays Dataset)", "Milestone 3 (Routes & Seasonality)"])
    with tabs[0]:
        df = load_m1()
        st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
        st.dataframe(df.head(20), use_container_width=True)
        st.write("Columns:")
        st.code(", ".join(df.columns))
    with tabs[1]:
        df = load_m2()
        st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
        st.dataframe(df.head(20), use_container_width=True)
        st.write("Columns:")
        st.code(", ".join(df.columns))
    with tabs[2]:
        df = load_m3()
        st.write(f"Rows: **{len(df):,}** | Columns: **{df.shape[1]}**")
        st.dataframe(df.head(20), use_container_width=True)
        st.write("Columns:")
        st.code(", ".join(df.columns))


def render_cleaning_preprocessing() -> None:
    st.subheader("Data Cleaning, Preprocessing and Feature Engineering")
    st.markdown(
        """
        - **Milestone 1**: datetime parsing (mixed formats), boolean flags for status, monthly aggregation
        - **Milestone 2**: unified `Flight Status` from `CANCELLED` + `ARR_DELAY`, derived `Month`, `Day`, and `Route`
        - **Milestone 3**: standardized cleaned dataset with `route`, `cancelled`, `month`, `day_of_week`
        """
    )

    st.markdown("Reference documentation already present in your milestones.")
    paths = _data_paths()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Milestone 1 README")
        st.code(str(paths["milestone1_readme"]))
    with col2:
        st.caption("Milestone 2 README")
        st.code(str(paths["milestone2_readme"]))
    with col3:
        st.caption("Milestone 3 README")
        st.code(str(paths["milestone3_readme"]))


def render_univariate_bivariate() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_m2()
    st.subheader("Univariate and Bivariate Analysis (Milestone 2)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Airlines by Number of Flights**")
        top_airlines = df["AIRLINE"].value_counts().head(10)
        fig, ax = plt.subplots()
        top_airlines.plot(kind="bar", ax=ax)
        ax.set_title("Top 10 Airlines by Flight Count")
        ax.set_xlabel("Airline")
        ax.set_ylabel("Number of Flights")
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Most Frequent Routes**")
        top_routes = df["Route"].value_counts().head(10)
        fig, ax = plt.subplots()
        top_routes.plot(kind="bar", ax=ax)
        ax.set_title("Top 10 Most Frequent Routes")
        ax.set_xlabel("Route")
        ax.set_ylabel("Number of Flights")
        plt.xticks(rotation=75)
        st.pyplot(fig, use_container_width=True)

    st.markdown("**Flight Distribution by Month**")
    monthly = df["MONTH"].value_counts().sort_index()
    fig, ax = plt.subplots()
    monthly.plot(kind="line", ax=ax)
    ax.set_title("Flight Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Flights")
    st.pyplot(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Flights by Day of the Week**")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x="Day", data=df, order=day_order, palette="magma", ax=ax)
        ax.set_title("Flights by Day of the Week")
        st.pyplot(fig, use_container_width=True)

    with col4:
        st.markdown("**Flight Distribution by Hour**")
        hour_counts = df["HOUR"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax)
        ax.set_title("Flight Distribution by Hour")
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)


def render_delay_cause_routes_airport() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Delay Cause Analysis, Routes, Airport Level Analysis")

    st.markdown("**Delay Cause Analysis (Milestone 2)**")
    df = load_m2()

    delay_cols = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS"]
    avg_delay = df.groupby("AIRLINE")[delay_cols].mean().reset_index()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = sns.color_palette("tab20", len(avg_delay))
    for i in range(len(avg_delay)):
        ax.scatter(
            avg_delay["DELAY_DUE_CARRIER"][i],
            avg_delay["DELAY_DUE_WEATHER"][i],
            s=avg_delay["DELAY_DUE_NAS"][i] * 20,
            color=colors[i],
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=avg_delay["AIRLINE"][i],
        )
    ax.set_title("Airline Delay Comparison (Bubble Size = NAS Delay)", fontsize=16, weight="bold")
    ax.set_xlabel("Average Carrier Delay (Minutes)", fontsize=12)
    ax.set_ylabel("Average Weather Delay (Minutes)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Percentage Contribution of Delay Causes**")
        delay_causes_cols = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]
        total_delay = df[delay_causes_cols].sum()
        colors = ["#FF6B6B", "#4ECDC4", "#556270", "#FFD166"]
        labels = ["Carrier", "Weather", "NAS", "Late Aircraft"]
        fig, ax = plt.subplots()
        ax.pie(total_delay, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, shadow=True)
        ax.set_title("Percentage Contribution of Delay Causes", fontsize=14, weight="bold")
        st.pyplot(fig, use_container_width=True)

    with col6:
        st.markdown("**Average Departure Delay by Hour**")
        hour_delay = df.groupby("HOUR")["DEP_DELAY"].mean()
        fig, ax = plt.subplots()
        sns.lineplot(x=hour_delay.index, y=hour_delay.values, marker="o", color="#2E86C1", ax=ax)
        ax.set_title("Average Departure Delay by Hour", fontsize=14, weight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Delay (Minutes)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig, use_container_width=True)

    st.markdown("**Top 10 Delay-Prone Airports (Milestone 2)**")
    airport_delay = df.groupby("ORIGIN")["DEP_DELAY"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=airport_delay.index, y=airport_delay.values, palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Top 10 Delay-Prone Airports", fontsize=14, weight="bold")
    ax.set_xlabel("Airport")
    ax.set_ylabel("Average Departure Delay (Minutes)")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Route + Airport Performance (Milestone 1)**")
    df1 = load_m1()
    route_perf = (
        df1.groupby(["Airport Name", "Arrival Airport"])
        .agg(total_flights=("Flight Status", "count"), on_time_flights=("is_on_time", "sum"), cancelled_flights=("is_cancelled", "sum"))
        .reset_index()
    )
    route_perf["OTP"] = (route_perf["on_time_flights"] / route_perf["total_flights"]) * 100
    route_perf["Cancellation Rate"] = (route_perf["cancelled_flights"] / route_perf["total_flights"]) * 100

    colr1, colr2 = st.columns(2)
    with colr1:
        top_otp_routes = route_perf.sort_values(by="OTP", ascending=False).head(10)
        st.dataframe(
            top_otp_routes[["Airport Name", "Arrival Airport", "total_flights", "OTP"]].reset_index(drop=True),
            use_container_width=True,
        )
    with colr2:
        top_cancel_routes = route_perf.sort_values(by="Cancellation Rate", ascending=False).head(10)
        st.dataframe(
            top_cancel_routes[["Airport Name", "Arrival Airport", "total_flights", "Cancellation Rate"]].reset_index(drop=True),
            use_container_width=True,
        )


def render_cancellation_seasonal() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Cancellation and Seasonal Trends (Milestone 3)")
    df = load_m3()

    cancel_by_month = df.groupby("month")["cancelled"].agg(total_cancelled="sum", total_flights="count").reset_index()
    cancel_by_month["cancel_rate"] = cancel_by_month["total_cancelled"] / cancel_by_month["total_flights"] * 100

    # Chart 12 – Cancellation Rate by Month
    st.markdown("#### Chart 12 – Cancellation Rate by Month (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cancel_by_month["month"], cancel_by_month["cancel_rate"],
            marker="o", linewidth=2, color="#E91E63", markersize=8)
    ax.set_title("Cancellation Rate by Month (%)", fontsize=14, weight="bold")
    ax.set_xlabel("Month (1=Jan … 12=Dec)"); ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 13 – Total Cancellations by Month
    st.markdown("#### Chart 13 – Total Cancellations by Month")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cancel_by_month["month"], cancel_by_month["total_cancelled"],
           color="#9C27B0", edgecolor="black")
    ax.set_title("Total Cancellations by Month", fontsize=14, weight="bold")
    ax.set_xlabel("Month (1=Jan … 12=Dec)"); ax.set_ylabel("Total Cancelled Flights")
    ax.set_xticks(range(1, 13))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 14 – Top 10 Routes with Most Cancellations
    st.markdown("#### Chart 14 – Top 10 Routes with the Most Cancellations")
    worst_routes = (
        df[df["cancelled"] == 1]
        .groupby("route").size().reset_index(name="cancel_count")
        .sort_values("cancel_count", ascending=False).head(10)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(worst_routes["route"], worst_routes["cancel_count"], color="#FF5722", edgecolor="black")
    ax.set_title("Top 10 Routes by Cancellation Count", fontsize=14, weight="bold")
    ax.set_xlabel("Route"); ax.set_ylabel("Number of Cancellations")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 15 – Cancellation Rate by Day of Week
    st.markdown("#### Chart 15 – Cancellation Rate by Day of Week")
    cancel_by_day = df.groupby("day_of_week")["cancelled"].agg(total_cancelled="sum", total_flights="count").reset_index()
    cancel_by_day["cancel_rate"] = cancel_by_day["total_cancelled"] / cancel_by_day["total_flights"] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cancel_by_day["day_of_week"].astype(str), cancel_by_day["cancel_rate"],
           color="#3F51B5", edgecolor="black")
    ax.set_title("Cancellation Rate by Day of Week (%)", fontsize=14, weight="bold")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Cancellation Rate (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 16 – Winter vs Overall Cancellation Rate
    st.markdown("#### Chart 16 – Winter vs Overall Cancellation Rate")
    winter = df[df["month"].isin([12, 1, 2])]
    winter_rate  = winter["cancelled"].mean() * 100
    overall_rate = df["cancelled"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Winter Months\n(Dec, Jan, Feb)", "Overall Average"],
                  [winter_rate, overall_rate],
                  color=["#1565C0", "#90CAF9"], edgecolor="black", width=0.4)
    ax.set_title("Winter vs Overall Cancellation Rate", fontsize=14, weight="bold")
    ax.set_ylabel("Cancellation Rate (%)")
    for bar, val in zip(bars, [winter_rate, overall_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", fontweight="bold", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_airport_route_exploration() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Airport and Route-Level Exploration (Milestone 3)")
    df = load_m3()

    # Chart 17 – Top 10 Origin-Destination Pairs
    st.markdown("#### Chart 17 – Top 10 Origin-Destination Pairs")
    od = df.groupby(["origin", "dest"]).size().reset_index(name="count").sort_values("count", ascending=False).head(10)
    od["od_pair"] = od["origin"] + " → " + od["dest"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_od = sns.color_palette("husl", len(od))
    ax.barh(od["od_pair"], od["count"], color=colors_od)
    ax.invert_yaxis()
    ax.set_title("Top 10 Origin-Destination Pairs by Flight Volume", fontsize=13, weight="bold")
    ax.set_xlabel("Number of Flights")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 18 – Top 10 Busiest Airports
    st.markdown("#### Chart 18 – Top 10 Busiest Airports")
    busy = df["origin"].value_counts().reset_index()
    busy.columns = ["Airport", "Flight Count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(busy.head(10)["Airport"], busy.head(10)["Flight Count"],
           color=sns.color_palette("Set2", 10), edgecolor="black")
    ax.set_title("Top 10 Busiest Airports by Departure Volume", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Number of Departing Flights")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 19 – Top 10 Airports by Average Arrival Delay
    st.markdown("#### Chart 19 – Top 10 Airports by Average Arrival Delay")
    avg_delay = df.groupby("origin")["arr_delay"].mean().reset_index().sort_values("arr_delay", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(avg_delay["origin"], avg_delay["arr_delay"],
           color=sns.color_palette("Reds_r", 10), edgecolor="black")
    ax.set_title("Top 10 Airports by Average Arrival Delay", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Average Arrival Delay (Minutes)")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_application() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader("Application (Final): Executive KPI + Actionable Insights (Professional View)")
    st.caption("This section synthesizes the full workflow into one decision-ready view using the Milestone 2 dataset.")

    df = load_m2()
    df = df.copy()

    airlines = ["All"] + sorted(df["AIRLINE"].dropna().unique().tolist())
    origins  = ["All"] + sorted(df["ORIGIN"].dropna().unique().tolist())
    months   = ["All"] + sorted(df["Month"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        airline = st.selectbox("Airline", airlines, index=0)
    with c2:
        origin = st.selectbox("Origin Airport", origins, index=0)
    with c3:
        month = st.selectbox("Month", months, index=0)

    if airline != "All":
        df = df[df["AIRLINE"] == airline]
    if origin != "All":
        df = df[df["ORIGIN"] == origin]
    if month != "All":
        df = df[df["Month"] == month]

    total         = len(df)
    cancel_rate   = float(df["CANCELLED"].mean()) if total else 0.0
    otp           = float(((df["CANCELLED"] == 0) & (df["ARR_DELAY"].fillna(0) <= 15)).mean()) if total else 0.0
    avg_arr_delay = float(df.loc[df["CANCELLED"] == 0, "ARR_DELAY"].mean()) if total else 0.0

    # Chart 20 – Executive KPI Summary
    st.markdown("#### Chart 20 – Executive KPI Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Flights", f"{total:,}")
    k2.metric("On-time Performance (≤15 min)", f"{otp*100:.2f}%")
    k3.metric("Cancellation Rate", f"{cancel_rate*100:.2f}%")
    k4.metric("Avg Arrival Delay (min)", f"{avg_arr_delay:.2f}")

    kpi_labels = ["Total Flights\n(thousands)", "OTP %\n(≤15 min)", "Cancellation\nRate %", "Avg Arrival\nDelay (min)"]
    kpi_values = [total / 1000, otp * 100, cancel_rate * 100, avg_arr_delay]
    kpi_colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(kpi_labels, kpi_values, color=kpi_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, kpi_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Executive KPI Summary – Full Dataset (Milestone 2)", fontsize=13, weight="bold")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 21 – Delay Cause Mix
    st.markdown("#### Chart 21 – Delay Cause Mix (Share of Total Delay Minutes)")
    delay_causes_cols = ["DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS", "DELAY_DUE_LATE_AIRCRAFT"]
    delay_mix = df[delay_causes_cols].sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(delay_mix.values, labels=["Carrier", "Weather", "NAS", "Late Aircraft"],
           autopct="%1.1f%%", startangle=140,
           colors=["#FF6B6B", "#4ECDC4", "#556270", "#FFD166"])
    ax.set_title("Delay Cause Contribution – Share of Total Delay Minutes", fontsize=13, weight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 22 – Top 15 Delay Hotspot Airports
    st.markdown("#### Chart 22 – Top 15 Delay Hotspot Origin Airports")
    hotspot = df.groupby("ORIGIN")["DEP_DELAY"].mean().sort_values(ascending=False).head(15).reset_index()
    hotspot.columns = ["Origin", "Avg Departure Delay (min)"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hotspot["Origin"], hotspot["Avg Departure Delay (min)"],
           color=sns.color_palette("YlOrRd", 15), edgecolor="black")
    ax.set_title("Top 15 Delay Hotspot Origin Airports", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Avg Departure Delay (min)")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Chart 23 – Top 15 Routes by Flight Volume
    st.markdown("#### Chart 23 – Top 15 Routes by Flight Volume")
    routes = df["Route"].value_counts().head(15).reset_index()
    routes.columns = ["Route", "Flights"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(routes["Route"], routes["Flights"],
           color=sns.color_palette("Blues_r", 15), edgecolor="black")
    ax.set_title("Top 15 Routes by Flight Volume (Route Demand)", fontsize=13, weight="bold")
    ax.set_xlabel("Route (Origin → Destination)"); ax.set_ylabel("Number of Flights")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_documentation_presentation() -> None:
    st.subheader("Documentation and Presentation")
    st.markdown(
        """
        - Your milestone documentation already exists as `README.md` and `presentation_slides.md`
        - This final dashboard also includes a generator to create **one combined Milestone 1–3 Word report**
        """
    )

    st.markdown("**Where the existing docs live**")
    p = _data_paths()
    st.code(str(p["milestone1_readme"]))
    st.code(str(p["milestone2_readme"]))
    st.code(str(p["milestone3_readme"]))

    st.markdown("**Generate the combined Word report**")
    st.code('python "generate_final_report.py"')

    out_docx = APP_DIR / "reports_output" / "AirFly_Insights_Milestones_1_2_3_Final_Report.docx"
    if out_docx.exists():
        st.success("Combined Word report found. You can download it below.")
        data = out_docx.read_bytes()
        st.download_button(
            label="Download Final Report (docx)",
            data=data,
            file_name=out_docx.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        st.info("Report not generated yet. Run the command above once, then refresh this page.")


def main() -> None:
    st.set_page_config(
        page_title="AirFly Insights: Data Visualization and Analysis of Airline Operations",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()

    st.markdown(
        """
        <div class="afi-hero">
          <div class="afi-kicker">AirFly Insights</div>
          <div class="afi-title">Data Visualization and Analysis of Airline Operations</div>
          <div class="afi-caption">Final Dashboard (Milestones 1–3 consolidated)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        "1. Overview",
        "2. Data Acquisition and Understanding",
        "3. Data Cleaning , Preprocessing and Feature Engineering",
        "4. Univariate and Bivariate Analysis",
        "5. Delay Cause Analysis , Routes ,Airport level Analysis",
        "6. Cancellation and Seasonal Trends",
        "7. Airport and Route-Level Exploration",
        "8. Application of this entire project",
        "9. Documentation and Presentation",
    ]

    st.sidebar.markdown('<div class="afi-sidebar-title"><span class="afi-sidebar-glow">Final Dashboard</span></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="afi-sidebar-subtitle">AirFly Insights • Professional Storyline</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="afi-badge">Navigation</div>', unsafe_allow_html=True)
    choice = st.sidebar.radio("Go to", nav_items)

    if choice.startswith("1."):
        render_overview()
    elif choice.startswith("2."):
        render_data_acquisition()
    elif choice.startswith("3."):
        render_cleaning_preprocessing()
    elif choice.startswith("4."):
        render_univariate_bivariate()
    elif choice.startswith("5."):
        render_delay_cause_routes_airport()
    elif choice.startswith("6."):
        render_cancellation_seasonal()
    elif choice.startswith("7."):
        render_airport_route_exploration()
    elif choice.startswith("8."):
        render_application()
    elif choice.startswith("9."):
        render_documentation_presentation()


if __name__ == "__main__":
    main()

