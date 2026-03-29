"""
AirFly Insights – Complete Final Report Generator
Generates a professional Word (.docx) report covering every section of the
Final Dashboard (Milestones 1-3) with all charts embedded and detailed
explanations written below each chart.

Run:
    python generate_final_report.py
Output:
    reports_output/AirFly_Insights_Final_Dashboard_Report.docx
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

APP_DIR   = Path(__file__).parent.resolve()
ROOT_DIR  = APP_DIR.parent
OUT_DIR   = APP_DIR / "reports_output"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# ── dataset paths (same as app.py) ──────────────────────────────────────────
def dpaths():
    return {
        "m1": ROOT_DIR / "Milestone-01" / "milestone-04" / "Airline Dataset Updated - v2.csv",
        "m2": ROOT_DIR / "Milestone-02" / "cleaned_flights.csv",
        "m3": ROOT_DIR / "Milestone-03" / "FInal_Flights_cleaned_data.csv",
    }

def savefig(path: Path, fig):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)

# ── loaders ──────────────────────────────────────────────────────────────────
def load_m1():
    df = pd.read_csv(dpaths()["m1"])
    df["Departure Date"] = pd.to_datetime(df["Departure Date"], format="mixed", errors="coerce")
    df["is_on_time"]   = df["Flight Status"].eq("On Time")
    df["is_delayed"]   = df["Flight Status"].eq("Delayed")
    df["is_cancelled"] = df["Flight Status"].eq("Cancelled")
    df["Month"] = df["Departure Date"].dt.to_period("M").astype(str)
    return df

def load_m2():
    df = pd.read_csv(dpaths()["m2"])
    def status(r):
        if r.get("CANCELLED", 0) == 1: return "Cancelled"
        if r.get("ARR_DELAY",  0) > 0: return "Delayed"
        return "On Time"
    df["Flight Status"] = df.apply(status, axis=1)
    mmap = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
            7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    dmap = {1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday",7:"Sunday"}
    df["Month"] = df["MONTH"].map(mmap)
    df["Day"]   = df["DAY_OF_WEEK"].map(dmap)
    df["Route"] = df["ORIGIN"].astype(str) + " → " + df["DEST"].astype(str)
    return df

def load_m3():
    return pd.read_csv(dpaths()["m3"])

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 – UNIVARIATE & BIVARIATE  (Milestone 2)
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_univariate(df2) -> list[tuple[str,str,str]]:
    out = []

    # 4-A  Top 10 Airlines by Flight Count
    top_airlines = df2["AIRLINE"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_airlines.plot(kind="bar", ax=ax, color="#2196F3", edgecolor="black")
    ax.set_title("Top 10 Airlines by Number of Flights", fontsize=14, weight="bold")
    ax.set_xlabel("Airline Code"); ax.set_ylabel("Number of Flights")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S4_01_top_airlines.png", fig)
    out.append((p, "Chart 1 – Top 10 Airlines by Number of Flights",
        "What this chart shows:\n"
        "This bar chart ranks the top 10 airlines by the total number of flights they operated "
        "in the dataset. Each bar on the X-axis represents an airline identified by its IATA code, "
        "and the height of the bar (Y-axis) shows how many flights that airline flew.\n\n"
        "Key Insights:\n"
        "• The tallest bar identifies the airline with the highest flight volume — the dominant "
        "carrier in this dataset.\n"
        "• Airlines with high flight counts have a larger influence on overall network performance "
        "metrics such as average delay and cancellation rate.\n"
        "• Comparing bar heights reveals the market share distribution among carriers.\n"
        "• This chart is the starting point for understanding which airlines drive the most "
        "operational activity and should be prioritized in performance improvement efforts."))

    # 4-B  Top 10 Most Frequent Routes
    top_routes = df2["Route"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_routes.plot(kind="bar", ax=ax, color="#FF9800", edgecolor="black")
    ax.set_title("Top 10 Most Frequent Routes", fontsize=14, weight="bold")
    ax.set_xlabel("Route (Origin → Destination)"); ax.set_ylabel("Number of Flights")
    plt.xticks(rotation=75, ha="right")
    p = savefig(PLOTS_DIR / "S4_02_top_routes.png", fig)
    out.append((p, "Chart 2 – Top 10 Most Frequent Routes",
        "What this chart shows:\n"
        "This bar chart displays the 10 most frequently operated origin-to-destination routes. "
        "Each bar represents a specific route (e.g., ATL → LAX) and its height shows how many "
        "times that route was flown.\n\n"
        "Key Insights:\n"
        "• High-frequency routes are the backbone of the airline network and generate the most revenue.\n"
        "• Any delay or disruption on these routes has a cascading effect on the entire schedule.\n"
        "• These routes are candidates for capacity upgrades (larger aircraft, more frequencies).\n"
        "• Comparing route frequencies helps identify geographic demand corridors — e.g., "
        "coast-to-coast or hub-to-hub routes that dominate traffic."))

    # 4-C  Flight Distribution by Month
    monthly = df2["MONTH"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(kind="line", ax=ax, marker="o", color="#9C27B0", linewidth=2)
    ax.set_title("Flight Distribution by Month", fontsize=14, weight="bold")
    ax.set_xlabel("Month (1=Jan … 12=Dec)"); ax.set_ylabel("Number of Flights")
    ax.set_xticks(range(1, 13))
    ax.grid(True, linestyle="--", alpha=0.5)
    p = savefig(PLOTS_DIR / "S4_03_monthly_dist.png", fig)
    out.append((p, "Chart 3 – Flight Distribution by Month",
        "What this chart shows:\n"
        "This line chart plots the total number of flights for each calendar month (1 = January "
        "through 12 = December). Each data point represents one month's total flight count, "
        "connected by a line to show the trend over the year.\n\n"
        "Key Insights:\n"
        "• Peaks in the line indicate high-demand travel months (typically summer: June–August, "
        "and holiday periods: November–December).\n"
        "• Troughs indicate low-demand months where airlines may reduce capacity.\n"
        "• Airlines use this pattern for seasonal capacity planning — adding flights in peak months "
        "and reducing in off-peak months.\n"
        "• Sudden dips may also indicate external disruptions such as weather events or economic factors."))

    # 4-D  Flights by Day of Week
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x="Day", data=df2, order=day_order, palette="magma", ax=ax)
    ax.set_title("Flights by Day of the Week", fontsize=14, weight="bold")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Number of Flights")
    p = savefig(PLOTS_DIR / "S4_04_day_of_week.png", fig)
    out.append((p, "Chart 4 – Flights by Day of the Week",
        "What this chart shows:\n"
        "This count plot shows how flight volume is distributed across the seven days of the week. "
        "Each bar represents one day, and the height shows the total number of flights on that day.\n\n"
        "Key Insights:\n"
        "• Weekdays (Monday–Friday) typically show higher flight volumes driven by business travel.\n"
        "• Weekends may show different patterns — Saturday often dips while Sunday can spike due "
        "to leisure travelers returning home.\n"
        "• Understanding daily demand helps airlines optimize gate assignments, crew scheduling, "
        "and ground handling resources.\n"
        "• Days with unusually high volumes are more prone to delays due to congestion."))

    # 4-E  Flight Distribution by Hour
    hour_counts = df2["HOUR"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax, color="#00BCD4")
    ax.set_title("Flight Distribution by Hour of Day", fontsize=14, weight="bold")
    ax.set_xlabel("Hour of Day (0–23)"); ax.set_ylabel("Number of Flights")
    plt.xticks(rotation=45)
    p = savefig(PLOTS_DIR / "S4_05_hourly_dist.png", fig)
    out.append((p, "Chart 5 – Flight Distribution by Hour of Day",
        "What this chart shows:\n"
        "This bar chart reveals how departures are spread across the 24 hours of the day. "
        "The X-axis shows each hour (0 = midnight, 12 = noon, 23 = 11 PM) and the Y-axis "
        "shows the number of flights departing in that hour.\n\n"
        "Key Insights:\n"
        "• Early morning hours (6–9 AM) typically see the highest departure volumes as airlines "
        "start their daily operations with freshly positioned aircraft.\n"
        "• Mid-day and afternoon hours maintain steady traffic.\n"
        "• Late-night hours (after 10 PM) have very few departures.\n"
        "• This pattern is critical for airport resource planning — security staffing, gate "
        "availability, and ground crew scheduling all depend on peak departure hours."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 – DELAY CAUSE ANALYSIS, ROUTES & AIRPORT LEVEL  (M1 + M2)
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_delay(df1, df2) -> list[tuple[str,str,str]]:
    out = []

    # 5-A  Airline Delay Bubble Chart
    delay_cols = ["DELAY_DUE_CARRIER","DELAY_DUE_WEATHER","DELAY_DUE_NAS"]
    avg_d = df2.groupby("AIRLINE")[delay_cols].mean().reset_index()
    colors_b = sns.color_palette("tab20", len(avg_d))
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, row in avg_d.iterrows():
        ax.scatter(row["DELAY_DUE_CARRIER"], row["DELAY_DUE_WEATHER"],
                   s=max(row["DELAY_DUE_NAS"] * 20, 30),
                   color=colors_b[i % len(colors_b)],
                   alpha=0.8, edgecolors="black", linewidth=1, label=row["AIRLINE"])
    ax.set_title("Airline Delay Comparison (Bubble Size = NAS Delay)", fontsize=14, weight="bold")
    ax.set_xlabel("Average Carrier Delay (Minutes)")
    ax.set_ylabel("Average Weather Delay (Minutes)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    p = savefig(PLOTS_DIR / "S5_01_airline_bubble.png", fig)
    out.append((p, "Chart 6 – Airline Delay Comparison Bubble Chart",
        "What this chart shows:\n"
        "This bubble chart compares all airlines across three delay dimensions simultaneously. "
        "The X-axis shows each airline's average Carrier Delay (minutes caused by the airline itself — "
        "maintenance, crew, fueling). The Y-axis shows average Weather Delay. The size of each bubble "
        "represents the average NAS (National Airspace System) Delay — congestion in air traffic control.\n\n"
        "Key Insights:\n"
        "• Airlines positioned in the upper-right corner with large bubbles are the most delay-affected "
        "across all three categories.\n"
        "• Airlines near the origin (0,0) with small bubbles are the best performers.\n"
        "• A large bubble but low X/Y position means the airline's main problem is airspace congestion, "
        "not internal operations.\n"
        "• This multi-dimensional view helps identify which airlines need improvement in which specific "
        "delay category — carrier delays are within airline control, while weather and NAS are external."))

    # 5-B  Delay Causes Pie
    dcols = ["DELAY_DUE_CARRIER","DELAY_DUE_WEATHER","DELAY_DUE_NAS","DELAY_DUE_LATE_AIRCRAFT"]
    total_d = df2[dcols].sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(total_d.values, labels=["Carrier","Weather","NAS","Late Aircraft"],
           autopct="%1.1f%%", colors=["#FF6B6B","#4ECDC4","#556270","#FFD166"],
           startangle=140, shadow=True)
    ax.set_title("Percentage Contribution of Delay Causes", fontsize=14, weight="bold")
    p = savefig(PLOTS_DIR / "S5_02_delay_pie.png", fig)
    out.append((p, "Chart 7 – Percentage Contribution of Delay Causes",
        "What this chart shows:\n"
        "This pie chart breaks down the total accumulated delay minutes across the entire dataset "
        "into four cause categories:\n"
        "  • Carrier Delay – caused by the airline (maintenance, crew issues, aircraft cleaning)\n"
        "  • Weather Delay – caused by meteorological conditions\n"
        "  • NAS Delay – caused by National Airspace System (air traffic control, airport operations)\n"
        "  • Late Aircraft Delay – caused by a previous flight arriving late, delaying the next departure\n\n"
        "Key Insights:\n"
        "• The largest slice is the primary driver of delays in the network.\n"
        "• Carrier and Late Aircraft delays are within airline control — these are actionable.\n"
        "• Weather and NAS delays are external — airlines can only mitigate these through better "
        "scheduling buffers and contingency planning.\n"
        "• This breakdown is essential for setting realistic improvement targets."))

    # 5-C  Avg Departure Delay by Hour
    hour_delay = df2.groupby("HOUR")["DEP_DELAY"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=hour_delay.index, y=hour_delay.values, marker="o", color="#2E86C1", ax=ax)
    ax.set_title("Average Departure Delay by Hour of Day", fontsize=14, weight="bold")
    ax.set_xlabel("Hour of Day (0–23)"); ax.set_ylabel("Average Delay (Minutes)")
    ax.grid(True, linestyle="--", alpha=0.6)
    p = savefig(PLOTS_DIR / "S5_03_delay_by_hour.png", fig)
    out.append((p, "Chart 8 – Average Departure Delay by Hour of Day",
        "What this chart shows:\n"
        "This line chart shows how the average departure delay changes throughout the 24 hours of "
        "the day. Each point on the line represents the mean delay for all flights departing in "
        "that specific hour.\n\n"
        "Key Insights:\n"
        "• Early morning flights (5–7 AM) typically have the lowest delays because aircraft are "
        "freshly positioned overnight and there is minimal congestion.\n"
        "• Delays accumulate as the day progresses — a phenomenon known as 'delay propagation' "
        "or 'schedule creep'.\n"
        "• Evening flights (after 6 PM) often show the highest average delays due to cascading "
        "disruptions from earlier in the day.\n"
        "• This insight supports scheduling strategies that build buffer time into afternoon and "
        "evening slots to absorb propagated delays."))

    # 5-D  Top 10 Delay-Prone Airports
    airport_delay = df2.groupby("ORIGIN")["DEP_DELAY"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=airport_delay.index, y=airport_delay.values, palette="viridis", ax=ax)
    ax.set_title("Top 10 Delay-Prone Airports (Avg Departure Delay)", fontsize=14, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Avg Departure Delay (Minutes)")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S5_04_delay_airports.png", fig)
    out.append((p, "Chart 9 – Top 10 Delay-Prone Airports",
        "What this chart shows:\n"
        "This bar chart identifies the 10 origin airports with the highest average departure delays. "
        "Each bar represents an airport (by IATA code) and its height shows the average number of "
        "minutes flights depart late from that airport.\n\n"
        "Key Insights:\n"
        "• Airports at the top of this list are operational bottlenecks in the network.\n"
        "• High delays at an airport may be caused by: runway congestion, limited gate capacity, "
        "high traffic volume, local weather patterns, or inefficient ground operations.\n"
        "• Airlines operating heavily from these airports should build extra buffer time into "
        "their schedules.\n"
        "• Airport authorities can use this data to prioritize infrastructure investments and "
        "operational process improvements."))

    # 5-E  Route OTP Table (top 10 best routes – M1)
    rp = (df1.groupby(["Airport Name","Arrival Airport"])
          .agg(total=("Flight Status","count"), on_time=("is_on_time","sum"), cancelled=("is_cancelled","sum"))
          .reset_index())
    rp["OTP %"]    = (rp["on_time"]    / rp["total"]) * 100
    rp["Cancel %"] = (rp["cancelled"]  / rp["total"]) * 100
    top_otp = rp.sort_values("OTP %", ascending=False).head(10)
    top_otp["Route"] = top_otp["Airport Name"].astype(str) + " → " + top_otp["Arrival Airport"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_otp["Route"], top_otp["OTP %"], color="#4CAF50")
    ax.invert_yaxis()
    ax.set_title("Top 10 Routes by On-Time Performance (Milestone 1)", fontsize=13, weight="bold")
    ax.set_xlabel("On-Time Performance (%)")
    p = savefig(PLOTS_DIR / "S5_05_top_otp_routes.png", fig)
    out.append((p, "Chart 10 – Top 10 Routes by On-Time Performance (Milestone 1)",
        "What this chart shows:\n"
        "This horizontal bar chart ranks the top 10 flight routes by their On-Time Performance (OTP) "
        "percentage, derived from the Milestone 1 dataset. Each bar represents a specific "
        "origin-to-destination route, and the length of the bar shows what percentage of flights "
        "on that route arrived on time.\n\n"
        "Key Insights:\n"
        "• Routes with bars approaching 100% are the most reliable in the network.\n"
        "• These best-performing routes can serve as benchmarks — studying what makes them reliable "
        "(shorter distance, less congested airports, favorable weather) can inform improvements "
        "on underperforming routes.\n"
        "• High OTP routes are valuable for marketing as premium reliable services."))

    # 5-F  Top 10 Routes by Cancellation Rate (M1)
    worst = rp.sort_values("Cancel %", ascending=False).head(10)
    worst["Route"] = worst["Airport Name"].astype(str) + " → " + worst["Arrival Airport"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(worst["Route"], worst["Cancel %"], color="#F44336")
    ax.invert_yaxis()
    ax.set_title("Top 10 Routes by Cancellation Rate (Milestone 1)", fontsize=13, weight="bold")
    ax.set_xlabel("Cancellation Rate (%)")
    p = savefig(PLOTS_DIR / "S5_06_top_cancel_routes.png", fig)
    out.append((p, "Chart 11 – Top 10 Routes by Cancellation Rate (Milestone 1)",
        "What this chart shows:\n"
        "This horizontal bar chart highlights the 10 routes with the highest cancellation rates "
        "from the Milestone 1 dataset. Each bar represents a route and its length shows the "
        "percentage of scheduled flights that were cancelled.\n\n"
        "Key Insights:\n"
        "• Routes with long red bars are the most problematic in the network.\n"
        "• High cancellation rates on specific routes may be caused by: geographic weather patterns "
        "(e.g., routes through storm-prone regions), low demand making routes economically unviable "
        "to operate, or infrastructure limitations at one of the airports.\n"
        "• Airlines and airport managers can use this chart to prioritize operational interventions "
        "and decide whether to restructure or discontinue certain routes."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 – CANCELLATION & SEASONAL TRENDS  (Milestone 3)
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_cancellation(df3) -> list[tuple[str,str,str]]:
    out = []

    cancel_by_month = (df3.groupby("month")["cancelled"]
                       .agg(total_cancelled="sum", total_flights="count").reset_index())
    cancel_by_month["cancel_rate"] = cancel_by_month["total_cancelled"] / cancel_by_month["total_flights"] * 100

    # 6-A  Cancellation Rate by Month (line)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cancel_by_month["month"], cancel_by_month["cancel_rate"],
            marker="o", linewidth=2, color="#E91E63", markersize=8)
    ax.set_title("Cancellation Rate by Month (%)", fontsize=14, weight="bold")
    ax.set_xlabel("Month (1=Jan … 12=Dec)"); ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xticks(range(1, 13)); ax.grid(True, linestyle="--", alpha=0.5)
    p = savefig(PLOTS_DIR / "S6_01_cancel_rate_month.png", fig)
    out.append((p, "Chart 12 – Cancellation Rate by Month",
        "What this chart shows:\n"
        "This line chart plots the monthly cancellation rate (percentage of scheduled flights that "
        "were cancelled) across all 12 months of the year. Each data point represents one month, "
        "and the line connects them to show the seasonal trend.\n\n"
        "Key Insights:\n"
        "• Peaks in the line (months with the highest cancellation rates) typically correspond to "
        "winter months (December, January, February) when severe weather — snowstorms, ice, and "
        "low visibility — forces airlines to cancel flights for safety.\n"
        "• Summer months generally show lower cancellation rates despite high traffic volumes.\n"
        "• Airlines can use this chart to pre-position contingency resources (spare aircraft, "
        "standby crews) during high-cancellation months.\n"
        "• Passengers can use this insight to choose more reliable travel months."))

    # 6-B  Total Cancellations by Month (bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cancel_by_month["month"], cancel_by_month["total_cancelled"],
           color="#9C27B0", edgecolor="black")
    ax.set_title("Total Cancellations by Month", fontsize=14, weight="bold")
    ax.set_xlabel("Month (1=Jan … 12=Dec)"); ax.set_ylabel("Total Cancelled Flights")
    ax.set_xticks(range(1, 13))
    p = savefig(PLOTS_DIR / "S6_02_total_cancel_month.png", fig)
    out.append((p, "Chart 13 – Total Cancellations by Month",
        "What this chart shows:\n"
        "This bar chart shows the absolute number of cancelled flights for each month. Unlike the "
        "cancellation rate chart (which shows percentage), this chart shows raw volume — how many "
        "flights were actually cancelled in each month.\n\n"
        "Key Insights:\n"
        "• Months with tall bars had the most cancellations in absolute terms.\n"
        "• A month can have a high cancellation rate but low absolute count if total flights are "
        "also low — this chart separates that distinction.\n"
        "• High absolute cancellation months create the most passenger disruption and require the "
        "most rebooking resources.\n"
        "• Comparing this chart with the rate chart reveals whether cancellations are driven by "
        "volume (more flights = more cancellations) or by genuine operational problems."))

    # 6-C  Top 10 Routes by Cancellation Count
    worst_routes = (df3[df3["cancelled"] == 1]
                    .groupby("route").size().reset_index(name="cancel_count")
                    .sort_values("cancel_count", ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(worst_routes["route"], worst_routes["cancel_count"], color="#FF5722", edgecolor="black")
    ax.set_title("Top 10 Routes with the Most Cancellations", fontsize=14, weight="bold")
    ax.set_xlabel("Route"); ax.set_ylabel("Number of Cancellations")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S6_03_top_cancel_routes.png", fig)
    out.append((p, "Chart 14 – Top 10 Routes with the Most Cancellations",
        "What this chart shows:\n"
        "This bar chart identifies the 10 specific routes (origin-destination pairs) that experienced "
        "the highest number of flight cancellations. Each bar represents a route and its height "
        "shows the total count of cancelled flights on that route.\n\n"
        "Key Insights:\n"
        "• Routes with the most cancellations are the biggest sources of passenger disruption.\n"
        "• These routes may pass through weather-prone regions, or connect airports with "
        "infrastructure or capacity constraints.\n"
        "• Airlines should investigate whether these cancellations are preventable (operational "
        "issues) or unavoidable (weather/external).\n"
        "• Passengers frequently travelling these routes should consider travel insurance and "
        "flexible booking options."))

    # 6-D  Cancellation Rate by Day of Week
    cancel_by_day = (df3.groupby("day_of_week")["cancelled"]
                     .agg(total_cancelled="sum", total_flights="count").reset_index())
    cancel_by_day["cancel_rate"] = cancel_by_day["total_cancelled"] / cancel_by_day["total_flights"] * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cancel_by_day["day_of_week"].astype(str), cancel_by_day["cancel_rate"],
           color="#3F51B5", edgecolor="black")
    ax.set_title("Cancellation Rate by Day of Week (%)", fontsize=14, weight="bold")
    ax.set_xlabel("Day of Week"); ax.set_ylabel("Cancellation Rate (%)")
    p = savefig(PLOTS_DIR / "S6_04_cancel_by_day.png", fig)
    out.append((p, "Chart 15 – Cancellation Rate by Day of Week",
        "What this chart shows:\n"
        "This bar chart shows the cancellation rate for each day of the week. The X-axis shows "
        "the day (represented as a number or name depending on the dataset encoding) and the "
        "Y-axis shows the percentage of flights cancelled on that day.\n\n"
        "Key Insights:\n"
        "• Days with higher cancellation rates may correspond to days with higher traffic volume "
        "(more congestion = more cancellations) or specific operational patterns.\n"
        "• Weekends sometimes show different cancellation patterns compared to weekdays due to "
        "different passenger mix (leisure vs. business) and staffing levels.\n"
        "• Airlines can use this to adjust staffing and contingency planning by day of week.\n"
        "• Passengers can choose lower-risk travel days based on this pattern."))

    # 6-E  Winter vs Overall Cancellation Rate (bar comparison)
    winter = df3[df3["month"].isin([12, 1, 2])]
    winter_rate  = winter["cancelled"].mean() * 100
    overall_rate = df3["cancelled"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Winter Months\n(Dec, Jan, Feb)", "Overall Average"],
           [winter_rate, overall_rate],
           color=["#1565C0", "#90CAF9"], edgecolor="black", width=0.4)
    ax.set_title("Winter vs Overall Cancellation Rate", fontsize=14, weight="bold")
    ax.set_ylabel("Cancellation Rate (%)")
    for i, v in enumerate([winter_rate, overall_rate]):
        ax.text(i, v + 0.1, f"{v:.2f}%", ha="center", fontweight="bold")
    p = savefig(PLOTS_DIR / "S6_05_winter_impact.png", fig)
    out.append((p, "Chart 16 – Winter vs Overall Cancellation Rate",
        "What this chart shows:\n"
        "This comparison bar chart directly contrasts the cancellation rate during winter months "
        "(December, January, February) against the overall average cancellation rate for the "
        "entire year. The exact percentage is labeled on top of each bar.\n\n"
        "Key Insights:\n"
        "• If the winter bar is significantly taller than the overall average bar, it confirms "
        "that winter weather is a major driver of flight cancellations.\n"
        "• This quantifies the seasonal risk — airlines and passengers can see exactly how much "
        "more likely a cancellation is in winter compared to the annual average.\n"
        "• Airlines operating in winter-affected regions should maintain larger contingency "
        "fleets and crew reserves during these months.\n"
        "• This finding supports the case for weather-resilient infrastructure investments at "
        "airports in cold-climate regions."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 – AIRPORT & ROUTE-LEVEL EXPLORATION  (Milestone 3)
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_airport_route(df3) -> list[tuple[str,str,str]]:
    out = []

    # 7-A  Top 10 Origin-Destination Pairs
    od = (df3.groupby(["origin","dest"]).size().reset_index(name="count")
          .sort_values("count", ascending=False).head(10))
    od["od_pair"] = od["origin"] + " → " + od["dest"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_od = sns.color_palette("husl", len(od))
    ax.barh(od["od_pair"], od["count"], color=colors_od)
    ax.invert_yaxis()
    ax.set_title("Top 10 Origin-Destination Pairs by Flight Volume", fontsize=13, weight="bold")
    ax.set_xlabel("Number of Flights")
    p = savefig(PLOTS_DIR / "S7_01_top_od_pairs.png", fig)
    out.append((p, "Chart 17 – Top 10 Origin-Destination Pairs",
        "What this chart shows:\n"
        "This horizontal bar chart shows the 10 most frequently flown origin-destination (OD) pairs "
        "in the Milestone 3 dataset. Each bar represents a specific route pair (e.g., ATL → ORD) "
        "and its length shows how many flights operated on that pair.\n\n"
        "Key Insights:\n"
        "• The longest bar represents the single busiest route in the network — the most critical "
        "connection between two airports.\n"
        "• These top OD pairs form the core of the airline network and are essential for connectivity.\n"
        "• High-volume OD pairs are prime candidates for larger aircraft deployment to meet demand.\n"
        "• Any disruption on these routes (weather, strikes, technical issues) affects the largest "
        "number of passengers and has the greatest revenue impact."))

    # 7-B  Top 10 Busiest Airports
    busy = df3["origin"].value_counts().reset_index()
    busy.columns = ["Airport","Flight Count"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(busy.head(10)["Airport"], busy.head(10)["Flight Count"],
           color=sns.color_palette("Set2", 10), edgecolor="black")
    ax.set_title("Top 10 Busiest Airports by Departure Volume", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Number of Departing Flights")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S7_02_busiest_airports.png", fig)
    out.append((p, "Chart 18 – Top 10 Busiest Airports",
        "What this chart shows:\n"
        "This bar chart ranks the 10 airports with the highest number of departing flights in the "
        "dataset. Each bar represents an airport (by IATA code) and its height shows the total "
        "number of flights that departed from that airport.\n\n"
        "Key Insights:\n"
        "• The tallest bar identifies the busiest hub airport in the network — typically a major "
        "connecting hub like ATL (Atlanta), ORD (Chicago O'Hare), or DFW (Dallas/Fort Worth).\n"
        "• Busy airports handle more passengers and flights, making them more susceptible to "
        "congestion-related delays.\n"
        "• These airports require the most resources: gates, ground crews, security personnel, "
        "and air traffic controllers.\n"
        "• Infrastructure investments at these airports have the highest network-wide impact."))

    # 7-C  Top 10 Airports by Average Arrival Delay
    avg_delay = (df3.groupby("origin")["arr_delay"].mean().reset_index()
                 .sort_values("arr_delay", ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(avg_delay["origin"], avg_delay["arr_delay"],
           color=sns.color_palette("Reds_r", 10), edgecolor="black")
    ax.set_title("Top 10 Airports by Average Arrival Delay", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Average Arrival Delay (Minutes)")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S7_03_avg_arrival_delay.png", fig)
    out.append((p, "Chart 19 – Top 10 Airports by Average Arrival Delay",
        "What this chart shows:\n"
        "This bar chart identifies the 10 airports where arriving flights experience the highest "
        "average delay. The X-axis shows airport codes and the Y-axis shows the average number "
        "of minutes flights arrive late at that airport.\n\n"
        "Key Insights:\n"
        "• Airports with the highest arrival delays are often congested destination hubs where "
        "incoming traffic exceeds runway and gate capacity.\n"
        "• High arrival delays at a destination airport also cause departure delays for the "
        "next leg of the aircraft's journey (late aircraft delay propagation).\n"
        "• Airlines scheduling connections through these airports should allow longer layover "
        "times to reduce missed connection risk.\n"
        "• This data can inform passengers about which destination airports to expect delays at, "
        "helping them plan ground transportation and connections accordingly."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 – APPLICATION (FINAL EXECUTIVE VIEW)  (Milestone 2)
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_application(df2) -> list[tuple[str,str,str]]:
    out = []

    # 8-A  KPI Summary Bar
    total       = len(df2)
    cancel_rate = float(df2["CANCELLED"].mean()) * 100
    otp         = float(((df2["CANCELLED"] == 0) & (df2["ARR_DELAY"].fillna(0) <= 15)).mean()) * 100
    avg_arr     = float(df2.loc[df2["CANCELLED"] == 0, "ARR_DELAY"].mean())
    kpi_labels  = ["Total Flights\n(thousands)", "OTP %\n(≤15 min)", "Cancellation\nRate %", "Avg Arrival\nDelay (min)"]
    kpi_values  = [total / 1000, otp, cancel_rate, avg_arr]
    kpi_colors  = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(kpi_labels, kpi_values, color=kpi_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, kpi_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Executive KPI Summary – Full Dataset (Milestone 2)", fontsize=13, weight="bold")
    ax.set_ylabel("Value")
    p = savefig(PLOTS_DIR / "S8_01_kpi_summary.png", fig)
    out.append((p, "Chart 20 – Executive KPI Summary",
        "What this chart shows:\n"
        "This bar chart presents the four most important Key Performance Indicators (KPIs) for "
        "the entire Milestone 2 dataset in a single view:\n"
        "  • Total Flights (in thousands) – the scale of operations\n"
        "  • On-Time Performance (OTP %) – percentage of flights arriving within 15 minutes of schedule\n"
        "  • Cancellation Rate % – percentage of scheduled flights that were cancelled\n"
        "  • Average Arrival Delay (minutes) – mean delay for non-cancelled flights\n\n"
        "Key Insights:\n"
        "• This is the executive summary view — a single glance tells stakeholders the overall "
        "health of airline operations.\n"
        "• A high OTP % (green bar) indicates good operational performance.\n"
        "• A low Cancellation Rate % (red bar) indicates reliability.\n"
        "• A low Average Arrival Delay (orange bar) indicates schedule adherence.\n"
        "• These four KPIs together form the standard airline performance scorecard used by "
        "regulators, investors, and airline management."))

    # 8-B  Delay Cause Pie (Application section)
    dcols = ["DELAY_DUE_CARRIER","DELAY_DUE_WEATHER","DELAY_DUE_NAS","DELAY_DUE_LATE_AIRCRAFT"]
    dmix  = df2[dcols].sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(dmix.values, labels=["Carrier","Weather","NAS","Late Aircraft"],
           autopct="%1.1f%%", startangle=140,
           colors=["#FF6B6B","#4ECDC4","#556270","#FFD166"])
    ax.set_title("Delay Cause Mix – Share of Total Delay Minutes", fontsize=13, weight="bold")
    p = savefig(PLOTS_DIR / "S8_02_delay_mix_pie.png", fig)
    out.append((p, "Chart 21 – Delay Cause Mix (Application View)",
        "What this chart shows:\n"
        "This pie chart in the Application section shows the proportional contribution of each "
        "delay cause to the total accumulated delay minutes across the full dataset. It answers "
        "the question: 'Of all the delay minutes experienced, what fraction came from each cause?'\n\n"
        "Key Insights:\n"
        "• The largest slice is the primary delay driver — the most impactful area for improvement.\n"
        "• Carrier Delay and Late Aircraft Delay are controllable by airlines through better "
        "maintenance scheduling, crew management, and turnaround optimization.\n"
        "• Weather and NAS delays require different strategies: better forecasting, flexible "
        "scheduling, and coordination with air traffic control.\n"
        "• This chart is used in the executive dashboard to quickly communicate where delay "
        "reduction efforts should be focused."))

    # 8-C  Top 15 Delay Hotspot Airports
    hotspot = (df2.groupby("ORIGIN")["DEP_DELAY"].mean()
               .sort_values(ascending=False).head(15).reset_index())
    hotspot.columns = ["Origin","Avg Dep Delay (min)"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hotspot["Origin"], hotspot["Avg Dep Delay (min)"],
           color=sns.color_palette("YlOrRd", 15), edgecolor="black")
    ax.set_title("Top 15 Delay Hotspot Origin Airports", fontsize=13, weight="bold")
    ax.set_xlabel("Airport Code"); ax.set_ylabel("Avg Departure Delay (min)")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S8_03_hotspot_airports.png", fig)
    out.append((p, "Chart 22 – Top 15 Delay Hotspot Origin Airports",
        "What this chart shows:\n"
        "This bar chart extends the delay-prone airport analysis to the top 15 origin airports "
        "with the highest average departure delays. It is presented in the Application section "
        "as an actionable operational insight for airline management.\n\n"
        "Key Insights:\n"
        "• The airports shown here are the most critical delay hotspots in the network.\n"
        "• Airlines should investigate root causes at each of these airports individually — "
        "some may be weather-driven, others congestion-driven, others operationally driven.\n"
        "• Targeted interventions at these 15 airports would have the greatest impact on "
        "improving overall network on-time performance.\n"
        "• This chart can be filtered by airline, month, or other dimensions in the live "
        "dashboard to drill down into specific operational scenarios."))

    # 8-D  Top 15 Routes by Flight Volume
    routes = df2["Route"].value_counts().head(15).reset_index()
    routes.columns = ["Route","Flights"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(routes["Route"], routes["Flights"],
           color=sns.color_palette("Blues_r", 15), edgecolor="black")
    ax.set_title("Top 15 Routes by Flight Volume (Route Demand)", fontsize=13, weight="bold")
    ax.set_xlabel("Route (Origin → Destination)"); ax.set_ylabel("Number of Flights")
    plt.xticks(rotation=45, ha="right")
    p = savefig(PLOTS_DIR / "S8_04_top_routes_volume.png", fig)
    out.append((p, "Chart 23 – Top 15 Routes by Flight Volume",
        "What this chart shows:\n"
        "This bar chart shows the 15 routes with the highest flight volume in the Milestone 2 "
        "dataset. It represents route demand — how many times each specific origin-to-destination "
        "pair was flown.\n\n"
        "Key Insights:\n"
        "• The tallest bar represents the single highest-demand route — the most commercially "
        "important connection in the network.\n"
        "• High-demand routes generate the most revenue and are the most critical to keep "
        "on-time and operational.\n"
        "• These routes are candidates for capacity upgrades (larger aircraft, more daily frequencies).\n"
        "• In the live dashboard, this chart updates dynamically when filtered by airline, "
        "origin, or month — allowing management to see demand patterns for specific scenarios."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# MILESTONE 1 OVERVIEW PLOTS
# ═══════════════════════════════════════════════════════════════════════════
def make_plots_m1_overview(df1) -> list[tuple[str,str,str]]:
    out = []

    # M1-A  Flight Status Distribution
    status_counts = df1["Flight Status"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.1f%%",
           startangle=90, colors=["#4CAF50","#FF9800","#F44336"])
    ax.set_title("Overall Flight Status Distribution (Milestone 1)", fontsize=13, weight="bold")
    p = savefig(PLOTS_DIR / "S2_01_flight_status_pie.png", fig)
    out.append((p, "Chart 24 – Overall Flight Status Distribution (Milestone 1)",
        "What this chart shows:\n"
        "This pie chart shows the overall proportion of flights in the Milestone 1 dataset that "
        "were classified as On Time, Delayed, or Cancelled. Each slice represents the percentage "
        "share of that flight status across the entire dataset.\n\n"
        "Key Insights:\n"
        "• The relative sizes of the slices give an immediate high-level view of airline reliability.\n"
        "• A large 'On Time' slice (green) indicates good overall operational performance.\n"
        "• A significant 'Delayed' slice (orange) highlights that delays are a common occurrence "
        "and warrant further investigation into causes.\n"
        "• A visible 'Cancelled' slice (red) indicates that cancellations, while less frequent "
        "than delays, are still a meaningful operational challenge.\n"
        "• This chart is the starting point of the entire analysis — it establishes the baseline "
        "performance picture before drilling into causes and patterns."))

    # M1-B  Monthly OTP vs Cancellation
    monthly = (df1.groupby("Month")
               .agg(total=("Flight Status","count"), on_time=("is_on_time","sum"), cancelled=("is_cancelled","sum"))
               .reset_index())
    monthly["OTP %"]    = (monthly["on_time"]   / monthly["total"]) * 100
    monthly["Cancel %"] = (monthly["cancelled"] / monthly["total"]) * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(monthly)); w = 0.35
    ax.bar([i - w/2 for i in x], monthly["OTP %"],    w, label="On-time %",      color="#2196F3")
    ax.bar([i + w/2 for i in x], monthly["Cancel %"], w, label="Cancellation %", color="#F44336")
    ax.set_xticks(list(x))
    ax.set_xticklabels(monthly["Month"], rotation=45, ha="right")
    ax.set_title("Monthly On-time Performance vs Cancellation Rate (Milestone 1)", fontsize=13, weight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Percentage (%)")
    ax.legend()
    p = savefig(PLOTS_DIR / "S2_02_monthly_otp_cancel.png", fig)
    out.append((p, "Chart 25 – Monthly On-time Performance vs Cancellation Rate (Milestone 1)",
        "What this chart shows:\n"
        "This grouped bar chart compares, month by month, the percentage of flights that arrived "
        "on time (blue bars) against the percentage that were cancelled (red bars) in the "
        "Milestone 1 dataset.\n\n"
        "Key Insights:\n"
        "• Months where the blue bar is tall and the red bar is short represent the best "
        "operational months — high reliability and low disruption.\n"
        "• Months where the red bar spikes indicate periods of operational disruption, possibly "
        "due to weather events, seasonal demand surges, or external factors.\n"
        "• The inverse relationship between OTP and cancellation rate is visible — when "
        "cancellations rise, on-time performance typically falls.\n"
        "• This trend analysis helps identify which months need the most operational improvement "
        "and resource pre-positioning."))

    return out

# ═══════════════════════════════════════════════════════════════════════════
# WORD DOCUMENT BUILDER
# ═══════════════════════════════════════════════════════════════════════════
def build_docx(sections: list[tuple[str, str, list[tuple[str,str,str]]]]) -> Path:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()

    # ── page margins ──
    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    def set_heading_color(para, r, g, b):
        for run in para.runs:
            run.font.color.rgb = RGBColor(r, g, b)

    def add_divider(doc):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(2)
        run = p.add_run("─" * 80)
        run.font.color.rgb = RGBColor(180, 180, 180)
        run.font.size = Pt(7)

    # ── COVER PAGE ──────────────────────────────────────────────────────────
    doc.add_paragraph()
    doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("AirFly Insights")
    tr.font.size = Pt(32); tr.font.bold = True
    tr.font.color.rgb = RGBColor(13, 71, 161)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Data Visualization and Analysis of Airline Operations")
    sr.font.size = Pt(18); sr.font.color.rgb = RGBColor(33, 33, 33)

    doc.add_paragraph()
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cap.add_run("Final Dashboard Report  |  Milestones 1 – 3  |  Infosys SpringBoard Internship 6.0")
    cr.font.size = Pt(12); cr.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}").font.size = Pt(11)

    doc.add_page_break()

    # ── TABLE OF CONTENTS (manual) ──────────────────────────────────────────
    h = doc.add_heading("Table of Contents", level=1)
    set_heading_color(h, 13, 71, 161)
    toc_items = [
        "1.  Project Overview",
        "2.  Data Acquisition and Understanding",
        "3.  Data Cleaning, Preprocessing and Feature Engineering",
        "4.  Univariate and Bivariate Analysis",
        "5.  Delay Cause Analysis, Routes and Airport Level Analysis",
        "6.  Cancellation and Seasonal Trends",
        "7.  Airport and Route-Level Exploration",
        "8.  Application – Executive KPI and Actionable Insights",
        "9.  Conclusion and Recommendations",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item, style="List Number")
        p.runs[0].font.size = Pt(11)
    doc.add_page_break()

    # ── SECTION 1: OVERVIEW ─────────────────────────────────────────────────
    h = doc.add_heading("1.  Project Overview", level=1)
    set_heading_color(h, 13, 71, 161)

    doc.add_heading("1.1  Project Statement", level=2)
    doc.add_paragraph(
        "The objective of this project is to analyze large-scale airline flight data to uncover "
        "operational trends, delay patterns, and cancellation reasons using data visualization "
        "techniques. The goal is to help understand airline and airport-level performance and "
        "contribute to actionable insights using visual analysis. This project was completed as "
        "part of the Infosys SpringBoard Internship 6.0 program across three progressive milestones."
    )

    doc.add_heading("1.2  Expected Outcomes", level=2)
    outcomes = [
        "Understand and preprocess aviation datasets for analysis",
        "Explore trends in flight schedules, delays, cancellations, and routes",
        "Visualize key metrics using bar charts, time series, pie charts, bubble charts, and comparisons",
        "Provide actionable insights for stakeholders including airline operators and analysts",
        "Summarize findings through a final visual dashboard and consolidated report",
    ]
    for o in outcomes:
        doc.add_paragraph(o, style="List Bullet")

    doc.add_heading("1.3  Dataset Sources", level=2)
    doc.add_paragraph(
        "Three datasets were used across the three milestones, all sourced from Kaggle's Airlines "
        "Flights Dataset collection:"
    )
    ds = [
        ("Milestone 1", "Airline Dataset Updated - v2.csv",
         "Contains flight status (On Time / Delayed / Cancelled), departure dates, airline names, "
         "origin and destination airports. Used for high-level operational status analysis."),
        ("Milestone 2", "cleaned_flights.csv",
         "A cleaned and enriched dataset containing detailed delay cause columns "
         "(DELAY_DUE_CARRIER, DELAY_DUE_WEATHER, DELAY_DUE_NAS, DELAY_DUE_LATE_AIRCRAFT), "
         "departure/arrival delay minutes, cancellation flags, hour of departure, and route information. "
         "Used for deep delay cause analysis and univariate/bivariate exploration."),
        ("Milestone 3", "FInal_Flights_cleaned_data.csv",
         "A further cleaned dataset with standardized column names (origin, dest, route, cancelled, "
         "month, day_of_week, arr_delay). Used for cancellation trend analysis and airport/route "
         "level exploration."),
    ]
    for ms, fname, desc in ds:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{ms} – {fname}: ").bold = True
        p.add_run(desc)

    doc.add_heading("1.4  Technology Stack", level=2)
    tech = [
        "Python 3.x – core programming language",
        "Pandas & NumPy – data loading, cleaning, and transformation",
        "Matplotlib & Seaborn – static chart generation",
        "Plotly Express – interactive charts in the live dashboard",
        "Streamlit – web-based interactive dashboard framework",
        "python-docx – Word document report generation",
    ]
    for t in tech:
        doc.add_paragraph(t, style="List Bullet")

    doc.add_heading("1.5  Dashboard Navigation Structure", level=2)
    doc.add_paragraph(
        "The Final Dashboard is organized into 9 navigation sections accessible from the sidebar:"
    )
    nav = [
        "1. Overview – Project statement, expected outcomes, dataset paths",
        "2. Data Acquisition and Understanding – Dataset previews and schema",
        "3. Data Cleaning, Preprocessing and Feature Engineering – Transformation steps",
        "4. Univariate and Bivariate Analysis – Distribution charts for airlines, routes, time",
        "5. Delay Cause Analysis, Routes, Airport Level Analysis – Delay breakdown and airport performance",
        "6. Cancellation and Seasonal Trends – Monthly and weekly cancellation patterns",
        "7. Airport and Route-Level Exploration – OD pairs, busiest airports, arrival delays",
        "8. Application of this entire project – Executive KPI dashboard with filters",
        "9. Documentation and Presentation – Report download and documentation links",
    ]
    for n in nav:
        doc.add_paragraph(n, style="List Number")

    doc.add_page_break()

    # ── SECTION 2: DATA ACQUISITION ─────────────────────────────────────────
    h = doc.add_heading("2.  Data Acquisition and Understanding", level=1)
    set_heading_color(h, 13, 71, 161)
    doc.add_paragraph(
        "This section of the dashboard provides a quick schema preview and sample rows for each "
        "of the three milestone datasets. The purpose is to familiarize the analyst with the "
        "structure, column names, data types, and scale of each dataset before analysis begins."
    )
    doc.add_heading("2.1  Milestone 1 Dataset – Operations Status", level=2)
    doc.add_paragraph(
        "Contains flight-level records with columns including: Airline Name, Airport Name, "
        "Arrival Airport, Departure Date, Flight Status (On Time / Delayed / Cancelled), "
        "and Passenger Count. This dataset is used for high-level operational status analysis "
        "and route performance benchmarking."
    )
    doc.add_heading("2.2  Milestone 2 Dataset – Delays Dataset", level=2)
    doc.add_paragraph(
        "A richer dataset with columns: AIRLINE, ORIGIN, DEST, MONTH, DAY_OF_WEEK, HOUR, "
        "DEP_DELAY, ARR_DELAY, CANCELLED, DELAY_DUE_CARRIER, DELAY_DUE_WEATHER, DELAY_DUE_NAS, "
        "DELAY_DUE_LATE_AIRCRAFT. This dataset enables granular delay cause analysis and "
        "temporal pattern exploration."
    )
    doc.add_heading("2.3  Milestone 3 Dataset – Routes and Seasonality", level=2)
    doc.add_paragraph(
        "A standardized cleaned dataset with columns: origin, dest, route, cancelled, month, "
        "day_of_week, arr_delay. This dataset is optimized for cancellation trend analysis "
        "and airport/route level exploration with consistent lowercase column naming."
    )
    doc.add_page_break()

    # ── SECTION 3: CLEANING ─────────────────────────────────────────────────
    h = doc.add_heading("3.  Data Cleaning, Preprocessing and Feature Engineering", level=1)
    set_heading_color(h, 13, 71, 161)
    doc.add_paragraph(
        "Before any analysis could be performed, each dataset required cleaning and feature "
        "engineering to ensure data quality and analytical readiness."
    )
    steps = [
        ("Milestone 1 – Datetime Parsing",
         "The 'Departure Date' column contained mixed date formats. pd.to_datetime() with "
         "format='mixed' and errors='coerce' was used to parse all dates, converting unparseable "
         "values to NaT. A 'Month' period column was derived for monthly aggregation."),
        ("Milestone 1 – Boolean Status Flags",
         "Three boolean columns were engineered: is_on_time, is_delayed, is_cancelled — each "
         "derived from the 'Flight Status' string column. These enable fast aggregation without "
         "repeated string comparisons."),
        ("Milestone 2 – Flight Status Derivation",
         "The Milestone 2 dataset did not have a direct 'Flight Status' column. It was derived "
         "programmatically: if CANCELLED==1 → 'Cancelled'; elif ARR_DELAY>0 → 'Delayed'; "
         "else → 'On Time'."),
        ("Milestone 2 – Month and Day Mapping",
         "Numeric MONTH (1–12) and DAY_OF_WEEK (1–7) columns were mapped to human-readable "
         "strings (e.g., 1 → 'January', 1 → 'Monday') for chart labeling."),
        ("Milestone 2 – Route Feature",
         "A 'Route' column was engineered by concatenating ORIGIN and DEST with an arrow: "
         "'ORIGIN → DEST'. This enables route-level grouping and frequency analysis."),
        ("Milestone 3 – Standardized Schema",
         "The Milestone 3 dataset was pre-cleaned with lowercase column names and standardized "
         "values. No additional cleaning was required beyond loading."),
    ]
    for title_s, desc_s in steps:
        doc.add_heading(title_s, level=2)
        doc.add_paragraph(desc_s)
    doc.add_page_break()

    # ── SECTIONS WITH CHARTS ────────────────────────────────────────────────
    for sec_title, sec_intro, plots in sections:
        h = doc.add_heading(sec_title, level=1)
        set_heading_color(h, 13, 71, 161)
        if sec_intro:
            doc.add_paragraph(sec_intro)
        doc.add_paragraph()

        for img_path, chart_title, chart_desc in plots:
            # chart heading
            ch = doc.add_heading(chart_title, level=2)
            set_heading_color(ch, 21, 101, 192)

            # embed image
            if os.path.exists(img_path):
                doc.add_picture(img_path, width=Inches(6.0))
                last_para = doc.paragraphs[-1]
                last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                doc.add_paragraph(f"[Image not found: {img_path}]")

            # explanation below chart
            doc.add_paragraph()
            for line in chart_desc.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":") and not line.startswith("•"):
                    p = doc.add_paragraph()
                    r = p.add_run(line)
                    r.bold = True; r.font.size = Pt(11)
                elif line.startswith("•"):
                    bp = doc.add_paragraph(style="List Bullet")
                    bp.add_run(line[1:].strip()).font.size = Pt(10)
                elif line.startswith("  •"):
                    bp = doc.add_paragraph(style="List Bullet 2")
                    bp.add_run(line[3:].strip()).font.size = Pt(10)
                else:
                    p = doc.add_paragraph(line)
                    p.runs[0].font.size = Pt(10)

            add_divider(doc)
            doc.add_paragraph()

        doc.add_page_break()

    # ── SECTION 9: CONCLUSION ───────────────────────────────────────────────
    h = doc.add_heading("9.  Conclusion and Recommendations", level=1)
    set_heading_color(h, 13, 71, 161)
    doc.add_paragraph(
        "This project successfully analyzed large-scale airline flight data across three progressive "
        "milestones, building from basic operational status analysis to deep delay cause investigation "
        "and finally to a professional executive dashboard. The following key findings and "
        "recommendations emerge from the analysis:"
    )
    findings = [
        ("Delay Propagation is the Biggest Challenge",
         "Delays accumulate throughout the day — early morning flights are most reliable while "
         "evening flights suffer from cascading disruptions. Airlines should build buffer time "
         "into afternoon and evening schedules."),
        ("Carrier and Late Aircraft Delays are Controllable",
         "The delay cause analysis shows that a significant portion of total delay minutes comes "
         "from carrier-controllable causes (maintenance, crew, turnaround). Investing in "
         "predictive maintenance and crew optimization can directly reduce these delays."),
        ("Winter Months Require Special Contingency Planning",
         "Cancellation rates spike significantly in winter months (December, January, February). "
         "Airlines should pre-position spare aircraft and standby crews at high-risk airports "
         "during these months."),
        ("A Small Number of Routes and Airports Drive Most Disruption",
         "The top 10 delay-prone airports and top 10 high-cancellation routes account for a "
         "disproportionate share of passenger disruption. Targeted interventions at these "
         "specific locations would have the highest network-wide impact."),
        ("High-Volume Routes Need Priority Reliability Management",
         "The top 15 routes by flight volume are the commercial backbone of the network. "
         "Any disruption on these routes affects the most passengers. These routes should "
         "have dedicated reliability monitoring and faster recovery protocols."),
        ("The Executive Dashboard Enables Data-Driven Decision Making",
         "The Application section of the dashboard provides a filterable, real-time KPI view "
         "that allows airline management to slice performance by airline, origin, and month. "
         "This enables rapid identification of underperforming segments and targeted action."),
    ]
    for ftitle, fdesc in findings:
        doc.add_heading(ftitle, level=2)
        doc.add_paragraph(fdesc)

    doc.add_paragraph()
    doc.add_paragraph(
        "This Final Dashboard and Report represent the complete analytical workflow from raw data "
        "ingestion through cleaning, exploration, visualization, and executive presentation — "
        "demonstrating the full data science pipeline applied to a real-world aviation dataset."
    )

    # ── save ────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "AirFly_Insights_Final_Dashboard_Report.docx"
    doc.save(str(out_path))
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  AirFly Insights – Final Report Generator")
    print("=" * 60)

    print("\n[1/6] Loading datasets...")
    df1 = load_m1()
    df2 = load_m2()
    df3 = load_m3()
    print(f"  M1: {len(df1):,} rows | M2: {len(df2):,} rows | M3: {len(df3):,} rows")

    print("\n[2/6] Generating Section 2 charts (Milestone 1 Overview)...")
    plots_s2 = make_plots_m1_overview(df1)

    print("[3/6] Generating Section 4 charts (Univariate & Bivariate)...")
    plots_s4 = make_plots_univariate(df2)

    print("[4/6] Generating Section 5 charts (Delay Cause & Airport Analysis)...")
    plots_s5 = make_plots_delay(df1, df2)

    print("[5/6] Generating Section 6 & 7 charts (Cancellation, Airport/Route)...")
    plots_s6 = make_plots_cancellation(df3)
    plots_s7 = make_plots_airport_route(df3)

    print("[5/6] Generating Section 8 charts (Application / Executive View)...")
    plots_s8 = make_plots_application(df2)

    total_charts = len(plots_s2) + len(plots_s4) + len(plots_s5) + len(plots_s6) + len(plots_s7) + len(plots_s8)
    print(f"  Total charts generated: {total_charts}")

    sections = [
        (
            "2.  Data Acquisition – Dataset Overview Charts",
            "The following charts provide a visual overview of the Milestone 1 dataset, "
            "establishing the baseline operational performance picture.",
            plots_s2,
        ),
        (
            "4.  Univariate and Bivariate Analysis",
            "This section explores the distributions of key variables in the Milestone 2 dataset — "
            "airlines, routes, monthly trends, day-of-week patterns, and hourly distributions. "
            "These charts form the foundation of understanding the dataset before deeper analysis.",
            plots_s4,
        ),
        (
            "5.  Delay Cause Analysis, Routes and Airport Level Analysis",
            "This section investigates the root causes of flight delays and identifies the airports "
            "and routes most affected. It combines data from both Milestone 1 and Milestone 2 "
            "to provide a comprehensive view of delay patterns across the network.",
            plots_s5,
        ),
        (
            "6.  Cancellation and Seasonal Trends",
            "This section analyzes flight cancellation patterns across months, days of the week, "
            "and specific routes using the Milestone 3 dataset. It also quantifies the impact "
            "of winter weather on cancellation rates.",
            plots_s6,
        ),
        (
            "7.  Airport and Route-Level Exploration",
            "This section provides a detailed exploration of airport and route performance using "
            "the Milestone 3 dataset — identifying the busiest airports, most popular routes, "
            "and airports with the highest arrival delays.",
            plots_s7,
        ),
        (
            "8.  Application – Executive KPI and Actionable Insights",
            "This is the final application section of the dashboard — a professional, "
            "stakeholder-ready executive view that synthesizes the entire analytical workflow "
            "into KPI metrics and actionable operational insights using the Milestone 2 dataset.",
            plots_s8,
        ),
    ]

    print("\n[6/6] Building Word document...")
    out_path = build_docx(sections)
    print(f"\n  Report saved to:\n  {out_path}")
    print("\nDone. Open the .docx file to view the complete report.")


if __name__ == "__main__":
    main()
