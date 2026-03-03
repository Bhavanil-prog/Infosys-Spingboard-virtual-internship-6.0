# AirFly Insights -- Milestone 2

## Week 3 & Week 4: Visual Exploration and Delay Trends

------------------------------------------------------------------------

## Introduction

This milestone focuses on exploratory visualization and delay analysis
of airline operations data.

------------------------------------------------------------------------

# WEEK 3 -- Univariate & Bivariate Analysis

## 1. Top Airlines

Code: df.groupby('AIRLINE').size()

Purpose: Identify dominant carriers.

------------------------------------------------------------------------

## 2. Top Routes

Code: df\['ROUTE'\].value_counts()

Purpose: Detect busiest origin-destination pairs.

------------------------------------------------------------------------

## 3. Monthly Trend

Code: df\['MONTH'\].value_counts().sort_index()

Purpose: Identify seasonal flight patterns.

------------------------------------------------------------------------

## 4. Flight Distribution by Hour

Code: plt.hist(df\['HOUR'\])

Purpose: Identify peak departure hours.

------------------------------------------------------------------------

## 5. Arrival Delay Distribution

Code: plt.boxplot(df\['ARR_DELAY'\])

Purpose: Analyze delay variability and outliers.

------------------------------------------------------------------------

# WEEK 4 -- Delay Analysis

## 6. Compare Delay Causes by Airline

Code: df.groupby('AIRLINE')\[delay_cols\].mean()

Purpose: Benchmark airline delay performance.

------------------------------------------------------------------------

## 7. Delay Contribution (Pie Chart)

Code: df\[delay_cols\].sum()

Purpose: Identify dominant delay driver.

------------------------------------------------------------------------

## 8. Delay by Time

Code: df.groupby('HOUR')\['DEP_DELAY'\].mean()

Purpose: Detect peak delay hours.

------------------------------------------------------------------------

## 9. Delay by Airport

Code: df.groupby('ORIGIN')\['DEP_DELAY'\].mean()

Purpose: Identify congestion hubs.

------------------------------------------------------------------------

## Key Observations

• Carrier delays dominate total delay minutes\
• Evening hours show higher delay accumulation\
• Certain airlines show higher variability\
• Hub airports experience congestion patterns

------------------------------------------------------------------------

## Conclusion

Milestone 2 successfully delivered professional visual insights into
airline delay patterns.
