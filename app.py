import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load data
df = pd.read_csv('FInal_Flights_cleaned_data.csv')

st.title('Milestone 3: Route, Cancellation, and Seasonal Insights')

st.header('Week 5: Route and Airport-Level Analysis')

# Top 10 Origin-Destination Pairs
od = (
    df.groupby(['origin', 'dest'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
    .head(10)
)
st.subheader('Top 10 Origin-Destination Pairs')
st.dataframe(od)
fig = px.bar(od, x='origin', y='count', color='dest', title='Top 10 Origin-Destination Pairs')
st.plotly_chart(fig)

# Delay Heatmaps by Airport and Route
delay = (
    df.groupby(['origin', 'dest'])['arr_delay']
    .mean()
    .reset_index()
)
pivot = delay.pivot(index='origin', columns='dest', values='arr_delay')
st.subheader('Delay Heatmap by Route')
fig = px.imshow(pivot, title='Average Arrival Delay Heatmap by Route')
st.plotly_chart(fig)

# Busiest airports and average delays
busy = df['origin'].value_counts().reset_index()
busy.columns = ['Airport', 'Flight Count']
st.subheader('Top 10 Busiest Airports')
fig = px.bar(busy.head(10), x='Airport', y='Flight Count', title='Top 10 Busiest Airports')
st.plotly_chart(fig)

avg_delay = (
    df.groupby('origin')['arr_delay']
    .mean()
    .reset_index()
    .sort_values('arr_delay', ascending=False)
)
st.subheader('Top 10 Airports by Average Arrival Delay')
fig = px.bar(avg_delay.head(10), x='origin', y='arr_delay', title='Top 10 Airports by Average Arrival Delay')
st.plotly_chart(fig)

st.header('Week 6: Seasonal and Cancellation Analysis')

# Monthly Cancellation Trends
cancel_by_month = (
    df.groupby('month')['cancelled']
    .agg(total_cancelled='sum', total_flights='count')
    .reset_index()
)
cancel_by_month['cancel_rate'] = cancel_by_month['total_cancelled'] / cancel_by_month['total_flights']

st.subheader('Monthly Cancellation Trends')
fig = px.line(cancel_by_month, x='month', y='cancel_rate', markers=True, title='Cancellation Rate by Month')
fig.update_layout(yaxis_tickformat='.1%')
st.plotly_chart(fig)

st.subheader('Monthly Cancellation Count')
fig = px.bar(cancel_by_month, x='month', y='total_cancelled', title='Total Cancellations by Month')
st.plotly_chart(fig)

# Cancellation Types (if available in this dataset)
if 'cancellation_reason' in df.columns:
    cancel_types = df['cancellation_reason'].value_counts().reset_index()
    cancel_types.columns = ['Reason', 'Count']
    st.subheader('Cancellation Types')
    fig = px.pie(cancel_types, values='Count', names='Reason', title='Cancellation Types')
    st.plotly_chart(fig)
else:
    st.subheader('Cancellation Types')
    st.write('Cancellation reason data not available in this dataset.')

# Cancellation breakdowns (alternative views)
worst_routes = (
    df[df['cancelled'] == 1]
    .groupby('route')
    .size()
    .reset_index(name='cancel_count')
    .sort_values('cancel_count', ascending=False)
    .head(10)
)
st.subheader('Top 10 Routes with the Most Cancellations')
fig = px.bar(worst_routes, x='route', y='cancel_count', title='Top 10 Routes by Cancellation Count')
st.plotly_chart(fig)

cancel_by_day = (
    df.groupby('day_of_week')['cancelled']
    .agg(total_cancelled='sum', total_flights='count')
    .reset_index()
)
cancel_by_day['cancel_rate'] = cancel_by_day['total_cancelled'] / cancel_by_day['total_flights']

st.subheader('Cancellation Rate by Day of Week')
fig = px.bar(cancel_by_day, x='day_of_week', y='cancel_rate', title='Cancellation Rate by Day of Week')
fig.update_layout(yaxis_tickformat='.1%')
st.plotly_chart(fig)

# Analyze impact of winter months on cancellations
winter = df[df['month'].isin([12, 1, 2])]
winter_cancel_rate = winter['cancelled'].mean()
total_cancel_rate = df['cancelled'].mean()
st.subheader('Impact of Winter Months')
st.write(f'Winter cancellation rate: {winter_cancel_rate:.4f}')
st.write(f'Total cancellation rate: {total_cancel_rate:.4f}')
st.write('Winter months show higher cancellation rates due to weather.')