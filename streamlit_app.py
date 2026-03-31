import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from data_manipulation import get_season_totals, create_metrics, add_seeds, add_FF, add_team_names, create_summary

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='March Madness Predictor',
    page_icon='🏀',
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def load_data():

    rdsr = pd.read_csv("./data/MRegularSeasonDetailedResults.csv")
    teams = pd.read_csv("./data/MTeams.csv")
    tourney = pd.read_csv("./data/MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv("./data/MNCAATourneySeeds.csv")

    return rdsr, teams, tourney, seeds

@st.cache_data
def manipulate_data():
    season_stats = get_season_totals(rdsr)
    team_stats = create_metrics(season_stats)
    team_stats = add_seeds(team_stats, seeds)
    stats_data = add_FF(team_stats, tourney)
    stats_data_with_names = add_team_names(stats_data, teams)
    return stats_data, stats_data_with_names


@st.cache_resource
def load_models():
    pca_model = joblib.load("./data/pca_pipeline.pkl")
    no_pca_model = joblib.load("./data/non_pca_pipeline.pkl")
    columns = joblib.load("./data/feature_columns.pkl")
    return pca_model, no_pca_model, columns


rdsr, teams, tourney, seeds = load_data()
stats_data, stats_data_with_names = manipulate_data()
pca_model, no_pca_model, columns = load_models()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 🏀 March Madness Final Four Predictor

This app demonstrates a machine learning model trained to predict Final Four teams using historical NCAA data.
'''

# Season Selection
seasons = sorted(stats_data["Season"].unique())
selected_season = st.selectbox("Select Season", seasons)

# Region Selection

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["Non-PCA", "PCA", "Compare Both"]
)

# Filter by season FIRST
season_df = stats_data_with_names[
    stats_data_with_names["Season"] == selected_season
].copy()

st.subheader(f"Data for {selected_season}")
st.dataframe(season_df)

'''
Choose what statistics to compare between Final Four and non-Final Four teams. You can select as many as you like! 
Hover over the points to see which team they represent.
'''

# Map FinalFour
season_df['FinalFour'] = season_df['FinalFour'].map({
    0: 'Not Final Four',
    1: 'Final Four'
})

# User selects stats
selected_stats = st.multiselect(
    "Select stats to compare",
    columns,
    default=columns[:1]
)

# Melt from the original (wide) data
plot_df = season_df[['Team', 'FinalFour'] + selected_stats].melt(
    id_vars=['Team', 'FinalFour'],
    var_name='Statistic',
    value_name='Value'
)

# Create plot
fig = px.strip(
    plot_df,
    x='Statistic',
    y='Value',
    color='FinalFour',
    hover_data=['Team'],
)

fig.update_layout(
    title=f"Stat Comparison ({selected_season})",
    xaxis_title="Statistic",
    yaxis_title="Value"
)

fig.update_traces(jitter=0.9)

# Show plot
st.plotly_chart(fig, use_container_width=True)