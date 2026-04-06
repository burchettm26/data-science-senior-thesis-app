import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
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


# Filter by season FIRST
season_df = stats_data_with_names[
    stats_data_with_names["Season"] == selected_season
].copy()

st.subheader(f"Data for {selected_season}")
st.dataframe(season_df)

'''
Choose what statistics to compare between the average of the Final Four teams and any other teams. You can select as many as you like! 
'''

# Map FinalFour
season_df['FinalFour'] = season_df['FinalFour'].map({
    0: 'Not Final Four',
    1: 'Final Four'
})

# User selects stats to compare
selected_stats = st.multiselect(
    "Select stats to compare",
    columns,
    default=columns[:1]
)

# select a team for the radar chart
selected_team = st.selectbox(
    "Select a team",
    season_df["Team"].unique()
)

percentile_df = season_df.copy()

for col in selected_stats:
    min_val = season_df[col].min()
    max_val = season_df[col].max()
    
    percentile_df[col + "_pct"] = (
        (season_df[col] - min_val) / (max_val - min_val)
    ) * 100

lower_is_better = ['DefRtg', 'OPPG', 'TORate']

for col in lower_is_better:
    if col in selected_stats:
        percentile_df[col + "_pct"] = 100 - percentile_df[col + "_pct"]

team_row = percentile_df[percentile_df["Team"] == selected_team].iloc[0]

categories = selected_stats

team_percentiles = [team_row[col + "_pct"] for col in selected_stats]
team_raw = [team_row[col] for col in selected_stats]

# Average Final Four team
ff_df = percentile_df[percentile_df["FinalFour"] == "Final Four"]

avg_percentiles = [
    ff_df[col + "_pct"].mean() for col in selected_stats
]

# Used to connect the lines in the radar chart around whatever stats the user selects. 
# We repeat the first stat at the end to close the loop.
categories = selected_stats + [selected_stats[0]]
team_percentiles += [team_percentiles[0]]
avg_percentiles += [avg_percentiles[0]]

fig = go.Figure()

# Team trace
fig.add_trace(go.Scatterpolar(
    r=team_percentiles,
    theta=categories,
    fill='toself',
    name=selected_team,
    hovertext=[
        f"{cat}<br>Percentile: {pct:.1f}<br>Value: {raw:.3f}"
        for cat, pct, raw in zip(categories, team_percentiles, team_raw)
    ],
    hoverinfo="text"
))

# Final Four average trace
fig.add_trace(go.Scatterpolar(
    r=avg_percentiles,
    theta=categories,
    fill='toself',
    name='Avg Final Four',
    line=dict(dash='dash')
))

fig.update_layout(
    title=f"{selected_team} Profile vs Final Four Teams ({selected_season})",
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# Model Selection
model_choice = st.selectbox(
    "Choose Model",
    ["Non-PCA", "PCA", "Compare Both"]
)

features = season_df.drop(columns=["FinalFour", "Season", "Team"])

if model_choice == "Non-PCA":
    season_df["Probability"] = no_pca_model.predict_proba(features)[:, 1]

elif model_choice == "PCA":
    season_df["Probability"] = pca_model.predict_proba(features)[:, 1]

else:
    season_df["Non-PCA Prob"] = no_pca_model.predict_proba(features)[:, 1]
    season_df["PCA Prob"] = pca_model.predict_proba(features)[:, 1]

st.subheader("Predictions")

if model_choice == "Compare Both":
    st.dataframe(
        season_df.sort_values("Non-PCA Prob", ascending=False)
    )
else:
    st.dataframe(
        season_df.sort_values("Probability", ascending=False)
    )

st.subheader("Final Four Predictions")

season_df = season_df.sort_values(by="Probability", ascending=True)

pred_fig = px.scatter(
    season_df,
    x='Probability',
    y='Team',
    color='Probability',
    hover_data=['Seed', 'FinalFour']
)

st.plotly_chart(pred_fig, use_container_width=True)