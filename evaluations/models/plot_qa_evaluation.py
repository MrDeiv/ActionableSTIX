import plotly.graph_objects as go
import pandas as pd
import json

with open("results.json", "r") as f:
    results = json.load(f)

fig = go.Figure()
for model in results:
    df = pd.DataFrame(dict(
        r = [results[model]['precision'], results[model]['recall'], results[model]['f1']],
        theta = ['precision', 'recall', 'f1']
    ))

    fig.add_trace(go.Scatterpolar(
        r=df['r'],
        theta=df['theta'],
        fill='toself',
        name=model,

    ))

fig.update_traces(fill='toself')
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.8, 0.9]
        )
    ),
    showlegend=True
)
fig.show()