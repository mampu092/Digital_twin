import plotly.graph_objects as go
from src.data.simulation import UNITS


def sensor_chart(frame,feature,reference=None,anomalies=None,comparison=None):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=frame.timestamp,y=frame[feature],mode="lines+markers",
        name="Synthetic observations",line=dict(color="#058b8c",width=2.5),marker=dict(size=5),connectgaps=False))
    if reference and feature in reference["center"]:
        center=reference["center"][feature];scale=reference["scale"][feature]
        fig.add_hrect(y0=center-2*scale,y1=center+2*scale,fillcolor="#058b8c",opacity=.09,line_width=0)
        fig.add_hline(y=center,line_dash="dot",line_color="#6f8897",annotation_text="Enrollment reference")
    if anomalies:
        flags={r["timestamp"] for r in anomalies if r.get("flag")}
        flagged=frame[frame.timestamp.isin(flags)]
        fig.add_trace(go.Scatter(x=flagged.timestamp,y=flagged[feature],mode="markers",
            name="Multisensor deviation",marker=dict(color="#df923d",size=10,symbol="diamond")))
    if comparison is not None:
        fig.add_trace(go.Scatter(x=comparison.timestamp,y=comparison[feature],mode="lines",
            name="Hypothetical trajectory",line=dict(color="#8e64b0",dash="dash",width=2)))
    fig.update_layout(height=285,margin=dict(l=15,r=15,t=35,b=15),title=feature.replace("_"," ").upper(),
        yaxis_title=UNITS[feature],paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=-.25),font=dict(color="#17354b",family="Arial"))
    fig.update_xaxes(showgrid=False);fig.update_yaxes(gridcolor="#e1e9ed")
    return fig
