import plotly.express as px
import pandas as pd
import numpy as np
from collections.abc import MutableMapping

def plot_interactive_precision_recall_curve(precision, recall, threshold):
    df = pd.DataFrame()
    df["threshold"] = np.concatenate([np.array([0]), threshold])
    df["precision"] = precision[::-1]
    df["recall"] = recall[::-1]
    fig = px.area(
        data_frame=df, 
        x="recall", 
        y="precision",
        hover_data=["threshold"], 
        title='Precision-Recall Curve'
    )
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=30, r=30, b=30, t=30, pad=4),
        title_x=.5, # Centre title
        hovermode = 'closest',
        xaxis=dict(hoverformat='.4f'),
        yaxis=dict(hoverformat='.4f')
    )
    hovertemplate = 'Recall=%{x}<br>Precision=%{y}<br>Threshold=%{customdata[0]:.4f}<extra></extra>'
    fig.update_traces(hovertemplate=hovertemplate)
    
    # Add dashed line with a slope of 1
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.show()


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)