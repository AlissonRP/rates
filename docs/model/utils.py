import matplotlib.pyplot as plt
import seaborn as sns


def bar_ploto(df, variable="Estado", metric="mean"):
    """
    metric: pode ser mean, min, max etc, e interessante pois tem municipios com prop=1
    """
    desmatamento = (
        df.groupby(variable, as_index=False)
        .agg(metric)
        .sort_values(["prop"], ascending=False)
    )

    desmatamento["prop"] = round(desmatamento["prop"], 2)

    ax = sns.barplot(data=desmatamento, x=variable, y="prop")
    for i in ax.containers:
        ax.bar_label(
            i,
        )
