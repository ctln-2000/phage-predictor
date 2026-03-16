import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_cs(df):

    matriz = df[["PHAGE","CS_cocktail"]].set_index("PHAGE").T

    fig, ax = plt.subplots(figsize=(14,4))

    sns.heatmap(
        matriz,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label":"Collection Score"},
        ax=ax
    )

    ax.set_ylabel("BACTERIA")
    ax.set_xlabel("")
    ax.set_title("Collection Score por Fago")

    plt.xticks(rotation=90)

    return fig
