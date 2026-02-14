from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_results(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def plot_alignment_distribution(results: pd.DataFrame):
    counts = results["llm_alignment_label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Alignment Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig
