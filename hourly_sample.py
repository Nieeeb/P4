import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn theme for consistent styling
sns.set_theme(style='whitegrid')


def read_json_file(filepath):
    """
    Reads a JSON file and returns the data as a Python object.
    :param filepath: Path to the JSON file.
    :return: Parsed JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_hour_histogram(filepath):
    """
    Aggregates image samples by hour of day across the entire dataset and plots a histogram,
    annotating it with the mean and standard deviation of the counts per hour (y-axis).

    :param filepath: Path to the JSON file containing 'images' with 'date_captured'.
    """
    # Load data
    data = read_json_file(filepath)
    if 'images' not in data:
        raise KeyError("JSON does not contain 'images' key.")

    # Extract hours
    hours = []
    for img in data['images']:
        ts = img.get('date_captured')
        if ts:
            try:
                hours.append(pd.to_datetime(ts).hour)
            except Exception:
                continue

    if not hours:
        raise ValueError("No valid 'date_captured' timestamps found.")

    # Count per hour
    df = pd.DataFrame({'hour': hours})
    counts = df['hour'].value_counts().sort_index()
    all_hours = pd.Series(
        index=range(24),
        data=[counts.get(h, 0) for h in range(24)]
    )

    # Compute y-axis stats
    mean_count = all_hours.mean()
    std_count  = all_hours.std()

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=all_hours.index, y=all_hours.values, palette='viridis')

    # Horizontal line at mean count
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=2,
               label=f'Mean count = {mean_count:.2f}')
    # Shaded band for ±1 std
    ax.axhspan(mean_count - std_count, mean_count + std_count,
               alpha=0.2, color='orange',
               label=f'±1 STD = {std_count:.2f}')

    plt.title('Image Sample Counts by Hour of Day')
    plt.xlabel('Hour of Day (0–23)')
    plt.ylabel('Number of Images')
    plt.xticks(range(24))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot sample counts aggregated by hour of day from image metadata, "
                    "with mean and standard deviation of the counts annotated on the y-axis."
    )
    parser.add_argument('filepath', help='Path to the JSON file containing image data.')
    args = parser.parse_args()

    plot_hour_histogram(args.filepath)
