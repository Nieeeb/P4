import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set Seaborn theme for consistent styling
sns.set_theme(style='whitegrid')


def read_json_file(filepath):
    """
    Reads a JSON file and returns the data as a Python object.
    :param filepath: Path to the JSON file.
    :return: Parsed JSON data.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError:
        raise


def plot_date_histogram(filepath, period='monthly'):
    """
    Plots a histogram (bar or line) of image counts aggregated by the given time period,
    styled with Seaborn, including a mean line and annotation.

    :param filepath: JSON file path containing 'images' list with 'date_captured' entries.
    :param period: One of 'daily', 'weekly', or 'monthly'.
    """
    # Load data
    data = read_json_file(filepath)
    if 'images' not in data:
        raise KeyError("JSON does not contain 'images' key.")

    # Build DataFrame
    dates = [img['date_captured'] for img in data['images'] if img.get('date_captured')]
    df = pd.DataFrame({'date_captured': pd.to_datetime(dates)})
    df.set_index('date_captured', inplace=True)

    # Resample rule mapping
    rule_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
    if period not in rule_map:
        raise ValueError("Invalid period. Choose from 'daily', 'weekly', or 'monthly'.")
    rule = rule_map[period]

    # Aggregate counts
    counts = df.resample(rule).size()

    # Compute mean on non-zero bins
    positive_counts = counts[counts > 0]
    mean_count = positive_counts.mean() if not positive_counts.empty else 0

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    if period == 'monthly':
        # Monthly barplot with month names
        months = counts.index.strftime('%B')
        sns.barplot(x=months, y=counts.values, ax=ax)
        ax.set_xticklabels(months, rotation=45, ha='right')
    else:
        # Daily/Weekly lineplot
        sns.lineplot(x=counts.index, y=counts.values, marker='o', ax=ax)

        # Date formatting
        if period == 'daily':
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        else:  # weekly
            locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

    # Add mean line
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=1)
    ax.annotate(
        f"Mean: {mean_count:.2f}",
        xy=(0.95, mean_count),
        xycoords=('axes fraction', 'data'),
        xytext=(0, 5),
        textcoords='offset points',
        color='red',
        ha='right',
        va='bottom',
        fontsize=10,
        backgroundcolor='white'
    )

    # Labels
    ax.set_title(f"Image Counts per {period.capitalize()}")
    ax.set_xlabel(period.capitalize())
    ax.set_ylabel('Number of Images')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot image counts by date period (daily, weekly, monthly) using Seaborn styling."
    )
    parser.add_argument('filepath', help='Path to the JSON file containing image data.')
    parser.add_argument(
        '--period',
        choices=['daily', 'weekly', 'monthly'],
        default='monthly',
        help='Aggregation period for the plot.'
    )
    args = parser.parse_args()

    plot_date_histogram(args.filepath, period=args.period)
