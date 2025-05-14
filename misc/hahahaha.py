#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Plot the 2nd column of a CSV against its row index and save as an image."
    )
    parser.add_argument(
        "csv_file",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_plot.png",
        help="Filename for the saved image (default: output_plot.png)"
    )
    args = parser.parse_args()

    # 1. Read CSV
    df = pd.read_csv(args.csv_file)

    # 2. Extract the 2nd column by position
    #    (iloc[:,1] is zero-based: 0=first, 1=second)
    series = df.iloc[:, 1]

    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, marker='o', linestyle='-')
    plt.xlabel("Index")
    plt.ylabel(df.columns[1])
    plt.title(f"{df.columns[1]} vs. Row Index")
    plt.grid(True)
    plt.tight_layout()

    # 4. Save to file
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as {args.output}")

if __name__ == "__main__":
    main()
