#!/usr/bin/env python3
"""Display the accuracy CSV without modifying it or running models."""

import argparse

from results_store import ROOT, SUMMARY_NAME, print_summary, read_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", help="all or a model name in the CSV")
    args = parser.parse_args()
    rows = read_summary()
    if args.model != "all" and args.model not in {row["model"] for row in rows}:
        parser.error("No saved result for this model")
    print_summary(rows, args.model)
    print("Summary CSV: " + str(ROOT / SUMMARY_NAME))


if __name__ == "__main__":
    main()
