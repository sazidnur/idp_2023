# Overview

`solution.py` is a Python command-line tool designed for signal analysis. It offers functionalities to analyze signal data from CSV files and generate optimal MIN-MAX water peak thresholds. The tool is equipped with two primary commands: `analyze` and `water`.

<br/>

# Prerequisites

\- Python (Version 3.x)

\- Required Python libraries: ***pandas, numpy, matplotlib, scipy, click.*** 

To install the required libraries, run the following command in your terminal:

```
pip install -r requirements.txt
```

This will install all the dependencies listed in the `requirements.txt` file.

<br/>

# Commands

`analyze`

\- Description: Analyzes signal data from a provided CSV file and optionally saves the analysis results to another CSV file.

\- Usage:

```
python main.py analyze INPUT\_FILE [--save OUTPUT\_FILE]
```

- `INPUT\_FILE`: Path to the CSV file containing signal data.

- `--save OUTPUT\_FILE` (optional): Path to save the analysis results. If not provided, results will not be saved.
---------------------------
`water`

\- Description: Generates optimal MIN-MAX water peak thresholds based on the provided signal data.

\- Usage:

```
python main.py water INPUT\_FILE [--non-baseline-fixed]
```

- `INPUT\_FILE`: Path to the CSV file containing signal data.

- `--non-baseline-fixed` (optional flag): If provided, processes the signal without baseline correction.

<br/>

# Usage Examples

**1.  Analyze Signal Data:**

- Command:

```
python main.py analyze "path/to/signal.csv"
```

\- This command analyzes the signal data in `signal.csv` and generates a ***peak\_detection.png*** image file to show the plot.

**2. Analyze and Save Results in csv file:**

\- Command:

```
python main.py analyze "path/to/signal.csv" --save "results.csv"
```

\- This command analyzes the signal data, generates a ***peak\_detection.png*** image file to show the plot and saves the results to `results.csv`.

**3. Generate Water Peak Thresholds:**

\- Command:

```
python main.py water "path/to/signal.csv"
```

\- Generates threshold and shows in console and saves the water peaks in ***water\_peak.png*** image file based on the provided signal data.

**4. Generate Water Peak Thresholds without Baseline Correction:**

\- Command:

```
python main.py water "path/to/signal.csv" --non-baseline-fixed
```

\- Generates threshold without fixing baseline and shows in console and saves the water peaks in ***water\_peak.png*** image file based on the provided signal data.

**5. Generate Water Peak Thresholds without Baseline Correction:**

\- Command:

```
python main.py --help<br>python main.py water --help<br>python main.py analyze --help
```

\- Each command will show help window for related their section

<br/>

# Notes

\- Ensure that the CSV file paths are correctly specified. If a file path contains spaces, enclose it in quotes.

\- The `--save` option in the `analyze` command is optional. If not used, the analysis results will only be displayed and not saved.
