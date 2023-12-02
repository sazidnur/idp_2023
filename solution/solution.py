import click

def solution(input_csv, output_csv=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, savgol_filter, peak_widths

    all_adc2 = []

    chunk_size = 50000
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        all_adc2.extend(chunk["adc2"].to_numpy())

    # Convert lists to numpy arrays
    all_adc2 = np.array(all_adc2)
    filtered_adc2 = all_adc2

    # optimally generated from water signal
    min_possible_water_peaks_threshold = 19.999644331937088
    max_possible_water_peaks_threshold = 46.17379768744214

    # optimally generated from water signal before baseline fix
    #actual_signal_min_possible_water_peaks_threshold = 702.736550291642
    actual_signal_max_possible_water_peaks_threshold = 733.4351469568594

    # Fix Baseline
    rolling_avg = pd.Series(filtered_adc2).rolling(window=50000, min_periods=1).mean()
    filtered_adc2 = filtered_adc2 - rolling_avg

    # Set negative values in filtered_adc2 to 0
    filtered_adc2 = np.where(filtered_adc2 < 0, 0, filtered_adc2)


    # Smoothing fix parameters
    window_size, poly_order = 1000, 5

    # Apply smoothing
    filtered_adc2 = savgol_filter(filtered_adc2, window_size, poly_order)

    # Find peaks
    final_peaks2, _ = find_peaks(filtered_adc2, height=min_possible_water_peaks_threshold)

    # Generating Time for Plot
    time_axis = np.linspace(0, all_adc2.shape[0] / 50000, all_adc2.shape[0])

    # Create plots for ADC2
    fig2, ax2 = plt.subplots(figsize=(100, 10))
    ax2.plot(time_axis, all_adc2, label="Signal")
    ax2.set_title("Water and Tissue Peak Detection")

    prev_peak_value = None
    prev_peak_index = None
    peak_indices = []
    eps_value = 1e-2
    eps_index = 2000

    def get_peak_color(value, actual_value, min_threshold, max_threshold):
        if (
            value >= min_threshold
            and value <= max_threshold
            and actual_value <= actual_signal_max_possible_water_peaks_threshold
        ):
            return "r"
        else:
            return "g"


    for peak in final_peaks2:
        color = get_peak_color(
            filtered_adc2[peak],
            all_adc2[peak],
            min_possible_water_peaks_threshold,
            max_possible_water_peaks_threshold,
        )

        if color != "none":
            # Add additional logic to ensure the peak is not too close to the previous one
            if (
                prev_peak_value is None
                or abs(filtered_adc2[peak] - prev_peak_value) >= eps_value
            ):
                if prev_peak_index is None or abs(peak - prev_peak_index) >= eps_index:
                    peak_indices.append(peak)
                    prev_peak_value = filtered_adc2[peak]
                    prev_peak_index = peak


    for peak in peak_indices:
        color = get_peak_color(
            filtered_adc2[peak],
            all_adc2[peak],
            min_possible_water_peaks_threshold,
            max_possible_water_peaks_threshold,
        )
        peak_time = time_axis[peak]
        peak_value = all_adc2[peak]
        ax2.annotate(
            "",
            xy=(peak_time, peak_value),
            xytext=(peak_time, peak_value + 50),  # 50 units above the peak
            arrowprops=dict(arrowstyle="->", color=color),
        )
    ax2.legend()
    fig2.savefig("peak_detection.png", dpi=300)

    # Calculate peak widths
    widths, width_heights, left_ips, right_ips = peak_widths(
        all_adc2, final_peaks2, rel_height=0.5
    )

    widths_time = widths / 50000  # Convert to time

    # Prepare data for result CSV
    data = []

    for i, peak in enumerate(final_peaks2):
        peak_time = time_axis[peak]
        start_time = peak_time - (widths_time[i] / 2)
        end_time = peak_time + (widths_time[i] / 2)

        # Round times to five decimal places
        start_time = round(start_time, 5)
        end_time = round(end_time, 5)

        # Ensure start time is not negative
        start_time = max(start_time, 0)

        color = get_peak_color(
            filtered_adc2[peak],
            all_adc2[peak],
            min_possible_water_peaks_threshold,
            max_possible_water_peaks_threshold,
        )
        label = "water" if color == "r" else "tissue"

        data.append([start_time, end_time, label])

    df = pd.DataFrame(data, columns=["startTime", "endTime", "label"])
    
    if output_csv:
        df.to_csv(output_csv, index=False)

    df["width"] = (
        widths_time / 2
    )  # since width_time is the full width, we divide by 2 for half-width

    # Calculate average widths
    avg_widths = df.groupby("label")["width"].mean()

    print("\n")
    print("Average Widths:")
    print(avg_widths)


def generate_water_threshold(input_file, non_baseline_fixed):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks, savgol_filter

    all_adc2 = []

    chunk_size = 50000
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        all_adc2.extend(chunk['adc2'].to_numpy())

    all_adc2 = np.array(all_adc2)

    filtered_adc2 = all_adc2

    multiply_min = 2.5
    multiply_max = 5

    if non_baseline_fixed is False:
        rolling_avg = pd.Series(filtered_adc2).rolling(window=50000, min_periods=1).mean()
        filtered_adc2 = filtered_adc2 - rolling_avg

        multiply_min = 4
        multiply_max = 4.3
        
        # Set negative values in filtered_adc2 to 0
        filtered_adc2 = np.where(filtered_adc2 < 0, 0, filtered_adc2)
    

    # Smoothing parameters
    window_size, poly_order = 1000, 5

    # Apply smoothing
    smooth_adc2 = savgol_filter(filtered_adc2, window_size, poly_order)
    #smooth_adc2 = filtered_adc2

    # Calculate optimal minimum thresholds
    optimal_min_threshold2 = np.mean(smooth_adc2) + multiply_min * np.std(smooth_adc2)

    # Detect initial peaks using minimum threshold
    initial_peaks2, _ = find_peaks(smooth_adc2, height=optimal_min_threshold2)

    # Calculate optimal maximum thresholds based on the initial peaks
    optimal_max_threshold2 = np.mean(smooth_adc2[initial_peaks2]) + multiply_max * np.std(smooth_adc2[initial_peaks2])

    # Detect peaks using both minimum and maximum thresholds
    final_peaks2, _ = find_peaks(smooth_adc2, height=(optimal_min_threshold2, optimal_max_threshold2))

    # Create plots for ADC2
    fig2, ax2 = plt.subplots(figsize=(100, 10))
    ax2.plot(smooth_adc2)
    ax2.axhline(y=optimal_min_threshold2, color='r', linestyle='--', label='Min Threshold')
    ax2.axhline(y=optimal_max_threshold2, color='g', linestyle='--', label='Max Threshold')
    ax2.scatter(final_peaks2, smooth_adc2[final_peaks2], color='r')
    ax2.legend()
    fig2.savefig('water_peak.png')

    ##print(f"Optimal Max Threshold 1 for entire data: {optimal_max_threshold1}")
    print(f"Optimal Min Threshold (adc2): {optimal_min_threshold2}")
    print(f"Optimal Max Threshold (adc2): {optimal_max_threshold2}")


# Command group
@click.group()
def cli():
    """A tool for signal analysis."""
    pass

# Analyze command
@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--save', 'output_file', type=click.STRING, help="Save the analysis results to a csv file.")
def analyze(input_file, output_file):
    """Analyze the signal from the provided signal. The result will be save on peak_detection.png file."""
    if output_file and not output_file.endswith('.csv'):
        output_file += '.csv'
    # Your code for the analyze function
    solution(input_file, output_file)

# Water command
@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--non-baseline-fixed', is_flag=True, help="Indicate if non-baseline-fixed processing is required for generating threshold.")
def water(input_file, non_baseline_fixed):
    """Generate Optimal MIN-MAX Water Peak Threshold"""
    generate_water_threshold(input_file, non_baseline_fixed)

# Main entry point
if __name__ == '__main__':
    cli()