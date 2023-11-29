import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt

all_adc2 = []

chunk_size = 50000
for chunk in pd.read_csv("water.csv", chunksize=chunk_size):
    all_adc2.extend(chunk['adc2'].to_numpy())

all_adc2 = np.array(all_adc2)

filtered_adc2 = all_adc2

rolling_avg = pd.Series(filtered_adc2).rolling(window=50000, min_periods=1).mean()
filtered_adc2 = filtered_adc2 - rolling_avg
# Set negative values in filtered_adc2 to 0
filtered_adc2 = np.where(filtered_adc2 < 0, 0, filtered_adc2)

# Smoothing parameters
window_size, poly_order = 1000, 5

# Apply smoothing
smooth_adc2 = savgol_filter(filtered_adc2, window_size, poly_order)
#smooth_adc2 = filtered_adc2

# Calculate optimal minimum thresholds
optimal_min_threshold2 = np.mean(smooth_adc2) + 4 * np.std(smooth_adc2)

# Detect initial peaks using minimum threshold
initial_peaks2, _ = find_peaks(smooth_adc2, height=optimal_min_threshold2)

# Calculate optimal maximum thresholds based on the initial peaks
optimal_max_threshold2 = np.mean(smooth_adc2[initial_peaks2]) + 4.3 * np.std(smooth_adc2[initial_peaks2])

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
