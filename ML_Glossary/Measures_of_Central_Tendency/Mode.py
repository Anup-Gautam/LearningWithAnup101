# Method 1: Using Python's built-in statistics module
import statistics
from collections import Counter

# Example dataset
data = [1, 2, 2, 3, 3, 4]

# --- Method 1: Using statistics.mode ---
try:
    mode_value = statistics.mode(data)
    print(f"[statistics.mode] Single mode: {mode_value}")
except statistics.StatisticsError as e:
    print(f"[statistics.mode] Error: {e}")

# --- Method 2: Handling multimodal datasets using Counter ---
def calculate_modes(data):
    frequency = Counter(data)
    max_count = max(frequency.values())
    modes = [key for key, count in frequency.items() if count == max_count]
    return modes

# Get all modes (safe for unimodal and multimodal data)
modes = calculate_modes(data)
print(f"[Counter] Mode(s): {modes}")
