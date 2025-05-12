import matplotlib.pyplot as plt
from collections import Counter

# Sample dataset
data = [1, 2, 2, 3, 3, 4, 4, 4, 5, 6]

# Count the frequency of each value
frequency = Counter(data)

# Find the highest frequency (i.e., the mode count)
max_freq = max(frequency.values())
modes = [val for val, count in frequency.items() if count == max_freq]

# Prepare data for plotting
x = list(frequency.keys())
y = list(frequency.values())
colors = ['orange' if val in modes else 'skyblue' for val in x]

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(x, y, color=colors)
for i, val in enumerate(y):
    plt.text(x[i], val + 0.1, str(val), ha='center')

plt.title("Frequency Distribution and Mode(s)")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
