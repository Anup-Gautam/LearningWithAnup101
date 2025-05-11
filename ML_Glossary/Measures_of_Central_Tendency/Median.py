import numpy as np

def calculate_median(numbers):
    """
    Calculate the median of a list of numbers.
    
    Args:
        numbers (list): A list of numbers
        
    Returns:
        float: The median of the numbers
    """
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)

    if n % 2 == 1:
        median_manual = sorted_numbers[n // 2]
    else:
        mid1 = sorted_numbers[n // 2 - 1]
        mid2 = sorted_numbers[n // 2]
        median_manual = (mid1 + mid2) / 2

    median_numpy = np.median(numbers)
    
    return median_manual, median_numpy


if __name__ == "__main__":
    data = [4, 8, 15, 16, 23, 42]
    
    median_manual, median_numpy = calculate_median(data)
    
    print(f"Data: {data}")
    print(f"Median (manual calculation): {median_manual}")
    print(f"Median (NumPy calculation): {median_numpy}")
    print(f"Both methods give the same result: {median_manual == median_numpy}")
