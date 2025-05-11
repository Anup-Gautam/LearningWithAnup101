import numpy as np

def calculate_mean(numbers):
    """
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers (list): A list of numbers
        
    Returns:
        float: The arithmetic mean of the numbers
    """
  
    mean_manual = sum(numbers) / len(numbers)
    mean_numpy = np.mean(numbers)
    
    return mean_manual, mean_numpy


if __name__ == "__main__":
    data = [4, 8, 15, 16, 23, 42]
    
    mean_manual, mean_numpy = calculate_mean(data)
    
    print(f"Data: {data}")
    print(f"Mean (manual calculation): {mean_manual}")
    print(f"Mean (NumPy calculation): {mean_numpy}")
    print(f"Both methods give the same result: {mean_manual == mean_numpy}")
    # this is testing for contribution in open source