import scipy.stats as stats

# Example: Sleep times (in minutes) before and after using a sleep app
before_app = [45, 50, 42, 48, 44, 46, 47, 49, 43, 45, 46, 50, 48, 47, 44]
after_app  = [38, 42, 36, 41, 39, 40, 42, 43, 37, 39, 40, 43, 41, 42, 39]

# Perform paired t-test
t_stat, p_val = stats.ttest_rel(before_app, after_app)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("Result: Significant difference (reject null hypothesis)")
else:
    print("Result: No significant difference (fail to reject null hypothesis)")
