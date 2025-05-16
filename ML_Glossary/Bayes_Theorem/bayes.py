def bayes_theorem(prior_A, likelihood_B_given_A, likelihood_B_given_not_A):
    """
    Calculate posterior probability using Bayes' Theorem:
    P(A|B) = [P(B|A) * P(A)] / [P(B|A)*P(A) + P(B|~A)*P(~A)]
    
    Parameters:
    - prior_A: P(A) - prior probability of A being true
    - likelihood_B_given_A: P(B|A) - likelihood of B if A is true
    - likelihood_B_given_not_A: P(B|~A) - likelihood of B if A is false
    
    Returns:
    - posterior: P(A|B) - updated probability of A given B
    """
    prior_not_A = 1 - prior_A
    marginal_B = (likelihood_B_given_A * prior_A) + (likelihood_B_given_not_A * prior_not_A)
    posterior = (likelihood_B_given_A * prior_A) / marginal_B
    return posterior

# ---------------------------
# Example 1:
# Question: You have a headache. What's the probability you have the flu?
# Given:
# P(flu) = 0.05
# P(headache | flu) = 0.90
# P(headache | no flu) = 0.10

prior_flu = 0.05
likelihood_headache_given_flu = 0.90
likelihood_headache_given_no_flu = 0.10

posterior_flu_given_headache = bayes_theorem(prior_flu, likelihood_headache_given_flu, likelihood_headache_given_no_flu)

print("Example 1:")
print(f"Probability of having the flu given a headache: {posterior_flu_given_headache:.2%}\n")

# ---------------------------
# Example 2:
# Question: A test for a rare disease is 99% accurate. If you test positive, what's the chance you actually have the disease?
# Given:
# P(disease) = 0.01
# P(positive | disease) = 0.99
# P(positive | no disease) = 0.01

prior_disease = 0.01
likelihood_positive_given_disease = 0.99
likelihood_positive_given_no_disease = 0.01

posterior_disease_given_positive = bayes_theorem(prior_disease, likelihood_positive_given_disease, likelihood_positive_given_no_disease)

print("Example 2:")
print(f"Probability of having the disease given a positive test: {posterior_disease_given_positive:.2%}")
