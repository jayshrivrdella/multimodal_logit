import numpy as np


def calculate_probabilities(parameters, data, utilities):
    """
    Calculate probabilities for each alternative based on the given parameters and utilities.

    Parameters:
    - parameters: Dictionary containing the β coefficients.
    - data: Dictionary containing the independent variables (X1, X2, Sero, etc.).
    - utilities: List of functions defining deterministic utilities for each alternative.

    Returns:
    Dictionary with keys representing each alternative and values as lists containing the calculated probabilities for each data point.
    """
    # Error handling for mismatched dimensions
    num_alternatives = len(utilities)
    num_data_points = len(data[next(iter(data))])

    if any(len(data[var]) != num_data_points for var in data):
        raise ValueError("Mismatched dimensions between parameters and data points.")

    probabilities = {}

    # Loop through each alternative
    for alternative, utility_function in enumerate(utilities, start=1):
        # Calculate deterministic utility for the current alternative
        utility_values = utility_function(parameters, data)

        # Calculate exponentials of utility values
        exp_utilities = np.exp(utility_values)

        # Calculate probability for the current alternative
        probability = exp_utilities / np.sum(exp_utilities, axis=0)

        # Store probabilities in the dictionary
        probabilities[f'P{alternative}'] = probability.tolist()

    return probabilities

# Given deterministic utilities
def V1(parameters, data):
    return parameters['β01'] + parameters['β1'] * np.array(data['X1']) + parameters['β2'] * np.array(data['X2'])

def V2(parameters, data):
    return parameters['β02'] + parameters['β1'] * np.array(data['X1']) + parameters['β2'] * np.array(data['X2'])

def V3(parameters, data):
    return parameters['β03'] + parameters['β1'] * np.array(data['Sero']) + parameters['β2'] * np.array(data['Sero'])

# Given sample data and parameters
data = {'X1': [2, 3, 5, 7, 1, 8, 4, 5, 6, 7],
        'X2': [1, 5, 3, 8, 2, 7, 5, 9, 4, 2],
        'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

parameters = {'β01': 0.1, 'β1': 0.5, 'β2': 0.5, 'β02': 1, 'β03': 0}

# List of utility functions
utilities = [V1, V2, V3]

try:
    # Calculate probabilities
    probabilities = calculate_probabilities(parameters, data, utilities)

    # Print the result
    print(probabilities)

except ValueError as e:
    print(f"Error: {e}")
