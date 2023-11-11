import numpy as np

def calculate_probabilities(parameters, data, utilities):
    """
    Calculate probabilities for each alternative based on the given parameters and utilities.

    Parameters:
    - parameters: Dictionary containing the β coefficients.
    - data: Dictionary containing the independent variables.
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

def get_user_input():
    """
    Get user input for parameters and independent variables.

    Returns:
    Tuple (parameters, data, utilities)
    """
    # Get the number of independent variables
    num_variables = int(input("Enter the number of independent variables: "))

    # Get parameters from the user
    parameters = {}
    for i in range(num_variables + 1):  # +1 for the intercept
        param_key = f'β{i}'
        parameters[param_key] = float(input(f"Enter the value for {param_key}: "))

    # Get the number of entries for each independent variable
    num_entries = int(input("Enter the number of entries for each independent variable: "))

    # Get data from the user
    data = {}
    for var_num in range(1, num_variables + 1):
        var_name = f'X{var_num}'
        data[var_name] = [float(input(f"Enter the value for {var_name}[{i + 1}]: ")) for i in range(num_entries)]

    # Compute utilities directly based on deterministic utility formulas
    utilities = [
        lambda params, d: params['β01'] + params['β1'] * np.array(d['X1']) + params['β2'] * np.array(d['X2']),
        lambda params, d: params['β02'] + params['β1'] * np.array(d['X1']) + params['β2'] * np.array(d['X2']),
        lambda params, d: params['β03'] + params['β1'] * np.array(d['Sero']) + params['β2'] * np.array(d['Sero'])
    ]

    return parameters, data, utilities

if __name__ == "__main__":
    try:
        # Get user input
        parameters, data, utilities = get_user_input()

        # Calculate probabilities
        probabilities = calculate_probabilities(parameters, data, utilities)

        # Print the result
        print(probabilities)

    except ValueError as e:
        print(f"Error: {e}")
