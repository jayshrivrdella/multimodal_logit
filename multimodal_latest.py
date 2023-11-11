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
        utility_values = utility_function(parameters, data, alternative)

        # Calculate exponentials of utility values
        exp_utilities = np.exp(utility_values)

        # Calculate probability for the current alternative
        probability = exp_utilities / np.sum(exp_utilities, axis=0)

        # Store probabilities in the dictionary
        probabilities[f'P{alternative}'] = probability.tolist()

    return probabilities

def generic_utility_function(params, data, num_alternatives):
    """
    Generic utility function to compute V1, V2, V3, ..., Vn.

    Parameters:
    - params: Dictionary containing the β coefficients.
    - data: Dictionary containing the independent variables.
    - num_alternatives: Number of alternatives.

    Returns:
    Deterministic utility values for each alternative.
    """
    # Initialize an array to store utility values for each alternative
    utility_values = np.zeros((num_alternatives, len(data[next(iter(data))])))

    # Loop through each alternative
    for alternative in range(1, num_alternatives + 1):
        # Construct the utility function dynamically based on the provided parameters
        utility_expr = f"{params[f'β0{alternative}']}"
        for i in range(1, len(params)):
            utility_expr += f" + {params[f'β{i}']} * np.array(data['X{i}'])"

        # Evaluate the utility expression and store the results
        utility_values[alternative - 1] = eval(utility_expr)

    return utility_values

def get_user_input():
    """
    Get user input for parameters and independent variables.

    Returns:
    Tuple (parameters, data, utilities, num_alternatives)
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

    # Get the number of alternatives from the user
    num_alternatives = int(input("Enter the number of alternatives: "))

    # Compute utilities directly based on generic utility function
    utilities = [generic_utility_function]

    return parameters, data, utilities, num_alternatives

if __name__ == "__main__":
    try:
        # Get user input
        parameters, data, utilities, num_alternatives = get_user_input()

        # Calculate probabilities
        probabilities = calculate_probabilities(parameters, data, utilities)

        # Print the result
        print(probabilities)

    except ValueError as e:
        print(f"Error: {e}")
