# Automatic Differentiation with Python

This Python script illustrates automatic differentiation, a fundamental technique in machine learning and optimization, using a custom implementation of the forward and backward passes.

## Overview

The script defines a custom `Value` class capable of representing real values with associated gradients. It demonstrates automatic differentiation by computing gradients through arithmetic operations and an example neural network regression scenario.

## Code Structure

- **`Value` Class**: 
  - Represents a real value with an associated gradient.
  - Supports arithmetic operations such as addition, subtraction, multiplication, and exponentiation.
  - Implements the ReLU activation function.
  - Facilitates automatic gradient computation through the backward pass.

- **Main Script**:
  - Reads data from a CSV file named "housing.csv".
  - Initializes weights and input features.
  - Executes gradient descent optimization to minimize the loss function.
  - Displays loss values during optimization.
  - Generates visualizations of the computational graph using Graphviz.

## Requirements

- Python 3.x
- pandas
- graphviz

## How to Use

1. Ensure Python is installed on your system.
2. Install necessary dependencies with pip: `pip install pandas graphviz`.
3. Prepare a CSV file named "housing.csv" containing relevant data.
4. Execute the script.

## Note

- The script assumes the availability of a CSV file named "housing.csv" with appropriate data.
- It relies on the `pandas` library to parse data from the CSV file.
- Visualization of the computational graph is facilitated by the `graphviz` library.

## Disclaimer

This script serves as an educational tool to understand automatic differentiation and its application in neural networks. Adaptations may be necessary for real-world applications, including error handling, input validation, and scalability considerations.
