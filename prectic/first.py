# A simple Python demo program


def greet(name):
    """Function to greet the user."""
    return f"Hello, {name}!"


def add_numbers(a, b):
    """Function to add two numbers."""
    return a + b


# Example usage
if __name__ == "__main__":
    # Greeting
    user_name = "Alice"
    print(greet(user_name))

    # Adding numbers
    num1 = 5
    num2 = 10
    print(f"The sum of {num1} and {num2} is {add_numbers(num1, num2)}.")
