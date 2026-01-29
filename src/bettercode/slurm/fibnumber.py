import argparse
import sys

def fibonacci(n):
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def main():
    parser = argparse.ArgumentParser(description="Print the ith Fibonacci number.")
    parser.add_argument('-i', type=int, required=True, help="Index of the Fibonacci number (non-negative integer)")
    args = parser.parse_args()
    sys.set_int_max_str_digits(10000000)

    print(fibonacci(args.i))

if __name__ == "__main__":
    main()