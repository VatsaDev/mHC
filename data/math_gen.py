import random
import os

# Configuration
FILENAME = "pemdas_expressions.txt"
TOTAL_LINES = 10_000_000
BATCH_SIZE = 100_000  # Write 100k lines at a time to optimize I/O

operators = ['+', '-', '*', '/', '**']

def generate_number():
    """Generates a number between 2 and 5 digits."""
    return str(random.randint(10, 99999))

def generate_expression():
    """Generates a random PEMDAS string."""
    # We create a mix of simple and complex structures
    # Structure: (num op num) op (num op num)
    
    def part():
        if random.random() > 0.7:
            # Single number
            return generate_number()
        else:
            # Number with an exponent or simple op
            op = random.choice(operators)
            n1 = generate_number()
            # Keep exponents small (2-4) so expressions look 'standard' 
            # even if the base is 5 digits
            n2 = str(random.randint(2, 4)) if op == '**' else generate_number()
            return f"({n1} {op} {n2})"

    # Combine parts into a full line
    op_mid = random.choice(['+', '-', '*', '/'])
    line = f"{part()} {op_mid} {part()}"
    
    # Randomly wrap the whole thing or add one more layer
    if random.random() > 0.5:
        line = f"({line}) {random.choice(operators)} {generate_number()}"
        
    return line

def main():
    print(f"Starting generation of {TOTAL_LINES} lines...")
    print(f"Output file: {FILENAME}")

    count = 0
    try:
        with open(FILENAME, "w") as f:
            buffer = []
            for i in range(1, TOTAL_LINES + 1):
                buffer.append(generate_expression() + "\n")
                
                # Periodically write the buffer to disk
                if i % BATCH_SIZE == 0:
                    f.writelines(buffer)
                    buffer = []
                    percent = (i / TOTAL_LINES) * 100
                    print(f"Progress: {percent:.1f}% ({i} lines generated)", end="\r")
            
            # Write any remaining lines in the buffer
            if buffer:
                f.writelines(buffer)
                
        print(f"\nSuccess! File '{FILENAME}' created.")
        print(f"Approximate file size: {os.path.getsize(FILENAME) / (1024**3):.2f} GB")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
