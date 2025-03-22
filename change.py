# This script reads an ARFF file, processes the food attribute columns,
# and writes the updated data to a new ARFF file.
# Specifically, for the attributes:
#    food_left, food_up, food_right, food_down (columns 5, 6, 7, and 8 in 1-indexed terms,
#    or indices 4,5,6,7 in 0-indexed Python lists),
# if the value is greater than 0, it is replaced by 1, otherwise it remains 0.

# Define the input and output file names
input_file = './Arffs/test_pruned.arff'
output_file = './Arffs/test_pruned_binary.arff'

# Read the ARFF file line by line
with open(input_file, 'r') as file:
    lines = file.readlines()

# Separate header and data sections
header_lines = []
data_lines = []
data_section = False

for line in lines:
    stripped_line = line.strip()
    # Detect the start of the data section
    if stripped_line.lower().startswith("@data"):
        data_section = True
        header_lines.append(line)  # Include the @data line in the header
        continue
    # If not in the data section, add line to header
    if not data_section:
        header_lines.append(line)
    else:
        # Skip any empty lines in the data section
        if stripped_line == "":
            continue
        data_lines.append(line)

# Process each data line to update food attributes
processed_data_lines = []
for line in data_lines:
    # Split the line by commas to get individual attribute values
    parts = line.strip().split(',')
    # Update columns for food_left, food_up, food_right, and food_down (indices 4,5,6,7)
    for idx in [4, 5, 6, 7]:
        # Convert the value to float to allow numeric comparison
        try:
            value = float(parts[idx])
        except ValueError:
            value = 0  # Default to 0 if conversion fails
        # Replace with "1" if value > 0, else "0"
        parts[idx] = "1" if value > 0 else "0"
    # Reconstruct the modified data line and add a newline character
    new_line = ",".join(parts) + "\n"
    processed_data_lines.append(new_line)

# Write the updated header and data sections to a new ARFF file
with open(output_file, 'w') as file:
    for line in header_lines:
        file.write(line)
    for line in processed_data_lines:
        file.write(line)

print("Modified ARFF file has been saved as", output_file)