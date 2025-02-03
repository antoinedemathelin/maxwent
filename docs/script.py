# Import necessary modules
import os

# Define file paths
HEADER_FILE = "header.html"
BANDEAU_FILE = "bandeau.html"
FOOTER_FILE = "footer.html"
CONTENT_FILE = "content_home.html"  # You can replace this for different pages
OUTPUT_FILE = "index.html"          # The output file

# Read the contents of a file
def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print(f"Error: {filepath} does not exist.")
        return ""

# Generate the page
def generate_page():
    # Read the individual components
    header = read_file(HEADER_FILE)
    bandeau = read_file(BANDEAU_FILE)
    footer = read_file(FOOTER_FILE)
    content = read_file(CONTENT_FILE)
    
    # Combine the components
    full_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
  <title>Home</title>
</head>
<body>
<div class="page-container">
  {bandeau}
  {header}
  <main class="main">
    {content}
  </main>
  {footer}
</div>
</body>
</html>
"""

    # Write the combined HTML to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
        output_file.write(full_page)
    print(f"Page generated successfully: {OUTPUT_FILE}")

# Run the script
if __name__ == "__main__":
    generate_page()
