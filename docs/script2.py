# Import necessary modules
import os

# Define file paths
HEADER_FILE = "header.html"
BANDEAU_FILE = "bandeau.html"
FOOTER_FILE = "footer.html"
CONTENT_DIR = "content"  # Folder where your content HTML files are stored
OUTPUT_DIR = "./"    # Folder to save the generated pages

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the contents of a file
def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print(f"Error: {filepath} does not exist.")
        return ""

# Generate the list of content files by scanning the content directory
def get_content_files(content_dir):
    # List all .html files in the content directory
    return [f for f in os.listdir(content_dir) if f.endswith('.html')]

# General function to generate a page
def generate_page(content_file, output_file):
    # Read the individual components
    header = read_file(HEADER_FILE)
    bandeau = read_file(BANDEAU_FILE)
    footer = read_file(FOOTER_FILE)
    content = read_file(content_file)

    # Combine the components to generate the full page
    full_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="notebook_style.css">
  <title>{output_file.replace('.html', '').title()}</title>
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
    with open(output_file, 'w', encoding='utf-8') as output_file_obj:
        output_file_obj.write(full_page)
    print(f"Page generated successfully: {output_file}")

# Generate pages for each content file
def generate_all_pages():
    # Get the list of content files from the content directory
    content_files = get_content_files(CONTENT_DIR)
    
    # Generate a page for each content file
    for content_file in content_files:
        # Define the output file path
        output_file = os.path.join(OUTPUT_DIR, content_file.replace(".html", "_generated.html"))
        content_file_path = os.path.join(CONTENT_DIR, content_file)  # Full path to content file
        generate_page(content_file_path, output_file)

# Run the script to generate all pages
if __name__ == "__main__":
    generate_all_pages()  # Generate all pages based on the HTML files in CONTENT_DIR
