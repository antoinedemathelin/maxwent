import os
import re
import subprocess
from bs4 import BeautifulSoup

# Define file paths
HEADER_FILE = "header.html"
BANDEAU_FILE = "bandeau.html"
FOOTER_FILE = "footer.html"
CONTENT_DIR = "content"  # Folder where your content files are stored
OUTPUT_DIR = "./"        # Folder to save the generated pages

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
    # List all .html and .ipynb files in the content directory
    return [f for f in os.listdir(content_dir) if f.endswith(('.html', '.ipynb'))]

# Convert a Jupyter notebook file to HTML using nbconvert
def convert_notebook_to_html(notebook_file):
    try:
        # Run the nbconvert command to convert the notebook to HTML
        subprocess.run(['jupyter', 'nbconvert', '--to', 'html', notebook_file], check=True)
        html_file = notebook_file.replace('.ipynb', '.html')
        return html_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting notebook {notebook_file}: {e}")
        return None

# Remove "In [X]:" labels from the HTML content
def remove_input_prompts(input_html_path):
    with open(input_html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # Find all div elements with the class 'jp-InputPrompt' and remove them
    for element in soup.find_all('div', class_='jp-InputPrompt'):
        element.decompose()  # Removes the entire <div> element

    # Save the modified HTML
    with open(input_html_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))

# Delete the temporary HTML file generated by nbconvert
def delete_temp_html_file(html_file):
    if os.path.exists(html_file):
        os.remove(html_file)
        print(f"Temporary HTML file {html_file} deleted.")

# Function to generate dynamic Table of Contents (ToC) from HTML content
def generate_dynamic_toc(content_html):
    soup = BeautifulSoup(content_html, 'html.parser')
    toc_items = []
    
    # Look for all headers <h1>, <h2>, <h3> etc.
    for header in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(header.name[1])  # Get the heading level (h1 -> 1, h2 -> 2, etc.)
        heading_text = header.get_text(strip=True)
        
        # Remove unwanted characters like ¶ using regex (if present)
        heading_text = re.sub(r'¶', '', heading_text)  # Remove the ¶ character
        
        # Clean up any other unwanted characters that might have been copied over
        heading_text = re.sub(r'[^\x00-\x7F]+', '', heading_text)  # Remove non-ASCII characters
        
        # Create an anchor-friendly version of the text (lowercase, replace spaces with hyphens)
        header_id = header.get('id')
        if header_id is not None:
            anchor = header["id"]
        else:
            anchor = heading_text.replace(' ', '-').lower()  
        
        # Add the header's ID to make it linkable and point to it in the ToC
        toc_items.append((level, anchor, heading_text))
    
    # Generate ToC HTML with internal links to the headers
    toc_html = '<div class="toc-container"><h3>Table of Contents</h3><ul>'
    for level, anchor, heading_text in toc_items:
        toc_html += f'<li style="margin-left: {5 + (level-2) * 20}px; font-size: {1. - 0.1 * (level-2)}rem;"><a href="#{anchor}">{heading_text}</a></li>'
    toc_html += '</ul></div>'
    
    return toc_html

# General function to generate a page
def generate_page(content_file, output_file):
    # Read the individual components
    header = read_file(HEADER_FILE)
    bandeau_left = read_file(BANDEAU_FILE)  # Left Bandeau
    bandeau_right = "<div class='bandeau-right'></div>"  # Right Bandeau (to insert ToC)
    footer = read_file(FOOTER_FILE)
    
    # Read content file (convert if it's a notebook)
    with open(content_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove the anchor link (¶) symbol from headers in the content
    content = re.sub(r'<a class="anchor-link" href="[^"]+">¶</a>', '', content)

    # Extract the title from the content file name (without extension)
    title = os.path.basename(content_file).split('.')[0].replace('-', ' ').title()
    title = title.replace("_", " ")

    if "Example" in content_file:
    # Replace the placeholder in the header with the actual title
        header = header.replace("Maximum Weight Entropy", title)

    # Generate the Table of Contents dynamically
    toc_html = generate_dynamic_toc(content)
    
    # Inject the ToC into the right bandeau (right sidebar)
    bandeau_right = "<div class='bandeau-right'>" + toc_html + "</div>"
    
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
  {bandeau_left}
  {bandeau_right}
  {header}
  <main class="main">
    {content}
  </main>
  {footer}
</div>
<script src="custom.js"></script>
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
        content_file_path = os.path.join(CONTENT_DIR, content_file)  # Full path to content file
        
        # Handle .ipynb files by converting them to HTML and removing input prompts
        if content_file.endswith('.ipynb'):
            html_file = convert_notebook_to_html(content_file_path)
            if html_file:
                # Remove "In [X]:" labels from the converted HTML
                remove_input_prompts(html_file)
                content_file_path = html_file  # Use the newly generated HTML
        
        # Define the output file path
        output_file = os.path.join(OUTPUT_DIR, content_file.replace(".ipynb", ".html"))
        generate_page(content_file_path, output_file)

        # Delete the temporary HTML file generated by nbconvert after page generation
        if content_file.endswith('.ipynb') and os.path.exists(content_file_path):
            delete_temp_html_file(content_file_path)

# Run the script to generate all pages
if __name__ == "__main__":
    generate_all_pages()  # Generate all pages based on the HTML files in CONTENT_DIR
