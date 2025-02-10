import inspect
import os
import textwrap
import re

from bs4 import BeautifulSoup


# Define file paths
HEADER_FILE = "header.html"
BANDEAU_FILE = "bandeau.html"
FOOTER_FILE = "footer.html"
CONTENT_DIR = "content"  # Folder where your content files are stored
OUTPUT_DIR = "./"        # Folder to save the generated pages

# Read the contents of a file
def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        print(f"Error: {filepath} does not exist.")
        return ""


class MaxWEnt:
    """
    MaxWEnt (Maximum Weight Entropy) model, extending TensorFlow's Keras Model.
    This model applies a weight entropy regularization term to encourage diversity
    in the learned parameters of a neural network.
    
    Args:
        network (tf.keras.Model): The base pretrained neural network model.
        lambda_ (float): A regularization coefficient for controlling weight entropy.
            Large `lambda_` generally implies more weight entropy.
    """
    
    def __init__(self, network, lambda_=1.):
        pass
    
    def call(self, inputs, training=False, clip=None, seed=None):
        """
        Performs a forward pass through the network.
        
        Args:
            inputs (tf.Tensor): Input data.
            training (bool, optional): Whether the model is in training mode. Defaults to False.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            seed (int, optional): Set the random seed in the layers. It is useful to use the same sampled
                network across multiple batches of data.
        
        Returns:
            tf.Tensor: The output of the network.
        """
        pass

    def _update_clip_in_layers(self, clip):
        """
        Updates the clipping value for all layers that support it.

        Args:
            clip (float or None): Clipping value to be applied to layers.
        """
        pass

    def build(self, input_shape):
        """
        Builds the model by initializing the underlying network if it hasn't been built yet.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        pass

    def fit_svd(self, x, batch_size=32):
        """
        Fit the Singular Value Decomposition (SVD) transition matrix on specific
        layers of the network.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
        """
        pass

    def predict(self, x, batch_size=32, clip=None, seed=None):
        """
        Makes predictions on input data.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            seed (int, optional): Set the random seed in the layers. It is useful to use the same sampled
                network across multiple batches of data.

        Returns:
            np.array: Predictions as a NumPy array.
        """
        pass

    def predict_mean(self, x, batch_size=32, clip=0., n_sample=1):
        """
        Computes the mean prediction over multiple stochastic forward passes.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            n_sample (int, optional): Number of stochastic forward passes. Defaults to 1.

        Returns:
            np.array: The mean prediction.
        """
        pass

    def predict_std(self, x, batch_size=32, clip=None, n_sample=10):
        """
        Computes the standard deviation of predictions over multiple stochastic forward passes.

        Args:
            x (np.array or tf.data.Dataset): The input data.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            clip (float, optional): Clipping value to use on the weight variance.
                If `clip` is `None`, no cliping is applied. If `clip = 0` there is no weight variance.
            n_sample (int, optional): Number of stochastic forward passes. Defaults to 10.

        Returns:
            np.array: The standard deviation of the predictions.
        """
        pass


def extract_args_from_docstring(docstring):
    """
    Extracts the Args section from the docstring and returns it as a list of tuples.
    Each tuple contains (argument_name, description).
    """
    args_section = re.search(r'Args:\s*([\s\S]+?)(Returns|$)', docstring)
    if not args_section:
        return []

    # Extracting the content inside the Args section
    args_content = args_section.group(1).strip()

    # Parsing each argument and its description
    args_list = []
    for line in args_content.splitlines():
        match = re.match(r'(\S+)\s*\(([^)]+)\):\s*(.*)', line.strip())
        if match:
            arg_name = match.group(1)
            arg_type = match.group(2)
            arg_desc = match.group(3)
            args_list.append((arg_name, arg_type, arg_desc))
    
    return args_list


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


def generate_docstring_html(class_obj):
    # Retrieve the class name and docstring
    class_name = class_obj.__name__
    class_doc = class_obj.__doc__

    # Get methods and their docstrings
    methods = inspect.getmembers(class_obj, predicate=inspect.isfunction)

    # Prepare method data (name, docstring, and argument details)
    method_docs = []
    for method_name, method in methods:
        method_doc = method.__doc__
        keep_method = (method_name[0] != "_") and (method_doc is not None)
        if keep_method:
        # Extract argument info
            signature = inspect.signature(method)

            args_list = extract_args_from_docstring(method_doc)

            # Formatting the arguments section as a bullet list
            formatted_args = "<ul>"
            for arg_name, arg_type, arg_desc in args_list:
                formatted_args += f"<li><b>{arg_name} ({arg_type}):</b> {arg_desc}</li>"
            formatted_args += "</ul>"

            method_docs.append({
                'name': method_name,
                'doc': textwrap.dedent(method_doc).strip(),
                'args': str(signature).replace("self, ", ""),  # Remove 'self' from the argument list
                'formatted_args': formatted_args
            })

    header = read_file(HEADER_FILE)
    bandeau_left = read_file(BANDEAU_FILE)  # Left Bandeau
    footer = read_file(FOOTER_FILE)
    
    args_list = extract_args_from_docstring(class_doc)
    formatted_args_object = "<ul>"
    for arg_name, arg_type, arg_desc in args_list:
        formatted_args_object += f"<li><b>{arg_name} ({arg_type}):</b> {arg_desc}</li>"
    formatted_args_object += "</ul>"

    signature = inspect.signature(class_obj.__init__)
    args = str(signature).replace("self, ", "")

    content = f"""
    <div class="object">
    <h1>{class_name}</h1>
    <div class="args">Arguments: {args} </div>
    <div class="formatted_args"> {formatted_args_object} </div>
    </div>
    """
    
    class_doc
    for method in method_docs:
        content += f"""
        <div class="method">
            <h2 id={method['name']}>{method['name']}</h2>
             <div class="args">Arguments: {method['args']} </div>
            <div class="formatted_args">{method['formatted_args']}</div>
        </div>
        """

    toc_html = generate_dynamic_toc(content)
    
    # Inject the ToC into the right bandeau (right sidebar)
    bandeau_right = "<div class='bandeau-right'>" + toc_html + "</div>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ class_name }} Documentation</title>
        <link rel="stylesheet" href="styles.css">
        <link rel="stylesheet" href="notebook_style.css">
        <link rel="stylesheet" href="docstring_style.css">
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

    # Save HTML to a file
    with open(f"{class_name}_documentation.html", "w") as f:
        f.write(html_content)

    print(f"Documentation saved to {class_name}_documentation.html")

# Example usage:
generate_doc_html(MaxWEnt)