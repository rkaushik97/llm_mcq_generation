from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape


def process_template(template_file: str, context: Dict[str, Any], template_dir:str="./") -> str:
    # Create a Jinja environment with a specified loader and autoescaping
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath=template_dir),
        autoescape=select_autoescape()  # Automatically escape HTML/XML content
    )

    # Load the specified template
    template = jinja_env.get_template(template_file)

    # Render the template with the provided context
    return template.render(**context)
