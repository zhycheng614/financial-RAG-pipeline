import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def convert_single_html(html_path, converted_dir, browser):
    """
    Convert a single HTML file to PDF.
    
    Args:
        html_path (Path): Path to the HTML file
        converted_dir (Path): Directory to save the PDF
        browser: Playwright browser instance
    
    Returns:
        str: Path to the generated PDF file
    """
    # Create the PDF filename (same name as HTML but with .pdf extension)
    pdf_filename = html_path.stem + ".pdf"
    pdf_path = converted_dir / pdf_filename
    
    # Convert HTML to PDF using Playwright
    print(f"Converting {html_path.name}...")
    
    page = browser.new_page()
    
    # Load the HTML file
    page.goto(f"file:///{html_path.absolute().as_posix()}")
    
    # Generate PDF with nice defaults
    page.pdf(
        path=str(pdf_path),
        format='A4',
        print_background=True,
        margin={
            'top': '20mm',
            'right': '20mm',
            'bottom': '20mm',
            'left': '20mm'
        }
    )
    
    page.close()
    
    print(f"  ✓ Saved to: {pdf_path}")
    return str(pdf_path)


def html_to_pdf(input_path):
    """
    Convert HTML file(s) to PDF and save in a 'converted' folder.
    
    - If input_path is a file: converts that single file
    - If input_path is a directory: converts all .html files in it (recursively)
    
    Args:
        input_path (str): Path to the HTML file or directory
    
    Returns:
        list: List of paths to the generated PDF files
    """
    # Convert to Path object for easier manipulation
    input_path = Path(input_path)
    
    # Check if the path exists
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    pdf_paths = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Handle directory input
        if input_path.is_dir():
            # Find all HTML files in the directory (recursively)
            html_files = list(input_path.rglob("*.html")) + list(input_path.rglob("*.htm"))
            
            if not html_files:
                browser.close()
                raise ValueError(f"No HTML files found in directory: {input_path}")
            
            print(f"\nFound {len(html_files)} HTML file(s) in {input_path}")
            print("=" * 60)
            
            # Create 'converted' folder in the input directory
            converted_dir = input_path / "converted"
            converted_dir.mkdir(exist_ok=True)
            
            # Convert each HTML file
            for i, html_file in enumerate(html_files, 1):
                print(f"\n[{i}/{len(html_files)}] ", end="")
                try:
                    pdf_path = convert_single_html(html_file, converted_dir, browser)
                    pdf_paths.append(pdf_path)
                except Exception as e:
                    print(f"  ✗ Error converting {html_file.name}: {e}")
        
        # Handle single file input
        else:
            if input_path.suffix.lower() not in ['.html', '.htm']:
                browser.close()
                raise ValueError(f"File must be an HTML file (.html or .htm): {input_path}")
            
            # Get the directory containing the HTML file
            html_dir = input_path.parent
            
            # Create 'converted' folder in the same directory
            converted_dir = html_dir / "converted"
            converted_dir.mkdir(exist_ok=True)
            
            print()
            pdf_path = convert_single_html(input_path, converted_dir, browser)
            pdf_paths.append(pdf_path)
        
        browser.close()
    
    return pdf_paths


if __name__ == "__main__":
    # Example usage: input HTML file or directory path
    input_path = input("Enter the path to an HTML file or directory: ").strip('"').strip("'")
    
    try:
        output_pdfs = html_to_pdf(input_path)
        
        print("\n" + "=" * 60)
        print(f"✓ Success! Converted {len(output_pdfs)} file(s)")
        print("=" * 60)
        
        if len(output_pdfs) == 1:
            print(f"\nPDF created at:\n{output_pdfs[0]}")
        else:
            print(f"\nAll PDFs saved to the 'converted' folder")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

