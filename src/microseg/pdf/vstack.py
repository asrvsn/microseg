'''
Vertically stack PDFs using width of first page
'''

import fitz  # pymupdf
import os
import tempfile
from typing import List, Tuple
from .trim import trim_pdf

def vstack_pdfs(pdf1_path: str, pdf2_path: str, output_path: str, trim1: str = "", trim2: str = "", fudge_factor: float = 1.0, verbose: bool = False):
    """
    Stack two PDFs vertically with pdf1 on top and pdf2 on bottom.
    The output width is determined by the width of the first PDF (after trimming).
    
    Args:
        pdf1_path: Path to the first (top) PDF
        pdf2_path: Path to the second (bottom) PDF
        output_path: Path for the output stacked PDF
        trim1: LaTeX-style trim for the first PDF: 'left bottom right top' in mm
        trim2: LaTeX-style trim for the second PDF: 'left bottom right top' in mm
        fudge_factor: Factor to multiply heights by to reduce unwanted spacing (0.9-0.99)
    """
    assert 0 < fudge_factor <= 1.0, f"Fudge factor must be between 0 and 1, got {fudge_factor}"
    # Check if files exist
    for path in [pdf1_path, pdf2_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDF file not found: {path}")
    
    doc1 = None
    doc2 = None
    new_doc = None
    temp_files = []
    
    try:
        pdf1_to_open = pdf1_path
        if trim1:
            fd, temp_pdf1_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            trim_pdf(pdf1_path, temp_pdf1_path, trim1)
            pdf1_to_open = temp_pdf1_path
            temp_files.append(temp_pdf1_path)

        pdf2_to_open = pdf2_path
        if trim2:
            fd, temp_pdf2_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            trim_pdf(pdf2_path, temp_pdf2_path, trim2)
            pdf2_to_open = temp_pdf2_path
            temp_files.append(temp_pdf2_path)

        # Open documents
        doc1 = fitz.open(pdf1_to_open)
        doc2 = fitz.open(pdf2_to_open)
        
        # Verify each PDF has exactly one page
        if len(doc1) != 1:
            raise ValueError(f"{pdf1_path} must have exactly one page, found {len(doc1)}")
        
        if len(doc2) != 1:
            raise ValueError(f"{pdf2_path} must have exactly one page, found {len(doc2)}")
        
        # Get pages
        page1 = doc1[0]
        page2 = doc2[0]
        
        # Calculate dimensions for the output from the (now trimmed) pages
        width = page1.rect.width
        
        # Apply fudge factor to the heights
        height1 = page1.rect.height 
        height2 = page2.rect.height * fudge_factor
        
        if verbose:
            print(f"Applied fudge factor {fudge_factor} to heights")
            print(f"PDF1 height: {page1.rect.height} → {height1}")
            print(f"PDF2 height: {page2.rect.height} → {height2}")
        
        # Create new PDF with combined height
        new_doc = fitz.open()
        new_page = new_doc.new_page(width=width, height=height1 + height2)
        
        # Insert PDF1 at the top (y=0)
        new_page.show_pdf_page(
            fitz.Rect(0, 0, width, height1),
            doc1, 0
        )
        
        # Insert PDF2 below PDF1 (y=height1)
        new_page.show_pdf_page(
            fitz.Rect(0, height1, width, height1 + height2),
            doc2, 0
        )
        
        # Save with maximum compression
        new_doc.save(output_path, garbage=4, deflate=True)
        if verbose:
            print(f"Saved stacked PDF to {output_path}")

    except Exception as e:
        raise Exception(f"Error stacking PDFs: {str(e)}")
    
    finally:
        # Ensure we close all documents even if an error occurs
        if doc1:
            doc1.close()
        if doc2:
            doc2.close()
        if new_doc:
            new_doc.close()
        # Clean up temporary files
        for f in temp_files:
            try:
                os.remove(f)
            except OSError:
                pass

if __name__ == '__main__':
    import argparse
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="PDF Stacker", description="Stack two PDF files vertically, with optional trims (LaTeX style, in mm).")
    parser.add_argument("pdf1", type=argparse.FileType('r'), help="Path to the first (top) PDF file.")
    parser.add_argument("pdf2", type=argparse.FileType('r'), help="Path to the second (bottom) PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output stacked PDF file.")
    parser.add_argument("--trim1", default="", help="Trim for first PDF: 'left bottom right top' in mm (LaTeX style).")
    parser.add_argument("--trim2", default="", help="Trim for second PDF: 'left bottom right top' in mm (LaTeX style).")
    parser.add_argument("--fudge", type=float, default=1.0, help="Fudge factor to reduce heights (default: 1.0)")
    args = parser.parse_args()

    vstack_pdfs(args.pdf1.name, args.pdf2.name, args.output, trim1=args.trim1, trim2=args.trim2, fudge_factor=args.fudge)