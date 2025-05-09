'''
Vertically stack PDFs using width of first page
'''

import fitz  # pymupdf
import os
from typing import List, Tuple

def parse_trim(trim_str: str) -> Tuple[float, float, float, float]:
    """
    Parse LaTeX-style trim string: 'left bottom right top' (in mm).
    Returns a tuple of (left, bottom, right, top) in points.
    """
    if not trim_str:
        return (0.0, 0.0, 0.0, 0.0)
    parts = trim_str.strip().split()
    if len(parts) != 4:
        raise ValueError("Trim must have 4 values: 'left bottom right top' (in mm)")
    # Convert mm to points (1 pt = 1/72 in, 1 in = 25.4 mm)
    return tuple(float(x) * 72 / 25.4 for x in parts)

def stack_two_pdfs(pdf1_path: str, pdf2_path: str, output_path: str, trim1: str = "", trim2: str = "", fudge_factor: float = 1.0):
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
    
    try:
        # Open documents
        doc1 = fitz.open(pdf1_path)
        doc2 = fitz.open(pdf2_path)
        
        # Verify each PDF has exactly one page
        if len(doc1) != 1:
            raise ValueError(f"{pdf1_path} must have exactly one page, found {len(doc1)}")
        
        if len(doc2) != 1:
            raise ValueError(f"{pdf2_path} must have exactly one page, found {len(doc2)}")
        
        # Get pages
        page1 = doc1[0]
        page2 = doc2[0]
        
        # Parse trims
        l1, b1, r1, t1 = parse_trim(trim1)
        l2, b2, r2, t2 = parse_trim(trim2)
        
        # Calculate visible content bounds (after trimming)
        rect1 = fitz.Rect(
            page1.cropbox.x0 + l1,
            page1.cropbox.y0 + t1,
            page1.cropbox.x1 - r1,
            page1.cropbox.y1 - b1
        )
        
        rect2 = fitz.Rect(
            page2.cropbox.x0 + l2,
            page2.cropbox.y0 + t2,
            page2.cropbox.x1 - r2,
            page2.cropbox.y1 - b2
        )
        
        # Calculate dimensions for the output
        width = rect1.width
        
        # Apply fudge factor to the heights
        height1 = rect1.height 
        height2 = rect2.height * fudge_factor
        
        print(f"Applied fudge factor {fudge_factor} to heights")
        print(f"PDF1 height: {rect1.height} → {height1}")
        print(f"PDF2 height: {rect2.height} → {height2}")
        
        # Create new PDF with combined height
        new_doc = fitz.open()
        new_page = new_doc.new_page(width=width, height=height1 + height2)
        
        # Insert PDF1 at the top (y=0)
        new_page.show_pdf_page(
            fitz.Rect(0, 0, width, height1),
            doc1, 0, clip=rect1
        )
        
        # Insert PDF2 below PDF1 (y=height1)
        new_page.show_pdf_page(
            fitz.Rect(0, height1, width, height1 + height2),
            doc2, 0, clip=rect2
        )
        
        # Save with maximum compression
        new_doc.save(output_path, garbage=4, deflate=True)
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Stack two PDF files vertically, with optional trims (LaTeX style, in mm).")
    parser.add_argument("pdf1", help="Path to the first (top) PDF file.")
    parser.add_argument("pdf2", help="Path to the second (bottom) PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output stacked PDF file.")
    parser.add_argument("--trim1", default="", help="Trim for first PDF: 'left bottom right top' in mm (LaTeX style).")
    parser.add_argument("--trim2", default="", help="Trim for second PDF: 'left bottom right top' in mm (LaTeX style).")
    parser.add_argument("--fudge", type=float, default=1.0, help="Fudge factor to reduce heights (default: 1.0)")
    args = parser.parse_args()

    stack_two_pdfs(args.pdf1, args.pdf2, args.output, trim1=args.trim1, trim2=args.trim2, fudge_factor=args.fudge)