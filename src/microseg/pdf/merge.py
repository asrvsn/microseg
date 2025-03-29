'''
merge pdfs
'''

import pymupdf
import os
from typing import List

def merge_pdfs(fpaths: List[str], output_path: str, delete: bool=False):
    merged = pymupdf.open()
    for fpath in fpaths:
        assert os.path.isfile(fpath), f'{fpath} is not a file'
        with pymupdf.open(fpath) as f:
            merged.insert_pdf(f)
    merged.save(output_path)
    merged.close()
    print(f'Saved merged PDF to {output_path}')
    if delete:
        for fpath in fpaths:
            os.remove(fpath)
        print(f'Deleted source PDF files: {fpaths}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge multiple PDF files into one.")
    parser.add_argument("pdf_files", nargs="+", help="Paths to the PDF files to merge, in the desired order.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output merged PDF file.")
    parser.add_argument("-d", "--delete", action="store_true", help="Delete source PDF files after merging.")
    args = parser.parse_args()
    
    merge_pdfs(args.pdf_files, args.output, delete=args.delete)