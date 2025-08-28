import argparse
import subprocess
from typing import Tuple
import pymupdf

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

def trim_pdf(in_path: str, out_path: str, trim_str: str, verbose: bool = False):
    """
    Trims a PDF file using the 'pdfcrop' command-line tool.

    Args:
        in_path: Path to the input PDF file.
        out_path: Path to save the trimmed PDF file.
        trim_str: LaTeX-style trim string: 'left bottom right top' in mm.
        verbose: If True, prints output from pdfcrop.
    """
    stdout_pipe = None if verbose else subprocess.DEVNULL
    stderr_pipe = None if verbose else subprocess.DEVNULL
    
    l, b, r, t = parse_trim(trim_str)
    
    # pdfcrop margins are 'left top right bottom'
    crop_cmd = [
        'pdfcrop',
        '--margins',
        f'{-l} {-t} {-r} {-b}',
        in_path,
        out_path
    ]
    
    try:
        subprocess.run(crop_cmd, check=True, stdout=stdout_pipe, stderr=stderr_pipe)
        if verbose:
            print(f'Trimmed "{in_path}" and saved to "{out_path}"')
    except FileNotFoundError:
        raise FileNotFoundError(
            "pdfcrop not found. Please ensure it is installed and in your PATH. "
            "It is usually part of a TeX/LaTeX distribution (e.g., TeX Live, MiKTeX)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pdfcrop failed with exit code {e.returncode}") from e

if __name__ == '__main__':
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="PDF Trimmer", description="Trim a PDF file using pdfcrop.")
    parser.add_argument("input", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output trimmed PDF.")
    parser.add_argument("--trim", required=True, help="Trim string: 'left bottom right top' in mm (LaTeX style).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print pdfcrop output.")
    args = parser.parse_args()

    trim_pdf(args.input, args.output, args.trim, verbose=args.verbose) 