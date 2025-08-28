import os
import subprocess
import argparse
import shutil

def resize_pdf(in_path: str, out_path: str, max_cm: float, target_dpi: int, verbose: bool = False):
    """
    Scales a PDF to a maximum dimension while preserving aspect ratio.

    Args:
        in_path (str): The path to the input PDF file to scale.
        out_path (str): The path to save the output PDF file.
        max_cm (float): The maximum dimension (width or height) in centimeters.
        target_dpi (int): The target resolution for any rasterized images.
        verbose (bool): If True, enables logging and console output.
    """
    temp_pdf_path = f"{os.path.splitext(out_path)[0]}.temp.pdf"
    stdout_pipe = None if verbose else subprocess.DEVNULL
    stderr_pipe = None if verbose else subprocess.DEVNULL

    try:
        # Get PDF dimensions using pdfinfo
        pdfinfo_cmd = ['pdfinfo', in_path]
        try:
            result = subprocess.run(pdfinfo_cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise FileNotFoundError("pdfinfo not found. Please ensure it is installed (part of poppler-utils) and in your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdfinfo failed for {in_path}: {e.stderr.strip()}") from e

        # Parse dimensions from pdfinfo output
        width_pt, height_pt = None, None
        for line in result.stdout.split('\n'):
            if line.startswith('Page size:'):
                size_str = line.split('Page size:')[1].strip()
                parts = size_str.split('x')
                try:
                    width_pt = float(parts[0].strip())
                    height_pt = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    raise ValueError(f"Could not parse page size from: '{size_str}'")
                break
        
        if width_pt is None or height_pt is None:
            raise ValueError(f"Could not parse page size from pdfinfo output for {in_path}")

        # Calculate new dimensions
        width_inches = width_pt / 72
        height_inches = height_pt / 72
        max_size_inches = max_cm / 2.54
        
        if max(width_inches, height_inches) <= max_size_inches:
            if verbose:
                print(f"PDF is already within the target dimensions. No scaling needed.")
            shutil.copyfile(in_path, out_path)
            if verbose:
                print(f'Copied "{in_path}" to "{out_path}".')
            return

        scale_factor = max_size_inches / max(width_inches, height_inches)
        new_width_pt = width_pt * scale_factor
        new_height_pt = height_pt * scale_factor
        
        if verbose:
            print(f'Original: {width_pt:.1f} x {height_pt:.1f} pts ({width_inches:.2f}" x {height_inches:.2f}")')
            print(f'Target max: {max_size_inches:.2f}" ({max_cm} cm)')
            print(f'Scale factor: {scale_factor:.3f}')
            print(f'New size: {new_width_pt:.1f} x {new_height_pt:.1f} pts')
        
        gs_command = [
            "gs",
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dAutoRotatePages=/None",
            "-dFIXEDMEDIA", f"-dDEVICEWIDTHPOINTS={int(new_width_pt)}", f"-dDEVICEHEIGHTPOINTS={int(new_height_pt)}",
            "-dFitPage", "-dSubsetFonts=true", "-dEmbedAllFonts=true",
            f"-dColorImageResolution={target_dpi}", f"-dGrayImageResolution={target_dpi}", f"-dMonoImageResolution={target_dpi}",
            "-dDownsampleColorImages=true", "-dDownsampleGrayImages=true", "-dDownsampleMonoImages=true",
            "-dNOPAUSE", "-dBATCH",
            f"-sOutputFile={temp_pdf_path}", in_path
        ]
        if not verbose:
            gs_command.append('-dQUIET')
        
        try:
            subprocess.run(gs_command, check=True, stdout=stdout_pipe, stderr=stderr_pipe)
        except FileNotFoundError:
            raise FileNotFoundError("gs (Ghostscript) not found. Please ensure it is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ghostscript failed with exit code {e.returncode}.") from e

        os.rename(temp_pdf_path, out_path)
        if verbose:
            print(f'Scaled "{in_path}" and saved to "{out_path}".')
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == '__main__':
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="PDF Resizer", description="Scales a PDF to a maximum dimension, preserving aspect ratio.")
    parser.add_argument("input", help="Path to the PDF file to scale.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output resized PDF.")
    parser.add_argument("--max-cm", type=float, required=True, help="Maximum dimension (width or height) in centimeters.")
    parser.add_argument("--dpi", type=int, default=300, help="Target resolution for any rasterized images (default: 300).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable logging and console output.")
    
    args = parser.parse_args()

    try:
        resize_pdf(args.input, args.output, args.max_cm, args.dpi, verbose=args.verbose)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)