'''
Use ghostscript with reasonable options to compress a PDF
'''
import os
import subprocess
import argparse
from microseg.utils.args import GuiArgumentParser

def compress_pdf(pdf_path: str, output_path: str, compressed_dpi: int, jpeg_quality: int, verbose: bool = False):
    """
    Compresses a PDF by downsampling images using Ghostscript.

    Args:
        pdf_path (str): The path to the PDF file to compress.
        output_path (str): Path for the compressed output PDF.
        compressed_dpi (int): The target resolution for downsampling images.
        jpeg_quality (int): The JPEG quality setting (0-100).
        verbose (bool): If True, enables logging and console output.
    """
    temp_pdf_path = f"{os.path.splitext(output_path)[0]}.temp.pdf"
    stdout_pipe = None if verbose else subprocess.DEVNULL
    stderr_pipe = None if verbose else subprocess.DEVNULL

    try:
        gs_command = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/printer",
            "-dAutoFilterColorImages=true",
            "-dColorImageDownsampleType=/Bicubic",
            f"-dColorImageResolution={compressed_dpi}",
            "-dAutoFilterGrayImages=true",
            "-dGrayImageDownsampleType=/Bicubic",
            f"-dGrayImageResolution={compressed_dpi}",
            f"-dJPEGQ={jpeg_quality}",
            "-dMonoImageFilter=/CCITTFaxEncode",
            "-dUseFlateCompression=true",
            "-dCompressPages=true",
            "-dDiscardDocumentStruct=true",
            "-dDiscardMetadata=true",
            "-dSubsetFonts=true",
            "-dEmbedAllFonts=true",
            "-dNOPAUSE",
            "-dBATCH",
            f"-sOutputFile={temp_pdf_path}",
            pdf_path
        ]
        if not verbose:
            gs_command.append('-dQUIET')
        
        try:
            subprocess.run(gs_command, check=True, stdout=stdout_pipe, stderr=stderr_pipe)
        except FileNotFoundError:
            raise FileNotFoundError("gs (Ghostscript) not found. Please ensure it is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ghostscript failed with exit code {e.returncode}.") from e

        os.rename(temp_pdf_path, output_path)
        if verbose:
            print(f'Compressed "{pdf_path}" and saved to "{output_path}".')
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == '__main__':
    parser = GuiArgumentParser(prog="PDF Compressor", description="Compress a PDF file using Ghostscript.")
    parser.add_argument("input_path", help="Path to the input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output compressed PDF.")
    parser.add_argument("-q", "--jpeg-quality", type=int, default=75, help="JPEG quality (0-100, default: 75).")
    parser.add_argument("-d", "--dpi", type=int, default=150, help="Resolution to downsample images to (default: 150).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable logging and console output.")
    args = parser.parse_args()

    try:
        compress_pdf(args.input_path, args.output, args.dpi, args.jpeg_quality, verbose=args.verbose)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        exit(1)