'''
Use ghostscript with reasonable options to compress a PDF
'''

if __name__ == '__main__':
    import argparse
    import subprocess
    import os

    parser = argparse.ArgumentParser(description="Compress a PDF file")
    parser.add_argument("path", type=str, help="Path to the PDF file")
    parser.add_argument("-q", "--jpeg-quality", type=int, default=75, help="JPEG quality")
    parser.add_argument("-d", "--dpi", type=int, default=300, help="DPI")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    gs_command = [
        "gs",
        "-sDEVICE=pdfwrite",  # Output device
        "-dCompatibilityLevel=1.4",  # PDF version compatibility
        f"-dPDFSETTINGS=/printer",  # Adjust settings for print-quality compression
        f"-dAutoFilterColorImages=true",
        # f"-dColorImageFilter=/DCTEncode",  # Force JPEG compression
        f"-dColorImageDownsampleType=/Bicubic",  # Downsample method
        f"-dColorImageResolution={args.dpi}",  # Downsample resolution
        f"-dAutoFilterGrayImages=true",
        # f"-dGrayImageFilter=/DCTEncode",  # Force JPEG compression
        f"-dGrayImageDownsampleType=/Bicubic",  # Downsample method
        f"-dGrayImageResolution={args.dpi}",  # Downsample resolution
        f"-dJPEGQ={args.jpeg_quality}",  # JPEG compression quality
        f"-dMonoImageFilter=/CCITTFaxEncode",
        f"-dUseFlateCompression=true",
        f"-dCompressPages=true",
        f"-dDiscardDocumentStruct=true",
        f"-dDiscardMetadata=true",
        f"-dSubsetFonts=true",
        f"-dEmbedAllFonts=true",
        "-dNOPAUSE",  # No pause between pages
        "-dBATCH",  # Batch mode (exit when done)
        "-dQUIET",  # Suppress messages
        f"-sOutputFile={args.output}",  # Output file
        args.path  # Input file
    ]
    subprocess.run(gs_command, check=True)