'''
Compress an image
'''
if __name__ == '__main__':
    import argparse
    import PIL
    import PIL.Image
    from microseg.utils.args import GuiArgumentParser
    
    parser = GuiArgumentParser(prog="Image Compressor", description="Compress an image")
    parser.add_argument("file_in", type=argparse.FileType('r'), help="Path to source image")
    parser.add_argument("file_out", type=str, help="Path to output image")
    parser.add_argument("-q", "--quality", type=int, default=85, help="JPEG quality")
    args = parser.parse_args()
    
    img = PIL.Image.open(args.file_in.name)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(args.file_out, quality=args.quality)