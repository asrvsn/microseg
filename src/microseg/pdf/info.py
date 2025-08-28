'''
Get PDF info 
'''

import argparse
import os
import pymupdf
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from microseg.utils.args import GuiArgumentParser

if __name__ == '__main__':
    parser = GuiArgumentParser(prog="PDF Info", description="Get PDF information for figure guidelines")
    parser.add_argument("path", type=argparse.FileType('r'), help="Path to the PDF file")
    parser.add_argument("--journal", choices=["PNAS"], help="Journal for specific guidelines")
    parser.add_argument("--show", action="store_true", help="Show embedded images")
    args = parser.parse_args()

    try:
        doc = pymupdf.open(args.path.name)
        page = doc[0]
        rect = page.rect
        
        # Convert points to inches and cm
        w_in, h_in = rect.width / 72, rect.height / 72
        w_cm, h_cm = w_in * 2.54, h_in * 2.54
        
        print(f"PDF: {os.path.basename(args.path.name)}")
        print(f"Size: {w_in:.2f}\" × {h_in:.2f}\" ({w_cm:.2f} × {h_cm:.2f} cm)")
        print(f"Pages: {len(doc)}")
        
        # Journal-specific guidelines
        if args.journal == "PNAS":
            print(f"PNAS sizes: Small (9×6 cm), Medium (11×11 cm), Large (18×22 cm)")
        
        # Check fonts
        fonts = page.get_fonts()
        if fonts:
            print(f"Fonts: {len(fonts)}")
            for i, font in enumerate(fonts):
                font_name = font[3] if font[3] else "Unknown"
                font_type = font[1]
                # Check if font is embedded (font[6] contains embedding info)
                is_embedded = font[6] if len(font) > 6 else "Unknown"
                embed_status = "Embedded" if is_embedded else "Not embedded"
                print(f"  Font {i+1}: {font_name} ({font_type}) - {embed_status}")
        else:
            print("No fonts detected")
        
        # DPI from embedded images
        images = page.get_images(full=True)
        if images:
            print(f"Embedded images: {len(images)}")
            for i, img in enumerate(images):
                xref = img[0]
                img_info = doc.extract_image(xref)
                img_rect = page.get_image_bbox(img)
                dpi_x = (img_info['width'] / img_rect.width) * 72
                dpi_y = (img_info['height'] / img_rect.height) * 72
                dpi_avg = (dpi_x + dpi_y) / 2
                size_mb = len(img_info['image']) / (1024 * 1024)
                
                if args.journal == "PNAS":
                    status = "✓" if dpi_avg >= 300 else "✗"
                else:
                    status = ""
                print(f"  Image {i+1}: {dpi_avg:.0f} DPI, {size_mb:.2f} MB {status}")
                
                # Show image if requested
                if args.show:
                    # Extract image data
                    imbytes = img_info['image']
                    pil_img = Image.open(io.BytesIO(imbytes))
                    img_array = np.array(pil_img)
                    
                    # Create and display figure
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
                    ax.set_title(f'Image {i+1}: {img_info["width"]}×{img_info["height"]} px, {dpi_avg:.0f} DPI')
                    ax.axis('off')
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)  # Properly close the figure
        else:
            print("No embedded images (pure vector content)")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

