"""
Convert one or more .heic files to .jpg in-place (same directory) using Pillow + pillow-heif.
"""

import os
from typing import List

import pillow_heif  # Registers HEIF opener for Pillow
from PIL import Image


pillow_heif.register_heif_opener()


def _derive_jpg_path(heic_path: str) -> str:
    base, _ = os.path.splitext(heic_path)
    return f"{base}.jpg"


def heic_to_jpg(
    input_paths: List[str],
    quality: int = 95,
    overwrite: bool = False,
    delete_original: bool = False,
    verbose: bool = False,
) -> List[str]:
    outputs: List[str] = []

    for in_path in input_paths:
        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"Input file not found: {in_path}")
        if not in_path.lower().endswith((".heic", ".heif")):
            raise ValueError(f"Input must be a .heic/.heif file: {in_path}")

        out_path = _derive_jpg_path(in_path)
        if os.path.exists(out_path) and not overwrite:
            if verbose:
                print(f"Skipping existing: {out_path}")
            outputs.append(out_path)
            continue

        with Image.open(in_path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")

            exif_bytes = im.info.get("exif")
            save_kwargs = {"quality": int(max(1, min(100, quality))), "optimize": True}
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            im.save(out_path, format="JPEG", **save_kwargs)

        if delete_original:
            try:
                os.remove(in_path)
            except OSError:
                pass

        if verbose:
            print(f"Converted '{in_path}' -> '{out_path}'")
        outputs.append(out_path)

    return outputs


if __name__ == '__main__':
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="HEIC to JPG", description="Convert one or more .heic files to .jpg in-place.")
    parser.add_argument("inputs", nargs="+", help="Path(s) to .heic/.heif file(s) to convert.")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100, default: 95).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .jpg files if present.")
    parser.add_argument("--delete-original", action="store_true", help="Delete original .heic files after conversion.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    try:
        heic_to_jpg(
            input_paths=args.inputs,
            quality=args.quality,
            overwrite=args.overwrite,
            delete_original=args.delete_original,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)


