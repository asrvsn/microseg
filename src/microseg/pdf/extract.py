import pymupdf

def extract_pdf(
        in_path: str,
        out_path: str,
        start_page: int,
        end_page: int,
        verbose: bool = False
    ):
    '''
    Extract a range of pages from a PDF file.
    '''
    with pymupdf.open(in_path) as doc:
        num_pages = len(doc)
        if start_page < 0 or end_page > num_pages or start_page >= end_page:
            raise ValueError(
                f"Invalid page range: {start_page}-{end_page}. "
                f"File has {num_pages} pages (0-{num_pages-1})."
            )
        doc.select(range(start_page, end_page))
        doc.save(out_path, garbage=4, deflate=True)
    if verbose:
        print(f"Extracted pages {start_page}-{end_page-1} from '{in_path}' to '{out_path}'")

if __name__ == '__main__':
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="PDF Page Extractor", description="Extract a range of pages from a PDF file.")
    parser.add_argument("input", help="Path to the input PDF file.")
    parser.add_argument("start_page", type=int, help="Start page number (0-indexed, inclusive).")
    parser.add_argument("end_page", type=int, help="End page number (0-indexed, exclusive).")
    parser.add_argument("-o", "--output", required=True, help="Path for the output extracted PDF.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    extract_pdf(args.input, args.output, args.start_page, args.end_page, verbose=args.verbose)
