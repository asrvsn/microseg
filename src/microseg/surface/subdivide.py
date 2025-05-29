'''
Perform surface subdivision
'''
if __name__ == '__main__':
    import sys
    import argparse
    import pickle
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser(prog="Surface Subdivider")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source triangulation (pickle file)')
    parser.add_argument('-n', '--nsubdiv', type=int, required=True, help='Number of subdivisions')
    parser.add_argument('-m', '--method', type=str, required=True, help='Subdivision method')
    args = parser.parse_args()

    tri = pickle.load(open(args.file.name, 'rb'))
    tri = tri.subdivide(args.nsubdiv, mode=args.method)
    path = f'{args.file.name}.subdivided'
    pickle.dump(tri, open(path, 'wb'))
    print(f'Subdivided surface by method {args.method} with {args.nsubdiv}-fold subdivision saved to {path}')