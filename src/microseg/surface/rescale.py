'''
Rescale a surface by a scale factor
'''
if __name__ == '__main__':
    import sys
    import argparse
    import pickle
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source triangulation (pickle file)')
    parser.add_argument('-s', '--scale', type=float, required=True, help='Scale factor')
    args = parser.parse_args()

    tri = pickle.load(open(args.file.name, 'rb'))
    tri = tri.rescale(args.scale)
    path = f'{args.file.name}.rescaled'
    pickle.dump(tri, open(path, 'wb'))
    print(f'Rescaled surface saved to {path}')
