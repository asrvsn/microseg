'''
Export triangulation as vertices and simplices
'''

if __name__ == '__main__':
    import argparse
    import pickle
    from matgeo import Triangulation
    from microseg.utils.args import GuiArgumentParser

    parser = GuiArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source triangulation file (pickle format)')
    args = parser.parse_args()

    tri = pickle.load(open(args.file.name, 'rb'))
    assert isinstance(tri, Triangulation), 'File must contain a Triangulation'
    tri.export(args.file.name)
    print(f'Exported triangulation from {args.file.name}')