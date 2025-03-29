'''
Export triangulation as vertices and simplices
'''

if __name__ == '__main__':
    import argparse
    import pickle
    from matgeo import Triangulation

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source triangulation file (pickle format)')
    args = parser.parse_args()

    tri = pickle.load(open(args.file, 'rb'))
    assert isinstance(tri, Triangulation), 'File must contain a Triangulation'
    tri.export(args.file)
    print(f'Exported triangulation from {args.file}')