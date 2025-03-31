'''
Rescale a surface by a scale factor
'''
if __name__ == '__main__':
    import sys
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source triangulation (pickle file)')
    parser.add_argument('-s', '--scale', type=float, required=True, help='Scale factor')
    args = parser.parse_args()

    tri = pickle.load(open(args.file, 'rb'))
    tri = tri.rescale(args.scale)
    path = f'{args.file}.rescaled'
    pickle.dump(tri, open(path, 'wb'))
    print(f'Rescaled surface saved to {path}')
