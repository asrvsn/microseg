'''
Perform surface subdivision
'''
if __name__ == '__main__':
    import sys
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source triangulation (pickle file)')
    parser.add_argument('-n', '--nsubdiv', type=int, required=True, help='Number of subdivisions')
    parser.add_argument('-m', '--method', type=str, required=True, help='Subdivision method')
    args = parser.parse_args()

    tri = pickle.load(open(args.file, 'rb'))
    tri = tri.subdivide(args.nsubdiv, mode=args.method)
    path = f'{args.file}.subdivided'
    pickle.dump(tri, open(path, 'wb'))
    print(f'Subdivided surface by method {args.method} with {args.nsubdiv}-fold subdivision saved to {path}')