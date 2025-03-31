'''
Report voxel size of data
'''
if __name__ == '__main__':
    import sys
    import argparse
    from microseg.utils.data import get_voxel_size
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    args = parser.parse_args()

    voxsize = get_voxel_size(args.file)
    print('Voxel size (XYZ):', voxsize.tolist())