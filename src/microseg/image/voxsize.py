'''
Report voxel size of data
'''
if __name__ == '__main__':
    import sys
    import argparse
    from microseg.utils.data import get_voxel_size
    from microseg.utils.args import GuiArgumentParser
    
    parser = GuiArgumentParser(prog="Voxel Size Tool")
    parser.add_argument('file', type=argparse.FileType('r'), help='Path to source img [tiff|jpg|png|czi|...]')
    args = parser.parse_args()

    voxsize = get_voxel_size(args.file.name)
    print('Voxel size (XYZ):', voxsize.tolist())