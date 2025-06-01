import cv2

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process the first positional argument.")
    parser.add_argument("cam", type=int, nargs="?", default=0, help="Camera index")
    args = parser.parse_args()
    
    cam = cv2.VideoCapture(args.cam)
    while True:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()