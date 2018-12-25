import os.path
from pathlib import Path

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            print("file: {0}".format(file))
            dirname = root.split(os.path.sep)[-1]
            print("directory : {0}".format(dirname))
                # if file.endswith("png") or file.endswith("jpg"):

main()