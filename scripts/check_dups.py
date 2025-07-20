import os
import argparse

def check_duplicate_filenames(dir1, dir2):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    duplicates = files1 & files2
    if duplicates:
        print(f"Found {len(duplicates)} duplicate filenames:")
        for f in sorted(duplicates):
            print(f"  {f}")
    else:
        print("No duplicate filenames found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="First directory path")
    parser.add_argument("dir2", help="Second directory path")
    args = parser.parse_args()

    check_duplicate_filenames(args.dir1, args.dir2)