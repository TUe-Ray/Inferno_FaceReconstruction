import sys
import subprocess

def main():
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
    else:
        input_path = input("Enter input path: ")

    # Run resize_npy.py
    subprocess.check_call([sys.executable, "resize_npy.py", input_path])

    # Run merge_rgb_depth_npy.py
    subprocess.check_call([sys.executable, "merge_rgb_depth_npy.py", input_path])

if __name__ == "__main__":
    main()
