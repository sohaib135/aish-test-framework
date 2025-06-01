# test_file_check.py
import os
import glob

# ---vvv--- Make absolutely sure this path string is IDENTICAL to your actual file path ---vvv---
file_path_to_check = r"D:\aish_test_app\data\processed_logs.csv"
directory_to_check = r"D:\aish_test_app\data"
# ---^^^-----------------------------------------------------------------------------------^^^---

print(f"--- File Check Script ---")
print(f"Python's current working directory: {os.getcwd()}")

print(f"\n1. Checking specific file path:")
print(f"   Path being checked: '{file_path_to_check}'")
file_exists = os.path.exists(file_path_to_check)
print(f"   os.path.exists() result: {file_exists}")
if file_exists:
    is_file = os.path.isfile(file_path_to_check)
    print(f"   os.path.isfile() result: {is_file}")
    if not is_file:
        print(f"   WARNING: Path exists but is not a file (it might be a directory).")
else:
    print(f"   File does not seem to exist at this path according to os.path.exists().")


print(f"\n2. Checking directory contents:")
print(f"   Directory being checked: '{directory_to_check}'")
dir_exists = os.path.exists(directory_to_check)
print(f"   Directory exists (os.path.exists()): {dir_exists}")
if dir_exists:
    is_dir = os.path.isdir(directory_to_check)
    print(f"   Is a directory (os.path.isdir()): {is_dir}")
    if is_dir:
        print(f"   Contents of '{directory_to_check}' (via glob):")
        try:
            for item in glob.glob(os.path.join(directory_to_check, '*')):
                print(f"     - {item}")
        except Exception as e:
            print(f"     Error during glob: {e}")
    else:
        print(f"   WARNING: Path exists but is not a directory.")
else:
    print(f"   Directory '{directory_to_check}' does not seem to exist.")

print(f"\n--- End of File Check Script ---")