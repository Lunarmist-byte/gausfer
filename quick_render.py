import os
import subprocess
import sys

def main():
    print("=== Gausfer Quick Render Utility ===")
    
    # 1. Check for images
    images_dir = "./images"
    rar_file = "room.rar"
    
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"Warning: '{images_dir}' is empty or does not exist.")
        if os.path.exists(rar_file):
            choice = input(f"Found '{rar_file}'. Extract it now? (y/n): ").lower()
            if choice == 'y':
                print("Extracting room.rar...")
                # Try common extraction tools
                try:
                    # Windows usually has tar (recent versions) or we can try 7z if in path
                    subprocess.run(["tar", "-xf", rar_file], check=True)
                    if not os.path.exists(images_dir) or not os.listdir(images_dir):
                         print("Extraction appeared to succeed but no images found in './images'.")
                         print("Note: 'tar' on Windows may not support .rar files directly.")
                         print("Please extract 'room.rar' MANUALLY using WinRAR or 7-Zip into the 'images' folder.")
                         return
                    print("Extraction complete.")
                except Exception as e:
                    print(f"Error extracting with tar: {e}")
                    print("Please extract 'room.rar' MANUALLY using WinRAR or 7-Zip into the 'images' folder.")
                    return
        else:
            print("No images found and 'room.rar' is missing.")
            print("Please place images in the 'images' folder first.")
            return

    # 2. Run the quick pipeline
    print("\nStarting Quick Pipeline (approx 5-10 mins)...")
    try:
        # Running main.py with --quick flag
        cmd = [sys.executable, "main.py", "--quick"]
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*40)
        print("SUCCESS: Quick rendering complete!")
        print(f"Check 'output/result.png' for your result image.")
        print("="*40)
        
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline failed with error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
