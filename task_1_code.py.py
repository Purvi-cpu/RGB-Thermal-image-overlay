import cv2
import numpy as np
import os

# STEP 1: Find thermal images
def find_thermal_images(input_dir):
    files = os.listdir(input_dir)
    thermal_files = [f for f in files if f.endswith('_T.JPG')]
    return thermal_files

# STEP 2: Process thermal image (colormap + border)
def process_thermal_image(thermal_img):
    gray_thermal = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray_thermal, cv2.COLORMAP_INFERNO)
    bordered = cv2.copyMakeBorder(colored, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return bordered

# STEP 3: Process all thermal images in folder
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    thermal_files = find_thermal_images(input_dir)

    for file in thermal_files:
        thermal_path = os.path.join(input_dir, file)
        thermal = cv2.imread(thermal_path)
        result = process_thermal_image(thermal)
        prefix = file[:-6]
        output_path = os.path.join(output_dir, f"{prefix}_overlay.jpg")
        cv2.imwrite(output_path, result)
        print(f"[✔] Saved overlay for: {prefix}")

# STEP 4: Run the script
if __name__ == "__main__":
    input_folder = "input-images"
    output_folder = "output-overlays"
    process_folder(input_folder, output_folder)
