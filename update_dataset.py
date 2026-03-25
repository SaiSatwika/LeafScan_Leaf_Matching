import os
import shutil

# Base dataset path
base_path = r"D:\updated dataset"

# Output folders
defoliated_output = os.path.join(base_path, "Defoliated_reconstruction")
healthy_output = os.path.join(base_path, "Healthy_reconstruction")

# Create output folders if they don't exist
os.makedirs(defoliated_output, exist_ok=True)
os.makedirs(healthy_output, exist_ok=True)

# Source folders mapping
mapping = {
    "defoliated": defoliated_output,
    "tattered_defo": defoliated_output,
    "healthy": healthy_output,
    "tattered": healthy_output
}

# Function to find reconstruction image
def find_reconstruction_image(folder_path):
    for file in os.listdir(folder_path):
        if "reconstruction" in file.lower():
            return file
    return None

# Main processing loop
for source_folder, output_folder in mapping.items():
    source_path = os.path.join(base_path, source_folder)

    if not os.path.exists(source_path):
        print(f"Skipping missing folder: {source_folder}")
        continue

    for subfolder in os.listdir(source_path):
        subfolder_path = os.path.join(source_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        # Find reconstruction image
        recon_image = find_reconstruction_image(subfolder_path)

        if recon_image is None:
            print(f"No reconstruction image found in {subfolder}")
            continue

        src_image_path = os.path.join(subfolder_path, recon_image)

        # Get extension (.png, .jpg, etc.)
        ext = os.path.splitext(recon_image)[1]

        # New filename = folder name + extension
        new_filename = subfolder + ext
        dst_image_path = os.path.join(output_folder, new_filename)

        # Copy and rename
        shutil.copy2(src_image_path, dst_image_path)

        print(f"Copied: {subfolder} → {new_filename}")

print("\n✅ Done! All reconstruction images organized.")