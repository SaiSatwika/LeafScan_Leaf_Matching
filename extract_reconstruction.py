import os
import shutil

def extract_reconstructions(source_dir, destination_dir):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            reconstruction_path = os.path.join(folder_path, "reconstruction.jpg")

            if os.path.exists(reconstruction_path):
                new_file_name = f"{folder_name}.jpg"
                destination_path = os.path.join(destination_dir, new_file_name)

                shutil.copy(reconstruction_path, destination_path)

                print(f"Copied: {folder_name}")
            else:
                print(f"Skipped (no reconstruction.jpg): {folder_name}")

    print(f"Done for {source_dir} ✅\n")


# -------- RUN FOR BOTH --------

# Defoliated
extract_reconstructions(
    r"D:\GreenhouseDataset\defoliated",
    r"D:\GreenhouseDataset\reconstructed_defoliated"
)

# Healthy
extract_reconstructions(
    r"D:\GreenhouseDataset\healthy",
    r"D:\GreenhouseDataset\reconstructed_healthy"
)