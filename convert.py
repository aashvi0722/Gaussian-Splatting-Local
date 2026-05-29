import os
import logging
from argparse import ArgumentParser
import shutil

# This script is a wrapper to run COLMAP SfM (Structure from Motion)
# Modified to work with COLMAP 3.8+ by removing outdated GPU flags.

def main():
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--colmap_executable", default="colmap", type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    args = parser.parse_args()

    # Paths
    colmap_command = args.colmap_executable
    use_gpu = 1 # We want to use your RTX 3050

    if not os.path.exists(os.path.join(args.source_path, "sparse")):
        os.makedirs(os.path.join(args.source_path, "sparse"), exist_ok=True)

    # 1. Feature Extraction
    if not args.skip_matching:
        print("Feature extraction")
        feat_extract_params = [
            colmap_command, "feature_extractor",
            "--database_path", os.path.join(args.source_path, "database.db"),
            "--image_path", os.path.join(args.source_path, "input"),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", args.camera,
            # Note: --SiftExtraction.use_gpu removed due to version incompatibility
        ]
        exit_code = os.system(" ".join(feat_extract_params))
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 2. Feature Matching
        print("Exhaustive matching")
        feat_matching_params = [
            colmap_command, "exhaustive_matcher",
            "--database_path", os.path.join(args.source_path, "database.db"),
            # Note: --SiftMatching.use_gpu removed due to version incompatibility
        ]
        exit_code = os.system(" ".join(feat_matching_params))
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

    # 3. Mapping
    print("Structure from Motion (SfM) mapping")
    mapper_params = [
        colmap_command, "mapper",
        "--database_path", os.path.join(args.source_path, "database.db"),
        "--image_path", os.path.join(args.source_path, "input"),
        "--output_path", os.path.join(args.source_path, "sparse"),
    ]
    exit_code = os.system(" ".join(mapper_params))
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    print("Conversion complete!")

if __name__ == "__main__":
    main()