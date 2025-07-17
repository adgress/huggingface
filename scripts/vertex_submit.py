# from google.cloud import aiplatform
# import os

# aiplatform.init(
#     project="helical-glass-466113-c2",
#     location="us-central1",  # Or your preferred region
#     staging_bucket="gs://huggingface-vertex-artifacts"
# )

# def main():
#     """Main function to submit training job to Vertex AI"""
#     huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
#     if not huggingface_token:
#         raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
    
#     # Create a custom training job
#     # job = aiplatform.CustomJob.from_local_script(
#     #     display_name="hf-train-job",
#     #     script_path="huggingface_training/train.py",
#     #     local_package_path="huggingface_training",
#     #     container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
#     #     requirements=["transformers", "datasets", "torch", "torchvision", "numpy", "pandas", "pillow", "tqdm", "huggingface_hub"],
#     #     machine_type="a2-highgpu-1g",
#     #     accelerator_type="NVIDIA_TESLA_A100",
#     #     accelerator_count=1,
#     #     replica_count=1,
#     #     environment_variables={"HUGGINGFACE_TOKEN": huggingface_token},
#     # )

#     job = aiplatform.CustomPythonPackageTrainingJob(
#         display_name="hf-train-job",
#         python_package_gcs_uri="gs://your-bucket/training_src/",
#         python_module_name="train",  # this is your `train.py`
#         container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310",
#         requirements=["transformers", "datasets", "torch", "torchvision", "numpy", "pandas", "pillow", "tqdm"],
#     )

#     # job.run(sync=True)

#     job.run(
#         args=[],
#         replica_count=1,
#         machine_type="a2-highgpu-1g",
#         accelerator_type="NVIDIA_TESLA_A100",
#         accelerator_count=1,
#     )


# if __name__ == "__main__":
#     main()

import os
import subprocess
from google.cloud import aiplatform

# === CONFIG ===
PACKAGE_NAME = "huggingface_training"  # Name of your package (src/your_package/)
PROJECT_ID = "helical-glass-466113-c2"  # Your GCP project ID
BUCKET_NAME = "huggingface-vertex-artifacts"
REGION = "us-central1"
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310"
GCS_PACKAGE_PATH = f"gs://{BUCKET_NAME}/packages"

# === STEP 1: Build package ===
print("Building Python package...")
subprocess.run(["python", "setup.py", "sdist"], check=True)

dist_files = os.listdir("dist")
tar_file = next(f for f in dist_files if f.endswith(".tar.gz"))
local_package_path = os.path.join("dist", tar_file)
gcs_package_uri = f"{GCS_PACKAGE_PATH}/{tar_file}"

# === STEP 2: Upload to GCS ===
print(f"Uploading {tar_file} to {gcs_package_uri}...")
subprocess.run(["gsutil", "cp", local_package_path, gcs_package_uri], check=True)

# === STEP 3: Submit to Vertex AI ===
print("Submitting job to Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name="vertex-train-job",
    python_package_gcs_uri=gcs_package_uri,
    python_module_name=f"{PACKAGE_NAME}.train",  # train.py must define a main()
    container_uri=CONTAINER_URI
)

job.run(
    args=[],  # CLI args to pass to train.py
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    replica_count=1,
    # requirements=["transformers", "datasets", "torch", "numpy", "pillow", "tqdm"],
)