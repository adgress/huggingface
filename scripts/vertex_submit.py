import datetime
import os
import subprocess
from google.cloud import aiplatform

import shutil
print("shutil.which('gsutil'):", shutil.which("gsutil"))


import os
print("os.environ['PATH']:", os.environ["PATH"])
# === CONFIG ===
PACKAGE_NAME = "huggingface_training"  # Name of your package (src/your_package/)
PROJECT_ID = "helical-glass-466113-c2"  # Your GCP project ID
BUCKET_NAME = "huggingface-vertex-artifacts"
STAGING_BUCKET = f"gs://{BUCKET_NAME}/staging"
REGION = "us-central1"
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310"
GCS_PACKAGE_PATH = f"gs://{BUCKET_NAME}/packages"

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
job_display_name = f"run-beans-vit-v1-{timestamp}"
experiment_name = f"beans-vit-experiment"
run_name = f"run-beans-vit-v1-{timestamp}"

# === STEP 1: Build package ===
print("Building Python package...")
subprocess.run(["python", "setup.py", "sdist"], check=True)

dist_files = os.listdir("dist")
tar_file = next(f for f in dist_files if f.endswith(".tar.gz"))
local_package_path = os.path.join("dist", tar_file)
gcs_package_uri = f"{GCS_PACKAGE_PATH}/{tar_file}"

# === STEP 2: Upload to GCS ===
print(f"Uploading {tar_file} to {gcs_package_uri}...")
subprocess.run(["gsutil", "cp", local_package_path, gcs_package_uri], check=True, shell=True)

# === STEP 3: Submit to Vertex AI ===
print("Submitting job to Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name=job_display_name,
    python_package_gcs_uri=gcs_package_uri,
    python_module_name=f"{PACKAGE_NAME}.train",  # train.py must define a main()
    container_uri=CONTAINER_URI
)

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if not huggingface_token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

job.run(
    args=[],  # CLI args to pass to train.py
    machine_type="a2-highgpu-1g",   
    accelerator_type="NVIDIA_TESLA_A100",
    
    # Logging doesn't work on these machines
    # machine_type="n1-standard-4",
    # accelerator_type="NVIDIA_TESLA_T4",  # Use V100 for better compatibility
    accelerator_count=1,
    replica_count=1,
    tensorboard="projects/82783227389/locations/us-central1/tensorboards/5185257254173016064",
    service_account="82783227389-compute@developer.gserviceaccount.com",
    environment_variables={ 
        "HUGGINGFACE_TOKEN": huggingface_token,
        "PROJECT_ID": PROJECT_ID,
        "REGION": REGION,
        "STAGING_BUCKET": STAGING_BUCKET,
        "VERTEX_EXPERIMENT_NAME": experiment_name,
        "VERTEX_RUN_NAME": run_name
    }
)