import subprocess
import os
from google.cloud import storage

def validate_run(dt_hash, sparkles_path, sparkles_config, save_pref):
    # Watch the ensemble
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "watch", f"ensemble_{dt_hash}", "--loglive"])
    
    # Reset the ensemble
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "reset", f"ensemble_{dt_hash}"])
    
    # Watch the ensemble again
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "watch", f"ensemble_{dt_hash}", "--loglive"])
    
    # Create data directory
    os.makedirs(os.path.join(save_pref, "data"), exist_ok=True)
    
    # Get default_url_prefix from sparkles_config
    with open(sparkles_config, 'r') as f:
        for line in f:
            if line.startswith("default_url_prefix"):
                default_url_prefix = line.split("=")[1].strip()
                break
    
    # Authenticate with Google Cloud
    subprocess.check_call(["gcloud", "auth", "activate-service-account", "--key-file", "/root/.sparkles-cache/service-keys/broad-achilles.json"])
    
    # List files in Google Cloud Storage
    storage_client = storage.Client()
    bucket_name = default_url_prefix.split("/")[2]
    prefix = f"ensemble_{dt_hash}/"
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    # Write completed jobs to file
    with open(os.path.join(save_pref, "completed_jobs.txt"), 'w') as f:
        for blob in blobs:
            if blob.name.endswith('.csv'):
                f.write(f"{default_url_prefix}/{blob.name}\n")
    
    # Validate jobs are complete
    subprocess.check_call([
        "/install/depmap-py/bin/python3.9", 
        "validate_jobs_complete.py", 
        os.path.join(save_pref, "completed_jobs.txt"),
        os.path.join(save_pref, "partitions.csv"),
        "features.csv",
        "predictions.csv"
    ])
    
    # Copy CSV files to local directory
    for blob in blobs:
        if blob.name.endswith('.csv'):
            destination_file_name = os.path.join(save_pref, "data", os.path.basename(blob.name))
            blob.download_to_filename(destination_file_name)
