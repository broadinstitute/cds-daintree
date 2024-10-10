import argparse
import subprocess
import os
import sys
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_docker_command(cmd, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True)
            return  # Command succeeded, exit the function
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Exiting.")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run daintree fit models in a Docker container")
    
    # Mandatory arguments
    parser.add_argument("--config", required=True, help="Model configuration file")
    parser.add_argument("--out", required=True, help="Output directory")
    
    # Optional arguments
    parser.add_argument("--image", default="us.gcr.io/broad-achilles/daintree-sparkles:v4", help="Docker image name")
    parser.add_argument("--taiga-dir", default="~/.taiga", help="Path to Taiga token and cache")
    parser.add_argument("--sparkles-cache", default="~/.sparkles-cache", help="Path to Sparkles cache")
    parser.add_argument("--sparkles-path", default="/install/sparkles/bin/sparkles", help="Path to Sparkles executable")
    parser.add_argument("--sparkles-config", default="/daintree/sparkles-config", help="Path to Sparkles config file")
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=True, help="Run in test mode")
    parser.add_argument("--skipfit", type=str2bool, nargs='?', const=True, default=False, help="Skip the model fitting process")
    
    args = parser.parse_args()
    
    # Construct the Docker command
    cmd = [
        "docker", "run",
        "-w", "/daintree",
        "--pull=always",
        "-v", f"{os.getcwd()}:/daintree",
        "-v", f"{os.path.expanduser(args.taiga_dir)}:/root/.taiga",
        "-v", f"{os.path.expanduser(args.sparkles_cache)}:/root/.sparkles-cache",
        args.image,
        "/install/depmap-py/bin/python3.9", "-u", "run_fit_models.py", "collect-and-fit-generate-config",
        "--input-files", args.config,
        "--sparkles-path", args.sparkles_path,
        "--sparkles-config", args.sparkles_config,
        "--save-dir", args.out,
        "--test", str(args.test),
        "--skipfit", str(args.skipfit)
    ]

    # Run the Docker command 
    run_docker_command(cmd)

if __name__ == "__main__":
    main()
    