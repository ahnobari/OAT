from huggingface_hub import snapshot_download

if __name__ == "__main__":
    print("Downloading OpenOAT dataset...")
    snapshot_download(
        repo_id="OpenTO/OpenOAT",
        repo_type="dataset",
        local_dir="."
    )
    print("Download complete.")