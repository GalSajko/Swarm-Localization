import kagglehub

# Download latest version
path = kagglehub.dataset_download("sabermalek/spacenetsi")

print("Path to dataset files:", path)