import os

# ===========================
# Create NeuralRecoNet Project Structure (Python version)
# ===========================

root = r"C:\Users\yagne\OneDrive\Desktop\JOB\projects\Recommender-System-with-Neural-Networks"

# Define directories
dirs = [
    f"{root}/data",
    f"{root}/notebooks",
    f"{root}/src/models",
    f"{root}/src/preprocessing",
    f"{root}/src/evaluation",
    f"{root}/config",
    f"{root}/app",
]

# Define placeholder files
files = [
    f"{root}/train.py",
    f"{root}/src/models/autoencoder.py",
    f"{root}/src/models/rnn_recommender.py",
    f"{root}/src/models/gnn_recommender.py",
    f"{root}/src/models/utils.py",
    f"{root}/src/preprocessing/data_loader.py",
    f"{root}/src/preprocessing/feature_engineering.py",
    f"{root}/src/evaluation/metrics.py",
    f"{root}/src/evaluation/visualizations.py",
    f"{root}/config/train_config.yaml",
    f"{root}/config/model_config.yaml",
    f"{root}/app/api.py",
    f"{root}/app/web_ui.py",
    f"{root}/requirements.txt"
]

# Create directories
for d in dirs:
    os.makedirs(d, exist_ok=True)



# Create placeholder files
for f in files:
    os.makedirs(os.path.dirname(f), exist_ok=True)
    open(f, "a", encoding="utf-8").close()

print("✅ NeuralRecoNet project structure created successfully at:")
print(root)