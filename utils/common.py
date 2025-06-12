import os
import torch

class GlobalConfig:
    """
    Global configuration settings for the EEG-Image project.
    Centralizes paths and other high-level settings.
    """
    # --- Base Directories ---
    # Adjust these paths based on your project structure and environment
    PROJECT_ROOT = "D:\\2025\\Projects\\KaggleDemo"
    DATA_BASE_DIR = "D:\\Datasets\\EEG DATASET\\things2\\NICE-EEG"
    MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, "model", "grok") # Base directory for saving models and checkpoints

    # --- Specific File/Directory Paths ---
    EEG_DATA_PATH = os.path.join(DATA_BASE_DIR, "Data", "Things-EEG2", "Preprocessed_data_250Hz")
    TEST_CENTER_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature")
    IMG_DATA_PATH = os.path.join(DATA_BASE_DIR, "dnn_feature") 

    # --- Wandb Settings ---
    WANDB_PROJECT_NAME = "EEG_Domain_Adv_Things2"
    WANDB_CODE_DIR = PROJECT_ROOT # Directory containing the code to be logged

    # --- Other Global Settings (optional) ---
    # Add any other settings that are constant across different training runs
    # e.g., default device, logging level, etc.
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Requires torch import if used here

# Instantiate the global configuration
global_config = GlobalConfig()


class TrainConfig:
    global_config = global_config  # Access the global config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 100 # subject data will be trained for local_epochs
    batch_size = 1024

    channels = 63
    time_points = 250
    sessions = 4
    mean_eeg_data = False # Important for VAE input shape
    keep_dim_after_mean = False # Important for VAE input shape
    cache_data = True # Enable memmapping
    image_feature_dim = 768 # should be same as encoder_output_dim
    num_subjects = 10 # Number of subjects in the dataset

    load_pre_trained_models = False

    learning_rate = 0.002
    discriminator_lr = 0.002

    weight_decay = 1e-5
    latent_dim = 768 # should be same as image features
    dropout_rate = 0.1


    # Populate TrainConfig with paths from GlobalConfig
    eeg_data_path = global_config.EEG_DATA_PATH
    test_center_path = global_config.TEST_CENTER_PATH
    model_save_base_dir = global_config.MODEL_BASE_DIR
    data_base_dir = global_config.DATA_BASE_DIR # Used by EEG_Dataset3
    
    Contrastive_augmentation = True
    nSub = 1
    nSub_Contrastive = 2
    
    EEG_Augmentation = False
    Total_Subjects = 2
    MultiSubject = True
    TestSubject = 1
    dnn = "clip"

    # adv training
    lambda_adv = 0.1 # If adv_loss dominates, reduce lambda_adv; if task performance(image features) suffers, increase it.
    max_lambda_adv = 0.1
    enable_adv_training = True

    #otho loss between cls feat and subj feat
    lambda_ortho = 0.0001

    alpha = 0.0
    
    log_test_data = True
    Train = True

    #debug
    profile_code = False
    encoder_output_dim = 768
    wandb_apikey = ""
    wandb_project = "things_eeg_adv_training"

    # Things DS attributes
    NUM_CLASSES = 1654
    TRAIN_SUBJECT_IDS = [1]
    VALIDATION_SUBJECT_IDS = [1]
    TRAIN_SESSION_IDS = [0,1]
    VALIDATION_SESSION_IDS = [2]
    ONE_SUBJECT_CLS = True

