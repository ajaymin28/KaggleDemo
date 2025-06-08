from pydantic import BaseModel, Field
from typing import List
import argparse
import yaml

class ModelConfig(BaseModel):
    data_path: str = Field("/kaggle/input/domainnet/DomainNet", description="Path to data")
    n_classes: int = Field(10, description="Number of output classes")
    image_size: int = Field(224, description="Input image size")
    num_domains: int = Field(2, description="Number of domains for adversarial training")
    adv_training: bool = Field(False, description="Whether to use adversarial training")

    DEBUG: bool = Field(False, description="debug prints")

    # training 
    use_fp16: bool = Field(False, description="use fp16 mixed precision during training")
    n_epochs: int = Field(10, description="use fp16 mixed precision during training")
    batch_size: int = Field(32, description="train batch size")
    learning_rate: float = Field(1e-3, description="train batch size")

    domainnet_classes: List[str] = Field(default_factory=lambda: ["airplane", "ant", "apple"], description="List of class names")
    TRAIN_DOMAINS: List[str] = Field(default_factory=lambda: ["clipart", "painting"], description="Training domains")
    TEST_DOMAINS: List[str] = Field(default_factory=lambda: ["sketch"], description="Test domains")

    # Adv Training Config
    adv_alpha: float = Field(0.5, description="Grad Reverse factor")
    schedule_alpha: bool = Field(True, description="schedule alpha during training based on epochs")
    alpha_smoothing_factor: float = Field(-10.0, description="alpha smoothing factor")
    adv_lambda: float = Field(0.5, description="domain desc loss weight")
    domain_method: str = Field("dann", description="dannm,cdan,mmd")  

    # logging
    wandb_project: str = Field("domain_adv_training_domainnet", description="project name")
    wandb_apikey: str = Field("", description="api key for wandb")

    early_stop_patience: int = Field(5, description="early_stop_patience")
    early_stop_min_delta: float = Field(1e-4, description="early_stop_min_delta")

    # class Config:
    #     extra = "forbid"   # Raises error on unexpected fields for strictness


def parse_args():
    parser = argparse.ArgumentParser()

    # Optionally, allow config file for initial values
    parser.add_argument("--config", type=str, default="config/mynet.yaml", help="Path to YAML config file")

    # Add all config fields as optional CLI args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--num_domains", type=int)
    parser.add_argument("--adv_training", type=lambda x: x.lower() == "true")
    parser.add_argument("--DEBUG", type=lambda x: x.lower() == "true")
    parser.add_argument("--use_fp16", type=lambda x: x.lower() == "true")
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--domainnet_classes", nargs="+", type=str)
    
    parser.add_argument("--adv_alpha", type=float)
    parser.add_argument("--schedule_alpha", type=lambda x: x.lower() == "true")
    parser.add_argument("--alpha_smoothing_factor", type=float)
    parser.add_argument("--adv_lambda", type=float)
    parser.add_argument("--domain_method", type=str, help="dann,cdan,mmd")

    parser.add_argument("--TRAIN_DOMAINS", nargs="+", type=str, help="Training domains")
    parser.add_argument("--TEST_DOMAINS", nargs="+", type=str, help="Test domains")

    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_apikey", type=str)

    args = parser.parse_args()
    return args


def load_config(args):
    # Start from config file if provided
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Override with CLI args (only those provided)
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config_dict[key] = value

    return ModelConfig(**config_dict)