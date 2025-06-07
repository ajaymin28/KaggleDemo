from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
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

    domainnet_classes: list = Field(["airplane", "ant", "apple"], "domain net classes to train")

    # Adv Training Config
    adv_alpha: float = Field(0.5, description="Grad Reverse factor")
    schedule_alpha: bool = Field(True, description="schedule alpha during training based on epochs")
    alpha_smoothing_factor: float = Field(-10.0, description="alpha smoothing factor")
    adv_lambda: float = Field(0.5, description="domain desc loss weight")

    # class Config:
    #     extra = "forbid"   # Raises error on unexpected fields for strictness