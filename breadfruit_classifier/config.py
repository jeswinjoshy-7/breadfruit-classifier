"""
Configuration module for breadfruit maturity and quality classification.
All hyperparameters and paths are centralized here for easy experimentation.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class DataConfig:
    data_root: str = "/home/jeswin/data_augmentation/training_dataset"
    annotation_file: str = ""
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"
    class_name: str = "breadfruit"
    image_size: Tuple[int, int] = (224, 224)
    input_channels: int = 3
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    maturity_classes: List[str] = field(default_factory=lambda: ["immature", "partially_mature", "mature"])
    quality_classes: List[str] = field(default_factory=lambda: ["poor", "good", "excellent"])
    freeze_backbone_epochs: int = 5


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_weights: Tuple[float, float] = (0.5, 0.5)
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-7
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42


@dataclass
class AugmentationConfig:
    train_augmentation: bool = True
    random_horizontal_flip_prob: float = 0.5
    random_vertical_flip_prob: float = 0.1
    random_rotation_degrees: int = 30
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    random_affine_degrees: float = 15.0
    random_affine_translate: Tuple[float, float] = (0.1, 0.1)
    random_affine_scale: Tuple[float, float] = (0.9, 1.1)
    random_affine_shear: float = 10.0
    random_erasing_prob: float = 0.25
    random_erasing_scale: Tuple[float, float] = (0.02, 0.33)
    cutmix_prob: float = 0.0
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.2


@dataclass
class LoggingConfig:
    experiment_name: str = "breadfruit_maturity_quality"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    plot_dir: str = "plots"
    tensorboard: bool = False
    log_interval: int = 10


@dataclass
class GradCAMConfig:
    target_layer: str = "features.8"
    use_cuda: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    gradcam: GradCAMConfig = field(default_factory=GradCAMConfig)

    def __post_init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
