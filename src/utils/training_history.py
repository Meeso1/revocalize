from dataclasses import dataclass


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None = None
    val_accuracy: float | None = None

    def to_wandb_dict(self) -> dict[str, float | None]:
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy
        }


@dataclass
class TrainingHistory:
    train_loss: list[float] = None
    train_accuracy: list[float] = None
    val_loss: list[float | None] = None
    val_accuracy: list[float | None] = None

    # Needed because default values in python are class-specific, not instance-specific
    def __post_init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    @staticmethod
    def from_entries(entries: list[TrainingHistoryEntry]) -> "TrainingHistory":
        ordered_entries = sorted(entries, key=lambda x: x.epoch)
        return TrainingHistory(
            train_loss=[entry.train_loss for entry in ordered_entries],
            train_accuracy=[entry.train_accuracy for entry in ordered_entries],
            val_loss=[entry.val_loss for entry in ordered_entries],
            val_accuracy=[entry.val_accuracy for entry in ordered_entries]
        )