from avalanche.core import SupervisedPlugin
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

# https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training#how-to-use-strategies-and-plugins


class OODDetectorPlugin(SupervisedPlugin):
    def __init__(self, detector):
        self.detector = detector

    def after_training_exp(self, strategy: "SupervisedTemplate", *args, **kwargs):
        # try after_training_exp
        self.detector.fit(strategy.dataloader, device=strategy.device)


# class OODDetectorPlugin(SupervisedPlugin):
#     def __init__(self, detector):
#         self.detector = detector

    def after_training_exp(self, strategy, **kwargs):
        # Assuming this is where the fit method of the detector is called
        custom_dataloader = DataLoader(strategy.experience.dataset, batch_size=32, shuffle=True)
        formatted_dataloader = self._format_dataloader(custom_dataloader)
        self.detector.fit(formatted_dataloader, device=strategy.device)

    def _format_dataloader(self, dataloader):
        # Function to format the dataloader to have batches of (x, y)
        for batch in dataloader:
            if len(batch) == 2:
                yield batch
            else:
                # Log or print the discarded elements for visualization
                # discarded_elements = batch[2:]
                # print("Discarded elements:", discarded_elements)
                yield batch[:2]
