# https://arxiv.org/pdf/1709.01604
from attacks import AbstractAttack
from datasets import Dataset


class LossAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {self.name: -x["nlloss"]})
        return dataset
