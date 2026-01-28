# https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
import zlib

from attacks import AbstractAttack
from datasets import Dataset


def zlib_score(record, ep = 1e-5):
    text = record["text"]
    loss = record["nlloss"]
    zlib_entropy = len(zlib.compress(text.encode()))/(len(text) + ep)
    zlib_score = -loss / (zlib_entropy + ep)
    return zlib_score

class ZlibAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {self.name: zlib_score(x)})
        return dataset