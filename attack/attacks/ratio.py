# https://www.usenix.org/system/files/sec21-carlini-extracting.pdf
import os
from attacks import AbstractAttack
from attacks.utils import batch_nlloss, compute_nlloss
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np

class RatioAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config, device):
        super().__init__(name, model, tokenizer, config, device)
        self.reference_model, self.reference_tokenizer = self._load_reference()
        self.reference_device = torch.device('cpu')

    def _load_reference(self):
        from huggingface_hub import login
        login(token=self.config.get('hf_token') or os.environ.get('HF_TOKEN', ''))
        reference_model = AutoModelForCausalLM.from_pretrained(self.config['reference_model_path'], device_map='auto')
        reference_tokenizer = AutoTokenizer.from_pretrained(self.config['reference_model_path'])
        reference_tokenizer.pad_token = reference_tokenizer.eos_token
        return reference_model, reference_tokenizer

    def _compute_loss_fp32(self, model, input_ids, attention_mask):
        """Compute loss in float32 precision"""
        with torch.cuda.amp.autocast(enabled=False):
            try:
                input_ids = input_ids.to(dtype=torch.long)
                attention_mask = attention_mask.to(dtype=torch.long)

                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.to(dtype=torch.float32)
                
                if isinstance(model, torch.nn.DataParallel):
                    vocab_size = model.module.config.vocab_size
                else:
                    vocab_size = model.config.vocab_size
                
                shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
                shift_targets = input_ids[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., :-1]
                
                shift_targets[shift_attention_mask == 0] = -100
                
                loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
                loss = loss.view(input_ids.shape[0], -1)
                loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)
                
                result = loss.detach().cpu().numpy()
                
                # Clean up memory
                del outputs, logits, shift_logits, shift_targets, shift_attention_mask, loss
                torch.cuda.empty_cache()
                
                return result
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                raise e

    def ratio_score(self, batch):
        target_tokenized = self.tokenizer.batch_encode_plus(
            batch["text"], 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=512
        )
        reference_tokenized = self.reference_tokenizer.batch_encode_plus(
            batch["text"], 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=512
        )
        
        target_losses = self._compute_loss_fp32(
            self.model,
            target_tokenized['input_ids'].to(self.device),
            target_tokenized['attention_mask'].to(self.device)
        )
        reference_losses = self._compute_loss_fp32(
            self.reference_model,
            reference_tokenized['input_ids'].to(self.reference_device),
            reference_tokenized['attention_mask'].to(self.reference_device)
        )
        
        scores = []
        for t_loss, r_loss in zip(target_losses, reference_losses):
            scores.append(-t_loss / r_loss)
        
        # Clean up memory
        del target_tokenized, reference_tokenized
        torch.cuda.empty_cache()
        
        return {self.name: scores}

    def extract_features(self, batch):
        # Target model loss
        target_tokenized = self.tokenizer.batch_encode_plus(batch["text"], return_tensors='pt', padding=True)
        target_losses = self._compute_loss_fp32(
            self.model,
            target_tokenized['input_ids'].to(self.device),
            target_tokenized['attention_mask'].to(self.device)
        )
        
        # Reference model loss
        reference_tokenized = self.reference_tokenizer.batch_encode_plus(batch["text"], return_tensors='pt', padding=True)
        reference_losses = self._compute_loss_fp32(
            self.reference_model,
            reference_tokenized['input_ids'].to(self.reference_device),
            reference_tokenized['attention_mask'].to(self.reference_device)
        )
        
        # Compute ratio feature
        ratio_feature = [-t_loss - r_loss for t_loss, r_loss in zip(target_losses, reference_losses)]
        return np.array(ratio_feature).reshape(-1, 1)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.ratio_score(x),
            batched=True,
            batch_size=self.config.get('batch_size', 4),  # Reduced default batch size
            new_fingerprint=f"{self.signature(dataset)}_v5",
        )
        return dataset
