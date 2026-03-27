import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
import numpy as np

class QAlignWorker:
    def __init__(self):
        self.model_id = "WT-MM/vit-base-blur"

        config = AutoConfig.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype="auto"
        )

        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def eval_rule(self, observed: float, rule: dict) -> dict:
        """
        Inverse Sigmoid for Blurry-High/Clear-Low signals.
        """
        k = -150
        x0 = 0.650

        normalized_score = 1 / (1 + np.exp(-k * (observed - x0)))

        quality_index = float(normalized_score)
        threshold = rule.get("threshold", 0.5)

        return {
            "is_passed": quality_index >= threshold,
            "quality_index": round(quality_index, 4),
            "raw_signal": round(observed, 6),
            "status": "APPROVED" if quality_index >= threshold else "REJECTED"
        }

    def get_signal(self, pil_image: Image.Image, rule: dict) -> dict:
        # Preprocess image to tensor [1, 3, 224, 224]
        img_tensor = self.preprocess(pil_image).unsqueeze(0)

        # Forward pass using pixel_values
        with torch.no_grad():
            outputs = self.model(pixel_values=img_tensor)
            cls_repr = outputs.last_hidden_state[:, 0, :]

            peak = cls_repr.max().item()
            avg = cls_repr.mean().item()

            raw_score = abs(peak - avg)

        validation = self.eval_rule(raw_score, rule)

        status = "APPROVED" if validation.get("is_passed", False) else "REJECTED"

        return {
            "id": "gen-uuid-placeholder",
            "jobId": "job_agent_001",
            "isValidated": True,
            "status": status,
            "cloudLocation": "s3://apple-vision-data/ingest/sample_01.jpg",
            "rulesPassed": ["iqatr_quality_check"] if validation.get("is_passed", False) else [],
            "rawOutputs": {
                "quality_index": validation.get("quality_index", None),
                "raw_signal": validation.get("raw_signal", raw_score)
            },
            "metadata": {
                "model_id": self.model_id,
                "threshold_applied": rule.get("threshold")
            },
            "collectionId": "col_spring_2026",
            "llmJudgeApplied": False
        }