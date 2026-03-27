import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


class QAlignWorker:
    def __init__(self):
        self.model_id = "Lee1219/iqatr-musique"
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def eval_rule(self, observed: float, rule: dict) -> dict:
        """
        Maps raw BERT signals to a 0.0 - 1.0 Quality Score.
        """

        val_min = -0.010
        val_max = 0.005

        normalized_score = (observed - val_min) / (val_max - val_min)

        normalized_score = max(0.0, min(1.0, normalized_score))

        threshold = rule.get("threshold", 0.5)
        operator = rule.get("operator", ">=")

        ops = {
            ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b, "<": lambda a, b: a < b, "==": lambda a, b: a == b
        }

        is_passed = ops.get(operator, lambda a, b: False)(normalized_score, threshold)

        return {
            "is_passed": is_passed,
            "quality_index": round(normalized_score, 4),
            "applied_operator": operator,
            "applied_threshold": threshold,
            "raw_signal": round(observed, 6)
        }

    def get_signal(self, pil_image: Image, rule: dict) -> dict:
        img_tensor = self.preprocess(pil_image).unsqueeze(0)
        patches = F.unfold(img_tensor, kernel_size=16, stride=16)
        inputs_bert = patches.transpose(1, 2)

        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_bert)
            raw_score = float(outputs.last_hidden_state.mean().item())

        validation = self.eval_rule(raw_score, rule)

        status = "APPROVED" if validation["is_passed"] else "REJECTED"

        return {
            "id": "gen-uuid-placeholder",
            "jobId": "job_agent_001",
            "isValidated": True,
            "status": status,
            "cloudLocation": "s3://apple-vision-data/ingest/sample_01.jpg",
            "rulesPassed": ["iqatr_quality_check"] if validation["is_passed"] else [],
            "rawOutputs": {
                "quality_index": validation["quality_index"],
                "raw_bert_signal": validation["raw_signal"]
            },
            "metadata": {
                "model_id": self.model_id,
                "threshold_applied": rule["threshold"]
            },
            "collectionId": "col_spring_2026",
            "llmJudgeApplied": False
        }