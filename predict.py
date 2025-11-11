import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, model_path='./fine-tuned-phobert-model'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            
            label_map_path = os.path.join(model_path, 'label_map.json')
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    self.label_map = json.load(f)
            else:
                raise FileNotFoundError("Không tìm thấy file label_map.json")
            
            logger.info("Đã khởi tạo model thành công")
            logger.info(f"Các nhãn: {self.label_map}")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Lỗi khởi tạo model: {str(e)}")
            raise

    def predict(self, text, threshold=0.5):
        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                values, indices = torch.topk(probabilities, 2)
                
                top_pred_idx = str(indices[0][0].item())
                confidence = float(values[0][0])
                
                if confidence < threshold:
                    result = "Không đủ tin cậy"
                else:
                    result = self.label_map[top_pred_idx]

                return {
                    "label": result,
                    "confidence": confidence,
                    "top_predictions": [
                        {
                            "label": self.label_map[str(indices[0][i].item())],
                            "confidence": float(values[0][i])
                        }
                        for i in range(2)
                    ]
                }
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            raise

def main():
    try:
        classifier = DocumentClassifier()
        
        test_texts = [
            "Quyết định về việc tăng lương cho nhân viên phòng kỹ thuật",
            "Báo cáo tài chính quý 3 năm 2023",
            "Hợp đồng mua bán thiết bị văn phòng"
        ]
        
        logger.info("Bắt đầu test dự đoán:")
        for text in test_texts:
            result = classifier.predict(text)
            logger.info("\nVăn bản: " + text[:100] + "...")
            logger.info(f"Kết quả: {result['label']}")
            logger.info(f"Độ tin cậy: {result['confidence']:.2%}")
            logger.info("Top predictions:")
            for pred in result['top_predictions']:
                logger.info(f"  {pred['label']}: {pred['confidence']:.2%}")
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình test: {str(e)}")

if __name__ == "__main__":
    main()