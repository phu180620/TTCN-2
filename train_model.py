import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_and_split_data(file_path, model_name="vinai/phobert-base"):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Đã đọc {len(df)} mẫu từ {file_path}")
        logger.info(f"Các cột: {df.columns.tolist()}")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("File CSV phải có cột 'text' và 'label'")
            
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].str.strip()
        unique_labels = sorted(df['label'].unique())
        label_map = {str(i): label for i, label in enumerate(unique_labels)}
        id_to_label = {v: k for k, v in label_map.items()}
        
        logger.info(f"Các nhãn phân loại: {label_map}")
        df['label_id'] = df['label'].map(lambda x: int(id_to_label[x]))
        
        label_dist = df['label'].value_counts()
        logger.info("\nPhân bố nhãn:")
        for label, count in label_dist.items():
            logger.info(f"{label}: {count} mẫu")
            
        X_train, X_val, y_train, y_val = train_test_split(
            df['text'],
            df['label_id'],
            test_size=0.2,
            random_state=42,
            stratify=df['label_id']
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_function(texts):
            return tokenizer(
                texts.tolist(),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
        
        logger.info("Đang tokenize dữ liệu...")
        train_encodings = tokenize_function(X_train)
        val_encodings = tokenize_function(X_val)
        
        return (train_encodings, y_train.tolist()), (val_encodings, y_val.tolist()), label_map
        
    except Exception as e:
        logger.error(f"Lỗi trong preprocess_and_split_data: {str(e)}")
        return None, None, None

class DocumentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    try:
        data_path = 'data/documents.csv'
        
        logger.info("Bắt đầu tiền xử lý dữ liệu...")
        train_data, val_data, label_map = preprocess_and_split_data(data_path)
        
        if not train_data:
            raise ValueError("Không thể xử lý dữ liệu training")
        
        train_dataset = DocumentDataset(train_data[0], train_data[1])
        val_dataset = DocumentDataset(val_data[0], val_data[1])
        
        num_labels = len(label_map)
        logger.info(f"Khởi tạo model với {num_labels} nhãn...")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            num_labels=num_labels
        )
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            do_eval=True,
            do_train=True,
            save_total_limit=2,
            save_steps=100,
            eval_steps=100
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        logger.info("Bắt đầu huấn luyện...")
        trainer.train()
        
        model_save_path = './fine-tuned-phobert-model'
        model.save_pretrained(model_save_path)
        
        with open(os.path.join(model_save_path, 'label_map.json'), 'w', encoding='utf-8') as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Đã lưu model tại {model_save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình training: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        logger.info("Training hoàn tất thành công!")
    else:
        logger.error("Training thất bại!")