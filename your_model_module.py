import os
import re
import csv
import logging
from collections import Counter

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'documents.csv')
MODEL_FILENAME = 'document_classifier_model.joblib'
VECTORIZER_FILENAME = 'tfidf_vectorizer.joblib'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
VECTORIZER_PATH = os.path.join(BASE_DIR, VECTORIZER_FILENAME)

MODEL = None
VECTORIZER = None

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model(test_size=0.2, max_features=5000, ngram_range=(1, 2), max_iter=2000):
    logger.info("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
    try:
        if not os.path.exists(DATA_PATH):
            logger.error(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
            return {"success": False, "message": "data file not found"}

        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error("CSV phải có cột 'text' và 'label'")
            return {"success": False, "message": "invalid csv columns"}

        df = df.dropna(subset=['text', 'label']).copy()
        df['cleaned_text'] = df['text'].apply(clean_text)
        X = df['cleaned_text']
        y = df['label'].astype(str)

        label_counts = Counter(y)
        logger.info(f"Phân bố nhãn: {dict(label_counts)}")

        stratify_param = y if len(label_counts) > 1 and min(label_counts.values()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_param
        )

        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                   token_pattern=r'(?u)\b\w+\b')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)

        os.makedirs(DATA_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        logger.info(f"Huấn luyện xong - Accuracy: {acc:.4f}")
        logger.info("Classification report:\n" + report)
        logger.info(f"Đã lưu model -> {MODEL_PATH}")
        logger.info(f"Đã lưu vectorizer -> {VECTORIZER_PATH}")

        return {"success": True, "accuracy": float(acc), "report": report}

    except Exception as e:
        logger.exception("Lỗi khi huấn luyện mô hình")
        return {"success": False, "message": str(e)}

def load_model_and_vectorizer(auto_train=False):
    global MODEL, VECTORIZER
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            VECTORIZER = joblib.load(VECTORIZER_PATH)
            logger.info("Đã tải mô hình và vectorizer thành công")
            return True
        except Exception as e:
            logger.exception("Lỗi khi tải model/vectorizer")
            MODEL = None
            VECTORIZER = None
            return False
    else:
        logger.warning("Không tìm thấy file model hoặc vectorizer")
        if auto_train:
            result = train_model()
            if result["success"]:
                return load_model_and_vectorizer(auto_train=False)
        return False

def predict(text):
    if MODEL is None or VECTORIZER is None:
        return {"error": "Model chưa được tải"}
        
    try:
        cleaned = clean_text(text)
        vec = VECTORIZER.transform([cleaned])
        label = MODEL.predict(vec)[0]
        
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(vec)[0]
            confidence = float(max(proba))
        else:
            confidence = None
            
        return {
            "label": label,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.exception("Lỗi khi dự đoán")
        return {"error": str(e)}

def predict_with_confidence(text, top_n=3):
    if MODEL is None or VECTORIZER is None:
        return {"error": "Model chưa được tải"}
        
    try:
        cleaned = clean_text(text)
        vec = VECTORIZER.transform([cleaned])
        
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(vec)[0]
            indices = proba.argsort()[-top_n:][::-1]
            
            predictions = [
                {
                    "label": MODEL.classes_[idx],
                    "confidence": float(proba[idx])
                }
                for idx in indices
            ]
            
            return {
                "label": predictions[0]["label"],
                "confidence": predictions[0]["confidence"],
                "top_predictions": predictions
            }
            
        else:
            label = MODEL.predict(vec)[0]
            return {
                "label": label,
                "confidence": None,
                "top_predictions": [{"label": label, "confidence": None}]
            }
            
    except Exception as e:
        logger.exception("Lỗi khi dự đoán")
        return {"error": str(e)}

def append_data_to_csv(text, label):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_exists = os.path.exists(DATA_PATH)
        
        with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['text', 'label'])
            writer.writerow([text, label])
            
        logger.info("Đã thêm dữ liệu vào CSV")
        return True
        
    except Exception as e:
        logger.exception("Lỗi khi ghi CSV")
        return False

# Load model khi import module
load_model_and_vectorizer(auto_train=False)

if __name__ == '__main__':
    result = train_model()
    if result["success"]:
        load_model_and_vectorizer()
    else:
        logger.error("Training thất bại")
