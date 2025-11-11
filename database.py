import sqlite3
import os
import json
from datetime import datetime

class Database:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), 'documents.db')
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_number TEXT,
                    sign TEXT,
                    text TEXT NOT NULL,
                    summary TEXT,
                    sender TEXT,
                    recipient TEXT,
                    department TEXT,
                    assignee TEXT,
                    processor TEXT,
                    processor_result TEXT,
                    priority TEXT DEFAULT 'normal',
                    progress TEXT,
                    note TEXT,
                    due_date TEXT,
                    received_date TEXT,
                    issued_date TEXT,
                    attachments TEXT DEFAULT '[]',
                    late_days INTEGER DEFAULT 0,
                    late_reason TEXT,
                    status TEXT DEFAULT 'Chưa xử lý',
                    outgoing INTEGER DEFAULT 0,
                    outgoing_status TEXT,
                    viewed INTEGER DEFAULT 0,
                    signer TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

    def add_document(self, doc_data):
        attachments = doc_data.get('attachments') or []
        try:
            attachments_json = json.dumps(attachments, ensure_ascii=False)
        except Exception:
            attachments_json = '[]'

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents
                (doc_number, sign, text, summary, sender, recipient, department, assignee,
                 processor, processor_result, priority, progress, note, due_date, received_date,
                 issued_date, attachments, late_days, late_reason, status, outgoing, outgoing_status,
                 viewed, signer, processed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                doc_data.get('doc_number'),
                doc_data.get('sign'),
                doc_data.get('text', ''),
                doc_data.get('summary') or (doc_data.get('text','')[:200] if doc_data.get('text') else ''),
                doc_data.get('sender'),
                doc_data.get('recipient'),
                doc_data.get('department'),
                doc_data.get('assignee'),
                doc_data.get('processor'),
                doc_data.get('processor_result'),
                doc_data.get('priority', 'normal'),
                doc_data.get('progress'),
                doc_data.get('note'),
                doc_data.get('due_date'),
                doc_data.get('received_date'),
                doc_data.get('issued_date'),
                attachments_json,
                doc_data.get('late_days', 0),
                doc_data.get('late_reason'),
                doc_data.get('status', 'Chưa xử lý'),
                1 if doc_data.get('outgoing') else 0,
                doc_data.get('outgoing_status'),
                1 if doc_data.get('viewed') else 0,
                doc_data.get('signer'),
                doc_data.get('processed_at')
            ))
            return cursor.lastrowid

    def get_documents(self, status=None, outgoing=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = 'SELECT * FROM documents'
            params = []
            conditions = []

            if status:
                conditions.append('status = ?')
                params.append(status)
            if outgoing is not None:
                conditions.append('outgoing = ?')
                params.append(1 if outgoing else 0)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' ORDER BY created_at DESC'
            cursor.execute(query, params)

            rows = [dict(r) for r in cursor.fetchall()]
            # parse attachments and convert ints to booleans
            for r in rows:
                try:
                    r['attachments'] = json.loads(r.get('attachments') or '[]')
                except Exception:
                    r['attachments'] = []
                r['outgoing'] = bool(r.get('outgoing', 0))
                r['viewed'] = bool(r.get('viewed', 0))
            return rows

    def mark_processed(self, doc_id, processor='system', result='Đã xử lý'):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE documents
                SET status = 'Đã xử lý',
                    processed_at = datetime('now'),
                    processor = ?,
                    processor_result = ?
                WHERE id = ?
            ''', (processor, result, doc_id))

    def add_user(self, user_data):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (email, name, password)
                VALUES (?, ?, ?)
            ''', (user_data['email'], user_data['name'], user_data['password']))
            return cursor.lastrowid

    def get_user_by_email(self, email):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            return dict(user) if user else None