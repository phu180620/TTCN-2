import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from database import Database
import your_model_module as model_mod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Production-oriented defaults
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['PREFERRED_URL_SCHEME'] = 'https'

# If behind reverse-proxy / CDN, allow correct external URL generation
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

db = Database()

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json() or {}
        text = (data.get('text') or '').strip()
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        result = model_mod.predict(text)
        resp = {
            'success': result.get('success', True),
            'label': result.get('label') or result.get('result'),
            'result': result.get('label') or result.get('result'),
            'ipa': result.get('ipa', '')
        }
        return jsonify(resp)
    except Exception as e:
        logger.exception("Classification error")
        return jsonify({'error': str(e)}), 500

@app.route('/send_document', methods=['POST'])
def send_document():
    try:
        attachments_list = []
        if request.content_type and 'multipart/form-data' in request.content_type:
            form = request.form
            files = request.files.getlist('attachments')
            for f in files:
                if f and f.filename:
                    filename = secure_filename(f.filename)
                    save_path = os.path.join(UPLOAD_DIR, filename)
                    base, ext = os.path.splitext(filename)
                    idx = 1
                    while os.path.exists(save_path):
                        filename = f"{base}_{idx}{ext}"
                        save_path = os.path.join(UPLOAD_DIR, filename)
                        idx += 1
                    f.save(save_path)
                    attachments_list.append('/uploads/' + filename)
            payload = {
                'text': form.get('text', ''),
                'department': form.get('department'),
                'priority': form.get('priority', 'normal'),
                'note': form.get('note', ''),
                'issued_date': form.get('date'),
                'doc_number': form.get('doc_number') or form.get('sign') or None,
                'sign': form.get('sign'),
                'recipient': form.get('receiver_name') or form.get('receiver'),
                'signer': form.get('signer'),
                'attachments': attachments_list,
                'outgoing': True,
                'outgoing_status': form.get('outgoing_status') or 'Đã phát hành',
                'status': form.get('status') or 'Đã phát hành'
            }
        else:
            data = request.get_json() or {}
            payload = {
                'text': data.get('text', ''),
                'department': data.get('department'),
                'priority': data.get('priority', 'normal'),
                'note': data.get('note', ''),
                'issued_date': data.get('date') or data.get('issued_date'),
                'doc_number': data.get('doc_number'),
                'sign': data.get('sign'),
                'recipient': data.get('receiver_name') or data.get('recipient'),
                'signer': data.get('signer'),
                'attachments': data.get('attachments', []),
                'outgoing': True,
                'outgoing_status': data.get('outgoing_status') or 'Đã phát hành',
                'status': data.get('status') or 'Đã phát hành'
            }

        if not payload['text'] or not payload['department']:
            return jsonify({'success': False, 'message': 'Text and department are required'}), 400

        doc_id = db.add_document(payload)
        return jsonify({'success': True, 'id': doc_id, 'message': 'Document sent successfully'})
    except Exception as e:
        logger.exception("Error sending document")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/incoming')
def get_incoming():
    try:
        status = request.args.get('status')
        docs = db.get_documents(status=status, outgoing=False)
        return jsonify(docs)
    except Exception as e:
        logger.exception("Error fetching incoming")
        return jsonify({'error': str(e)}), 500

@app.route('/api/outgoing')
def get_outgoing():
    try:
        docs = db.get_documents(outgoing=True)
        for d in docs:
            if d.get('issued_date') and not d.get('date'):
                d['date'] = d.get('issued_date')
        return jsonify(docs)
    except Exception as e:
        logger.exception("Error fetching outgoing")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add_sample', methods=['POST'])
def add_sample():
    try:
        data = request.get_json() or {}
        text = data.get('text','').strip()
        label = data.get('label','').strip()
        if not text or not label:
            return jsonify({'success': False, 'message': 'Missing fields'}), 400
        try:
            db.add_sample(text, label)
        except Exception:
            logger.exception("DB add_sample failed, continuing")
        model_mod.add_sample(text, label, retrain=True)
        return jsonify({'success': True, 'message': 'Sample added'})
    except Exception as e:
        logger.exception("Error add_sample")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/mark_processed', methods=['POST'])
def mark_processed():
    try:
        data = request.get_json() or {}
        doc_id = data.get('id')
        if not doc_id:
            return jsonify({'success': False, 'message': 'Document id required'}), 400
        db.mark_processed(doc_id, processor=data.get('processor','system'), result=data.get('result','Đã xử lý'))
        return jsonify({'success': True})
    except Exception as e:
        logger.exception("Error mark_processed")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        name = data.get('name','').strip()
        email = data.get('email','').strip()
        password = data.get('password','')
        if not name or not email or not password:
            return jsonify({'success': False, 'message': 'Missing fields'}), 400
        if db.get_user_by_email(email):
            return jsonify({'success': False, 'message': 'Email already exists'}), 400
        db.add_user({'email': email, 'name': name, 'password': generate_password_hash(password)})
        return jsonify({'success': True})
    except Exception as e:
        logger.exception("Register error")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        email = data.get('email','').strip()
        password = data.get('password','')
        if not email or not password:
            return jsonify({'success': False, 'message': 'Missing credentials'}), 400
        user = db.get_user_by_email(email)
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
        return jsonify({'success': True, 'name': user['name'], 'email': user['email']})
    except Exception as e:
        logger.exception("Login error")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/sitemap.xml')
def sitemap():
    try:
        pages = []
        now = datetime.utcnow().strftime('%Y-%m-%d')
        pages.append({
            'loc': url_for('home', _external=True),
            'lastmod': now,
            'changefreq': 'daily',
            'priority': '1.0'
        })
        docs = db.get_documents(limit=50)
        for doc in docs:
            created = doc.get('created_at') or now
            if isinstance(created, str):
                try:
                    created_dt = datetime.strptime(created, '%Y-%m-%d %H:%M:%S')
                    created = created_dt.strftime('%Y-%m-%d')
                except Exception:
                    created = now
            pages.append({
                'loc': url_for('home', _external=True) + f'#doc-{doc.get("id")}',
                'lastmod': created,
                'changefreq': 'weekly',
                'priority': '0.8'
            })

        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
        for page in pages:
            xml.append('  <url>')
            xml.append(f'    <loc>{page["loc"]}</loc>')
            xml.append(f'    <lastmod>{page["lastmod"]}</lastmod>')
            xml.append(f'    <changefreq>{page["changefreq"]}</changefreq>')
            xml.append(f'    <priority>{page["priority"]}</priority>')
            xml.append('  </url>')
        xml.append('</urlset>')
        return Response('\n'.join(xml), mimetype='application/xml')
    except Exception as e:
        logger.exception("Sitemap generation error")
        return Response('<?xml version="1.0"?><urlset></urlset>', mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    return Response(
        f"""User-agent: *
Allow: /
Disallow: /uploads/
Disallow: /admin/
Sitemap: {url_for('sitemap', _external=True)}""",
        mimetype='text/plain'
    )

@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    logger.info("Starting server on http://0.0.0.0:5000 (debug=False)")
    app.run(host='0.0.0.0', port=5000, debug=False)