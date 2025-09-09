from flask import Flask, request, redirect, url_for, session, jsonify
import numpy as np
import os
import math
from typing import Tuple

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

MODEL_PATH = os.environ.get("MODEL_PATH", "keras_model.h5")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.txt")
USERNAME = os.environ.get("APP_USERNAME", "admin")
PASSWORD = os.environ.get("APP_PASSWORD", "1234")


def _try_import_tf():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        return tf, load_model
    except Exception:
        return None, None


tf, load_model = _try_import_tf()


class ModelManager:
    def __init__(self, model_path: str, labels_path: str):
        self.labels = self._load_labels(labels_path)
        self.model = None
        self.mode = "fallback"
        if os.path.exists(model_path) and load_model is not None:
            try:
                self.model = load_model(model_path)
                self.mode = "tensorflow"
            except Exception:
                self.model = None
                self.mode = "fallback"
        if not self.labels:
            self.labels = ["Kirli Deniz", "Temiz Deniz"]
        elif len(self.labels) == 1:
            self.labels = [self.labels[0], self.labels[0]]

    @staticmethod
    def _load_labels(path: str):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception:
                return []
        return []

    def predict(self, lat: float, lng: float) -> Tuple[str, float, str]:
        if self.model is not None:
            try:
                x = coords_to_input(lat, lng)
                preds = self.model.predict(x, verbose=0)
                cls = int(np.argmax(preds[0]))
                conf = float(np.max(preds[0]))
                label = self.labels[cls] if 0 <= cls < len(self.labels) else str(cls)
                return label, conf, "tensorflow"
            except Exception:
                pass
        label, conf = fallback_predict(lat, lng, self.labels)
        return label, conf, "fallback"


def coords_to_input(lat: float, lng: float) -> np.ndarray:
    x = np.array([[lat / 90.0, lng / 180.0]], dtype=np.float32)
    return x


def fallback_predict(lat: float, lng: float, labels: list) -> Tuple[str, float]:
    s = 0.5 * (math.sin(math.radians(lat * 3.1)) + math.cos(math.radians(lng * 2.3)))
    prob_clean = (s + 1.0) / 2.0
    clean_label = labels[1] if len(labels) > 1 else "Temiz Deniz"
    dirty_label = labels[0] if len(labels) > 0 else "Kirli Deniz"
    label = clean_label if prob_clean >= 0.5 else dirty_label
    conf = prob_clean if label == clean_label else (1.0 - prob_clean)
    return label, conf


model_mgr = ModelManager(MODEL_PATH, LABELS_PATH)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("map_page"))
        return "Login Failed", 401
    return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>Login</title>
            <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; margin: 0; padding: 0; display: grid; place-items: center; height: 100vh; background: #0b1f2a; color: #e8eef2; }
                form { background: rgba(255,255,255,0.06); padding: 24px; border-radius: 16px; backdrop-filter: blur(8px); box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
                input { display:block; width: 260px; padding: 10px 12px; margin: 8px 0; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.2); color: #e8eef2; }
                button { width: 100%; padding: 10px 12px; border: none; border-radius: 10px; background: #25a; color: white; font-weight: 600; cursor: pointer; }
                .hint { font-size: 12px; opacity: 0.8; margin-top: 6px; }
            </style>
        </head>
        <body>
            <form method="POST">
                <h2>Ocean Quality Map</h2>
                <input type="text" name="username" placeholder="Username" required value="admin" />
                <input type="password" name="password" placeholder="Password" required value="1234" />
                <button type="submit">Login</button>
                <div class="hint">Demo creds: admin / 1234</div>
            </form>
        </body>
        </html>
    '''


@app.route("/map")
def map_page():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>World Ocean Quality</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
        <style>
            html, body, #map {{ height: 100%; margin: 0; }}
            .badge {{ position: fixed; top: 10px; right: 10px; background: #fff; padding: 6px 10px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15); font-family: system-ui, sans-serif; }}
            .result-popup {{ font-family: system-ui, sans-serif; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div class="badge">Mode: {model_mgr.mode.upper()}</div>
        <script>
            var map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {{
                maxZoom: 18,
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }}).addTo(map);
            var lastMarker = null;
            map.on('click', function(e) {{
                if (lastMarker) {{ map.removeLayer(lastMarker); }}
                var lat = e.latlng.lat; var lng = e.latlng.lng;
                lastMarker = L.marker([lat, lng]).addTo(map);
                fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ lat: lat, lng: lng }})
                }})
                .then(res => res.json())
                .then(data => {{
                    var label = data.result;
                    var conf = (data.confidence !== undefined) ? (Math.round(data.confidence * 100)) + '%' : '—';
                    var mode = data.mode ? data.mode.toUpperCase() : 'UNKNOWN';
                    var html = `<div class="result-popup"><b>Sonuç:</b> ${label}<br/><b>Güven:</b> ${conf}<br/><b>Çalışma Modu:</b> ${mode}</div>`;
                    lastMarker.bindPopup(html).openPopup();
                }})
                .catch(err => {{
                    lastMarker.bindPopup('Hata: ' + err).openPopup();
                }});
            }});
        </script>
    </body>
    </html>
    '''


@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True, silent=True) or {}
    try:
        lat = float(data.get("lat"))
        lng = float(data.get("lng"))
    except Exception:
        return jsonify({"error": "invalid_coordinates"}), 400
    label, conf, mode = model_mgr.predict(lat, lng)
    return jsonify({
        "result": label,
        "confidence": conf,
        "mode": mode,
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "tensorflow_available": bool(load_model is not None and tf is not None),
        "model_loaded": model_mgr.mode == "tensorflow",
        "labels": model_mgr.labels,
        "mode": model_mgr.mode,
    })


def _run_tests():
    import json
    print("Running smoke tests...")
    with app.test_client() as c:
        r = c.get("/map", follow_redirects=False)
        assert r.status_code in (301, 302)
        r = c.get("/")
        assert r.status_code == 200
        assert b"Login" in r.data
        r = c.post("/", data={"username": "x", "password": "y"})
        assert r.status_code == 401
        r = c.post("/", data={"username": USERNAME, "password": PASSWORD}, follow_redirects=False)
        assert r.status_code in (301, 302)
        assert "/map" in r.headers.get("Location", "")
        c.post("/", data={"username": USERNAME, "password": PASSWORD})
        r = c.get("/map")
        assert r.status_code == 200
        assert b"World Ocean Quality" in r.data
        payload = {"lat": 10.0, "lng": 20.0}
        r = c.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        js = r.get_json()
        assert js and js.get("result") in ("Kirli Deniz", "Temiz Deniz")
        assert "confidence" in js and "mode" in js
        r = c.get("/health")
        assert r.status_code == 200
        js = r.get_json()
        assert js["status"] == "ok"
    print("All smoke tests passed.")


if __name__ == "__main__":
    if os.environ.get("RUN_TESTS") == "1":
        _run_tests()
    else:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
