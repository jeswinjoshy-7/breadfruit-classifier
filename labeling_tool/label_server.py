"""Simple web-based labeling tool. Run: python label_server.py"""
import csv
import json
import os
import re
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

AUG_PATTERN = re.compile(r'(aug|flip|rot|crop|noise|blur|shadow|elastic|gamma|zoom|jitter|trans)')
DATA_ROOT = Path(__file__).parent.parent / "training_dataset"
CSV_PATH = Path(__file__).parent / "annotations.csv"

MATURITY_CLASSES = ["immature", "partially_mature", "mature"]
QUALITY_CLASSES = ["poor", "good", "excellent"]

class LabelHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/api/images":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            images = get_unlabeled()
            self.wfile.write(json.dumps({"images": images, "total": len(images),
                                          "maturity_classes": MATURITY_CLASSES,
                                          "quality_classes": QUALITY_CLASSES}).encode())
        elif self.path.startswith("/images/"):
            fname = self.path[len("/images/"):]
            for split in ["train", "val", "test"]:
                p = DATA_ROOT / split / "breadfruit" / fname
                if p.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "image/jpeg")
                    self.end_headers()
                    with open(p, "rb") as f:
                        self.wfile.write(f.read())
                    return
            self.send_response(404)
            self.end_headers()
        elif self.path == "/api/save":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            save_label(data["filename"], data["maturity"], data["quality"])
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            remaining = count_unlabeled()
            self.wfile.write(json.dumps({"status": "ok", "remaining": remaining}).encode())
        elif self.path.startswith("/api/skip"):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            remaining = count_unlabeled()
            self.wfile.write(json.dumps({"status": "ok", "remaining": remaining}).encode())
        else:
            super().do_GET()

def get_unlabeled():
    labels = load_existing()
    images = []
    for split in ["train", "val", "test"]:
        img_dir = DATA_ROOT / split / "breadfruit"
        for f in sorted(img_dir.glob("*")):
            if not AUG_PATTERN.search(f.name) and f.name not in labels:
                images.append({"filename": f.name, "split": split})
    return images

def load_existing():
    labels = {}
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                if row.get("maturity") and row.get("quality"):
                    labels[row["filename"]] = row
    return labels

def save_label(filename, maturity, quality):
    labels = load_existing()
    labels[filename] = {"filename": filename, "maturity": maturity, "quality": quality}
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "maturity", "quality"])
        writer.writeheader()
        writer.writerows(labels.values())

def count_unlabeled():
    return len(get_unlabeled())

HTML = """<!DOCTYPE html>
<html>
<head><title>Breadfruit Labeler</title>
<style>
  body { font-family: sans-serif; max-width: 700px; margin: 40px auto; text-align: center; }
  img { max-width: 500px; max-height: 500px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
  .btn { padding: 12px 24px; margin: 8px; border: 2px solid #333; border-radius: 6px; cursor: pointer; font-size: 16px; }
  .btn:hover { background: #eee; }
  .btn.selected { background: #4CAF50; color: white; border-color: #388E3C; }
  #progress { margin: 20px 0; font-size: 18px; color: #666; }
  .section { margin: 20px 0; }
  h3 { color: #333; }
  #saveBtn { background: #2196F3; color: white; border: none; padding: 14px 40px; font-size: 18px; border-radius: 8px; cursor: pointer; }
  #saveBtn:hover { background: #1976D2; }
  .skipBtn { background: #9E9E9E; color: white; border: none; padding: 10px 20px; font-size: 14px; border-radius: 6px; cursor: pointer; }
</style></head>
<body>
  <h1>🍞 Breadfruit Labeler</h1>
  <div id="progress">Loading...</div>
  <img id="image" src="" alt="Breadfruit" style="display:none" />
  <p id="filename" style="color:#666;"></p>
  <div class="section">
    <h3>Maturity</h3>
    <div id="maturityBtns"></div>
  </div>
  <div class="section">
    <h3>Quality</h3>
    <div id="qualityBtns"></div>
  </div>
  <br/>
  <button id="saveBtn" onclick="saveLabel()" disabled>Save & Next</button>
  <br/><br/>
  <button class="skipBtn" onclick="skipImage()">Skip</button>
  <script>
    let images = [], currentIdx = 0, selectedMaturity = null, selectedQuality = null;
    let maturityClasses = [], qualityClasses = [];

    function load() {
      fetch('/api/images').then(r => r.json()).then(data => {
        images = data.images;
        maturityClasses = data.maturity_classes;
        qualityClasses = data.quality_classes;
        createButtons('maturityBtns', maturityClasses, 'maturity');
        createButtons('qualityBtns', qualityClasses, 'quality');
        if (images.length === 0) {
          document.getElementById('progress').innerText = '🎉 All images labeled!';
          return;
        }
        showImage(0);
      });
    }

    function createButtons(containerId, classes, type) {
      const div = document.getElementById(containerId);
      div.innerHTML = '';
      classes.forEach(cls => {
        const btn = document.createElement('button');
        btn.className = 'btn';
        btn.innerText = cls;
        btn.onclick = () => selectType(type, cls, btn);
        div.appendChild(btn);
      });
    }

    function selectType(type, value, btn) {
      const container = btn.parentElement;
      container.querySelectorAll('.btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      if (type === 'maturity') selectedMaturity = value;
      else selectedQuality = value;
      document.getElementById('saveBtn').disabled = !(selectedMaturity && selectedQuality);
    }

    function showImage(idx) {
      if (idx >= images.length) {
        document.getElementById('progress').innerText = '🎉 All images labeled!';
        document.getElementById('image').style.display = 'none';
        return;
      }
      currentIdx = idx;
      const img = images[idx];
      document.getElementById('progress').innerText = `${idx + 1} / ${images.length}`;
      document.getElementById('image').src = '/images/' + encodeURIComponent(img.filename);
      document.getElementById('image').style.display = 'block';
      document.getElementById('filename').innerText = img.filename + ' (' + img.split + ')';
      selectedMaturity = null;
      selectedQuality = null;
      document.querySelectorAll('.btn').forEach(b => b.classList.remove('selected'));
      document.getElementById('saveBtn').disabled = true;
    }

    function saveLabel() {
      const img = images[currentIdx];
      fetch('/api/save', {
        method: 'POST',
        body: JSON.stringify({filename: img.filename, maturity: selectedMaturity, quality: selectedQuality})
      }).then(r => r.json()).then(data => {
        document.getElementById('progress').innerText = `${currentIdx + 1} / ${images.length} — ${data.remaining} remaining`;
        showImage(currentIdx + 1);
      });
    }

    function skipImage() {
      showImage(currentIdx + 1);
    }

    load();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    port = 8765
    webbrowser.open(f"http://localhost:{port}")
    print(f"🚀 Open http://localhost:{port} in your browser")
    print(f"📝 Labels saved to: {CSV_PATH}")
    HTTPServer(("", port), LabelHandler).serve_forever()
