#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

OUTPUT_DIR = Path("/app/output")
MODEL_PATH = OUTPUT_DIR / "model_best_v2.pt"


# Must match 02_train_v2.py
class GAPCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.gap(x)
        x = self.head(x)
        return x


def preprocess(img: Image.Image, img_size: int) -> torch.Tensor:
    # No rotation. Resize only.
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    return torch.from_numpy(arr).unsqueeze(0)         # NCHW


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {MODEL_PATH}. Train first with 02_train_v2.py"
        )
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    img_size = int(ckpt.get("img_size", 224))
    id_to_label = ckpt.get("id_to_label") or {0: "1_Pronacio", 1: "2_Neutralis", 2: "3_Szupinacio"}
    id_to_label = {int(k): str(v) for k, v in id_to_label.items()}

    model = GAPCNN(num_classes=len(id_to_label))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, img_size, id_to_label


MODEL, IMG_SIZE, ID_TO_LABEL = load_model()

app = FastAPI(title="AnkleAlign Service", version="1.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "img_size": IMG_SIZE, "classes": ID_TO_LABEL}


@app.get("/", response_class=HTMLResponse)
def gui():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AnkleAlign – Demo</title>
  <style>
    :root{
      --bg:#0b1220;
      --card:#121a2b;
      --muted:#9aa4b2;
      --text:#e6edf7;
      --accent:#5dd6ff;
      --ok:#2dd4bf;
      --warn:#fbbf24;
      --err:#fb7185;
      --border:rgba(255,255,255,.08);
      --shadow: 0 18px 40px rgba(0,0,0,.45);
      --radius:16px;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background: radial-gradient(1200px 600px at 10% 10%, rgba(93,214,255,.15), transparent 60%),
                  radial-gradient(900px 500px at 80% 0%, rgba(45,212,191,.10), transparent 60%),
                  var(--bg);
      color:var(--text);
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:28px;
    }
    .wrap{width:min(1100px, 100%)}
    .header{
      display:flex;
      align-items:flex-end;
      justify-content:space-between;
      gap:16px;
      margin-bottom:18px;
    }
    .title{
      display:flex; flex-direction:column; gap:6px;
    }
    h1{margin:0; font-size:28px; letter-spacing:.2px}
    .sub{color:var(--muted); font-size:14px; line-height:1.4}
    .pill{
      border:1px solid var(--border);
      background: rgba(255,255,255,.03);
      padding:10px 12px;
      border-radius:999px;
      color:var(--muted);
      font-size:12px;
      display:flex; gap:10px; align-items:center;
    }
    .dot{width:8px; height:8px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 4px rgba(45,212,191,.18)}
    .grid{
      display:grid;
      grid-template-columns: 1.2fr .8fr;
      gap:18px;
    }
    @media (max-width: 920px){
      .grid{grid-template-columns:1fr}
    }
    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .card .inner{padding:18px}
    .card h2{margin:0 0 10px 0; font-size:16px}
    .uploader{
      display:flex;
      gap:12px;
      align-items:center;
      flex-wrap:wrap;
      margin-bottom:12px;
    }
    input[type=file]{
      flex:1;
      padding:10px;
      background: rgba(255,255,255,.03);
      border:1px dashed rgba(255,255,255,.18);
      border-radius: 12px;
      color: var(--muted);
    }
    .btn{
      background: linear-gradient(180deg, rgba(93,214,255,.25), rgba(93,214,255,.12));
      border:1px solid rgba(93,214,255,.35);
      color: var(--text);
      padding:10px 14px;
      border-radius: 12px;
      cursor:pointer;
      font-weight:600;
      transition: transform .05s ease;
    }
    .btn:active{transform: translateY(1px)}
    .btn.secondary{
      background: rgba(255,255,255,.04);
      border:1px solid var(--border);
      color: var(--muted);
      font-weight:600;
    }
    .preview{
      display:grid;
      grid-template-columns: 220px 1fr;
      gap:14px;
      align-items:start;
      margin-top:14px;
    }
    @media (max-width: 520px){
      .preview{grid-template-columns:1fr}
    }
    .imgbox{
      width:220px; height:220px;
      border-radius: 14px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.02);
      display:flex;
      align-items:center;
      justify-content:center;
      overflow:hidden;
    }
    .imgbox img{width:100%; height:100%; object-fit:cover}
    .placeholder{
      color: var(--muted);
      font-size:12px;
      padding:14px;
      text-align:center;
    }
    .resultTop{
      display:flex; gap:10px; align-items:center; flex-wrap:wrap;
      margin-bottom:10px;
    }
    .badge{
      padding:6px 10px;
      border-radius:999px;
      font-size:12px;
      font-weight:700;
      border:1px solid var(--border);
      background: rgba(255,255,255,.04);
    }
    .badge.pred{border-color: rgba(45,212,191,.35); background: rgba(45,212,191,.12); color: #bff8ef}
    .badge.conf{border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.12); color: #ffe6a6}
    .bars{display:flex; flex-direction:column; gap:10px; margin-top:8px}
    .barRow{
      display:grid;
      grid-template-columns: 140px 1fr 70px;
      gap:10px;
      align-items:center;
    }
    .labelName{color: var(--muted); font-size:13px}
    .barTrack{
      height:10px;
      border-radius:999px;
      background: rgba(255,255,255,.06);
      border:1px solid rgba(255,255,255,.08);
      overflow:hidden;
    }
    .barFill{
      height:100%;
      width:0%;
      background: linear-gradient(90deg, rgba(93,214,255,.95), rgba(45,212,191,.95));
      border-radius:999px;
      transition: width .25s ease;
    }
    .pct{font-variant-numeric: tabular-nums; color: var(--text); font-size:13px}
    .status{
      margin-top:12px;
      padding:10px 12px;
      border-radius:12px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.03);
      color: var(--muted);
      font-size:13px;
      display:none;
    }
    .status.err{border-color: rgba(251,113,133,.35); background: rgba(251,113,133,.10); color:#ffd0da}
    .status.ok{border-color: rgba(45,212,191,.35); background: rgba(45,212,191,.10); color:#c6fff4}
    .footer{
      margin-top:14px;
      color: var(--muted);
      font-size:12px;
      display:flex;
      justify-content:space-between;
      gap:10px;
      flex-wrap:wrap;
    }
    a{color: var(--accent); text-decoration:none}
    a:hover{text-decoration:underline}
    code{background: rgba(255,255,255,.05); padding:2px 6px; border-radius:8px; border:1px solid var(--border)}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div class="title">
        <h1>AnkleAlign</h1>
        <div class="sub">Upload an image and get a 3-class prediction (Pronáció / Neutrális / Szupináció). Preprocessing is <b>resize-only</b> (no rotation).</div>
      </div>
      <div class="pill"><span class="dot"></span><span>Service running</span><span>•</span><span><a href="/health">/health</a></span></div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="inner">
          <h2>Input</h2>

          <div class="uploader">
            <input id="file" type="file" accept="image/*"/>
            <button class="btn" onclick="predict()">Predict</button>
            <button class="btn secondary" onclick="clearAll()">Clear</button>
          </div>

          <div class="preview">
            <div class="imgbox">
              <div id="ph" class="placeholder">No image selected</div>
              <img id="preview" style="display:none" />
            </div>

            <div>
              <div class="resultTop">
                <span class="badge">Top class</span>
                <span id="pred" class="badge pred">—</span>
                <span class="badge">Confidence</span>
                <span id="conf" class="badge conf">—</span>
              </div>

              <div class="bars" id="bars"></div>

              <div id="status" class="status"></div>
            </div>
          </div>

          <div class="footer">
            <div>API endpoint: <code>POST /predict</code> (multipart form field <code>file</code>)</div>
            <div>Tip: open DevTools → Network to see the JSON response.</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="inner">
          <h2>How it works</h2>
          <div class="sub" style="margin-bottom:10px">
            This GUI calls the backend API. The backend loads the trained model from <code>/app/output/model_best_v2.pt</code>.
          </div>
          <ul class="sub" style="margin:0; padding-left:18px; line-height:1.7">
            <li><b>Input:</b> Image upload</li>
            <li><b>Preprocess:</b> RGB conversion + resize to model size</li>
            <li><b>Output:</b> Probabilities for all classes + top prediction</li>
            <li><b>Note:</b> No rotation augmentation is applied</li>
          </ul>

          <div class="sub" style="margin-top:14px">
            If you want to test via CLI:
            <div style="margin-top:8px">
              <code>curl -F "file=@image.jpg" http://localhost:8000/predict</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
const fileEl = document.getElementById("file");
const previewEl = document.getElementById("preview");
const phEl = document.getElementById("ph");
const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");
const barsEl = document.getElementById("bars");
const statusEl = document.getElementById("status");

fileEl.addEventListener("change", () => {
  const f = fileEl.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  previewEl.src = url;
  previewEl.style.display = "block";
  phEl.style.display = "none";
  setStatus("Ready. Click Predict.", "ok");
});

function clearAll() {
  fileEl.value = "";
  previewEl.src = "";
  previewEl.style.display = "none";
  phEl.style.display = "block";
  predEl.textContent = "—";
  confEl.textContent = "—";
  barsEl.innerHTML = "";
  statusEl.style.display = "none";
}

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (kind || "");
  statusEl.style.display = "block";
}

function makeBars(probabilities) {
  // probabilities: {label: score}
  const entries = Object.entries(probabilities)
    .sort((a,b) => b[1] - a[1]);

  barsEl.innerHTML = "";
  for (const [label, score] of entries) {
    const pct = Math.round(score * 1000) / 10; // 0.1%
    const row = document.createElement("div");
    row.className = "barRow";
    row.innerHTML = `
      <div class="labelName">${escapeHtml(label)}</div>
      <div class="barTrack"><div class="barFill"></div></div>
      <div class="pct">${pct.toFixed(1)}%</div>
    `;
    barsEl.appendChild(row);
    // animate width
    requestAnimationFrame(() => {
      row.querySelector(".barFill").style.width = (score * 100).toFixed(1) + "%";
    });
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (m) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[m]));
}

async function predict() {
  const f = fileEl.files[0];
  if (!f) { setStatus("No file selected.", "err"); return; }

  const form = new FormData();
  form.append("file", f);

  setStatus("Running inference...", "");
  predEl.textContent = "—";
  confEl.textContent = "—";
  barsEl.innerHTML = "";

  try {
    const resp = await fetch("/predict", { method: "POST", body: form });
    const text = await resp.text();
    let data;
    try { data = JSON.parse(text); } catch { throw new Error(text); }

    if (!resp.ok) {
      setStatus(data.error || ("Request failed: " + resp.status), "err");
      return;
    }

    predEl.textContent = data.prediction_label ?? "—";
    const c = data.confidence ?? 0;
    confEl.textContent = (c * 100).toFixed(1) + "%";

    if (data.probabilities) makeBars(data.probabilities);

    setStatus("Done.", "ok");
  } catch (e) {
    setStatus("Error: " + e.message, "err");
  }
}
</script>
</body>
</html>
"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {type(e).__name__}: {e}"})

    x = preprocess(img, IMG_SIZE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    return JSONResponse(
        content={
            "prediction_id": pred_id,
            "prediction_label": ID_TO_LABEL.get(pred_id, str(pred_id)),
            "confidence": float(probs[pred_id]),
            "probabilities": {ID_TO_LABEL.get(i, str(i)): float(probs[i]) for i in range(len(probs))},
        }
    )
