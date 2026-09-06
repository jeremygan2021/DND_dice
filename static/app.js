// YOLO Web Studio 前端逻辑

// 在摄像头模式下显示本机 LAN URL（手机扫这个访问）
(function showLanUrl() {
  const host = location.hostname || 'localhost';
  const port = location.port || '1111';
  const proto = location.protocol.replace(':', '');
  document.getElementById('lanUrl').textContent = `${proto}://${host}:${port}`;
})();

const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const resultImg = document.getElementById('resultImg');
const resultSummary = document.getElementById('resultSummary');
const emptyState = document.getElementById('emptyState');
const btnRun = document.getElementById('btnRun');
const btnClear = document.getElementById('btnClear');
const statusText = document.getElementById('statusText');
const taskBtns = document.querySelectorAll('.task-btn');
const modeBtns = document.querySelectorAll('.mode-btn');
const uploadPane = document.getElementById('uploadPane');
const cameraPane = document.getElementById('cameraPane');

let currentTask = 'dice';
let currentFile = null;
let currentMode = 'upload';

// -------- 任务切换 --------
taskBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    taskBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentTask = btn.dataset.task;
    // 摄像头模式只支持 dice
    if (currentMode === 'camera' && currentTask !== 'dice') {
      setMode('upload');
    }
  });
});

// -------- 模式切换 --------
modeBtns.forEach(btn => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

function setMode(mode) {
  if (mode === 'camera' && currentTask !== 'dice') {
    setStatus('摄像头模式仅支持骰子识别', 'err');
    return;
  }
  currentMode = mode;
  modeBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  uploadPane.classList.toggle('hidden', mode !== 'upload');
  cameraPane.classList.toggle('hidden', mode !== 'camera');
  document.body.dataset.mode = mode;  // 给 CSS 用
  if (mode === 'camera') {
    stopCamera();
  }
}

// -------- 拖拽 / 选择 --------
dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fi.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(file) {
  if (!file.type.startsWith('image/')) {
    setStatus('请选择图片文件', 'err'); return;
  }
  currentFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dz.classList.add('has-image');
  btnRun.disabled = false;
  resultImg.removeAttribute('src');
  resultImg.parentElement.classList.remove('has-image');
  resultSummary.innerHTML = '';
  setStatus(`已选: ${file.name}`, 'ok');
}

btnClear.addEventListener('click', () => {
  currentFile = null;
  previewImg.removeAttribute('src');
  dz.classList.remove('has-image');
  resultImg.removeAttribute('src');
  resultImg.parentElement.classList.remove('has-image');
  resultSummary.innerHTML = '';
  fi.value = '';
  btnRun.disabled = true;
  setStatus('', '');
});

// -------- 推理 --------
btnRun.addEventListener('click', async () => {
  if (!currentFile) return;
  btnRun.disabled = true;
  setStatus('推理中...', 'busy');

  const fd = new FormData();
  fd.append('task', currentTask);
  fd.append('mode', 'upload');
  fd.append('image', currentFile);

  try {
    const t0 = performance.now();
    const r = await fetch('/api/infer', { method: 'POST', body: fd });
    const dt = (performance.now() - t0) / 1000;

    if (!r.ok) {
      const e = await r.json().catch(() => ({ error: '请求失败' }));
      setStatus(`失败: ${e.error}`, 'err');
      btnRun.disabled = false;
      return;
    }
    const data = await r.json();
    resultImg.src = data.result_url;
    resultImg.parentElement.classList.add('has-image');
    renderSummary(data);
    setStatus(`完成 · ${data.elapsed_ms}ms · ${dt.toFixed(2)}s`, 'ok');
  } catch (err) {
    setStatus(`异常: ${err.message}`, 'err');
  } finally {
    btnRun.disabled = false;
  }
});

// -------- 结果渲染 --------
function renderSummary(data) {
  const s = data.summary || data;
  let html = `<div class="summary-title">${data.task === 'dice' ? '🎲 骰子结果' : '📖 版面分析'}</div>`;
  html += '<div class="summary-stats">';

  if (data.task === 'dice') {
    html += stat('骰子数', s.num_dice);
    html += stat('总点数', s.total_value);
  } else {
    html += stat('区域数', s.num_blocks);
    const counts = {};
    s.blocks.forEach(b => counts[b.class] = (counts[b.class] || 0) + 1);
    Object.entries(counts).forEach(([k, v]) => html += stat(k, v));
  }
  html += '</div>';

  if (data.task === 'dice') {
    const diceList = s.dice || [];
    if (diceList.length === 0) {
      html += '<div style="color:var(--muted);font-size:12px;">未识别到骰子，换张更清晰的照片试试（俯视、正方形面）。</div>';
    } else {
      html += '<div class="block-list">';
      diceList.forEach((d, i) => {
        html += `<div class="dice-row">
          <div class="dice-value">${d.value}</div>
          <div style="color:var(--muted);font-size:11px;">#${i+1}</div>
          <div class="block-bbox">[${d.bbox.join(', ')}]</div>
        </div>`;
      });
      html += '</div>';
    }
  } else {
    if (s.blocks.length === 0) {
      html += '<div style="color:var(--muted);font-size:12px;">未识别到版面区域。</div>';
    } else {
      html += '<div class="block-list">';
      s.blocks.forEach(b => {
        html += `<div class="block-row">
          <div class="block-class">${b.class}</div>
          <div class="block-conf">conf: ${b.conf}</div>
          <div class="block-bbox">[${b.bbox.join(', ')}]</div>
        </div>`;
      });
      html += '</div>';
    }
  }
  resultSummary.innerHTML = html;
}

function stat(k, v) {
  return `<div class="stat-chip"><span class="k">${k}</span><span class="v">${v}</span></div>`;
}

function setStatus(msg, cls) {
  statusText.textContent = msg;
  statusText.className = 'status' + (cls ? ' ' + cls : '');
}

// =====================================================================
// 摄像头实时识别
// =====================================================================
const camVideo = document.getElementById('camVideo');
const camOverlay = document.getElementById('camOverlay');
const camTotal = document.getElementById('camTotal');
const camFps = document.getElementById('camFps');
const camMs = document.getElementById('camMs');
const camError = document.getElementById('camError');
const btnCamStart = document.getElementById('btnCamStart');
const btnCamStop = document.getElementById('btnCamStop');
const camIntervalSel = document.getElementById('camInterval');
const camUseApi = document.getElementById('camUseApi');

let camStream = null;
let camTimer = null;
let camInFlight = false;
let camFrameCount = 0;
let camFpsT0 = 0;

btnCamStart.addEventListener('click', startCamera);
btnCamStop.addEventListener('click', stopCamera);

async function startCamera() {
  camError.classList.add('hidden');
  try {
    // 手机网络宽高小一些，省流量 + 加快后端处理
    const isMobile = /Mobi|Android|iPhone/i.test(navigator.userAgent);
    camStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: isMobile ? 640 : 1280 },
        height: { ideal: isMobile ? 480 : 720 },
        facingMode: 'environment',
      },
      audio: false,
    });
  } catch (e) {
    camError.textContent = `无法访问摄像头: ${e.message}（HTTPS 或 localhost 才能调用 getUserMedia）`;
    camError.classList.remove('hidden');
    return;
  }
  camVideo.srcObject = camStream;
  await new Promise(r => camVideo.onloadedmetadata = r);
  // canvas 同步 video 原生尺寸（坐标用它做参照）
  camOverlay.width = camVideo.videoWidth;
  camOverlay.height = camVideo.videoHeight;
  camVideo.play().catch(() => {});

  btnCamStart.disabled = true;
  btnCamStop.disabled = false;
  camFrameCount = 0;
  camFpsT0 = performance.now();

  const tick = () => {
    if (!camStream) return;
    if (!camInFlight) {
      camInFlight = true;
      captureAndInfer().finally(() => { camInFlight = false; });
    }
  };
  // 间隔根据实际耗时自适应：处理 2s 自动降到 ~1 FPS，避免雪崩
  const baseInterval = parseInt(camIntervalSel.value, 10);
  camTimer = setInterval(tick, baseInterval);
  // 同步显示当前目标
  camFps.textContent = (1000 / baseInterval).toFixed(0);
}

function stopCamera() {
  if (camTimer) { clearInterval(camTimer); camTimer = null; }
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  camVideo.srcObject = null;
  btnCamStart.disabled = false;
  btnCamStop.disabled = true;
  drawOverlay([]);  // 清空
  camTotal.textContent = '0';
  camFps.textContent = '--';
  camMs.textContent = '--';
}

// 稳定 bbox 跟踪：相同 bbox 出现 N 帧后，调用 API 校准一次，结果缓存到位置变化
const stableTracker = new Map();  // key: "x1,y1,x2,y2" → { value, frames, lastSeen, source }
let apiPending = false;
let apiCooldownUntil = 0;

function bboxKey(b) { return b.map(v => Math.round(v / 20) * 20).join(','); }

async function captureAndInfer() {
  const targetW = Math.min(camVideo.videoWidth, 1280);
  const targetH = Math.round(camVideo.videoHeight * targetW / camVideo.videoWidth);
  const off = document.createElement('canvas');
  off.width = targetW;
  off.height = targetH;
  const ctx = off.getContext('2d');
  ctx.drawImage(camVideo, 0, 0, targetW, targetH);
  // 质量 0.82：摄像头帧质量关键，OCR 误读很大程度来自压缩
  const blob = await new Promise(r => off.toBlob(r, 'image/jpeg', 0.82));

  // 决策：是否走 API
  //  - 手动勾选：每帧都走（费配额）
  //  - 自动：同 bbox 稳定 2 帧后切 API 校准，缓存结果直到 bbox 移动
  const useApi = camUseApi.checked || (stableTracker.size > 0 && !apiPending && performance.now() > apiCooldownUntil);

  const fd = new FormData();
  fd.append('task', 'dice');
  fd.append('mode', 'camera');
  if (useApi) fd.append('api', '1');
  fd.append('image', blob, 'frame.jpg');

  const t0 = performance.now();
  try {
    if (useApi) apiPending = true;
    const r = await fetch('/api/infer', { method: 'POST', body: fd });
    if (!r.ok) return;
    const data = await r.json();
    const scaleX = camVideo.videoWidth / targetW;
    const scaleY = camVideo.videoHeight / targetH;
    let scaled = (data.dice || []).map(d => ({
      bbox: [d.bbox[0]*scaleX, d.bbox[1]*scaleY, d.bbox[2]*scaleX, d.bbox[3]*scaleY],
      value: d.value,
    }));

    // 自动模式：把 API 值缓存到稳定的 bbox
    if (!camUseApi.checked && useApi) {
      const apiMap = new Map(scaled.map(d => [bboxKey(d.bbox), d.value]));
      scaled = scaled.map(d => {
        const k = bboxKey(d.bbox);
        const cached = stableTracker.get(k);
        const apiV = apiMap.get(k);
        if (apiV) {
          stableTracker.set(k, { value: apiV, frames: 999, source: 'api' });
          return { ...d, value: apiV, source: 'api' };
        }
        return d;
      });
      apiCooldownUntil = performance.now() + 3000;  // 3s 冷却，避免连续 API 调用
    } else if (!camUseApi.checked) {
      // 本地帧：更新稳定计数
      const seen = new Set();
      scaled.forEach(d => {
        const k = bboxKey(d.bbox);
        seen.add(k);
        const cur = stableTracker.get(k);
        if (cur) stableTracker.set(k, { ...cur, frames: cur.frames + 1, lastSeen: performance.now() });
        else stableTracker.set(k, { value: d.value, frames: 1, lastSeen: performance.now(), source: 'local' });
      });
      // 清理超时
      for (const [k, v] of stableTracker) {
        if (performance.now() - v.lastSeen > 2000) stableTracker.delete(k);
      }
    }

    camMs.textContent = data.elapsed_ms;
    camTotal.textContent = scaled.reduce((s, d) => s + d.value, 0);
    drawOverlay(scaled);
    camFrameCount++;
    const fpsDt = (performance.now() - camFpsT0) / 1000;
    if (fpsDt >= 1) {
      camFps.textContent = (camFrameCount / fpsDt).toFixed(1);
      camFrameCount = 0;
      camFpsT0 = performance.now();
    }
  } catch (e) {
  } finally {
    if (useApi) apiPending = false;
  }
}

function drawOverlay(dice) {
  const c = camOverlay.getContext('2d');
  c.clearRect(0, 0, camOverlay.width, camOverlay.height);
  c.lineWidth = 3;
  c.font = 'bold 28px sans-serif';
  for (const d of dice) {
    const [x1, y1, x2, y2] = d.bbox;
    c.strokeStyle = '#00c864';
    c.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = String(d.value);
    const w = c.measureText(text).width;
    c.fillStyle = '#00c864';
    c.fillRect(x1, y1 - 36, w + 16, 36);
    c.fillStyle = '#fff';
    c.fillText(text, x1 + 8, y1 - 8);
  }
}

// -------- 模型状态轮询 --------
async function refreshStatus() {
  try {
    const r = await fetch('/healthz');
    const d = await r.json();
    document.getElementById('devName').textContent = d.device;
    document.getElementById('msDoc').textContent = d.models.doclayout ? '已就绪' : '未加载';
    document.getElementById('msDice').textContent = d.models.dice ? '已就绪' : '未加载';
  } catch {}
}
refreshStatus();
setInterval(refreshStatus, 5000);
