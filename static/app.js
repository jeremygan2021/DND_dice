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
  fd.append('dice_type', document.getElementById('diceType').value);
  fd.append('task', currentTask);
  fd.append('mode', 'upload');
  fd.append('detector', document.getElementById('detectorSel').value);
  if (vlmEnable && vlmEnable.checked) {
    fd.append('vlm', '1');
    fd.append('vlm_model', vlmModel.value);
  }
  if (currentTask === 'dice' && upUseApi && upUseApi.checked) fd.append('api', '1');
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
          <div class="dice-value">${d.value ?? "待确认"}</div>
          <div style="color:var(--muted);font-size:11px;">#${i+1} · ${d.dice_type || 'unknown'}${d.source ? ' · ' + d.source : ''}</div>
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
// 摄像头实时识别(两阶段:实时拉框跟随 → 稳定后后台 OCR)
// 服务端按 session 保留跟踪状态,每帧响应带每颗骰的 state:
//   moving  = 骰子在动,只拉框不读值
//   settling= 刚停下,正在累计稳定帧
//   stable  = 已稳定,value 有值(OCR 缓存)或正在后台 OCR
// =====================================================================
const camVideo = document.getElementById('camVideo');
const camOverlay = document.getElementById('camOverlay');
const camTotal = document.getElementById('camTotal');
const camFps = document.getElementById('camFps');
const camMs = document.getElementById('camMs');
const camError = document.getElementById('camError');
const camStateEl = document.getElementById('camState');
const btnCamStart = document.getElementById('btnCamStart');
const btnCamStop = document.getElementById('btnCamStop');
const camIntervalSel = document.getElementById('camInterval');
const camDeviceSel = document.getElementById('camDevice');
const camUseApi = document.getElementById('camUseApi');
const upUseApi = document.getElementById('upUseApi');
const detectorSel = document.getElementById('detectorSel');
const detectorStatus = document.getElementById('detectorStatus');
const vlmEnable = document.getElementById('vlmEnable');
const vlmModel = document.getElementById('vlmModel');

let camStream = null;
let camTimer = null;          // setTimeout 句柄(自适应循环,非固定间隔)
let camRunning = false;
let camSessionId = null;
let camFrameCount = 0;
let camFpsT0 = 0;
let lastFrameWasStable = false;

btnCamStart.addEventListener('click', startCamera);
btnCamStop.addEventListener('click', stopCamera);
camDeviceSel.addEventListener('change', async () => {
  // 切换设备: 若在运行, 重建 stream
  if (camRunning) {
    stopCamera();
    await startCamera();
  }
});

function newSessionId() {
  return 'cam-' + Date.now().toString(36) + '-' +
         Math.random().toString(36).slice(2, 10);
}

// -------- 枚举摄像头设备 --------
async function refreshDevices() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    camDeviceSel.innerHTML = '<option value="">浏览器不支持设备枚举</option>';
    return;
  }
  try {
    // enumerateDevices 需要先获取一次权限,否则 label 会空
    if (!camStream) {
      try {
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        tmp.getTracks().forEach(t => t.stop());
      } catch (_) {}
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videos = devices.filter(d => d.kind === 'videoinput');
    const prev = camDeviceSel.value;
    camDeviceSel.innerHTML = '';
    if (videos.length === 0) {
      camDeviceSel.innerHTML = '<option value="">未检测到摄像头</option>';
      return;
    }
    videos.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || `摄像头 ${i + 1}`;
      camDeviceSel.appendChild(opt);
    });
    if (prev && [...camDeviceSel.options].some(o => o.value === prev)) {
      camDeviceSel.value = prev;
    }
  } catch (e) {
    camDeviceSel.innerHTML = `<option value="">${e.message}</option>`;
  }
}
refreshDevices();
if (navigator.mediaDevices) {
  navigator.mediaDevices.addEventListener('devicechange', refreshDevices);
}

async function startCamera() {
  camError.classList.add('hidden');
  const isMobile = /Mobi|Android|iPhone/i.test(navigator.userAgent);
  const videoConstraints = {
    width: { ideal: Number(document.getElementById("camResolution").value) },
    height: { ideal: Math.round(Number(document.getElementById("camResolution").value) * 9 / 16) },
  };
  const devId = camDeviceSel.value;
  if (devId) {
    videoConstraints.deviceId = { exact: devId };
  } else {
    videoConstraints.facingMode = 'environment';
  }
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
  } catch (e) {
    camError.textContent = `无法访问摄像头: ${e.message}（HTTPS 或 localhost 才能调用 getUserMedia）`;
    camError.classList.remove('hidden');
    return;
  }
  // 设备列表里 label 可能刚被解锁,刷新一次
  refreshDevices();
  camVideo.srcObject = camStream;
  await new Promise(r => camVideo.onloadedmetadata = r);
  camOverlay.width = camVideo.videoWidth;
  camOverlay.height = camVideo.videoHeight;
  camVideo.play().catch(() => {});

  btnCamStart.disabled = true;
  btnCamStop.disabled = false;
  camFrameCount = 0;
  camFpsT0 = performance.now();
  camSessionId = newSessionId();     // 每个会话独立跟踪状态
  camRunning = true;
  lastFrameWasStable = false;
  camStateEl.textContent = '启动中';
  camStateEl.className = 'cam-state working';

  // 自适应循环:处理完一帧再调度下一帧,间隔 = max(目标, 实测耗时+余量),
  // 避免请求积压雪崩;全部骰子已稳定读出时自动降频省资源
  const loop = async () => {
    if (!camRunning || !camStream) return;
    const started = performance.now();
    await captureAndInfer();
    if (!camRunning) return;
    const cost = performance.now() - started;
    let base = parseInt(camIntervalSel.value, 10) || 125;
    if (lastFrameWasStable) base = Math.max(base, 220);   // 稳定后不必高帧率
    const next = Math.max(0, base - cost);
    camTimer = setTimeout(loop, next);
  };
  loop();
  // 下一次调度直接读取新档位，不并发启动第二个推理循环。
  camIntervalSel.onchange = null;
}

function stopCamera() {
  camRunning = false;
  if (camTimer) { clearTimeout(camTimer); camTimer = null; }
  camIntervalSel.onchange = null;
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
  camStateEl.textContent = '已停止';
  camStateEl.className = 'cam-state waiting';
}

async function captureAndInfer() {
  // 取视频当前帧画到离屏 canvas(与 video 同尺寸,避免重复缩放)
  const vw = camVideo.videoWidth, vh = camVideo.videoHeight;
  if (!vw || !vh) return;
  const off = document.createElement('canvas');
  off.width = vw;
  off.height = vh;
  const ctx = off.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(camVideo, 0, 0, vw, vh);
  // 质量 0.82:太高影响速度,太低影响 OCR
  const blob = await new Promise(r => off.toBlob(r, 'image/jpeg', 0.92));

  const fd = new FormData();
  fd.append('dice_type', document.getElementById('diceType').value);
  fd.append('task', 'dice');
  fd.append('mode', 'camera');
  fd.append('session', camSessionId);
  fd.append('detector', detectorSel.value);
  if (vlmEnable && vlmEnable.checked) {
    fd.append('vlm', '1');
    fd.append('vlm_model', vlmModel.value);
  }
  if (camUseApi.checked) fd.append('api', '1');  // 稳定后读值走百炼
  fd.append('image', blob, 'frame.jpg');

  const t0 = performance.now();
  try {
    const r = await fetch('/api/infer', { method: 'POST', body: fd });
    if (!r.ok) return;
    const data = await r.json();
    // 后端 bbox 在原始帧(与 canvas 同尺寸)坐标系,overlay 与其一致,无需缩放
    const dice = data.dice || [];
    lastFrameWasStable = !!data.all_settled;

    camMs.textContent = data.det_ms != null ? data.det_ms : data.elapsed_ms;
    // 总值只统计已稳定读出的骰子
    const stableVals = dice.filter(d => d.state === 'stable' && d.value != null);
    camTotal.textContent = stableVals.reduce((s, d) => s + d.value, 0);

    // 状态徽标
    const gaveUp = dice.filter(d => d.state === 'stable' && d.ocr_gave_up).length;
    if (dice.length === 0) {
      camStateEl.textContent = '未检测到骰子';
      camStateEl.className = 'cam-state waiting';
    } else if (stableVals.length === dice.length) {
      camStateEl.textContent = `✓ ${stableVals.length} 颗已读出`;
      camStateEl.className = 'cam-state stable';
    } else if (gaveUp > 0) {
      camStateEl.textContent = `${gaveUp} 颗读不出(角度/光线)`;
      camStateEl.className = 'cam-state working';
    } else if (dice.some(d => d.state === 'stable')) {
      const moving = dice.filter(d => d.state !== 'stable').length;
      camStateEl.textContent = moving > 0 ? `稳定中 · ${moving} 颗在动` : '稳定,读取中…';
      camStateEl.className = 'cam-state working';
    } else {
      camStateEl.textContent = '骰子滚动中…';
      camStateEl.className = 'cam-state working';
    }

    drawOverlay(dice);
    camFrameCount++;
    const fpsDt = (performance.now() - camFpsT0) / 1000;
    if (fpsDt >= 1) {
      camFps.textContent = (camFrameCount / fpsDt).toFixed(1);
      camFrameCount = 0;
      camFpsT0 = performance.now();
    }
  } catch (e) {
    // 网络抖动忽略,下一帧会继续
  }
}

function drawOverlay(dice) {
  const c = camOverlay.getContext('2d');
  const W = camOverlay.width, H = camOverlay.height;
  c.clearRect(0, 0, W, H);
  c.lineWidth = 3;
  for (const d of dice) {
    const [x1, y1, x2, y2] = d.bbox.map(Math.round);
    const w = x2 - x1, h = y2 - y1;
    const st = d.state === 'stable' ? 'stable'
             : d.state === 'settling' ? 'settling' : 'moving';
    c.strokeStyle = st === 'stable' ? '#00d68f'
                  : st === 'settling' ? '#ffb547' : '#ffd54a';
    c.setLineDash(st === 'moving' ? [10, 7] : []);
    c.strokeRect(x1, y1, w, h);
    c.setLineDash([]);

    // 标签:稳定且有值 → 绿色实心;云端采纳 → 蓝色带值;VLM 跑完但越界 → 蓝色"raw?";VLM 进行中 → 蓝"☁";OCR 失败 → 待确认/?
    let label = null;
    let tagColor = null;
    if (st === 'stable') {
      if (d.value != null) {
        label = String(d.value);
        tagColor = d.source === 'vlm' ? 'rgba(120,170,255,0.95)' : '#00d68f';
      } else if (d.vlm_pending) {
        label = '☁'; tagColor = 'rgba(120,170,255,0.95)';
      } else if (d.vlm_text) {
        // VLM 跑完了但越界被拒 → 把原始返回显示给用户看,提示"这是云端说的但不合规"
        label = d.vlm_text.length <= 4 ? d.vlm_text : (d.vlm_text.slice(0,3) + '…');
        tagColor = 'rgba(120,170,255,0.75)';
      } else if (d.reason && !d.ocr_pending) { label = '待确认'; tagColor = 'rgba(255,180,80,0.9)'; }
      else if (d.ocr_gave_up) { label = '?'; tagColor = 'rgba(255,180,80,0.9)'; }
      else { label = '…'; tagColor = 'rgba(120,126,140,0.9)'; }
    }
    if (label) {
      c.font = 'bold 26px sans-serif';
      const tw = c.measureText(label).width;
      const bx = x1, by = y1 - 40;
      c.fillStyle = tagColor;
      c.fillRect(bx, by, tw + 18, 36);
      c.fillStyle = '#06130d';
      c.fillText(label, bx + 9, by + 26);
    }
  }
}

// -------- 模型状态轮询 --------
async function refreshStatus() {
  try {
    const r = await fetch('/healthz');
    const d = await r.json();
    document.getElementById('devName').textContent = d.device;
    const backend = (d.dice_detector.backend || '?').toUpperCase();
    const warn = d.dice_detector.warning ? ' · ' + d.dice_detector.warning : '';
    document.getElementById('diceBackend').textContent = backend + warn;
    document.getElementById('msDoc').textContent = d.models.doclayout ? '已就绪' : '未加载';
    if (detectorStatus) {
      const onnx = d.onnx_dice || {};
      if (onnx.available) {
        const dev = onnx.device ? `· ${onnx.device}` : '· 未加载';
        detectorStatus.textContent = `onnx ✓ ${dev}`;
        detectorStatus.className = 'detector-status on';
      } else {
        detectorStatus.textContent = 'onnx ✗ 模型未下载';
        detectorStatus.className = 'detector-status off';
      }
    }
  } catch {}
}
refreshStatus();
setInterval(refreshStatus, 5000);
