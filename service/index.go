package service

import (
	"net/http"
)

// indexPage 返回 HTML 测试页，包含 TTS 合成与 Voice 注册两个 tab。
// 对应 Python service.py 的 GET / 路由。
// 页面为单文件内嵌，无外部依赖，zh-CN 界面。
func (s *Server) indexPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(indexHTML))
}

const indexHTML = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArkTTS 测试</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f5f5f7; color: #1d1d1f; padding: 20px; }
  .container { max-width: 720px; margin: 0 auto; }
  h1 { font-size: 24px; margin-bottom: 16px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab { padding: 8px 20px; background: #fff; border: 1px solid #d2d2d7; border-radius: 8px 8px 0 0;
         cursor: pointer; font-size: 14px; color: #6e6e73; }
  .tab.active { background: #0071e3; color: #fff; border-color: #0071e3; }
  .panel { display: none; background: #fff; border: 1px solid #d2d2d7; border-radius: 0 8px 8px 8px;
           padding: 20px; }
  .panel.active { display: block; }
  .field { margin-bottom: 12px; }
  label { display: block; font-size: 13px; color: #6e6e73; margin-bottom: 4px; }
  input[type=text], input[type=number], select, textarea {
    width: 100%; padding: 8px 10px; border: 1px solid #d2d2d7; border-radius: 6px;
    font-size: 14px; font-family: inherit; }
  textarea { resize: vertical; min-height: 80px; }
  .row { display: flex; gap: 12px; }
  .row .field { flex: 1; }
  button { padding: 10px 24px; background: #0071e3; color: #fff; border: none;
           border-radius: 8px; font-size: 14px; cursor: pointer; }
  button:hover { background: #0058b9; }
  button:disabled { background: #b0b0b5; cursor: not-allowed; }
  button.secondary { background: #e5e5ea; color: #1d1d1f; }
  button.secondary:hover { background: #d2d2d7; }
  .actions { display: flex; gap: 8px; margin-top: 8px; }
  audio { width: 100%; margin-top: 12px; }
  .status { margin-top: 12px; padding: 8px 12px; border-radius: 6px; font-size: 13px;
            display: none; }
  .status.info { background: #e8f0fe; color: #1a73e8; display: block; }
  .status.error { background: #fce8e6; color: #d93025; display: block; }
  .status.success { background: #e6f4ea; color: #188038; display: block; }
  .voices-list { margin-top: 8px; font-size: 13px; color: #6e6e73; }
  .progress { margin-top: 8px; height: 4px; background: #e5e5ea; border-radius: 2px; overflow: hidden; }
  .progress-bar { height: 100%; background: #0071e3; width: 0%; transition: width 0.2s; }
  input[type=file] { width: 100%; padding: 6px 0; font-size: 13px; }
</style>
</head>
<body>
<div class="container">
  <h1>ArkTTS 测试</h1>
  <div class="tabs">
    <div class="tab active" onclick="switchTab('tts')">语音合成</div>
    <div class="tab" onclick="switchTab('register')">Voice 注册</div>
    <div class="tab" onclick="switchTab('system')">系统状态</div>
  </div>

  <!-- TTS Panel -->
  <div id="panel-tts" class="panel active">
    <div class="field">
      <label>文本</label>
      <textarea id="tts-text" placeholder="输入要合成的文本">你好，这是 ArkTTS 的语音合成测试。</textarea>
    </div>
    <div class="row">
      <div class="field">
        <label>Voice</label>
        <select id="tts-voice"></select>
      </div>
      <div class="field">
        <label>Max Tokens</label>
        <input type="number" id="tts-max-tokens" value="1024" min="16" max="2048">
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label>Temperature</label>
        <input type="number" id="tts-temp" value="0.3" min="0" max="2" step="0.1">
      </div>
      <div class="field">
        <label>Top P</label>
        <input type="number" id="tts-top-p" value="0.9" min="0" max="1" step="0.05">
      </div>
      <div class="field">
        <label>Top K</label>
        <input type="number" id="tts-top-k" value="50" min="1" max="4096">
      </div>
      <div class="field">
        <label>Seed</label>
        <input type="number" id="tts-seed" value="42">
      </div>
    </div>
    <div class="actions">
      <button onclick="synthesize()">合成</button>
      <button onclick="streamSynth()" class="secondary">流式合成</button>
      <button onclick="cancelStream()" class="secondary">取消</button>
      <button onclick="loadVoices()" class="secondary">刷新 Voice</button>
    </div>
    <div id="tts-progress" class="progress" style="display:none">
      <div id="tts-progress-bar" class="progress-bar"></div>
    </div>
    <div id="tts-status" class="status"></div>
    <audio id="tts-audio" controls></audio>
  </div>

  <!-- Register Panel -->
  <div id="panel-register" class="panel">
    <div class="field">
      <label>Voice 名称</label>
      <input type="text" id="reg-name" placeholder="如 speaker_a">
    </div>
    <div class="field">
      <label>参考文本</label>
      <textarea id="reg-ref-text" placeholder="上传音频对应的文本内容"></textarea>
    </div>
    <div class="field">
      <label>音频文件（wav/mp3/flac/m4a）</label>
      <input type="file" id="reg-audio" accept="audio/*">
    </div>
    <div class="field">
      <label><input type="checkbox" id="reg-overwrite"> 覆盖已存在的 Voice</label>
    </div>
    <div class="actions">
      <button onclick="registerVoice()">注册</button>
      <button onclick="checkRegStatus()" class="secondary">检查注册器状态</button>
    </div>
    <div id="reg-status" class="status"></div>
  </div>

  <!-- System Panel -->
  <div id="panel-system" class="panel">
    <div class="actions">
      <button onclick="loadSystem()">刷新系统状态</button>
      <button onclick="reloadRuntime()" class="secondary">重新加载模型</button>
    </div>
    <pre id="sys-output" style="margin-top:12px;font-size:13px;white-space:pre-wrap;color:#1d1d1f;background:#f5f5f7;padding:12px;border-radius:6px;"></pre>
  </div>
</div>

<script>
const api = location.origin;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const panels = ['tts', 'register', 'system'];
    t.classList.toggle('active', panels[i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  if (name === 'system') loadSystem();
}

function showStatus(id, type, msg) {
  const el = document.getElementById(id);
  el.className = 'status ' + type;
  el.textContent = msg;
}

// --- Voice 列表 ---
async function loadVoices() {
  try {
    const resp = await fetch(api + '/api/voices');
    const voices = await resp.json();
    const sel = document.getElementById('tts-voice');
    sel.innerHTML = '';
    if (!voices || voices.length === 0) {
      sel.innerHTML = '<option value="">（无 voice，请先注册）</option>';
      showStatus('tts-status', 'info', '未发现已注册 voice，请切换到「Voice 注册」tab。');
      return;
    }
    voices.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = v.name + (v.reference_text ? ' — ' + v.reference_text.slice(0, 30) : '');
      sel.appendChild(opt);
    });
    showStatus('tts-status', 'info', '已加载 ' + voices.length + ' 个 voice。');
  } catch (e) {
    showStatus('tts-status', 'error', '加载 voice 失败：' + e.message);
  }
}

// --- 合成 ---
async function synthesize() {
  const body = buildTtsBody();
  if (!body) return;
  const btn = event.target;
  btn.disabled = true;
  showStatus('tts-status', 'info', '合成中…');
  try {
    const resp = await fetch(api + '/api/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({error: resp.statusText}));
      throw new Error(err.error + (err.detail ? ': ' + err.detail : ''));
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    document.getElementById('tts-audio').src = url;
    showStatus('tts-status', 'success', '合成完成，大小 ' + (blob.size / 1024).toFixed(1) + ' KB。');
  } catch (e) {
    showStatus('tts-status', 'error', e.message);
  } finally {
    btn.disabled = false;
  }
}

// --- 流式合成 ---
async function streamSynth() {
  const body = buildTtsBody();
  if (!body) return;
  showStatus('tts-status', 'info', '流式合成中…');
  document.getElementById('tts-progress').style.display = 'block';
  document.getElementById('tts-progress-bar').style.width = '0%';

  try {
    const resp = await fetch(api + '/api/tts/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    const chunks = [];

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.event === 'start') {
          showStatus('tts-status', 'info', '开始流式合成，采样率 ' + ev.sample_rate + ' Hz');
        } else if (ev.event === 'audio_chunk') {
          const pcm = base64ToArrayBuffer(ev.pcm_b64);
          chunks.push(pcm);
          const pct = Math.min(100, (ev.frame_count / body.max_new_tokens) * 100);
          document.getElementById('tts-progress-bar').style.width = pct + '%';
          showStatus('tts-status', 'info', '已接收 ' + (ev.seq + 1) + ' 块，共 ' + ev.frame_count + ' 帧');
        } else if (ev.event === 'complete') {
          showStatus('tts-status', 'success', '流式完成，共 ' + ev.frame_count + ' 帧');
          document.getElementById('tts-progress-bar').style.width = '100%';
        } else if (ev.event === 'cancelled') {
          showStatus('tts-status', 'info', '已取消');
          break;
        }
      }
    }

    if (chunks.length > 0) {
      const wav = encodeWAV(concatBuffers(chunks), 44100);
      document.getElementById('tts-audio').src = URL.createObjectURL(new Blob([wav], {type: 'audio/wav'}));
    }
  } catch (e) {
    showStatus('tts-status', 'error', e.message);
  } finally {
    document.getElementById('tts-progress').style.display = 'none';
  }
}

async function cancelStream() {
  try {
    const resp = await fetch(api + '/api/tts/cancel', {method: 'POST'});
    if (resp.ok) {
      showStatus('tts-status', 'info', '已发送取消请求');
    } else {
      showStatus('tts-status', 'info', '无活跃流式任务');
    }
  } catch (e) {
    showStatus('tts-status', 'error', e.message);
  }
}

function buildTtsBody() {
  const text = document.getElementById('tts-text').value.trim();
  const voice = document.getElementById('tts-voice').value;
  if (!text) { showStatus('tts-status', 'error', '请输入文本'); return null; }
  if (!voice) { showStatus('tts-status', 'error', '请选择 voice'); return null; }
  return {
    text: text,
    voice_name: voice,
    max_new_tokens: parseInt(document.getElementById('tts-max-tokens').value),
    temperature: parseFloat(document.getElementById('tts-temp').value),
    top_p: parseFloat(document.getElementById('tts-top-p').value),
    top_k: parseInt(document.getElementById('tts-top-k').value),
    seed: parseInt(document.getElementById('tts-seed').value),
  };
}

// --- 注册 ---
async function registerVoice() {
  const name = document.getElementById('reg-name').value.trim();
  const refText = document.getElementById('reg-ref-text').value.trim();
  const audioFile = document.getElementById('reg-audio').files[0];
  const overwrite = document.getElementById('reg-overwrite').checked;
  if (!name || !refText || !audioFile) {
    showStatus('reg-status', 'error', '请填写所有字段并选择音频文件');
    return;
  }
  const btn = event.target;
  btn.disabled = true;
  showStatus('reg-status', 'info', '注册中…');
  try {
    const fd = new FormData();
    fd.append('name', name);
    fd.append('reference_text', refText);
    fd.append('audio', audioFile);
    fd.append('overwrite', overwrite ? 'true' : 'false');
    const resp = await fetch(api + '/api/voices/register', {method: 'POST', body: fd});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error + (data.detail ? ': ' + data.detail : ''));
    showStatus('reg-status', 'success', 'Voice "' + name + '" 注册成功');
    loadVoices();
  } catch (e) {
    showStatus('reg-status', 'error', e.message);
  } finally {
    btn.disabled = false;
  }
}

async function checkRegStatus() {
  try {
    const resp = await fetch(api + '/api/registration/status');
    const data = await resp.json();
    const lines = [
      '可用: ' + (data.available ? '是' : '否'),
      'Encoder 模型: ' + (data.encoder_model || '不存在'),
      'Manifest: ' + (data.manifest_exists ? '存在' : '不存在'),
      '指纹匹配: ' + (data.fingerprint_match ? '是' : '否'),
      'FFmpeg: ' + (data.ffmpeg_available ? '可用' : '不可用'),
    ];
    showStatus('reg-status', data.available ? 'success' : 'error', lines.join('\n'));
  } catch (e) {
    showStatus('reg-status', 'error', e.message);
  }
}

// --- 系统 ---
async function loadSystem() {
  try {
    const resp = await fetch(api + '/api/system');
    const data = await resp.json();
    const fmt = (b) => (b / 1024 / 1024).toFixed(1) + ' MB';
    const lines = [
      '引擎就绪: ' + (data.engine_ready ? '是' : '否'),
      '运行时长: ' + data.uptime_seconds.toFixed(0) + ' s',
      '线程数: ' + data.threads,
      'Go 版本: ' + data.go_version,
      'GOMAXPROCS: ' + data.gomaxprocs,
      '',
      '当前堆分配: ' + fmt(data.alloc_bytes),
      '累计堆分配: ' + fmt(data.total_alloc_bytes),
      '从 OS 获取: ' + fmt(data.sys_bytes),
      '堆在用: ' + fmt(data.heap_inuse_bytes),
      '栈在用: ' + fmt(data.stack_inuse_bytes),
      'GC 次数: ' + data.num_gc,
    ];
    document.getElementById('sys-output').textContent = lines.join('\n');
  } catch (e) {
    document.getElementById('sys-output').textContent = '加载失败: ' + e.message;
  }
}

async function reloadRuntime() {
  if (!confirm('确定要重新加载模型吗？这会中断所有进行中的推理。')) return;
  try {
    const resp = await fetch(api + '/api/runtime/reload', {method: 'POST'});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'reload failed');
    alert('重新加载成功');
    loadSystem();
    loadVoices();
  } catch (e) {
    alert('重新加载失败: ' + e.message);
  }
}

// --- PCM → WAV 工具 ---
function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}

function concatBuffers(buffers) {
  let total = 0;
  for (const b of buffers) total += b.byteLength;
  const result = new Uint8Array(total);
  let offset = 0;
  for (const b of buffers) { result.set(new Uint8Array(b), offset); offset += b.byteLength; }
  return result.buffer;
}

function encodeWAV(pcmBuffer, sampleRate) {
  const samples = new Int16Array(pcmBuffer);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(offset, samples[i], true);
    offset += 2;
  }
  return buffer;
}

// 初始化
loadVoices();
</script>
</body>
</html>`
