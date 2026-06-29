"""
apply_mesh_features.py — add tactical drawing tools + MP4 recording to
the MESH tab inside static/FORMA B.dc.html.

What this does
  Patches `app/static/FORMA B.dc.html` in place, adding:
    1. A "● Record MP4" button in the top header (next to Export Videos).
    2. A tactical drawing tool panel (Arrow / Line / Path / Animated) inside
       the MESH tab's viewport.
    3. A recording overlay modal.
    4. ~700 lines of JS that wire everything to the existing meshScene /
       meshCamera / _meshUpdateFrame globals.

Safety
  - Creates `FORMA B.dc.html.bak` before any change. Never overwrites a
    previous backup.
  - Idempotent. Refuses to run twice (looks for MESH_FEATURES_V1 marker).
  - Validates every anchor before touching the file. If any anchor is
    missing, prints which one and aborts WITHOUT modifying anything.
  - Restores from .bak on any failure mid-patch.

Where to put it / how to run
  Drop this script anywhere. From a terminal:

      python apply_mesh_features.py
      python apply_mesh_features.py --file path\\to\\FORMA B.dc.html
      python apply_mesh_features.py --revert     (restore from .bak)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Default file location, resolved relative to the script's parent.
# Works whether you run the script from the project root or from app/.
# ─────────────────────────────────────────────────────────────────────────────

def _default_target() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "static" / "FORMA B.dc.html",
        here / "app"    / "static" / "FORMA B.dc.html",
        Path.cwd() / "static" / "FORMA B.dc.html",
        Path.cwd() / "app" / "static" / "FORMA B.dc.html",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]  # report the first one as the expected default


MARKER = "MESH_FEATURES_V1"   # idempotency token written into the patched file


# ─────────────────────────────────────────────────────────────────────────────
# Anchor strings. Each must appear EXACTLY ONCE in the original file or the
# patch refuses to run. They were chosen to be distinctive enough to survive
# minor edits to surrounding lines.
# ─────────────────────────────────────────────────────────────────────────────

ANCHOR_HEADER_BUTTONS = (
    '<button onclick="toggleExportPanel()" class="tool-btn" id="btn-export">'
    '↓ Export Videos</button>'
)

ANCHOR_MESH_EMPTY = '<div id="empty-mesh"'

ANCHOR_BEFORE_STATUS_MSG = '<div id="status-msg"'

ANCHOR_END_SCRIPT = "</script>\n</body>"


# ─────────────────────────────────────────────────────────────────────────────
# Inserts
# ─────────────────────────────────────────────────────────────────────────────

# 1) New Record MP4 button to drop right after the Export Videos button.
INSERT_RECORD_BUTTON = (
    '<button onclick="recordMp4Mesh()" class="tool-btn" id="btn-record-mp4" '
    'disabled title="Record the MESH tab as MP4">● Record MP4</button>'
)


# 2) Floating tactical drawing panel — placed inside #view-mesh, just before
#    the empty-state placeholder so it stacks above the canvas at z-index 6.
INSERT_TACTICAL_PANEL = '''
<!-- ╔══ MESH-TAB TACTICAL TOOLS (added by apply_mesh_features.py) ═════════ -->
<div id="mesh-tactics-panel" style="position:absolute;top:180px;left:10px;background:rgba(15,23,42,.94);border:1px solid #334155;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:6px;min-width:138px;box-shadow:0 4px 12px rgba(0,0,0,.4);z-index:6;font-family:'Sora',system-ui,sans-serif;">
  <div style="font-size:9px;color:#f59e0b;letter-spacing:.12em;font-weight:600;margin-bottom:2px;">TACTICAL TOOLS</div>
  <button class="mt-btn active" data-mt="select"   onclick="meshSetTool('select')"  style="background:#3b82f6;color:#fff;border-color:#3b82f6;text-align:left;padding:5px 9px;font-size:11px;border:1px solid #334155;border-radius:5px;cursor:pointer;">↖ Select</button>
  <button class="mt-btn"        data-mt="arrow"    onclick="meshSetTool('arrow')"    style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;">→ Arrow</button>
  <button class="mt-btn"        data-mt="line"     onclick="meshSetTool('line')"     style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;">━ Line</button>
  <button class="mt-btn"        data-mt="path"     onclick="meshSetTool('path')"     style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;">∿ Path</button>
  <button class="mt-btn"        data-mt="animpath" onclick="meshSetTool('animpath')" style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;">⟿ Animated</button>
  <div style="display:flex;align-items:center;gap:6px;font-size:10px;color:#94a3b8;margin-top:4px;">
    <span>Color</span><input type="color" id="mt-color" value="#ff3030" style="width:28px;height:20px;border:none;border-radius:3px;cursor:pointer;padding:0;">
  </div>
  <div style="display:flex;align-items:center;gap:6px;font-size:10px;color:#94a3b8;">
    <span>Size</span><input type="range" id="mt-thick" min="1" max="10" value="4" style="flex:1;">
  </div>
  <div id="mt-dur-row" style="display:none;align-items:center;gap:6px;font-size:10px;color:#94a3b8;">
    <span>Dur</span><input type="number" id="mt-dur" min="0.5" max="20" step="0.5" value="2.0" style="width:42px;padding:2px 4px;background:#374151;color:#e2e8f0;border:1px solid #4b5563;border-radius:3px;font-size:10px;"><span>s</span>
  </div>
  <button onclick="meshUndoDrawing()" style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;margin-top:4px;">↶ Undo</button>
  <button onclick="meshClearDrawings()" style="background:#1e293b;color:#d1d5db;border:1px solid #334155;border-radius:5px;padding:5px 9px;font-size:11px;text-align:left;cursor:pointer;">✕ Clear All</button>
  <div id="mt-count" style="font-size:9px;color:#475569;text-align:center;margin-top:2px;font-family:'IBM Plex Mono',monospace;">0 drawings</div>
</div>
'''


# 3) Recording overlay modal — full-screen blocker shown during record + transcode.
INSERT_REC_OVERLAY = '''
<!-- ╔══ MESH RECORDING OVERLAY (added by apply_mesh_features.py) ══════════ -->
<div id="rec-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:300;align-items:center;justify-content:center;backdrop-filter:blur(4px);">
  <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:28px 36px;min-width:360px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.6);">
    <div id="rec-title" style="font-size:14px;color:#f59e0b;font-weight:600;letter-spacing:.06em;margin-bottom:14px;font-family:'Sora',system-ui,sans-serif;"><span id="rec-dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px;vertical-align:middle;animation:recpulse 1.2s infinite;"></span>Recording…</div>
    <div style="height:6px;background:#263344;border-radius:3px;overflow:hidden;margin:10px 0;"><div id="rec-bar" style="height:100%;background:linear-gradient(90deg,#3b82f6,#10b981);width:0%;transition:width .2s;"></div></div>
    <div id="rec-stage" style="font-size:11px;color:#94a3b8;font-family:'IBM Plex Mono',monospace;margin:8px 0 4px;">Frame 0 / 0</div>
    <div id="rec-detail" style="font-size:10px;color:#475569;font-family:'IBM Plex Mono',monospace;min-height:14px;"></div>
    <button onclick="meshCancelRecording()" style="margin-top:14px;padding:6px 16px;background:#334155;color:#e2e8f0;border:none;border-radius:5px;font-size:11px;cursor:pointer;font-family:'Sora',system-ui,sans-serif;">Cancel</button>
  </div>
</div>
<style>@keyframes recpulse{0%,100%{opacity:1}50%{opacity:.3}}</style>
'''


# 4) The big JS payload. Wired against the existing mesh-tab globals
#    (meshScene, meshCamera, meshCtrl, meshFrames, meshTotalFrames,
#     meshCurFrame, meshPW, meshPH, meshMetadata, _meshUpdateFrame).
#    Wraps _meshUpdateFrame so animated drawings keep in sync.
INSERT_JS = '''
// ╔══════════════════════════════════════════════════════════════════════════
// ║  MESH-TAB TACTICAL DRAWINGS + MP4 RECORDING (added by patch script)
// ║  Tag: ''' + MARKER + '''
// ║
// ║  Lives entirely inside the MESH tab. Uses the existing meshScene,
// ║  meshCamera, meshCtrl, _meshUpdateFrame, meshPW, meshPH globals so it
// ║  matches the pitch and player meshes that are already on screen.
// ║
// ║  Drawings:
// ║    - 4 primitive types (arrow, line, path, animated path), all on the
// ║      pitch plane at y=0.05, raycasted from screen-space mouse clicks.
// ║    - Persisted across frames. Animated paths animate from the frame
// ║      they were created on, using setDrawRange on a tube geometry.
// ║    - OrbitControls are disabled while a drawing tool is active.
// ║
// ║  Recording:
// ║    - canvas.captureStream(fps) + MediaRecorder for WebM (VP9 preferred,
// ║      VP8 fallback).
// ║    - Drives _meshUpdateFrame() at the source fps via setTimeout so
// ║      output duration matches video duration.
// ║    - POSTs the WebM to /transcode-to-mp4 for ffmpeg → MP4 conversion.
// ║    - If transcode fails the WebM is still returned to the user as
// ║      a fallback so the recording is never lost.
// ╚══════════════════════════════════════════════════════════════════════════

(function() {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  const drawings = [];
  const drawingObjects = {};
  let currentTool = 'select';
  let drawingInProgress = null;
  const raycaster = new THREE.Raycaster();
  const pitchPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

  // ── Geometry builders ────────────────────────────────────────────────────
  function buildTube(points, thickness) {
    if (points.length < 2) return null;
    const v3 = points.map(p => new THREE.Vector3(p[0], 0.05, p[1]));
    const curve = (points.length === 2)
      ? new THREE.LineCurve3(v3[0], v3[1])
      : new THREE.CatmullRomCurve3(v3, false, 'catmullrom', 0.1);
    const segs = Math.max(8, Math.min(256, points.length * 8));
    return new THREE.TubeGeometry(curve, segs, thickness, 8, false);
  }

  function buildArrowHead(start, end, thickness, color) {
    const dir = new THREE.Vector3(end[0] - start[0], 0, end[1] - start[1]);
    const len = dir.length();
    if (len < 1e-4) return null;
    dir.normalize();
    const headLen = Math.max(thickness * 4, 0.8);
    const headRad = Math.max(thickness * 2.5, 0.5);
    const geom = new THREE.ConeGeometry(headRad, headLen, 16);
    const mat = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.Mesh(geom, mat);
    const up = new THREE.Vector3(0, 1, 0);
    mesh.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(up, dir));
    mesh.position.set(
      end[0] + dir.x * (headLen / 2 - thickness),
      0.05,
      end[1] + dir.z * (headLen / 2 - thickness),
    );
    return mesh;
  }

  // ── Drawing lifecycle ────────────────────────────────────────────────────
  function instantiate(d) {
    if (!window.meshScene) return;
    const color = new THREE.Color(d.color);
    const mat = new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide });
    const meshes = [];
    let totalIndices = 0;

    const tubeGeom = buildTube(d.points, d.thickness);
    if (tubeGeom) {
      const tube = new THREE.Mesh(tubeGeom, mat);
      window.meshScene.add(tube);
      meshes.push(tube);
      totalIndices = tubeGeom.index ? tubeGeom.index.count : 0;
    }

    if (d.type === 'arrow' && d.points.length >= 2) {
      const head = buildArrowHead(
        d.points[d.points.length - 2],
        d.points[d.points.length - 1],
        d.thickness, color
      );
      if (head) { window.meshScene.add(head); meshes.push(head); }
    }

    drawingObjects[d.id] = { meshes, totalIndices, material: mat };
  }

  function removeObjects(id) {
    const rec = drawingObjects[id];
    if (!rec) return;
    for (const m of rec.meshes) {
      if (window.meshScene) window.meshScene.remove(m);
      if (m.geometry) m.geometry.dispose();
    }
    if (rec.material) rec.material.dispose();
    delete drawingObjects[id];
  }

  function updateCountUI() {
    const el = document.getElementById('mt-count');
    if (el) el.textContent = drawings.length + ' drawing' + (drawings.length === 1 ? '' : 's');
  }

  // ── Mouse → pitch ────────────────────────────────────────────────────────
  function mouseToPitch(e) {
    const canvas = document.getElementById('c-mesh');
    if (!canvas || !window.meshCamera) return null;
    const rect = canvas.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(mouse, window.meshCamera);
    const hit = new THREE.Vector3();
    if (!raycaster.ray.intersectPlane(pitchPlane, hit)) return null;
    const PW = window.meshPW || 105, PH = window.meshPH || 68;
    if (Math.abs(hit.x) > PW || Math.abs(hit.z) > PH) return null;
    return [hit.x, hit.z];
  }

  // ── Public-ish API exposed on window for inline onclick handlers ─────────
  window.meshSetTool = function(tool) {
    currentTool = tool;
    document.querySelectorAll('.mt-btn').forEach(b => {
      const active = b.dataset.mt === tool;
      b.classList.toggle('active', active);
      // Inline-style buttons need their background updated too
      b.style.background = active ? '#3b82f6' : '#1e293b';
      b.style.color      = active ? '#fff'    : '#d1d5db';
      b.style.borderColor = active ? '#3b82f6' : '#334155';
    });
    const durRow = document.getElementById('mt-dur-row');
    if (durRow) durRow.style.display = (tool === 'animpath') ? 'flex' : 'none';
    if (window.meshCtrl) window.meshCtrl.enabled = (tool === 'select');
    const c = document.getElementById('c-mesh');
    if (c) c.style.cursor = (tool === 'select') ? '' : 'crosshair';
  };

  window.meshUndoDrawing = function() {
    if (drawings.length === 0) return;
    const d = drawings.pop();
    removeObjects(d.id);
    updateCountUI();
  };

  window.meshClearDrawings = function() {
    if (drawings.length === 0) return;
    if (!confirm('Remove all ' + drawings.length + ' drawing(s)?')) return;
    for (const id of Object.keys(drawingObjects)) removeObjects(id);
    drawings.length = 0;
    updateCountUI();
  };

  function addDrawing(type, points) {
    if (points.length < 2) return;
    const color = document.getElementById('mt-color').value;
    const slider = parseInt(document.getElementById('mt-thick').value, 10);
    const thickness = 0.05 + (slider - 1) / 9 * 0.45;
    const d = {
      id: 'd' + Date.now() + '_' + Math.random().toString(36).slice(2, 7),
      type, points, color, thickness,
    };
    if (type === 'animpath') {
      d.duration = parseFloat(document.getElementById('mt-dur').value) || 2.0;
      d.startFrame = (typeof window.meshCurFrame === 'number') ? window.meshCurFrame : 0;
    }
    drawings.push(d);
    instantiate(d);
    updateAnimated(window.meshCurFrame || 0);
    updateCountUI();
  }

  function updateAnimated(frame) {
    const fps = (window.meshMetadata && window.meshMetadata.playback_fps) || 25;
    for (const d of drawings) {
      if (d.type !== 'animpath') continue;
      const rec = drawingObjects[d.id];
      if (!rec || rec.meshes.length === 0) continue;
      const durFrames = Math.max(1, Math.round(d.duration * fps));
      let progress;
      if (frame < d.startFrame) progress = 0;
      else if (frame >= d.startFrame + durFrames) progress = 1;
      else progress = (frame - d.startFrame) / durFrames;
      const tube = rec.meshes[0];
      if (tube && tube.geometry && tube.geometry.index) {
        tube.geometry.setDrawRange(0, Math.floor(rec.totalIndices * progress));
      }
      tube.visible = (frame >= d.startFrame);
    }
  }

  // ── Mouse handlers on the mesh canvas ────────────────────────────────────
  function wirePointerEvents() {
    const canvas = document.getElementById('c-mesh');
    if (!canvas) {
      // The mesh canvas only exists after the mesh tab is initialised; wait.
      setTimeout(wirePointerEvents, 300);
      return;
    }
    if (canvas.dataset.tacticalWired === '1') return;
    canvas.dataset.tacticalWired = '1';

    let pointerDown = false;

    canvas.addEventListener('pointerdown', (e) => {
      if (currentTool === 'select') return;
      if (!window.meshTotalFrames) {
        if (typeof window._showToast === 'function') window._showToast('Load a mesh JSON first', 4000);
        return;
      }
      e.preventDefault();
      const pt = mouseToPitch(e);
      if (!pt) return;
      pointerDown = true;
      drawingInProgress = { type: currentTool, points: [pt], previewObj: null };
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!pointerDown || !drawingInProgress) return;
      e.preventDefault();
      const pt = mouseToPitch(e);
      if (!pt) return;
      const dip = drawingInProgress;
      if (dip.type === 'path' || dip.type === 'animpath') {
        const last = dip.points[dip.points.length - 1];
        const dx = pt[0] - last[0], dz = pt[1] - last[1];
        if (dx * dx + dz * dz < 0.04) return;
        dip.points.push(pt);
      } else {
        dip.points = [dip.points[0], pt];
      }
      if (dip.previewObj) {
        if (window.meshScene) window.meshScene.remove(dip.previewObj);
        if (dip.previewObj.geometry) dip.previewObj.geometry.dispose();
        if (dip.previewObj.material) dip.previewObj.material.dispose();
        dip.previewObj = null;
      }
      if (dip.points.length >= 2) {
        const colorHex = document.getElementById('mt-color').value;
        const slider = parseInt(document.getElementById('mt-thick').value, 10);
        const thickness = 0.05 + (slider - 1) / 9 * 0.45;
        const geom = buildTube(dip.points, thickness);
        if (geom) {
          const mat = new THREE.MeshBasicMaterial({
            color: new THREE.Color(colorHex), transparent: true, opacity: 0.7,
          });
          const mesh = new THREE.Mesh(geom, mat);
          if (window.meshScene) window.meshScene.add(mesh);
          dip.previewObj = mesh;
        }
      }
    });

    function endStroke(e) {
      if (!pointerDown || !drawingInProgress) { pointerDown = false; return; }
      pointerDown = false;
      if (drawingInProgress.previewObj) {
        if (window.meshScene) window.meshScene.remove(drawingInProgress.previewObj);
        if (drawingInProgress.previewObj.geometry) drawingInProgress.previewObj.geometry.dispose();
        if (drawingInProgress.previewObj.material) drawingInProgress.previewObj.material.dispose();
      }
      if (drawingInProgress.points.length >= 2) {
        addDrawing(drawingInProgress.type, drawingInProgress.points);
      }
      drawingInProgress = null;
    }
    canvas.addEventListener('pointerup', endStroke);
    canvas.addEventListener('pointercancel', endStroke);
    canvas.addEventListener('pointerleave', endStroke);
  }
  wirePointerEvents();

  // ── Hook _meshUpdateFrame to refresh animated drawings every frame ───────
  function installFrameHook() {
    if (typeof window._meshUpdateFrame !== 'function') {
      setTimeout(installFrameHook, 300);
      return;
    }
    if (window._meshUpdateFrame.__tacticalWrapped) return;
    const orig = window._meshUpdateFrame;
    const wrapped = function(idx) {
      const out = orig(idx);
      try { updateAnimated(window.meshCurFrame || 0); } catch(_) {}
      return out;
    };
    wrapped.__tacticalWrapped = true;
    window._meshUpdateFrame = wrapped;
  }
  installFrameHook();

  // ╔══════════════════════════════════════════════════════════════════════
  // ║  MP4 RECORDING
  // ╚══════════════════════════════════════════════════════════════════════

  const recState = { active: false, recorder: null, chunks: [], cancelled: false };

  function pickMimeType() {
    const list = [
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8,opus',
      'video/webm;codecs=vp8',
      'video/webm',
    ];
    for (const m of list) {
      if (typeof MediaRecorder !== 'undefined'
          && MediaRecorder.isTypeSupported
          && MediaRecorder.isTypeSupported(m)) return m;
    }
    return null;
  }

  function showOverlay(show) {
    const el = document.getElementById('rec-overlay');
    if (el) el.style.display = show ? 'flex' : 'none';
  }
  function setProgress(pct, stage, detail) {
    const bar = document.getElementById('rec-bar');
    if (bar) bar.style.width = pct + '%';
    if (stage !== undefined) {
      const s = document.getElementById('rec-stage');
      if (s) s.textContent = stage;
    }
    if (detail !== undefined) {
      const d = document.getElementById('rec-detail');
      if (d) d.textContent = detail;
    }
  }
  function setTitle(t) {
    const el = document.getElementById('rec-title');
    if (el) el.innerHTML = '<span id="rec-dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px;vertical-align:middle;animation:recpulse 1.2s infinite;"></span>' + t;
  }

  window.recordMp4Mesh = async function() {
    if (recState.active) return;
    if (!window.meshTotalFrames) {
      if (typeof window._showToast === 'function') window._showToast('Load a mesh JSON first', 4000);
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      if (typeof window._showToast === 'function') window._showToast('Your browser does not support MediaRecorder.', 5000);
      return;
    }
    const mime = pickMimeType();
    if (!mime) {
      if (typeof window._showToast === 'function') window._showToast('No supported WebM codec in this browser.', 5000);
      return;
    }
    // Switch to mesh tab if not already (recording requires c-mesh to be visible
    // and rendering — captureStream returns blank frames for hidden canvases)
    if (window.activeTab !== 'mesh' && typeof window.switchTab === 'function') {
      window.switchTab('mesh');
      // give the layout a frame to settle
      await new Promise(r => setTimeout(r, 200));
    }
    // Pause any normal playback
    if (window.meshPlaying) window.meshPlaying = false;

    const canvas = document.getElementById('c-mesh');
    if (!canvas) {
      if (typeof window._showToast === 'function') window._showToast('Mesh canvas not found.', 5000);
      return;
    }
    const fps = Math.max(1, Math.round((window.meshMetadata && window.meshMetadata.playback_fps) || 25));
    let stream;
    try { stream = canvas.captureStream(fps); }
    catch (err) {
      if (typeof window._showToast === 'function') window._showToast('Canvas capture failed: ' + err.message, 6000);
      return;
    }

    recState.active = true;
    recState.cancelled = false;
    recState.chunks = [];

    let recorder;
    try {
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6_000_000 });
    } catch (err) {
      recState.active = false;
      if (typeof window._showToast === 'function') window._showToast('MediaRecorder failed: ' + err.message, 6000);
      return;
    }
    recState.recorder = recorder;
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recState.chunks.push(e.data); };
    recorder.onerror = (e) => { console.error('MediaRecorder error:', e); };

    showOverlay(true);
    setTitle('Recording…');
    setProgress(0, 'Frame 0 / ' + window.meshTotalFrames, mime);
    recorder.start(1000);

    // Drive playback through _meshUpdateFrame at fps
    const total = window.meshTotalFrames;
    const dur = 1000 / fps;
    const tStart = performance.now();
    let frame = 0;

    await new Promise((resolve) => {
      function tick() {
        if (recState.cancelled) { resolve(); return; }
        if (frame >= total) { resolve(); return; }
        try { window._meshUpdateFrame(frame); } catch(_) {}
        setProgress(Math.round(100 * frame / total),
          'Frame ' + (frame + 1) + ' / ' + total,
          fps + ' fps • ' + mime);
        frame++;
        const expected = tStart + frame * dur;
        const delay = Math.max(0, expected - performance.now());
        setTimeout(tick, delay);
      }
      tick();
    });

    await new Promise(r => setTimeout(r, 400));

    const stopped = new Promise(res => { recorder.onstop = () => res(); });
    try { recorder.stop(); } catch(_) {}
    await stopped;

    try { stream.getTracks().forEach(t => t.stop()); } catch(_) {}
    recState.recorder = null;
    recState.active = false;

    if (recState.cancelled) {
      showOverlay(false);
      if (typeof window._showToast === 'function') window._showToast('Recording cancelled.', 3000);
      return;
    }

    const blob = new Blob(recState.chunks, { type: mime });
    recState.chunks = [];

    setTitle('Transcoding to MP4…');
    setProgress(0, 'Uploading to server…', 'WebM size: ' + (blob.size / 1024 / 1024).toFixed(1) + ' MB');

    try {
      const mp4Url = await uploadAndTranscode(blob, mime);
      setProgress(100, 'Ready — download starting', '');
      setTitle('Done');
      const a = document.createElement('a');
      a.href = mp4Url; a.download = 'tactical_analysis.mp4';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => showOverlay(false), 1400);
      if (typeof window._showToast === 'function') window._showToast('MP4 download started.', 4000);
    } catch (err) {
      setTitle('Transcode failed');
      setProgress(0, err.message || 'Unknown error', 'Falling back to WebM…');
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'tactical_analysis.webm';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      setTimeout(() => showOverlay(false), 2500);
      if (typeof window._showToast === 'function') window._showToast('Server transcode failed; WebM downloaded.', 6000);
    }
  };

  window.meshCancelRecording = function() {
    if (!recState.active) { showOverlay(false); return; }
    recState.cancelled = true;
    if (recState.recorder && recState.recorder.state !== 'inactive') {
      try { recState.recorder.stop(); } catch(_) {}
    }
  };

  async function uploadAndTranscode(blob, mime) {
    const form = new FormData();
    form.append('video', blob, 'recording.webm');
    form.append('mime_type', mime);
    let resp;
    try { resp = await fetch('/transcode-to-mp4', { method: 'POST', body: form }); }
    catch (err) { throw new Error('Network error: ' + err.message); }
    if (!resp.ok) {
      let detail = '';
      try { detail = await resp.text(); } catch(_) {}
      throw new Error('Server returned ' + resp.status + ': ' + (detail || resp.statusText).slice(0, 200));
    }
    const out = await resp.blob();
    if (out.size === 0) throw new Error('Server returned an empty MP4');
    return URL.createObjectURL(out);
  }

  // ── Enable the Record button when mesh data lands ────────────────────────
  // loadMeshData runs once per file load; wrap it to enable our button after.
  function installLoadHook() {
    if (typeof window.loadMeshData !== 'function') {
      setTimeout(installLoadHook, 300);
      return;
    }
    if (window.loadMeshData.__recordHooked) return;
    const orig = window.loadMeshData;
    const wrapped = function(data) {
      const r = orig(data);
      const btn = document.getElementById('btn-record-mp4');
      if (btn) btn.disabled = false;
      return r;
    };
    wrapped.__recordHooked = true;
    window.loadMeshData = wrapped;
  }
  installLoadHook();

  console.log('[mesh-features] tactical drawings + MP4 recording loaded');
})();
'''


# ─────────────────────────────────────────────────────────────────────────────
# Patch driver
# ─────────────────────────────────────────────────────────────────────────────

def _validate_anchors(html: str, anchors: dict[str, str]) -> list[str]:
    """Return list of error messages, empty on success."""
    errs = []
    for name, anchor in anchors.items():
        n = html.count(anchor)
        if n == 0:
            errs.append(f"  - {name}: anchor not found")
        elif n > 1:
            errs.append(f"  - {name}: anchor appears {n} times (must be unique)")
    return errs


def _insert_after(html: str, anchor: str, payload: str) -> str:
    """Insert payload immediately after the first occurrence of anchor."""
    i = html.find(anchor)
    if i < 0:
        raise RuntimeError(f"anchor not found at insert time: {anchor[:80]!r}")
    j = i + len(anchor)
    return html[:j] + payload + html[j:]


def _insert_before(html: str, anchor: str, payload: str) -> str:
    """Insert payload immediately before the first occurrence of anchor."""
    i = html.find(anchor)
    if i < 0:
        raise RuntimeError(f"anchor not found at insert time: {anchor[:80]!r}")
    return html[:i] + payload + html[i:]


def patch(target: Path) -> int:
    if not target.is_file():
        print(f"[patch] ERROR: file not found: {target}")
        print(f"[patch] Pass --file path/to/FORMA B.dc.html explicitly.")
        return 2

    html = target.read_text(encoding="utf-8")

    if MARKER in html:
        print(f"[patch] Already patched (marker '{MARKER}' present). "
              f"Use --revert to undo.")
        return 0

    anchors = {
        "header buttons":  ANCHOR_HEADER_BUTTONS,
        "mesh empty msg":  ANCHOR_MESH_EMPTY,
        "status msg":      ANCHOR_BEFORE_STATUS_MSG,
        "end of <script>": ANCHOR_END_SCRIPT,
    }
    errs = _validate_anchors(html, anchors)
    if errs:
        print(f"[patch] ERROR: anchor validation failed.")
        print("\n".join(errs))
        print(f"[patch] Your FORMA B.dc.html may have been hand-edited or "
              f"differs from the expected baseline.")
        return 3

    # Backup
    bak = target.with_suffix(target.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(target, bak)
        print(f"[patch] backup saved: {bak.name}")
    else:
        print(f"[patch] backup already exists, leaving as-is: {bak.name}")

    try:
        new_html = html
        # 1. Record MP4 button after the Export Videos button
        new_html = _insert_after(new_html, ANCHOR_HEADER_BUTTONS,
                                 "\n      " + INSERT_RECORD_BUTTON)
        # 2. Tactical panel before the empty-mesh placeholder
        new_html = _insert_before(new_html, ANCHOR_MESH_EMPTY, INSERT_TACTICAL_PANEL + "        ")
        # 3. Recording overlay before the status-msg toast
        new_html = _insert_before(new_html, ANCHOR_BEFORE_STATUS_MSG, INSERT_REC_OVERLAY + "\n")
        # 4. JS payload just before </script>
        new_html = _insert_before(new_html, ANCHOR_END_SCRIPT, INSERT_JS + "\n")
    except Exception as e:
        # Restore on any failure
        print(f"[patch] ERROR during patch: {e}")
        shutil.copy2(bak, target)
        print(f"[patch] target restored from backup, no changes applied.")
        return 4

    target.write_text(new_html, encoding="utf-8")
    delta = len(new_html) - len(html)
    print(f"[patch] ✓ patched successfully ({delta:+d} bytes)")
    print(f"[patch] file: {target}")
    print(f"\nNext step:")
    print(f"  1) Restart the server: python main.py")
    print(f"  2) Open http://localhost:8000")
    print(f"  3) Load a mesh JSON. The tactical panel appears in the MESH tab")
    print(f"     and the 'Record MP4' button enables in the top header.")
    return 0


def revert(target: Path) -> int:
    bak = target.with_suffix(target.suffix + ".bak")
    if not bak.is_file():
        print(f"[patch] ERROR: no backup found at {bak}")
        return 2
    shutil.copy2(bak, target)
    print(f"[patch] ✓ reverted from {bak.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(_default_target()),
                    help="path to FORMA B.dc.html (auto-detected by default)")
    ap.add_argument("--revert", action="store_true",
                    help="restore from .bak instead of patching")
    args = ap.parse_args()

    target = Path(args.file)
    if args.revert:
        return revert(target)
    return patch(target)


if __name__ == "__main__":
    sys.exit(main())
