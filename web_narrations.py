#!/usr/bin/env python3
"""
HD-EPIC Web Narration Player

Opens a browser UI with the YouTube video on the left and a live
narration feed on the right.  The video clock drives everything.

Usage:
    python web_narrations.py --video-id P01-20240202-110250

Controls  (keyboard shortcuts work too):
    Space       play / pause
    ←  →        seek ±10 s
"""

import argparse
import csv
import os
import sys
import threading
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from flask import Flask, jsonify, render_template_string
except ImportError:
    sys.exit("flask is required.  Run:  pip install flask")

app = Flask(__name__)
_state: dict = {}   # video_id, yt_id, narrations, fixtures

# ── Noun filter ───────────────────────────────────────────────────────────────

TARGET_NOUN_KEYS = {
    "tap", "cupboard", "lid", "drawer", "fridge", "container", "box", "hob",
    "pot", "tray", "bin", "kettle", "oven", "maker:coffee", "rubbish", "sink",
    "heat", "dishwasher", "filter", "blender", "mat", "grater", "juicer",
    "scale", "rest", "rack:drying", "alarm", "freezer", "cap", "machine:washing",
    "holder", "pin:rolling", "processor:food", "cooker:slow", "plug", "utensil",
    "phone", "thermometer", "spinner:salad", "presser", "opener:bottle",
    "toaster", "knob", "handle", "whisk", "slicer", "control:remote", "label",
    "hoover", "tv", "shelf", "stand", "machine:sous:vide", "masher", "cork",
    "pestle", "window", "heater", "watch", "power", "airer", "computer",
    "door:kitchen", "tape", "camera", "cd", "hook", "machine:candy:floss",
    "machine:hot:chocolate", "top", "light"
}


def get_target_noun_ids(noun_classes_path: str) -> set:
    """Return numeric noun class IDs that match TARGET_NOUN_KEYS."""
    import csv as _csv
    ids = set()
    try:
        with open(noun_classes_path, newline="") as f:
            for row in _csv.DictReader(f):
                if row["key"] in TARGET_NOUN_KEYS:
                    ids.add(int(row["id"]))
    except FileNotFoundError:
        pass
    return ids


def build_instance_to_key_map(csv_path: str) -> dict:
    """Return {instance_string: canonical_key} from a verb/noun classes CSV."""
    import ast as _ast
    import csv as _csv
    m = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                key = row["key"]
                m[key] = key
                try:
                    instances = _ast.literal_eval(row["instances"])
                except Exception:
                    instances = []
                for inst in instances:
                    m[inst] = key
    except FileNotFoundError:
        pass
    return m

# ── HTML / JS template ────────────────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HD-EPIC · {{ video_id }}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #0d1117;
      --surface: #161b22;
      --border:  #30363d;
      --text:    #c9d1d9;
      --muted:   #8b949e;
      --accent:  #58a6ff;
      --green:   #3fb950;
      --red:     #f85149;
      --yellow:  #d29922;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      user-select: none;
    }

    /* ── header ── */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 8px 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
      height: 44px;
    }
    .logo   { font-size: 15px; font-weight: 700; color: var(--accent); letter-spacing: 1px; }
    .vid-id { font-size: 12px; color: var(--muted); }
    .hints  { font-size: 11px; color: var(--muted); display: flex; gap: 6px; align-items: center; }
    kbd {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 10px;
      font-family: inherit;
      color: var(--muted);
    }
    .clock {
      margin-left: auto;
      text-align: right;
      font-variant-numeric: tabular-nums;
      line-height: 1.2;
    }
    .clock-ts {
      display: block;
      font-size: 18px;
      font-weight: 700;
      color: var(--green);
      letter-spacing: 2px;
    }
    .clock-frame {
      display: block;
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 1px;
    }

    /* ── main ── */
    .main { display: flex; flex: 1; overflow: hidden; }

    /* ── left ── */
    .left {
      width: 56%;
      display: flex;
      flex-direction: column;
      border-right: 1px solid var(--border);
      padding: 12px;
      gap: 10px;
    }

    .player-wrap {
      flex: 1;
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    #player { position: absolute; inset: 0; width: 100%; height: 100%; }

    /* ── controls ── */
    .controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      flex-shrink: 0;
    }

    .btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 8px;
      padding: 9px 18px;
      font-family: inherit;
      font-size: 13px;
      cursor: pointer;
      transition: background .15s, border-color .15s, transform .08s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover  { background: var(--border); border-color: var(--accent); }
    .btn:active { transform: scale(.96); }

    #btnPlay {
      background: #1a7f37;
      border-color: var(--green);
      color: #fff;
      padding: 9px 30px;
      font-size: 15px;
      min-width: 140px;
      justify-content: center;
    }
    #btnPlay:hover { background: var(--green); }
    #btnPlay.pausing {
      background: #6e1c1c;
      border-color: var(--red);
    }
    #btnPlay.pausing:hover { background: var(--red); }

    /* ── right ── */
    .right {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .term-header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 7px 14px;
      font-size: 11px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }
    .dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 6px var(--green);
      transition: background .3s, box-shadow .3s;
    }
    .dot.paused { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }

    .terminal {
      flex: 1;
      overflow-y: auto;
      padding: 10px 14px;
    }
    .terminal::-webkit-scrollbar       { width: 5px; }
    .terminal::-webkit-scrollbar-track { background: var(--bg); }
    .terminal::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .line {
      display: flex;
      gap: 10px;
      padding: 5px 0;
      border-bottom: 1px solid #161b22;
      animation: pop .2s ease-out;
      line-height: 1.6;
    }
    .line:last-child { border-bottom: none; }
    .line.current {
      background: rgba(56, 139, 253, .07);
      border-radius: 4px;
      padding: 5px 8px;
      margin: 0 -8px;
    }
    @keyframes pop {
      from { opacity: 0; transform: translateY(5px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .ts  { color: var(--accent); font-size: 11px; white-space: nowrap; flex-shrink: 0; padding-top: 3px; }
    .txt { color: var(--text); font-size: 13px; }

    /* ── fixture lines ── */
    .line-fixture {
      border-left: 2px solid transparent;
      padding-left: 8px;
      margin-left: -10px;
    }
    .line-fixture.move-start {
      border-color: var(--yellow);
      background: rgba(210, 153, 34, .06);
    }
    .line-fixture.move-end {
      border-color: var(--green);
      background: rgba(63, 185, 80, .06);
    }
    .fx-icon { font-size: 13px; flex-shrink: 0; padding-top: 2px; }
    .fx-name { font-weight: 600; font-size: 13px; }
    .fx-loc  { font-size: 12px; color: var(--muted); }
    .fx-arrow { font-size: 11px; color: var(--muted); margin: 0 4px; }

    /* ── noun / verb tags ── */
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 4px;
    }
    .tag {
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 10px;
      white-space: nowrap;
    }
    .tag-verb {
      background: rgba(88, 166, 255, .12);
      color: #79c0ff;
      border: 1px solid rgba(88, 166, 255, .25);
    }
    .tag-noun {
      background: rgba(210, 153, 34, .12);
      color: #e3b341;
      border: 1px solid rgba(210, 153, 34, .25);
    }
    .tag-pair {
      display: inline-flex;
      align-items: center;
      border-radius: 10px;
      overflow: hidden;
      font-size: 10px;
      white-space: nowrap;
      border: 1px solid rgba(88, 166, 255, .25);
    }
    .tag-pair .pair-verb {
      background: rgba(88, 166, 255, .18);
      color: #79c0ff;
      padding: 1px 6px;
    }
    .tag-pair .pair-sep {
      background: rgba(88, 166, 255, .06);
      color: #555e6a;
      padding: 1px 3px;
      font-size: 9px;
    }
    .tag-pair .pair-noun {
      background: rgba(210, 153, 34, .18);
      color: #e3b341;
      padding: 1px 6px;
      border-left: 1px solid rgba(88, 166, 255, .15);
    }

    /* loading overlay */
    #loading {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000a;
      color: var(--muted);
      font-size: 13px;
      border-radius: 8px;
      pointer-events: none;
      transition: opacity .4s;
    }
    #loading.hidden { opacity: 0; }
  </style>
</head>
<body>

<header>
  <span class="logo">HD-EPIC</span>
  <span class="vid-id">{{ video_id }}</span>
  <span class="hints">
    <kbd>Space</kbd> play/pause &nbsp;
    <kbd>←</kbd> <kbd>→</kbd> ±10 s
  </span>
  <span class="clock" id="clock">
    <span class="clock-ts"  id="clock-ts">00:00.000</span>
    <span class="clock-frame" id="clock-frame">frame 0</span>
  </span>
</header>

<div class="main">

  <!-- ── left panel ── -->
  <div class="left">
    <div class="player-wrap">
      <div id="player"></div>
      <div id="loading">Loading player…</div>
    </div>
    <div class="controls">
      <button class="btn" id="btnBack" title="−10 s  [←]">⏮ −10 s</button>
      <button class="btn" id="btnPlay" onclick="togglePlay()">▶ Play</button>
      <button class="btn" id="btnFwd"  title="+10 s  [→]">+10 s ⏭</button>
    </div>
  </div>

  <!-- ── right panel ── -->
  <div class="right">
    <div class="term-header">
      <span class="dot" id="dot"></span>
      <span id="status">READY</span>
      <span style="margin-left:auto" id="count"></span>
    </div>
    <div class="terminal" id="terminal"></div>
  </div>

</div>

<!-- YouTube IFrame API -->
<script>
  /* ── state ── */
  let player      = null;
  let narrations  = [];
  let fixtures    = [];
  let shownSet    = new Set();   // "n{i}" or "f{i}"
  let lastT       = 0;
  let isPlaying   = false;

  /* ── load narrations ── */
  fetch("/api/narrations")
    .then(r => r.json())
    .then(data => {
      narrations = data;
      updateCount();
    });

  /* ── load fixtures ── */
  fetch("/api/fixtures")
    .then(r => r.json())
    .then(data => {
      fixtures = data;
      updateCount();
    });

  function updateCount() {
    const n = narrations.length, f = fixtures.length;
    document.getElementById("count").textContent =
      n + " narrations" + (f ? " · " + f + " moves" : "");
  }

  /* ── YouTube IFrame API bootstrap ── */
  (function () {
    const s = document.createElement("script");
    s.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(s);
  })();

  function onYouTubeIframeAPIReady() {
    player = new YT.Player("player", {
      videoId: "{{ yt_id }}",
      playerVars: {
        enablejsapi:    1,
        rel:            0,
        modestbranding: 1,
        controls:       0,   // hide native controls; we use our own
        disablekb:      0,
        iv_load_policy: 3,
        fs:             0,
      },
      events: {
        onReady:       onPlayerReady,
        onStateChange: onStateChange,
      }
    });
  }

  function onPlayerReady() {
    document.getElementById("loading").classList.add("hidden");
  }

  function onStateChange(e) {
    isPlaying = e.data === YT.PlayerState.PLAYING;
    updatePlayBtn();
    updateStatus();
  }

  /* ── controls ── */
  function togglePlay() {
    if (!player) return;
    isPlaying ? player.pauseVideo() : player.playVideo();
  }

  function seek(delta) {
    if (!player) return;
    const t = player.getCurrentTime();
    player.seekTo(Math.max(0, t + delta), true);
  }

  document.getElementById("btnBack").addEventListener("click", () => seek(-10));
  document.getElementById("btnFwd" ).addEventListener("click", () => seek( 10));

  document.addEventListener("keydown", e => {
    if (e.key === " ")            { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft")  { e.preventDefault(); seek(-10); }
    else if (e.key === "ArrowRight") { e.preventDefault(); seek( 10); }
  });

  /* ── UI helpers ── */
  function updatePlayBtn() {
    const btn = document.getElementById("btnPlay");
    if (isPlaying) {
      btn.innerHTML = "⏸ Pause";
      btn.classList.add("pausing");
    } else {
      btn.innerHTML = "▶ Play";
      btn.classList.remove("pausing");
    }
  }

  function updateStatus() {
    const dot  = document.getElementById("dot");
    const stat = document.getElementById("status");
    if (isPlaying) {
      dot.classList.remove("paused");
      stat.textContent = "PLAYING";
    } else {
      dot.classList.add("paused");
      stat.textContent = "PAUSED";
    }
  }

  function fmtTs(s) {
    const h   = Math.floor(s / 3600);
    const m   = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    const ms  = Math.round((s - Math.floor(s)) * 1000);
    const p2  = n => String(n).padStart(2, "0");
    const p3  = n => String(n).padStart(3, "0");
    return h > 0
      ? `${p2(h)}:${p2(m)}:${p2(sec)}.${p3(ms)}`
      : `${p2(m)}:${p2(sec)}.${p3(ms)}`;
  }

  function esc(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function bumpCurrent(el) {
    const term = document.getElementById("terminal");
    const prev = term.querySelector(".current");
    if (prev) prev.classList.remove("current");
    el.classList.add("current");
  }

  function makePairTags(pairs) {
    return (pairs || []).map(([v, n]) =>
      `<span class="tag-pair">` +
        `<span class="pair-verb">${esc(v)}</span>` +
        `<span class="pair-sep">→</span>` +
        `<span class="pair-noun">${esc(n)}</span>` +
      `</span>`
    ).join("");
  }

  function appendNarrationLine(n, idx) {
    const term = document.getElementById("terminal");
    const el = document.createElement("div");
    el.className = "line";
    el.id = "n" + idx;

    const pairTags = makePairTags(n.pairs);

    el.innerHTML =
      `<span class="ts">[${fmtTs(n.start_timestamp)} · f${Math.floor(n.start_timestamp * 30)}]</span>` +
      `<span class="txt">` +
        `${esc(n.narration)}` +
        (pairTags ? `<div class="tags">${pairTags}</div>` : "") +
      `</span>`;
    term.appendChild(el);
    bumpCurrent(el);
    term.scrollTop = term.scrollHeight;
  }

  function cleanFixture(raw) {
    if (!raw || raw === "Null") return "unknown location";
    // strip participant prefix e.g. "P01_counter.008" → "counter.008"
    return raw.replace(/^P\d+_/, "");
  }

  function appendFixtureLine(f, idx) {
    const term  = document.getElementById("terminal");
    const isStart = f.type === "start";
    const color   = isStart ? "var(--yellow)" : "var(--green)";
    const icon    = isStart ? "↑" : "↓";
    const prep    = isStart ? "from" : "to";
    const loc     = cleanFixture(f.fixture);

    const el = document.createElement("div");
    el.className = "line line-fixture " + (isStart ? "move-start" : "move-end");
    el.id = "f" + idx;
    el.innerHTML =
      `<span class="ts">[${fmtTs(f.timestamp)} · f${Math.floor(f.timestamp * 30)}]</span>` +
      `<span class="fx-icon" style="color:${color}">${icon}</span>` +
      `<span class="txt">` +
        `<span class="fx-name" style="color:${color}">${esc(f.name)}</span>` +
        `<span class="fx-arrow">—</span>` +
        `<span class="fx-loc">${prep} <em>${esc(loc)}</em></span>` +
      `</span>`;
    term.appendChild(el);
    bumpCurrent(el);
    term.scrollTop = term.scrollHeight;
  }

  function clearTerminal() {
    document.getElementById("terminal").innerHTML = "";
    shownSet.clear();
  }

  /* ── main loop (100 ms) ── */
  setInterval(() => {
    if (!player || typeof player.getCurrentTime !== "function") return;

    const t = player.getCurrentTime();

    // Update clock
    document.getElementById("clock-ts").textContent = fmtTs(t);
    document.getElementById("clock-frame").textContent = "frame " + Math.floor(t * 30);

    // Backward seek detected → reset terminal
    if (t < lastT - 2) clearTerminal();
    lastT = t;

    // Narrations
    for (let i = 0; i < narrations.length; i++) {
      const key = "n" + i;
      if (!shownSet.has(key) && narrations[i].start_timestamp <= t) {
        appendNarrationLine(narrations[i], i);
        shownSet.add(key);
      }
    }

    // Fixture events
    for (let i = 0; i < fixtures.length; i++) {
      const key = "f" + i;
      if (!shownSet.has(key) && fixtures[i].timestamp <= t) {
        appendFixtureLine(fixtures[i], i);
        shownSet.add(key);
      }
    }
  }, 100);
</script>

</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(
        TEMPLATE,
        video_id=_state["video_id"],
        yt_id=_state["yt_id"],
    )


@app.route("/api/narrations")
def api_narrations():
    return jsonify(_state["narrations"])


@app.route("/api/fixtures")
def api_fixtures():
    return jsonify(_state.get("fixtures", []))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_urls(path: str) -> dict:
    m = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m[row["video_id"]] = row["youtube_url"]
    return m


def extract_yt_id(url: str) -> str:
    url = url.strip()
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0].strip()
    return url


def load_fixture_events(fixtures_dir: str, video_id: str, noun_map: dict = None) -> list:
    """
    For each track in assoc_info.json, emit two events:
      - type="start"  at time_segment[0]  with the fixture of the chronologically
                      first mask  (where the object was before being moved)
      - type="end"    at time_segment[1]  with the fixture of the chronologically
                      last  mask  (where the object was placed)
    Returns a list sorted by timestamp.
    """
    import json

    assoc_path = os.path.join(fixtures_dir, "assoc_info.json")
    mask_path  = os.path.join(fixtures_dir, "mask_info.json")

    if not os.path.exists(assoc_path) or not os.path.exists(mask_path):
        return []

    with open(assoc_path) as f:
        assoc_data = json.load(f)
    with open(mask_path) as f:
        mask_data = json.load(f)

    if video_id not in assoc_data or video_id not in mask_data:
        return []

    video_masks  = mask_data[video_id]
    video_assocs = assoc_data[video_id]

    _nm = noun_map or {}
    events = []
    for assoc in video_assocs.values():
        raw_name = assoc.get("name", "object")
        name = _nm.get(raw_name, raw_name)
        for track in assoc.get("tracks", []):
            time_seg = track.get("time_segment", [])
            mask_ids = track.get("masks", [])
            if len(time_seg) < 2 or not mask_ids:
                continue

            # Resolve masks → (frame_number, fixture), sorted chronologically
            resolved = []
            for mid in mask_ids:
                m = video_masks.get(mid)
                if m:
                    resolved.append((m["frame_number"], m.get("fixture", "Null")))
            if not resolved:
                continue
            resolved.sort(key=lambda x: x[0])

            start_fixture = resolved[0][1]
            end_fixture   = resolved[-1][1]

            events.append({
                "timestamp": time_seg[0],
                "type":      "start",
                "name":      name,
                "fixture":   start_fixture,
            })
            events.append({
                "timestamp": time_seg[1],
                "type":      "end",
                "name":      name,
                "fixture":   end_fixture,
            })

    events.sort(key=lambda e: e["timestamp"])
    return events


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HD-EPIC web narration player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video-id", required=True, metavar="ID",
        help="e.g. P01-20240202-110250",
    )
    parser.add_argument(
        "--narrations",
        default=os.path.join(HERE, "narrations-and-action-segments", "HD_EPIC_Narrations.pkl"),
        metavar="PATH",
        help="Path to HD_EPIC_Narrations.pkl",
    )
    parser.add_argument(
        "--urls",
        default=os.path.join(HERE, "youtube-links", "HD_EPIC_YouTube_URLs.csv"),
        metavar="PATH",
        help="Path to HD_EPIC_YouTube_URLs.csv",
    )
    parser.add_argument(
        "--fixtures-dir",
        default=os.path.join(HERE, "scene-and-object-movements"),
        metavar="PATH",
        help="Directory containing assoc_info.json and mask_info.json",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Show all narrations instead of only appliance/fixture-related ones",
    )
    parser.add_argument(
        "--raw-tags", action="store_true",
        help="Show raw verb/noun strings from the narration instead of canonical class keys",
    )
    args = parser.parse_args()

    # ── validate video id ──
    url_map = load_urls(args.urls)
    if args.video_id not in url_map:
        sys.exit(f"Error: video_id '{args.video_id}' not found in {args.urls}")

    yt_url = url_map[args.video_id]
    yt_id  = extract_yt_id(yt_url)

    # ── load narrations ──
    if not os.path.exists(args.narrations):
        sys.exit(
            f"Error: narrations file not found:\n  {args.narrations}\n"
            "Download HD_EPIC_Narrations.pkl from the project webpage and place it in\n"
            "  narrations-and-action-segments/"
        )
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required.  Run:  pip install pandas")

    df  = pd.read_pickle(args.narrations)
    sub = df[df["video_id"] == args.video_id].sort_values("start_timestamp")
    total_count = len(sub)

    # ── optional noun filter ──
    if not args.no_filter:
        noun_csv = os.path.join(HERE, "narrations-and-action-segments", "HD_EPIC_noun_classes.csv")
        target_ids = get_target_noun_ids(noun_csv)
        if target_ids:
            sub = sub[sub["noun_classes"].apply(
                lambda nc: bool(nc) and any(x in target_ids for x in nc)
            )]

    # ── optional tag normalisation ──
    if not args.raw_tags:
        classes_dir = os.path.join(HERE, "narrations-and-action-segments")
        verb_map = build_instance_to_key_map(os.path.join(classes_dir, "HD_EPIC_verb_classes.csv"))
        noun_map = build_instance_to_key_map(os.path.join(classes_dir, "HD_EPIC_noun_classes.csv"))
    else:
        verb_map = noun_map = {}

    def _norm_pair(v, n):
        return [verb_map.get(v, v), noun_map.get(n, n)]

    sub = sub[["start_timestamp", "narration", "nouns", "verbs", "pairs"]]
    rows = []
    for _, r in sub.iterrows():
        raw_pairs = [list(p) for p in r["pairs"]] if r["pairs"] is not None else []
        rows.append({
            "start_timestamp": float(r["start_timestamp"]),
            "narration":       str(r["narration"]),
            "verbs":           list(r["verbs"]) if r["verbs"] is not None else [],
            "nouns":           list(r["nouns"]) if r["nouns"] is not None else [],
            "pairs":           [_norm_pair(v, n) for v, n in raw_pairs],
        })
    if not rows:
        sys.exit(f"No narrations found for '{args.video_id}'")

    fixture_events = load_fixture_events(args.fixtures_dir, args.video_id, noun_map)

    _state["video_id"]   = args.video_id
    _state["yt_id"]      = yt_id
    _state["narrations"] = rows
    _state["fixtures"]   = fixture_events

    filter_info = f"{len(rows)}/{total_count}" if not args.no_filter else f"{len(rows)} (unfiltered)"
    print(f"  Video:      {args.video_id}")
    print(f"  YouTube:    {yt_url}")
    print(f"  Narrations: {filter_info}")
    print(f"  Moves:      {len(fixture_events) // 2} tracks → {len(fixture_events)} fixture events")
    print(f"  UI:         http://localhost:{args.port}")
    print()

    def _open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{args.port}")

    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
