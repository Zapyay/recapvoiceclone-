#@title 🎬 Auto Movie Recap + Dubbing + Voice Clone (AssemblyAI + Gemini)
# ============ ဒီ ၂ ကြောင်းကို GitHub ပေါ်မှာ တစ်ခါပဲ ပြင်ပါ ============
SERVER_URL = "https://script.google.com/macros/s/AKfycbxN-nSr2CV6f2nsClYzrTYBzaCQzeCEfCETMvvqOmwqNdEmTLa5RcQiQHG1g-6zu0kn/exec"  # Apps Script Web App URL (Admin quota server) — ဗလာထားရင် ကန့်သတ်ချက်မရှိ (စမ်းသပ်ရန်သာ)
ID_SALT = "zapyartun-recap-2026-k9x"   # တစ်ခါသတ်မှတ်ပြီးရင် မပြောင်းပါနဲ့ (ပြောင်းရင် User ID အားလုံး ပြောင်းသွားမယ်)
# ====================================================================
# API key တွေကို Colab cell ထဲမှာ ကြိုသတ်မှတ်ထားရင် ယူသုံးမယ်၊ မရှိရင် Run တဲ့အခါ မေးမယ် (GitHub ပေါ်မှာ key မထည့်ပါနဲ့)
GEMINI_API_KEY = globals().get("GEMINI_API_KEY", "") or ""
ASSEMBLYAI_API_KEY = globals().get("ASSEMBLYAI_API_KEY", "") or ""
COOKIES_FILE = globals().get("COOKIES_FILE", "") or "/content/cookies.txt"

import os, re, sys, json, time, math, glob, shutil, subprocess, threading, uuid, traceback, getpass, unicodedata, tempfile, hashlib
from concurrent.futures import ThreadPoolExecutor
if not shutil.which("nvidia-smi"): raise SystemExit("❌ GPU မရှိပါ။ Runtime → Change runtime type → T4 GPU ရွေးပြီး ပြန် Run ပါ")
if not GEMINI_API_KEY: GEMINI_API_KEY = getpass.getpass("Gemini API key: ").strip()
if not ASSEMBLYAI_API_KEY: ASSEMBLYAI_API_KEY = getpass.getpass("AssemblyAI API key: ").strip()
import requests

# ---- User ID + ထုတ်ခွင့် (Admin server)
def srv(**p):
    if not SERVER_URL: return {"ok": True, "remaining": None}
    last = ""
    for _ in range(3):
        try: return requests.get(SERVER_URL, params=p, timeout=30).json()
        except Exception as e: last = str(e)[:100]; time.sleep(1.5)
    return {"ok": False, "msg": "Server ချိတ်မရပါ: " + last}

def make_uid():
    email = ""
    try:
        from google.colab import auth
        auth.authenticate_user()
        out = subprocess.run(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], capture_output=True, text=True).stdout.strip()
        email = (out.splitlines() or [""])[0].strip().lower()
        if "@" not in email:
            tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip()
            email = str(requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers={"Authorization": "Bearer " + tok}, timeout=20).json().get("email", "")).lower()
    except Exception as e:
        print("Google account ဖတ်မရပါ:", str(e)[:120])
    if "@" in email:
        return "U-" + hashlib.sha256(f"{ID_SALT}|{email}".encode()).hexdigest()[:8].upper()
    fp = "/content/.uid"
    try:
        if os.path.exists(fp): return open(fp).read().strip()
    except Exception: pass
    u = "U-" + uuid.uuid4().hex[:8].upper()
    try: open(fp, "w").write(u)
    except Exception: pass
    print("⚠️ Google account မရလို့ ID ကို ယာယီထုတ်ထားတယ် (Runtime ပြန်စရင် ပြောင်းနိုင်ပါတယ်)")
    return u

UID = make_uid() if SERVER_URL else "U-TEST0000"
_q = srv(action="check", id=UID)
print("🆔 User ID:", UID, "| ထုတ်ခွင့်ကျန်:", (_q.get("remaining") if (SERVER_URL and _q.get("ok")) else "ကန့်သတ်မထား"))
if SERVER_URL and not _q.get("ok"): print("⚠️", _q.get("msg"))

import importlib.metadata as md
pins = []
for n in ["numpy", "torch", "torchaudio", "torchvision", "scipy", "numba", "pandas", "scikit-learn"]:
    try: pins.append(f"{n}=={md.version(n)}")
    except Exception: pass
open("constraints.txt", "w").write("\n".join(pins))
print("🔧 Package တပ်ဆင်နေပါတယ်…")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "constraints.txt", "voxcpm", "soundfile", "fastapi", "uvicorn", "python-multipart", "google-genai", "requests"], capture_output=True, text=True)
if r.returncode: print(r.stderr[-2000:]); raise SystemExit("❌ Package တပ်ဆင်မှု မအောင်မြင်ပါ")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp[default]", "curl_cffi"], capture_output=True, text=True)
# မြန်မာ စာတန်းကို video ပေါ်ရိုက်ထည့်ဖို့ font
subprocess.run("apt-get install -y -qq fonts-noto-core >/dev/null 2>&1; fc-cache -f >/dev/null 2>&1", shell=True)
import numpy as np, soundfile as sf, uvicorn
from voxcpm import VoxCPM
from google import genai
from google.genai import types
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
print("⏳ VoxCPM2 load နေပါတယ်…")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
SR = getattr(model.tts_model, "sample_rate", 48000)
gc = genai.Client(api_key=GEMINI_API_KEY)
MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]
def discover_models():
    try:
        names = []
        for m in gc.models.list():
            n = (getattr(m, "name", "") or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if n.startswith("gemini") and "flash" in n and (not acts or "generateContent" in acts) and not any(x in n for x in ("image", "tts", "live", "audio", "embed", "exp", "robotics", "computer", "native", "thinking")):
                names.append(n)
        def key(n):
            v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return (-(float(v.group(1)) if v else 0), "lite" in n, "preview" in n, n)
        return sorted(set(names), key=key)[:8]
    except Exception as e:
        print("model list မရပါ:", str(e)[:120]); return []
_d = discover_models()
if _d: MODELS = _d + [m for m in ("gemini-flash-latest",) if m not in _d]
print("🤖 Gemini models:", MODELS[:5])
DEAD, COOL = set(), {}
AAI = "https://api.assemblyai.com/v2"
WORK, FPS = "/content/recap", 30
GAP = 8.0
MIN_SPEED = 0.4
LINE_MAX = 26
MAX_MIN = 15   # video အရှည်ဆုံး (မိနစ်)
# scene ခွဲပုံ — စကားတစ်ကြောင်းချင်း အချိန်မှန်အောင် scene တိုတိုခွဲမယ်
MERGE_SPAN, MERGE_MAX, MERGE_GAP = 3.5, 7.0, 1.0
os.makedirs(WORK, exist_ok=True)
JOBS, GPU = {}, threading.Lock()
DL = {}   # link ဒေါင်းတဲ့ အလုပ်တွေ

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise RuntimeError((r.stderr or "")[-400:])
    return r.stdout

def sh_prog(cmd, total, cb):
    ef = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:], stdout=subprocess.PIPE, stderr=ef, text=True)
    for line in p.stdout:
        if line.startswith("out_time_ms="):
            try: cb(min(.99, int(line.split("=")[1]) / 1e6 / max(1.0, total)))
            except Exception: pass
    p.wait(); ef.seek(0); err = ef.read().decode(errors="ignore"); ef.close()
    if p.returncode: raise RuntimeError(err[-400:])

def probe(p):
    j = json.loads(sh(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", p]))
    v = [s for s in j["streams"] if s["codec_type"] == "video"][0]
    w, h = int(v["width"]), int(v["height"])
    rot = int(v.get("tags", {}).get("rotate", 0) or 0)
    for sd in v.get("side_data_list", []):
        if "rotation" in sd: rot = int(sd["rotation"])
    if abs(rot) in (90, 270): w, h = h, w
    return float(j["format"]["duration"]), w, h

def mmss(x): return f"{int(x // 60):02d}:{int(x % 60):02d}"

def split(text, n=240):
    out, cur = [], ""
    for p in re.split(r"(?<=[။?!.\n])", text):
        if len(cur) + len(p) > n and cur: out.append(cur.strip()); cur = ""
        if len(p) > n: out += [p[i:i+n].strip() for i in range(0, len(p), n)]
        else: cur += p
    if cur.strip(): out.append(cur.strip())
    return [x for x in out if x]

def jparse(t): return json.loads(re.sub(r"^```(?:json)?|```$", "", t.strip()).strip())

TH = {}
def _think_variants():
    v = []
    try: v.append(types.ThinkingConfig(thinking_budget=0))
    except Exception: pass
    try: v.append(types.ThinkingConfig(thinking_level="low"))
    except Exception: pass
    v.append(None)
    return v
THV = _think_variants()

def gcall(contents, temp=0.3):
    last, errs = "", {}
    for rnd in range(4):
        tried = False
        for m in MODELS:
            if m in DEAD or COOL.get(m, 0) > time.time(): continue
            tried = True
            for vi in range(TH.get(m, 0), len(THV)):
                try:
                    kw = {"thinking_config": THV[vi]} if THV[vi] is not None else {}
                    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temp, **kw)
                    txt = gc.models.generate_content(model=m, contents=contents, config=cfg).text
                    TH[m] = vi
                    return txt
                except Exception as e:
                    s = str(e); last = f"{m}: {s[:250]}"; errs[m] = s[:110]
                    if any(x in s for x in ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED")):
                        raise RuntimeError("Gemini API key မှားနေ/ခွင့်မပြုပါ: " + s[:250])
                    if "404" in s or "NOT_FOUND" in s: DEAD.add(m); break
                    if "thinking" in s.lower() and vi < len(THV) - 1: continue
                    COOL[m] = time.time() + 30; break
        time.sleep(10 if not tried else 3)
    raise RuntimeError("Gemini ခေါ်လို့မရပါ — " + (" | ".join(f"{k}: {v}" for k, v in list(errs.items())[:4]) or last))

try:
    gcall(["Reply with JSON array [1]"])
    print("✅ Gemini OK:", [m for m in MODELS if m not in DEAD and COOL.get(m, 0) <= time.time()][:1])
except Exception as e:
    raise SystemExit("❌ " + str(e))

# ---- မြန်မာစာ helper
def clusters(t):
    out = []
    for i, ch in enumerate(t):
        nx = t[i + 1] if i + 1 < len(t) else ""
        if out and (unicodedata.category(ch).startswith("M") or out[-1].endswith("\u1039") or nx == "\u103a"): out[-1] += ch
        else: out.append(ch)
    return out

def fix_my(t):
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub("[\u200b\u200c\u200d\u2060\ufeff*_#`]", "", t)
    t = t.replace("\u1039\u1039", "\u1039").replace("\u103a\u103a", "\u103a")
    t = re.sub(r"([\u102b-\u103e])\1+", r"\1", re.sub(r"[ \t]+", " ", t))
    t = re.sub(r"\s+([၊။])", r"\1", t)
    return t.strip()

# ---- စာတန်း (cue တစ်ခု အများဆုံး ၂ ကြောင်း)
def break_text(cl, maxn):
    out, cur = [], list(cl)
    while cur and cur[0] == " ": cur.pop(0)
    while cur:
        if len(cur) <= maxn: out.append("".join(cur).strip()); break
        j = next((k for k in range(maxn - 1, maxn // 2 - 1, -1) if cur[k] in (" ", "၊", "။", ",", ".")), None)
        if j is None: j = maxn - 1
        out.append("".join(cur[:j + 1]).strip()); cur = cur[j + 1:]
        while cur and cur[0] == " ": cur.pop(0)
    return [x for x in out if x]

def make_cues(text, start, speech, lm=LINE_MAX):
    text = " ".join(fix_my(text).split())
    sents = [s.strip() for s in re.split(r"(?<=[။?!])", text) if s.strip()]
    maxc, groups, buf, bn = lm * 2, [], [], 0
    for s in sents:
        n = len(clusters(s))
        if buf and bn + n + 1 > maxc: groups.append(" ".join(buf)); buf, bn = [], 0
        buf.append(s); bn += n + 1
    if buf: groups.append(" ".join(buf))
    cues = []
    for g in groups:
        ls = break_text(clusters(g), lm)
        for i in range(0, len(ls), 2): cues.append(ls[i:i + 2])
    if not cues: return []
    w = [max(1, sum(len(clusters(x)) for x in c)) for c in cues]; tot = sum(w)
    res, t0 = [], start
    for c, x in zip(cues, w):
        t1 = t0 + speech * x / tot
        res.append((t0, t1, c)); t0 = t1
    return res

def all_cues(texts, starts, alens, durs, lm):
    out = []
    for t, st, L, dd in zip(texts, starts, alens, durs):
        if t: out += make_cues(t, st, min(L + 0.1, dd), lm)
    return out

def ass_color(h):
    h = (h or "#ffffff").lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h): h = "ffffff"
    return ("&H00" + h[4:6] + h[2:4] + h[0:2]).upper().replace("&H", "&H")

def ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000); m, cs = divmod(cs, 6000); s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def build_ass(cues, W, H, sub):
    fs = max(10, int(sub["size"] * H)); x = int(sub["x"] * W); y = int(sub["y"] * H)
    ol = max(1, round(fs / 18))
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 2\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Noto Sans Myanmar,%d,%s,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,%d,1,5,10,10,10,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n") % (W, H, fs, ass_color(sub.get("color")), ol)
    lines = [head]
    for a, b, ls in cues:
        lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{{\\an5\\pos({x},{y})}}" + "\\N".join(ls) + "\n")
    return "".join(lines)

# ---- Link ကနေ video ဒေါင်း (TikTok / YouTube / Facebook / Rednote …)
def tikwm_dl(url, base, prog):
    j = requests.get("https://www.tikwm.com/api/", params={"url": url, "hd": 1}, timeout=40, headers={"User-Agent": "Mozilla/5.0"}).json()
    dt = j.get("data") or {}
    pu = dt.get("hdplay") or dt.get("play")
    if not pu: raise RuntimeError(str(j.get("msg"))[:100])
    if pu.startswith("/"): pu = "https://www.tikwm.com" + pu
    path = base + ".mp4"
    with requests.get(pu, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        tb = int(r.headers.get("content-length", 0) or 0); got = 0
        with open(path, "wb") as f:
            for ch in r.iter_content(1 << 16):
                f.write(ch); got += len(ch)
                if tb: prog(min(.95, got / tb), f"{got / 1e6:.0f}/{tb / 1e6:.0f} MB")
    return path, (dt.get("title") or "tiktok").strip()[:100]

def fetch_video(url, jid, prog):
    import yt_dlp
    base = os.path.join(WORK, "dl_" + jid)
    def hook(dd):
        if dd.get("status") == "downloading":
            tb = dd.get("total_bytes") or dd.get("total_bytes_estimate") or 0
            got = dd.get("downloaded_bytes", 0) or 0
            if tb: prog(min(.95, got / tb), f"{got / 1e6:.0f}/{tb / 1e6:.0f} MB")
            else: prog(.5, f"{got / 1e6:.0f} MB")
    common = {"merge_output_format": "mp4", "outtmpl": base + ".%(ext)s", "noplaylist": True, "quiet": True, "no_warnings": True,
              "progress_hooks": [hook], "retries": 5, "fragment_retries": 5, "socket_timeout": 30, "concurrent_fragment_downloads": 4,
              "js_runtimes": {"node": {}}}
    if os.path.exists(COOKIES_FILE): common["cookiefile"] = COOKIES_FILE
    FMT = "bv*[height<=720]+ba/b[height<=720]/bv*+ba/b"
    variants = [{"format": FMT}]
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        variants.append({"format": FMT, "impersonate": ImpersonateTarget("chrome")})
    except Exception: pass
    variants += [{"format": FMT, "extractor_args": {"youtube": {"player_client": ["tv", "web_safari"]}}},
                 {"format": FMT, "extractor_args": {"youtube": {"player_client": ["android_vr", "ios"]}}},
                 {"format": "b[height<=720]/b"}]
    info, last, files = None, "", []
    for v in variants:
        for f in glob.glob(base + ".*"):
            try: os.remove(f)
            except Exception: pass
        try:
            with yt_dlp.YoutubeDL({**common, **v}) as y:
                info = y.extract_info(url, download=True)
            files = [f for f in glob.glob(base + ".*") if not f.endswith((".part", ".ytdl", ".json"))]
            if files: break
            info = None
        except Exception as e:
            last = re.sub(r"\x1b\[[0-9;]*m", "", str(e))[-300:]; info = None
    if not info and re.search(r"tiktok\.com|douyin\.com", url):
        try: return tikwm_dl(url, base, prog)
        except Exception as e: last += " | tikwm: " + str(e)[:100]
    if not info:
        raise RuntimeError("Link ကနေ video ဒေါင်းမရပါ (နည်းလမ်းအားလုံး စမ်းပြီး): " + last + "\n💡 YouTube/Facebook ဆိုရင် cookies.txt ကို /content/cookies.txt မှာ တင်ပြီး ပြန် Run ပါ၊ (သို့) video ကို ဖိုင်အဖြစ် တင်ပါ")
    ents = info.get("entries") if isinstance(info, dict) else None
    if ents: info = next((x for x in ents if x), info)
    files.sort(key=os.path.getsize, reverse=True)
    return files[0], (info.get("title") or "video").strip()

def dl_worker(did, url):
    d = DL[did]
    try:
        path, title = fetch_video(url, did, lambda f, s: d.update(f=f, d=s))
        d.update(state="done", path=path, title=title, f=1.0, d="")
    except Exception as e:
        d.update(state="err", msg=str(e)[-400:])

# ---- AssemblyAI: video ထဲက စကားကို စာထုတ် (Dubbing မှာ စကားပြောသူ ခွဲ)
def aai_transcribe(audio_path, lang, prog, diar=False):
    H = {"authorization": ASSEMBLYAI_API_KEY}
    prog(.1, "အသံ တင်နေ")
    with open(audio_path, "rb") as f:
        r = requests.post(f"{AAI}/upload", headers=H, data=f, timeout=900)
    if r.status_code == 401: raise RuntimeError("AssemblyAI API key မှားနေပါတယ်")
    r.raise_for_status()
    url = r.json()["upload_url"]
    base = {"audio_url": url}
    if lang and lang != "auto": base["language_code"] = lang
    else: base["language_detection"] = True
    if diar: base["speaker_labels"] = True
    tries = [base, {**base, "speech_models": ["universal-3-pro", "universal-2"]}, {**base, "speech_model": "best"}]
    err = ""
    for body in tries:
        r = requests.post(f"{AAI}/transcript", headers={**H, "content-type": "application/json"}, json=body, timeout=60)
        if r.status_code == 200: break
        err = r.text
    else: raise RuntimeError("AssemblyAI: " + err[:300])
    tid = r.json()["id"]
    t0 = time.time()
    while True:
        time.sleep(3)
        j = requests.get(f"{AAI}/transcript/{tid}", headers=H, timeout=60).json()
        if j.get("status") == "completed": break
        if j.get("status") == "error": raise RuntimeError("AssemblyAI: " + str(j.get("error"))[:300])
        prog(min(.9, .3 + (time.time() - t0) / 300), "စာထုတ်နေ")
    s = requests.get(f"{AAI}/transcript/{tid}/sentences", headers=H, timeout=60).json().get("sentences", [])
    order, out = {}, []
    for x in s:
        t = str(x.get("text", "")).strip()
        if not t: continue
        k = 0
        if diar:
            lab = x.get("speaker") or "A"
            if lab not in order: order[lab] = len(order)
            k = order[lab] % 2      # စကားပြောသူ ၁ → Voice1 ၊ ၂ → Voice2
        out.append((x["start"] / 1000.0, x["end"] / 1000.0, t, k))
    return out, j.get("language_code", "")

# ---- Scene ဖွဲ့
# အသံနဲ့ video ကိုက်အောင်: scene တစ်ခုချင်းစီက မူရင်းစကား စတဲ့အချိန်မှာပဲ စတယ်၊
# ဒါကြောင့် ဘာသာပြန်အသံ ဟာ အဲ့ဒီစကားပြောတဲ့ အပိုင်းမှာပဲ စပြောမယ် (ကြိုမပြောတော့ဘူး)
def build_scenes(sents, dur):
    sents = sorted([(max(0.0, min(a, dur)), max(0.0, min(b, dur)), t, k) for a, b, t, k in sents if t], key=lambda x: x[0])
    groups, cur = [], None
    for a, b, t, k in sents:
        if cur and cur[3] == k and (cur[1] - cur[0] < MERGE_SPAN) and (b - cur[0] <= MERGE_MAX) and (a - cur[1] < MERGE_GAP):
            cur[1] = max(cur[1], b); cur[2].append(t)
        else:
            if cur: groups.append(cur)
            cur = [a, b, [t], k]
    if cur: groups.append(cur)
    sc = lambda a, b, t, k=0, sd=0.0: {"a": a, "b": b, "src": t, "t": "", "spk": k, "sd": sd}
    out, pos = [], 0.0
    for gi, (s, e, tl, k) in enumerate(groups):
        if s - pos > 0.3: out.append(sc(pos, s, "")); pos = s
        nxt = groups[gi + 1][0] if gi + 1 < len(groups) else None
        txt, sd = " ".join(tl), max(0.5, e - s)
        if nxt is None:
            if dur - e > GAP: out.append(sc(pos, e + .5, txt, k, sd)); out.append(sc(e + .5, dur, "")); pos = dur
            else: out.append(sc(pos, dur, txt, k, sd)); pos = dur
        elif nxt - e > GAP:
            out.append(sc(pos, e + .5, txt, k, sd)); out.append(sc(e + .5, nxt, "")); pos = nxt
        else:
            out.append(sc(pos, nxt, txt, k, sd)); pos = nxt
    return [x for x in out if x["src"] or x["b"] - x["a"] > 0.3]

# ---- Gemini ဘာသာပြန်
TR = """သင်ဟာ မြန်မာစာ ကျွမ်းကျင်တဲ့ video dubbing ဘာသာပြန်သူပါ။ အောက်က JSON array က video ထဲက စကားပြောတွေ (မူရင်းဘာသာစကား ဘာဖြစ်ဖြစ်) ကို အစီအစဉ်အတိုင်း ပေးထားတာပါ။ တစ်ခုချင်းကို မြန်မာလို ဘာသာပြန်ပြီး အသံထွက်ဖတ်ဖို့ သင့်တော်အောင် ရေးပါ။
- string array တစ်ခုတည်း ပြန်ပါ။ အရေအတွက်၊ အစီအစဉ် အတိအကျ တူရမယ်။
- item တစ်ခုချင်းစီရဲ့ "chars" က ရေးရမယ့် မြန်မာစာလုံး (character) အရေအတွက် ပစ်မှတ်။ ±15% ထဲ ကိုက်ပါစေ၊ ကျော်လို့မရ (အသံကို မြန်မြန်ဖတ်ခိုင်းမှာ မဟုတ်လို့ ကျော်ရင် အချိန်ပိုကြာသွားမယ်)။ ပိုရှည်နေရင် အသေးစိတ်ကို ချုံ့ပါ၊ တိုနေရင် မူရင်းအဓိပ္ပာယ်ကို သဘာဝကျကျ ရှင်းပြပါ (မူရင်းမှာ မပါတာ မဖန်တီးနဲ့)။
- စာလုံးပေါင်း တိတိကျကျ မှန်ရမယ်။ စာလုံးအသစ်ဖန်တီးခြင်း၊ မသေချာတဲ့ ရှားပါးစကားလုံး မသုံးနဲ့။ ရိုးရိုး နေ့စဉ်ပြောတဲ့ စကားလုံးတွေသာ သုံးပါ။ ဝါကျတစ်ကြောင်းချင်း ပြည့်စုံပြီး သဒ္ဒါမှန်ရမယ်။
- Unicode မြန်မာစာသာ (Zawgyi မသုံးနဲ့)။ ဂဏန်း/English ကို မြန်မာစာလုံးနဲ့ ရေးပါ။ စာကြောင်းအဆုံးမှာ ။ ထည့်ပါ။ ဇာတ်ကောင်နာမည် တစ်ပုံစံတည်း ရေးပါ။
CTX
INPUT:
"""
RECAP_HINT = "- ဒီ video ကို ဇာတ်လမ်းအကျဉ်းပြောပြတဲ့ narrator တစ်ယောက်တည်း အသံနဲ့ ဖတ်မှာပါ။ ဇာတ်ကြောင်းပြောပြသလို တစ်ဆက်တည်း သဘာဝကျကျ ရေးပါ။ item တစ်ခုချင်းစီက video ရဲ့ အဲ့ဒီအချိန်မှာ ဖြစ်နေတာကိုသာ ပြောပါ (နောက်ပိုင်း ဖြစ်မယ့်အကြောင်း ကြိုမပြောနဲ့)။"
DUB_HINT = "- ဒီ video မှာ လူနှစ်ယောက် စကားပြောနေတာပါ ('spk' က စကားပြောသူ နံပါတ်)။ အပြန်အလှန် စကားပြောသလို စကားပြော ပုံစံနဲ့ ရေးပါ။"
SHORT = "အောက်က မြန်မာစာကို အဓိပ္ပာယ်မပျက်ဘဲ {N} စာလုံးခန့်အထိ ချုံ့ပါ။ ရိုးရိုးစကားလုံး၊ စာလုံးပေါင်းမှန်၊ Unicode၊ အဆုံးမှာ ။ ထည့်ပါ။ string တစ်ခုပါတဲ့ JSON array ပြန်ပါ။\n"

def as_list(data):
    if isinstance(data, dict): data = next((v for v in data.values() if isinstance(v, list)), [])
    return [(x.get("text", "") if isinstance(x, dict) else str(x)) for x in data]

def translate_batch(items, hint):
    p = TR.replace("CTX", hint) + json.dumps(items, ensure_ascii=False)
    for _ in range(2):
        try:
            out = as_list(jparse(gcall([p])))
            if len(out) == len(items) and all(x.strip() for x in out): return [fix_my(x) for x in out]
        except Exception as e:
            if "Gemini ခေါ်လို့မရပါ" in str(e) or "API key" in str(e): raise
    res = []
    for it in items:
        q = TR.replace("CTX", hint) + json.dumps([it], ensure_ascii=False)
        try: o = as_list(jparse(gcall([q])))
        except Exception: o = []
        res.append(fix_my(o[0]) if o and o[0].strip() else "")
    return res

def shorten(text, n):
    o = as_list(jparse(gcall([SHORT.replace("{N}", str(int(n))) + json.dumps([text], ensure_ascii=False)], temp=0.2)))
    return fix_my(o[0]) if o and o[0].strip() else text

# ---- Job state
WT = [2, 1, 8, 3, 10, 32, 30, 14, 0]
def pct(j):
    if j["done"]: return 100
    p = 0.0
    for w, x in zip(WT, j["steps"]):
        if x["s"] == "done": p += w
        elif x["s"] == "run": p += w * (x.get("f") or 0)
    return min(99, int(round(p)))

def P(job, i, state, d="", f=0.0):
    job["steps"][i] = {"s": state, "d": d, "f": f}; job["upd"] = time.time()

def refund_job(job):
    t = job.pop("ticket", None)
    if t: srv(action="refund", id=UID, ticket=t)

CAL = "မင်္ဂလာပါ။ ဒီနေ့ ရာသီဥတု အရမ်းကောင်းပါတယ်။ ကျွန်တော်တို့ အတူတူ ဇာတ်ကားကြည့်ကြရအောင်နော်။"

def tempo(au, spd, d=None):
    # အသံမြန်နှုန်း ချိန်ညှိ (အသံအနိမ့်အမြင့် မပြောင်းဘဲ)
    if abs(spd - 1.0) < 0.02 or len(au) < 200: return au
    base = os.path.join(d or WORK, "t_" + uuid.uuid4().hex[:8])
    try:
        sf.write(base + "_i.wav", au, SR)
        sh(["ffmpeg", "-y", "-i", base + "_i.wav", "-filter:a", f"atempo={max(0.5, min(2.0, spd)):.3f}", base + "_o.wav"])
        o, _ = sf.read(base + "_o.wav", dtype="float32")
        return (o if o.ndim == 1 else o.mean(axis=1)).astype(np.float32)
    finally:
        for z in ("_i.wav", "_o.wav"):
            try: os.remove(base + z)
            except Exception: pass

def synth(t, ref, steps=6, spd=1.0, d=None):
    ws = []
    for piece in split(t):
        w = model.generate(text=piece, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=steps)
        ws += [np.asarray(w, dtype=np.float32).reshape(-1), np.zeros(int(SR * 0.08), np.float32)]
    au = np.concatenate(ws) if ws else np.zeros(int(SR * 0.2), np.float32)
    return tempo(au, spd, d)

def run_job(jid, vid, dlid, voice, voice2, lang, steps, mode, style, spd=1.0):
    job = JOBS[jid]
    d = os.path.join(WORK, jid); os.makedirs(d, exist_ok=True)
    src = os.path.join(WORK, "up_" + vid) if vid else None
    vsrc = os.path.join(WORK, "up_" + voice)
    vsrc2 = os.path.join(WORK, "up_" + voice2) if voice2 else None
    try:
        try: stl = json.loads(style)
        except Exception: stl = {}
        sub = {"on": True, "x": .5, "y": .88, "size": .055, "color": "#ffffff", **(stl.get("sub") or {})}
        bl = {"on": False, "x": .1, "y": .8, "w": .8, "h": .16, "s": 12, **(stl.get("bl") or {})}
        cl = lambda v, a, b: max(a, min(b, float(v)))
        bgb = bool(stl.get("bgb", True))                       # Blur နောက်ခံ
        size = stl.get("size", "orig")
        if size not in ("orig", "9:16", "3:4", "16:9", "1:1"): size = "orig"
        Q = 1080 if int(stl.get("q", 1080) or 1080) == 1080 else 720
        cr = {"on": False, "x": 0, "y": 0, "w": 1, "h": 1, **(stl.get("crop") or {})}

        # 1) Link ဒေါင်းပြီးသား / upload ပြီးသား ဖိုင်
        if dlid:
            P(job, 0, "run", "Link ဒေါင်းနေ", .02)
            while True:
                dd = DL.get(dlid)
                if not dd: raise RuntimeError("ဒေါင်းထားတဲ့ video မတွေ့ပါ")
                if dd["state"] == "err": raise RuntimeError(dd["msg"])
                if dd["state"] == "done": break
                P(job, 0, "run", dd.get("d", ""), dd.get("f", 0)); time.sleep(1)
            src = dd["path"]; job["name"] = (dd.get("title") or "video")[:120]
        P(job, 0, "done", "", 1); P(job, 1, "run")
        dur, sw, sh_ = probe(src)
        if dur < 3: raise RuntimeError("Video တိုလွန်းပါတယ် (၃ စက္ကန့်အထက် လိုပါတယ်)")
        if dur > MAX_MIN * 60 + 5: raise RuntimeError(f"Video က {MAX_MIN} မိနစ်ထက် ပိုရှည်နေပါတယ် ({dur/60:.1f} မိနစ်) — တိုတိုဖြတ်ပြီး ထပ်ထည့်ပါ")
        # Crop (မူရင်း pixel၊ ဂဏန်းစုံကိုပဲ သုံး)
        crop_on = bool(cr.get("on"))
        if crop_on:
            cx = int(cl(cr["x"], 0, .95) * sw) // 2 * 2; cy = int(cl(cr["y"], 0, .95) * sh_) // 2 * 2
            cw = max(16, int(cl(cr["w"], .05, 1) * sw) // 2 * 2); ch_ = max(16, int(cl(cr["h"], .05, 1) * sh_) // 2 * 2)
            cw = min(cw, (sw - cx) // 2 * 2); ch_ = min(ch_, (sh_ - cy) // 2 * 2)
        else: cx = cy = 0; cw, ch_ = sw, sh_
        # ထွက်မယ့် Size (အတိုဘက် = Quality)
        ar = (cw / ch_) if size == "orig" else (int(size.split(":")[0]) / int(size.split(":")[1]))
        if ar >= 1: H, W = Q, round(Q * ar)
        else: W, H = Q, round(Q / ar)
        W, H = W // 2 * 2, H // 2 * 2
        need_bg = bgb and abs(cw / ch_ - W / H) > 0.01       # Size မကိုက်ရင် Blur နောက်ခံ ဖြည့်
        bw, bh = max(2, (W // 10) // 2 * 2), max(2, (H // 10) // 2 * 2)
        crop_f = f"crop={cw}:{ch_}:{cx}:{cy}," if crop_on else ""
        P(job, 1, "done", f"{dur/60:.1f} မိနစ်" if dur >= 60 else f"{dur:.0f} စက္ကန့်", 1)

        # 2) AssemblyAI
        P(job, 2, "run", "အသံထုတ်နေ", .05)
        aud = os.path.join(d, "audio.mp3")
        try: sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", aud])
        except RuntimeError: raise RuntimeError("Video ထဲမှာ အသံ မတွေ့ပါ")
        sents, lc = aai_transcribe(aud, lang, lambda f, s: P(job, 2, "run", s, f), diar=(mode == "dubbing"))
        if not sents: raise RuntimeError("Video ထဲမှာ စကားပြောသံ မတွေ့ပါ (သို့) AssemblyAI က မဖတ်နိုင်ပါ")
        scenes = build_scenes(sents, dur)
        nspk = len({x["spk"] for x in scenes if x["src"]})
        P(job, 2, "done", f"{len(sents)} ကြောင်း ({lc})" + (f" · {nspk} ယောက်" if mode == "dubbing" else ""), 1)

        # 3) နမူနာအသံ
        P(job, 3, "run", "တန်းစီစောင့်နေ", 0)
        refs = []
        for k, vs in enumerate([vsrc, vsrc2]):
            rp = os.path.join(d, f"ref{k}.wav")
            if vs and os.path.exists(vs): sh(["ffmpeg", "-y", "-i", vs, "-vn", "-t", "30", "-ac", "1", "-ar", "44100", rp])
            else: rp = refs[0]      # Voice 2 မထည့်ရင် Voice 1 နဲ့ပဲ ပြောမယ်
            refs.append(rp)
        with GPU:
            P(job, 3, "run", "", .4)
            w = synth(CAL, refs[0], steps, spd, d)
        cps = max(6.0, min(20.0, len(CAL) / max(0.5, len(w) / SR))) * 0.8
        P(job, 3, "done", f"{cps:.1f} လုံး/စက္ကန့်", 1)

        # 4) ဘာသာပြန်
        P(job, 4, "run", "", 0)
        sp = [x for x in scenes if x["src"]]
        hint = RECAP_HINT if mode == "recap" else DUB_HINT
        def tr_chunk(chunk):
            items = []
            for x in chunk:
                it = {"text": x["src"], "chars": int(max(6, x["sd"] * cps))}
                if mode == "dubbing": it["spk"] = x["spk"] + 1
                items.append(it)
            for x, t in zip(chunk, translate_batch(items, hint)): x["t"] = t
        chunks = [sp[i:i + 15] for i in range(0, len(sp), 15)]
        with ThreadPoolExecutor(max_workers=3) as ex2:
            f2 = [ex2.submit(tr_chunk, c) for c in chunks]
            for k, fu in enumerate(f2):
                fu.result(); P(job, 4, "run", f"{k+1}/{len(chunks)}", (k + 1) / len(chunks))
        for x in scenes:
            if x["src"] and not x["t"]: x["t"] = fix_my(x["src"])
        N = len(scenes)
        P(job, 4, "done", f"{N} scene", 1)

        # 5) အသံထုတ် (GPU)  +  6) video ဖြတ် (CPU)
        P(job, 5, "run", "တန်းစီစောင့်နေ", 0); P(job, 6, "run", f"0/{N}", 0)
        segs, full, tot, cnt, lk = [None] * N, [None] * N, [0.0] * N, {"n": 0}, threading.Lock()
        texts, alens = [""] * N, [0.0] * N
        def enc(i, a, b, t, au):
            T = b - a
            if t: dd = max(T, len(au) / SR + 0.15); s = max(MIN_SPEED, min(1.0, T / dd))
            else: dd, s = T, 1.0
            n = max(FPS, math.ceil(dd * FPS)); dd = n / FPS
            need = max(0.3, min(dd * s, T, dur - a))
            sp_ = os.path.join(d, f"seg{i}.mp4")
            pre = f"{crop_f}setpts=(PTS-STARTPTS)/{s:.5f},fps={FPS},tpad=stop_mode=clone:stop_duration=30,setsar=1"
            fit = f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2"
            if need_bg:
                bgf = f"scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},gblur=sigma=2,scale={W}:{H}"
                fc = f"[0:v]{pre},split[a][b];[b]{bgf}[bg];[a]{fit}[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
            else:
                fc = f"[0:v]{pre},{fit},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[v]"
            cmd = ["nice", "-n", "19", "ffmpeg", "-y", "-ss", f"{min(a, dur - 0.5):.3f}", "-t", f"{need:.3f}", "-i", src, "-an", "-filter_complex", fc, "-map", "[v]"]
            sh(cmd + ["-frames:v", str(n), "-c:v", "libx264", "-preset", "superfast", "-crf", "24", "-threads", "1", "-r", str(FPS), sp_])
            Tn = int(round(n * SR / FPS)); full[i] = np.pad(au, (0, max(0, Tn - len(au))))[:Tn]; segs[i] = sp_; tot[i] = dd
            with lk:
                cnt["n"] += 1; P(job, 6, "run", f"{cnt['n']}/{N}", cnt["n"] / N)
        ex = ThreadPoolExecutor(max_workers=2); futs = []
        with GPU:
            t_loop = time.time()
            for i, x in enumerate(scenes):
                for fu in futs:
                    if fu.done() and fu.exception(): raise fu.exception()
                eta = (time.time() - t_loop) / i * (N - i) if i >= 2 else 0
                P(job, 5, "run", f"{i}/{N}" + (f" · ~{int(eta // 60) + 1} မိနစ်ကျန်" if eta else ""), i / N)
                a, b, t = x["a"], x["b"], x["t"]
                ref = refs[x.get("spk", 0)] if mode == "dubbing" else refs[0]
                if t:
                    T = b - a
                    au = synth(t, ref, steps, spd, d)
                    L = len(au) / SR
                    if L > T * 1.5 and len(t) > 25:
                        try:
                            t2 = shorten(t, len(t) * T * 1.2 / L)
                            au2 = synth(t2, ref, steps, spd, d)
                            if len(au2) < len(au): au, t = au2, t2
                        except Exception: pass
                else:
                    au = np.zeros(int(SR * (b - a)), np.float32)
                texts[i], alens[i] = t, len(au) / SR
                futs.append(ex.submit(enc, i, a, b, t, au))
        P(job, 5, "done", f"{N}/{N}", 1)
        for fu in futs: fu.result()
        ex.shutdown(); P(job, 6, "done", f"{N}/{N}", 1)

        # 7) ပေါင်း + Blur + မြန်မာစာတန်း ရိုက်ထည့် → MP4 တစ်ခုတည်း
        P(job, 7, "run", "ပေါင်းနေ", .02)
        lst = os.path.join(d, "list.txt"); open(lst, "w").write("".join(f"file '{p}'\n" for p in segs))
        vv, wv, out = os.path.join(d, "video.mp4"), os.path.join(d, "audio.wav"), os.path.join(d, "recap.mp4")
        sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", vv])
        sf.write(wv, np.concatenate(full), SR)
        starts, acc = [], 0.0
        for x in tot: starts.append(acc); acc += x
        total = sum(tot)
        fs = max(10, int(sub["size"] * H))
        lm = int(max(10, min(40, (W * 0.92) / (fs * 0.65))))
        cues = all_cues(texts, starts, alens, tot, lm) if sub.get("on", True) else []
        ap = os.path.join(d, "sub.ass")
        open(ap, "w", encoding="utf-8").write(build_ass(cues, W, H, sub))
        bx = int(max(0, min(1, bl["x"])) * W); by = int(max(0, min(1, bl["y"])) * H)
        bw = max(2, min(W - bx, int(bl["w"] * W))); bh = max(2, min(H - by, int(bl["h"] * H)))
        sig = max(1.0, float(bl["s"]) * W / 720)
        def final_cmd(use_sub):
            chain, last = [], "[0:v]"
            if bl.get("on"):
                chain.append(f"{last}split[a][b];[b]crop={bw}:{bh}:{bx}:{by},gblur=sigma={sig:.1f}[c];[a][c]overlay={bx}:{by}[v1]"); last = "[v1]"
            if use_sub and cues:
                chain.append(f"{last}ass={ap}[vo]"); last = "[vo]"
            if chain: vc = ["-filter_complex", ";".join(chain), "-map", last, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"]
            else: vc = ["-map", "0:v", "-c:v", "copy"]
            return ["ffmpeg", "-y", "-i", vv, "-i", wv] + vc + ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
        cb = lambda f: P(job, 7, "run", "စာတန်း + Blur ထည့်နေ", .05 + .93 * f)
        note = ""
        try: sh_prog(final_cmd(True), total, cb)
        except RuntimeError as e:
            print("စာတန်းထည့်မရ၊ စာတန်းမပါဘဲ ထပ်စမ်းမယ်:", str(e)[-300:])
            sh_prog(final_cmd(False), total, cb); note = " (စာတန်းမထည့်နိုင်)"
        for p in segs + [vv, wv]:
            try: os.remove(p)
            except Exception: pass
        P(job, 7, "done", f"{mmss(total)} / မူရင်း {mmss(dur)}" + note, 1); P(job, 8, "done", "", 1); job["out"] = out; job["done"] = True
    except Exception as e:
        traceback.print_exc(); job["err"] = str(e)[-500:]
        for x in job["steps"]:
            if x["s"] == "run": x["s"] = "err"
        refund_job(job)
    finally:
        job["fin"] = time.time(); job["upd"] = time.time()
        for p in (src, vsrc, vsrc2):
            if not p: continue
            try: os.remove(p)
            except Exception: pass

# ---- web page
HTML = r'''<!DOCTYPE html><html lang="my"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Auto Movie Recap</title>
<style>
:root{color-scheme:light;--r:#e11d2e}*{box-sizing:border-box}
body{margin:0;background:#fff;color:#000;font-family:system-ui,"Noto Sans Myanmar","Padauk",sans-serif;line-height:1.7}
main{max-width:720px;margin:0 auto;padding:16px 14px 60px}h1{font-size:1.5rem;margin:6px 0 12px}
.card{border:1px solid #e6e6e6;border-radius:14px;padding:16px;margin-bottom:12px}h2{font-size:1.05rem;margin:0 0 10px}h3{font-size:.95rem;margin:14px 0 6px}
input[type=file],select,input[type=text]{width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;font:inherit;color:#000;background:#fff}
video,audio{width:100%;margin-top:10px}video{background:#000;border-radius:8px}
.btn{display:block;text-align:center;text-decoration:none;color:#fff;background:var(--r);border:0;border-radius:10px;padding:13px;width:100%;font:inherit;font-weight:700;cursor:pointer;margin-top:10px}
.btn.ol{background:#fff;color:#000;border:1px solid var(--r)}.btn:disabled{opacity:.4}.row{display:flex;gap:8px;align-items:center}
.note{font-size:.82rem;color:#000;margin-top:6px;opacity:.75}ul{list-style:none;padding:0;margin:0}
li{padding:8px;display:flex;gap:10px;align-items:center;border-bottom:1px dashed #e6e6e6;border-radius:8px;color:#777}.ic{width:24px;text-align:center}
.sp{display:inline-block;width:20px;height:20px;border:3px solid #e3e3e3;border-top-color:var(--r);border-radius:50%;animation:r .7s linear infinite;flex:none}
@keyframes r{to{transform:rotate(360deg)}}li.done{color:#000}li.done .ic{color:#1a9d5c}li.err{color:var(--r)}
li.run{background:#fff1f2;color:#000;font-weight:600}
.d{margin-left:auto;font-size:.82rem;text-align:right}
.live{display:flex;align-items:center;gap:14px;margin-bottom:8px}.big{width:46px;height:46px;border:5px solid #eee;border-top-color:var(--r);border-radius:50%;animation:r .8s linear infinite;flex:none}
.big.fin{animation:none;border-color:#1a9d5c;display:flex;align-items:center;justify-content:center;color:#1a9d5c;font-size:1.4rem}.big.bad{animation:none;border-color:var(--r);color:var(--r);display:flex;align-items:center;justify-content:center;font-size:1.4rem}
#pct{font-size:1.9rem;font-weight:800;line-height:1.1}#el{margin-left:auto;font-size:1rem;font-weight:700}
.bar{height:14px;background:#eee;border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;width:0;border-radius:10px;transition:width .6s;background:var(--r)}
.hide{display:none!important}pre{white-space:pre-wrap;font-size:.8rem;color:var(--r)}
.opt{display:flex;gap:10px;align-items:flex-start;border:1px solid #ddd;border-radius:12px;padding:12px;margin:8px 0;cursor:pointer}.opt input{margin-top:6px;accent-color:var(--r)}
.ctl{display:flex;align-items:center;gap:10px;margin-top:10px}.ctl label{min-width:92px;font-size:.9rem}.ctl input[type=range]{flex:1;accent-color:var(--r)}.ctl input[type=color]{width:56px;height:36px;border:1px solid #ccc;border-radius:8px;padding:2px;background:#fff}
.top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.top .id{font-weight:800;letter-spacing:.5px;user-select:all}
.mini{border:1px solid var(--r);background:#fff;border-radius:8px;padding:2px 10px;font:inherit;cursor:pointer}
.hi{border:1px solid #e6e6e6;border-radius:12px;padding:12px;margin:8px 0}.hi b{word-break:break-all}.hi .bar{height:8px;margin-top:6px}
.drop{display:block;text-align:center;border:2px dashed #f1a1a8;background:#fff5f6;border-radius:16px;padding:24px 12px;cursor:pointer}
.drop .ico{width:48px;height:48px;margin:0 auto 8px;background:#fff;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.08);display:flex;align-items:center;justify-content:center;font-size:1.4rem;color:var(--r)}
.or{display:flex;align-items:center;gap:10px;color:#666;font-size:.85rem;margin:12px 0}.or:before,.or:after{content:"";flex:1;height:1px;background:#e6e6e6}
.lrow{display:flex;gap:8px}.lbox{flex:1;display:flex;align-items:center;border:1px solid #ddd;border-radius:12px;padding:3px 3px 3px 10px;min-width:0}
.lbox input{border:0;padding:8px 4px;min-width:0;flex:1;outline:0;font:inherit;background:#fff;color:#000}
.pst{border:0;background:#fdecee;color:var(--r);font-weight:700;border-radius:9px;padding:8px 10px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}
.dlb{border:0;background:#1b1b2b;color:#fff;font-weight:700;border-radius:12px;padding:0 14px;font:inherit;font-weight:700;cursor:pointer;white-space:nowrap}
.tabs3{display:flex;border-bottom:1px solid #e6e6e6;margin:14px 0 12px}
.tabs3 button{flex:1;background:none;border:0;border-bottom:3px solid transparent;padding:12px 4px;font:inherit;font-weight:700;color:#666;cursor:pointer}
.tabs3 button.on{color:#000;border-bottom-color:#3b8bff}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{border:1px solid #ccc;border-radius:10px;padding:8px 14px;cursor:pointer;font-weight:600}.chip input{display:none}.chip.on{background:var(--r);border-color:var(--r);color:#fff}
.sw{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 0;border-bottom:1px solid #eee;font-weight:600}
.sw input{appearance:none;-webkit-appearance:none;width:46px;height:26px;border-radius:20px;background:#ccc;position:relative;flex:none;cursor:pointer;transition:.2s;margin:0}
.sw input:before{content:"";position:absolute;left:3px;top:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:.2s}
.sw input:checked{background:var(--r)}.sw input:checked:before{left:23px}
.pane{padding:6px 0 4px}
.stg{position:relative;line-height:0;user-select:none;-webkit-user-select:none;overflow:hidden;border-radius:8px;background:#000;width:100%;max-width:420px;margin:10px auto 0;aspect-ratio:16/9}
.stg canvas{position:absolute;left:0;top:0;width:100%;height:100%}
#bz,#cz{position:absolute;touch-action:none;cursor:move}
#bz{border:2px dashed var(--r)}
#cz{border:2px solid #fff;box-shadow:0 0 0 999px rgba(0,0,0,.55)}
.hd{position:absolute;width:30px;height:30px;background:var(--r);touch-action:none}
#sb{position:absolute;transform:translate(-50%,-50%);text-align:center;white-space:pre;font-weight:700;line-height:1.5;font-family:"Noto Sans Myanmar","Padauk",system-ui,sans-serif;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000,0 2px 4px #000;touch-action:none;cursor:move;padding:0 6px;outline:1px dashed rgba(255,255,255,.8)}
</style></head><body><main>
<h1>🎬 Auto Movie Recap</h1>
<div class="card"><div class="top"><span>🆔 User ID:</span><span class="id" id="uid">…</span><button class="mini" id="cp">📋 Copy</button></div>
<div class="note" style="opacity:1">🎞 ထုတ်ခွင့်ကျန်: <span id="rem">…</span></div>
<div class="row"><button class="btn ol" id="hbtn">📜 History</button><button class="btn ol" id="newbtn">➕ Video အသစ်</button></div></div>

<div id="v-main">
<div class="card">
<label class="drop" for="vf"><div class="ico">⬆</div><b id="dropt">Video ဖိုင် ရွေးပါ</b><div class="note">MP4, MOV, MKV · ၁၅ မိနစ်အထိ</div></label>
<input id="vf" type="file" accept="video/*" class="hide">
<div class="or">သို့မဟုတ် Link ထည့်ပါ</div>
<div class="lrow"><div class="lbox"><input id="lk" type="text" placeholder="TikTok / YouTube link ထည့်ပါ" autocomplete="off"><button class="pst" id="pst">📋 Paste</button></div><button class="dlb" id="dlb">🔗 ဒေါင်းမည်</button></div>
<div id="dlw" class="hide" style="margin-top:10px"><div class="bar"><i id="dpb"></i></div><div class="note" id="dtx"></div><pre id="der"></pre></div>
<details style="margin-top:8px"><summary class="note">🍪 cookies.txt (YouTube/Facebook က ပိတ်ရင်သာ ထည့်ပါ)</summary><input id="ck" type="file" accept=".txt,text/plain"></details>
<video id="pv" controls playsinline class="hide"></video>
<div class="note" id="vmsg"></div>
</div>

<div class="tabs3"><button data-t="1" class="on">ဗီဒီယို</button><button data-t="2">စာတန်း</button><button data-t="3">အသံ</button></div>

<div class="card" id="t1">
<h3 style="margin-top:0">Model</h3>
<label class="opt"><input type="radio" name="md" value="recap" checked><div><b>🎙 Recap</b><div class="note">ဇာတ်လမ်းတစ်ပုဒ်လုံးကို လူတစ်ယောက်တည်း နောက်ခံအသံနဲ့ ပြောပြသလို ထုတ်မယ်။ (အသံ ၁ ခု)</div></div></label>
<label class="opt"><input type="radio" name="md" value="dubbing"><div><b>🎭 Dubbing</b><div class="note">လူနှစ်ယောက် စကားပြောတာကို ခွဲပြီး ထုတ်မယ်။ (အသံ ၂ ခု ထည့်လို့ရ)</div></div></label>
<h3>Video Size</h3>
<div class="chips"><label class="chip on"><input type="radio" name="sz" value="orig" checked>မူရင်း</label><label class="chip"><input type="radio" name="sz" value="9:16">9:16</label><label class="chip"><input type="radio" name="sz" value="3:4">3:4</label><label class="chip"><input type="radio" name="sz" value="16:9">16:9</label><label class="chip"><input type="radio" name="sz" value="1:1">1:1</label></div>
<h3>Video Quality</h3>
<div class="chips"><label class="chip on"><input type="radio" name="ql" value="1080" checked>HD 1080P</label><label class="chip"><input type="radio" name="ql" value="720">720P</label></div>
<div class="note" id="osz"></div>
<h3>🌐 Video ထဲက စကားပြောဘာသာ</h3><select id="lang"><option value="auto" selected>အလိုအလျောက် သိမယ် (Auto)</option><option value="en">English</option><option value="zh">Chinese</option><option value="ko">Korean</option><option value="ja">Japanese</option><option value="th">Thai</option><option value="hi">Hindi</option><option value="es">Spanish</option><option value="fr">French</option></select>
<div class="sw" style="margin-top:12px"><span>✂ Crop (Video ကို လိုသလောက် ဖြတ်မယ်)</span><input type="checkbox" id="t-cr"></div>
<div id="cpane" class="pane hide"><div class="stg" id="cstage"><canvas id="cv2"></canvas><div id="cz"><div class="hd" id="ch1" style="left:-2px;top:-2px;border-radius:0 0 8px 0"></div><div class="hd" id="ch2" style="right:-2px;bottom:-2px;border-radius:8px 0 0 0"></div></div></div>
<div class="note">အဖြူစတုရန်းကို ဆွဲရွှေ့ပါ။ ထောင့်နှစ်ခု (အနီ) ကို ဆွဲပြီး ဖြတ်မယ့်နေရာ အရွယ်ပြောင်းပါ။ ဖြတ်ထားတဲ့အပိုင်းကို ရွေးထားတဲ့ Size ထဲ ထည့်ပေးမယ်။</div></div>
</div>

<div class="card hide" id="t2">
<div class="stg" id="stage"><canvas id="cv"></canvas><div id="bz"><div class="hd" id="bh" style="right:-2px;bottom:-2px;border-radius:8px 0 0 0"></div></div><div id="sb">မြန်မာစာတန်း နမူနာ
ဒီနေရာမှာ ပေါ်ပါမယ်</div></div>
<button class="btn ol" id="pp">▶ / ⏸ Video ဖွင့်ကြည့်</button>
<div class="note">စာတန်းကို လက်နဲ့ ဆွဲရွှေ့ပါ။ Blur ဘောက်စ်ကို ဆွဲရွှေ့ပါ၊ အောက်ညာထောင့် အနီစတုရန်းကို ဆွဲပြီး အရွယ်စား ပြောင်းပါ။</div>
<div class="sw" style="margin-top:8px"><span>စာတန်း / အရောင် ချိန်ညှိ</span><input type="checkbox" id="t-sub" checked></div>
<div id="subctl" class="pane"><div class="ctl"><label>🎨 စာအရောင်</label><input type="color" id="sc" value="#ffffff"></div><div class="ctl"><label>🔠 စာအရွယ်</label><input type="range" id="ss" min="20" max="120" value="55"></div></div>
<div class="sw"><span>Blur ချိန်ညှိ</span><input type="checkbox" id="t-bl"></div>
<div id="blctl" class="pane hide"><div class="ctl"><label>💧 Blur အားစိုက်</label><input type="range" id="bs" min="2" max="40" value="12"></div></div>
<div class="sw"><span>Blur နောက်ခံ</span><input type="checkbox" id="t-bg" checked></div>
<div class="note">Blur နောက်ခံ = ရွေးထားတဲ့ Size နဲ့ Video မကိုက်လို့ ကျန်တဲ့နေရာကို Video ကိုယ်တိုင်ကို blur လုပ်ပြီး ဖြည့်ပေးတာ။ ပိတ်ထားရင် အနက်ရောင်ပဲ ဖြည့်မယ်။ စာတန်း/Blur မလိုရင် အပေါ်က ခလုတ်တွေကို ပိတ်ထားပါ။</div>
</div>

<div class="card hide" id="t3">
<h3 style="margin-top:0" id="v1t">🎤 အသံနမူနာ ထည့်ပါ</h3><input id="af" type="file" accept="audio/*,video/*"><audio id="ap" controls class="hide"></audio>
<div id="v2w" class="hide"><h3>🎤 Voice 2 (ဒုတိယလူအတွက် — မထည့်လည်းရ)</h3><input id="af2" type="file" accept="audio/*,video/*"><audio id="ap2" controls class="hide"></audio>
<div class="note">Voice 2 မထည့်ရင် နှစ်ယောက်လုံး Voice 1 နဲ့ပဲ ပြောပါမယ်။</div></div>
<div id="spw" class="hide"><h3>⏱ အနှေး အမြန် ချိန်ညှိ</h3>
<div class="ctl"><span>🐢</span><input type="range" id="sp" min="0.6" max="1.5" step="0.05" value="1"><span>🐇</span></div>
<div class="note">အသံမြန်နှုန်း: <b id="spv">1.00x</b> (1.00x = ပုံမှန်)။ မြန်လေ အသံတိုလေ၊ နှေးရင် အသံရှည်လို့ Video ကို နှေးပေးရနိုင်တယ်။</div></div>
<h3>⚡ အသံထုတ် အရည်အသွေး</h3><select id="q"><option value="4">အမြန်ဆုံး (အရည်အသွေး နည်းနည်းလျော့)</option><option value="6" selected>ပုံမှန်</option><option value="10">အကောင်းဆုံး (နှေးတယ်)</option></select>
<div class="note">သန့်တဲ့ အသံ စက္ကန့် ၁၀–၃၀။ ကိုယ့်အသံ (သို့) ခွင့်ပြုချက်ရတဲ့ အသံသာ သုံးပါ။</div>
<div class="note">ℹ️ ⏳ ၁၀ မိနစ် video ဆို ၁၅–၃၀ မိနစ်ခန့် ကြာနိုင်ပါတယ်။ Colab tab ကို မပိတ်ပါနဲ့။</div>
</div>
<button class="btn" id="go" disabled>▶ စတင်မည်</button>
<div class="note" id="gon">Video နဲ့ အသံနမူနာ နှစ်ခုလုံး ထည့်ပြီးမှ စလို့ရပါမယ်။</div>
</div>

<div id="v-job" class="card hide"><h2>လုပ်ဆောင်ချက် — <span id="jn"></span></h2>
<div class="live"><div class="big" id="bigsp"></div><div><div id="pct">0%</div><div class="note" id="cur" style="margin:0">စတင်နေပါတယ်…</div></div><div id="el">⏱ 00:00</div></div>
<div class="bar"><i id="pb"></i></div><div class="note" id="upn"></div><div class="note" id="hb"></div>
<ul id="st" style="margin-top:8px"></ul><pre id="er"></pre>
<button class="btn ol" id="nx">➕ Video အသစ်ထုတ်မည်</button>
<div class="note">ဒီ video ကို နောက်ကွယ်မှာ ဆက်လုပ်နေမယ်။ နောက်တစ်ပုဒ် စထုတ်လို့ရတယ်၊ 📜 History မှာ ပြန်ကြည့်နိုင်တယ်။</div></div>

<div id="v-res" class="card hide"><h2>ရလဒ် — <span id="rn"></span></h2><video id="rv" controls playsinline></video>
<input id="fn" type="text" placeholder="ဖိုင်အမည် (မပေးလည်းရ)" style="margin-top:10px">
<a id="dl" class="btn">⬇ MP4 ဒေါင်းမည်</a>
<button class="btn ol" id="nx2">➕ Video အသစ်ထုတ်မည်</button></div>

<div id="v-hist" class="card hide"><h2>📜 History</h2><div id="hl"></div>
<button class="btn ol" id="hb8">➕ Video အသစ်ထုတ်မည်</button></div>
</main><script>
const $=i=>document.getElementById(i);
let vf=null,af=null,af2=null,dlid=null,vid=null,upP=null,srcReady=false,dur=0,objU=null,watching=null,view='main',dT=null,curId=null,seen={},REM=null;
const ST={sub:{on:true,x:.5,y:.88,size:.055,color:'#ffffff'},bl:{on:false,x:.1,y:.8,w:.8,h:.16,s:12},bgb:true,size:'orig',q:1080,crop:{on:false,x:.1,y:.1,w:.8,h:.8}};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const STEPS=["Video ရယူခြင်း","Video စစ်ဆေးခြင်း","AssemblyAI နဲ့ စကားကို စာထုတ်ခြင်း","နမူနာအသံ ပြင်ဆင်ခြင်း","Gemini နဲ့ မြန်မာလို ဘာသာပြန်ခြင်း","Voice Clone နဲ့ အသံထုတ်ခြင်း","Video ဖြတ်ညှိခြင်း","စာတန်း + Blur ထည့်ပြီး MP4 ထုတ်ခြင်း","ပြီးစီးပါပြီ"];
const mm=s=>String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0');
const c=(v,a,b)=>Math.max(a,Math.min(b,v));
const rid=()=>Array.from(crypto.getRandomValues(new Uint8Array(12)),b=>b.toString(16).padStart(2,'0')).join('');
const MAXS=900;
function show(v){view=v;['main','job','res','hist'].forEach(n=>$('v-'+n).classList.toggle('hide',n!=v));if(v!='res')$('rv').pause();if(v!='main')$('pv').pause();window.scrollTo(0,0);if(v=='main')setTimeout(paint,80);if(v=='hist')loadHist()}
function setTab(n){document.querySelectorAll('.tabs3 button').forEach(b=>b.classList.toggle('on',+b.dataset.t==n));[1,2,3].forEach(i=>$('t'+i).classList.toggle('hide',i!=n));setTimeout(paint,60)}
document.querySelectorAll('.tabs3 button').forEach(b=>b.onclick=()=>setTab(+b.dataset.t));

// ---- User ID + ထုတ်ခွင့်
async function refreshMe(){let m;try{m=await(await fetch('/api/me')).json()}catch(e){return}
$('uid').textContent=m.uid;REM=(m.limited&&m.ok)?m.rem:null;
$('rem').innerHTML=!m.ok?'<span style="color:#e11d2e">'+esc(m.msg||'Server ချိတ်မရပါ')+'</span>':m.limited?'<b>'+m.rem+'</b> ပုဒ်':'ကန့်သတ်မထား (စမ်းသပ်ခြင်း)'}
$('cp').onclick=()=>{try{navigator.clipboard.writeText($('uid').textContent);$('cp').textContent='✔ Copied'}catch(e){}};
$('hbtn').onclick=()=>show('hist');

async function upload(f,id,cb){const CH=8*1024*1024,n=Math.max(1,Math.ceil(f.size/CH));for(let i=0;i<n;i++){const fd=new FormData();fd.append('id',id);fd.append('idx',i);fd.append('n',n);fd.append('chunk',f.slice(i*CH,(i+1)*CH),'c');
const r=await fetch('/api/up',{method:'POST',body:fd});if(!r.ok)throw new Error('တင်လို့မရပါ ('+r.status+')');if(cb)cb((i+1)/n)}}
function upd(){const long=dur>MAXS;$('vmsg').style.color=long?'#e11d2e':'';
if(long)$('vmsg').textContent='⚠️ Video က ၁၅ မိနစ်ထက် ပိုရှည်နေပါတယ် ('+mm(dur)+') — တိုတိုလေး ဖြတ်ပြီး ထပ်ထည့်ပါ';
$('go').disabled=!(srcReady&&af&&!long);
$('gon').textContent=!srcReady?'Video (သို့) Link ကို အရင်ထည့်ပါ':!af?'အသံနမူနာကို "အသံ" tab မှာ ထည့်ပါ':long?'Video ရှည်လွန်းနေပါတယ်':'အားလုံးအဆင်သင့်ပါ — စတင်လို့ရပါပြီ'}

// ---- Video / Link
function resetSrc(){if(objU){URL.revokeObjectURL(objU);objU=null}clearInterval(dT);dlid=null;vid=null;vf=null;srcReady=false;dur=0;$('vmsg').textContent='';$('dlw').classList.add('hide');$('der').textContent=''}
function setSrc(u){const p=$('pv');p.src=u;p.classList.remove('hide');p.load()}
$('vf').onchange=e=>{const f=e.target.files[0]||null;if(!f)return;resetSrc();vf=f;vid=rid();srcReady=true;$('dropt').textContent='✔ '+f.name;objU=URL.createObjectURL(f);setSrc(objU);
upP=upload(f,vid,pr=>{if(!vf||vf!==f)return;$('vmsg').textContent=pr<1?'📤 Video ဖိုင်တင်နေ '+Math.round(pr*100)+'%':''});upP.catch(()=>{});upd()};
$('pst').onclick=async()=>{try{$('lk').value=await navigator.clipboard.readText()}catch(e){$('lk').focus();alert('Paste မရပါ — Link ကို ကြာကြာဖိပြီး Paste လုပ်ပါ')}};
$('dlb').onclick=async()=>{const u=$('lk').value;if(!/https?:\/\//i.test(u)){alert('Link မမှန်ပါ');return}
const ckf=$('ck').files[0];if(ckf){const cf=new FormData();cf.append('f',ckf);try{await fetch('/api/cookies',{method:'POST',body:cf})}catch(e){}}
const fd=new FormData();fd.append('url',u);let j={};try{j=await(await fetch('/api/dl',{method:'POST',body:fd})).json()}catch(e){}
if(!j.id){alert(j.msg||'Link မမှန်ပါ');return}
resetSrc();$('pv').pause();$('pv').classList.add('hide');$('dropt').textContent='Video ဖိုင် ရွေးပါ';$('vf').value='';
dlid=j.id;$('dlw').classList.remove('hide');$('dpb').style.width='0';$('dtx').textContent='ဒေါင်းနေပါတယ်… 0%';upd();pollDl()};
function pollDl(){clearInterval(dT);const id=dlid;dT=setInterval(async()=>{if(id!=dlid){clearInterval(dT);return}
let s;try{s=await(await fetch('/api/dlst/'+id)).json()}catch(e){return}
const p=Math.round((s.f||0)*100);$('dpb').style.width=p+'%';
$('dtx').textContent=s.state=='done'?'ဒေါင်းပြီးပါပြီ ✔':s.state=='err'?'ဒေါင်းမရပါ':'ဒေါင်းနေပါတယ်… '+p+'% '+(s.d||'');
if(s.state=='done'){clearInterval(dT);srcReady=true;setSrc('/api/dlv/'+id);upd()}
if(s.state=='err'){clearInterval(dT);$('der').textContent=s.msg||''}},1000)}

// ---- Preview (output frame) + crop editor
const BG=document.createElement('canvas');
function outAR(){const p=$('pv');let w=p.videoWidth||16,h=p.videoHeight||9;if(ST.crop.on){w*=ST.crop.w;h*=ST.crop.h}
if(ST.size=='orig')return w/h;const a=ST.size.split(':');return a[0]/a[1]}
function outDims(){const ar=outAR(),Q=ST.q;let W,H;if(ar>=1){H=Q;W=Math.round(Q*ar)}else{W=Q;H=Math.round(Q/ar)}return [W-W%2,H-H%2]}
function drawOut(cv,W,H){W=Math.round(W);H=Math.round(H);if(cv.width!=W||cv.height!=H){cv.width=W;cv.height=H}
const x=cv.getContext('2d'),p=$('pv'),vw=p.videoWidth,vh=p.videoHeight;x.fillStyle='#000';x.fillRect(0,0,W,H);if(!vw||p.readyState<2)return;
const k=ST.crop.on?ST.crop:{x:0,y:0,w:1,h:1},sx=k.x*vw,sy=k.y*vh,sw=k.w*vw,sh=k.h*vh,s=Math.min(W/sw,H/sh),dw=sw*s,dh=sh*s;
if(ST.bgb&&(Math.abs(dw-W)>1||Math.abs(dh-H)>1)){BG.width=Math.max(2,Math.round(W/14));BG.height=Math.max(2,Math.round(H/14));const s2=Math.max(BG.width/sw,BG.height/sh);
BG.getContext('2d').drawImage(p,sx,sy,sw,sh,(BG.width-sw*s2)/2,(BG.height-sh*s2)/2,sw*s2,sh*s2);x.imageSmoothingEnabled=true;x.drawImage(BG,0,0,W,H)}
x.drawImage(p,sx,sy,sw,sh,(W-dw)/2,(H-dh)/2,dw,dh)}
function paint(){const p=$('pv'),vw=p.videoWidth;
if(vw){$('stage').style.aspectRatio=String(outAR());const d=outDims();$('osz').textContent='Output: '+d[0]+' × '+d[1]}
const r=$('stage').getBoundingClientRect();
if(r.width){const b=ST.bl,s=ST.sub,z=$('bz'),t=$('sb');
z.style.display=b.on?'block':'none';z.style.left=b.x*100+'%';z.style.top=b.y*100+'%';z.style.width=b.w*100+'%';z.style.height=b.h*100+'%';
const bp='blur('+(b.s*r.width/720)+'px)';z.style.backdropFilter=bp;z.style.webkitBackdropFilter=bp;
t.style.display=s.on?'block':'none';t.style.left=s.x*100+'%';t.style.top=s.y*100+'%';t.style.fontSize=(s.size*r.height)+'px';t.style.color=s.color;
drawOut($('cv'),r.width,r.height)}
if(vw){$('cstage').style.aspectRatio=String(vw/p.videoHeight)}
const r2=$('cstage').getBoundingClientRect();
if(r2.width&&vw){const cv=$('cv2'),W=Math.round(r2.width),H=Math.round(r2.height);if(cv.width!=W||cv.height!=H){cv.width=W;cv.height=H}
const x=cv.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,W,H);if(p.readyState>=2)x.drawImage(p,0,0,W,H);
const k=ST.crop,z=$('cz');z.style.left=k.x*100+'%';z.style.top=k.y*100+'%';z.style.width=k.w*100+'%';z.style.height=k.h*100+'%'}}
window.addEventListener('resize',paint);
function loop(){const p=$('pv');if(p.paused||p.ended)return;paint();requestAnimationFrame(loop)}
(function(){const p=$('pv');p.addEventListener('loadedmetadata',()=>{dur=p.duration||0;upd();paint()});
['loadeddata','seeked','timeupdate'].forEach(e=>p.addEventListener(e,paint));p.addEventListener('play',loop)})();
function drag(el,stage,start,move){el.addEventListener('pointerdown',e=>{e.preventDefault();e.stopPropagation();el.setPointerCapture(e.pointerId);
const r=stage.getBoundingClientRect(),sx=e.clientX,sy=e.clientY,o=start();
const mv=ev=>move(o,(ev.clientX-sx)/r.width,(ev.clientY-sy)/r.height);
const up=()=>{el.removeEventListener('pointermove',mv);el.removeEventListener('pointerup',up);el.removeEventListener('pointercancel',up)};
el.addEventListener('pointermove',mv);el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up)})}
const S2=$('stage'),CS=$('cstage'),MINC=.1;
drag($('bz'),S2,()=>({x:ST.bl.x,y:ST.bl.y}),(o,dx,dy)=>{ST.bl.x=c(o.x+dx,0,1-ST.bl.w);ST.bl.y=c(o.y+dy,0,1-ST.bl.h);paint()});
drag($('bh'),S2,()=>({w:ST.bl.w,h:ST.bl.h}),(o,dx,dy)=>{ST.bl.w=c(o.w+dx,.05,1-ST.bl.x);ST.bl.h=c(o.h+dy,.03,1-ST.bl.y);paint()});
drag($('sb'),S2,()=>({x:ST.sub.x,y:ST.sub.y}),(o,dx,dy)=>{ST.sub.x=c(o.x+dx,0,1);ST.sub.y=c(o.y+dy,0,1);paint()});
drag($('cz'),CS,()=>({x:ST.crop.x,y:ST.crop.y}),(o,dx,dy)=>{ST.crop.x=c(o.x+dx,0,1-ST.crop.w);ST.crop.y=c(o.y+dy,0,1-ST.crop.h);paint()});
drag($('ch2'),CS,()=>({w:ST.crop.w,h:ST.crop.h}),(o,dx,dy)=>{ST.crop.w=c(o.w+dx,MINC,1-ST.crop.x);ST.crop.h=c(o.h+dy,MINC,1-ST.crop.y);paint()});
drag($('ch1'),CS,()=>({x:ST.crop.x,y:ST.crop.y,w:ST.crop.w,h:ST.crop.h}),(o,dx,dy)=>{const nx=c(o.x+dx,0,o.x+o.w-MINC),ny=c(o.y+dy,0,o.y+o.h-MINC);ST.crop.x=nx;ST.crop.y=ny;ST.crop.w=o.w+(o.x-nx);ST.crop.h=o.h+(o.y-ny);paint()});

// ---- controls
function chips(name,cb){const L=document.querySelectorAll('input[name='+name+']');L.forEach(r=>r.onchange=()=>{L.forEach(x=>x.parentElement.classList.toggle('on',x.checked));cb(r.value)})}
chips('sz',v=>{ST.size=v;paint()});chips('ql',v=>{ST.q=+v;paint()});
$('t-cr').onchange=e=>{ST.crop.on=e.target.checked;$('cpane').classList.toggle('hide',!ST.crop.on);setTimeout(paint,30)};
$('t-sub').onchange=e=>{ST.sub.on=e.target.checked;$('subctl').classList.toggle('hide',!ST.sub.on);paint()};
$('t-bl').onchange=e=>{ST.bl.on=e.target.checked;$('blctl').classList.toggle('hide',!ST.bl.on);paint()};
$('t-bg').onchange=e=>{ST.bgb=e.target.checked;paint()};
$('sc').oninput=e=>{ST.sub.color=e.target.value;paint()};
$('ss').oninput=e=>{ST.sub.size=e.target.value/1000;paint()};
$('bs').oninput=e=>{ST.bl.s=+e.target.value;paint()};
$('pp').onclick=()=>{const p=$('pv');if(!p.src)return;p.paused?p.play():p.pause()};
const curMd=()=>document.querySelector('input[name=md]:checked').value;
function setVoiceUI(){const d=curMd()=='dubbing';$('v2w').classList.toggle('hide',!d);$('v1t').textContent=d?'🎤 Voice 1 (မဖြစ်မနေလို)':'🎤 အသံနမူနာ ထည့်ပါ'}
document.querySelectorAll('input[name=md]').forEach(r=>r.onchange=setVoiceUI);setVoiceUI();
$('af').onchange=e=>{af=e.target.files[0]||null;if(af){$('ap').src=URL.createObjectURL(af);$('ap').classList.remove('hide')}$('spw').classList.toggle('hide',!af);upd()};
$('af2').onchange=e=>{af2=e.target.files[0]||null;if(af2){$('ap2').src=URL.createObjectURL(af2);$('ap2').classList.remove('hide')}};
$('sp').oninput=()=>{$('spv').textContent=(+$('sp').value).toFixed(2)+'x'};

// ---- စတင်မည်
$('go').onclick=async()=>{const V=vf,DL=dlid,L=$('lk').value.trim(),lang=$('lang').value,steps=$('q').value,md=curMd(),spd=$('sp').value;
const VID=vid,UPP=upP,AF=af,AF2=af2,STY=JSON.stringify({sub:ST.sub,bl:ST.bl,bgb:ST.bgb,size:ST.size,q:ST.q,crop:ST.crop});
const isFile=!!V;
$('go').disabled=true;let started=false,J=null;
try{const f0=new FormData();f0.append('name',isFile?V.name:L);const j0=await(await fetch('/api/new',{method:'POST',body:f0})).json();J=j0.id;if(!J)throw new Error(j0.msg||'စတင်မရပါ');
started=true;watching=J;show('job');poll();
const o1=rid();const a1=upload(AF,o1);a1.catch(()=>{});let o2='',a2=null;if(md=='dubbing'&&AF2){o2=rid();a2=upload(AF2,o2);a2.catch(()=>{})}
if(isFile)await UPP;await a1;if(a2)await a2;
const fd=new FormData();fd.append('jid',J);fd.append('vid',isFile?VID:'');fd.append('dlid',isFile?'':DL);fd.append('voice',o1);fd.append('voice2',o2);
fd.append('lang',lang);fd.append('steps',steps);fd.append('mode',md);fd.append('style',STY);fd.append('spd',spd);
const j=await(await fetch('/api/start',{method:'POST',body:fd})).json();if(!j.ok)throw new Error(j.msg||'စတင်မရပါ');refreshMe();poll()
}catch(e){const m=String((e&&e.message)||e);if(started){const f=new FormData();f.append('msg',m);fetch('/api/fail/'+J,{method:'POST',body:f})}else alert(m)}
upd()};

// ---- progress
function paintJob(j){
const ico={wait:'○',run:'<i class="sp"></i>',done:'✔',err:'✖'},done=j.status=='done',bad=j.status=='err';
$('jn').textContent=j.name;
$('st').innerHTML=STEPS.map((n,i)=>{const x=j.steps[i]||{s:'wait'};const pc=x.s=='run'&&x.f?' '+Math.round(x.f*100)+'%':'';return `<li class="${x.s}"><span class="ic">${ico[x.s]}</span><span>${i+1}. ${n}</span><span class="d">${esc(x.d||'')}${pc}</span></li>`}).join('');
$('pb').style.width=j.pct+'%';$('pct').textContent=j.pct+'%';
const run=j.steps.map((x,i)=>x.s=='run'?STEPS[i]:null).filter(Boolean);
$('cur').textContent=done?'ပြီးစီးပါပြီ ✔':bad?'အမှားဖြစ်သွားပါတယ်':(run.join(' + ')||'စတင်နေပါတယ်…');
const bg=$('bigsp');bg.className='big'+(done?' fin':bad?' bad':'');bg.textContent=done?'✔':bad?'✖':'';
$('el').textContent='⏱ '+mm(j.el);$('er').textContent=j.err||'';
$('hb').textContent=(done||bad)?'':(j.ago>150?'🟡 ကြာနေပေမယ့် Server အလုပ်လုပ်နေဆဲပါ — နောက်ဆုံး update '+Math.round(j.ago)+' စက္ကန့်':'🟢 Server အလုပ်လုပ်နေပါတယ်');
document.title=(done?'✔ ':j.pct+'% ')+'Auto Movie Recap'}
async function poll(){if(!watching)return;let j;try{j=await(await fetch('/api/job/'+watching)).json()}catch(e){return}
if(j.msg){watching=null;if(view=='job')show('main');return}
if(j.status!='run'&&!seen[j.id]){seen[j.id]=1;refreshMe()}
if(view!='job')return;paintJob(j);if(j.status=='done')openResult(j)}

// ---- ရလဒ်
function setLink(){if(!curId)return;const n=$('fn').value.trim().replace(/[\\\/:*?"<>|]/g,'');$('dl').href='/api/result/'+curId+'?dl=1'+(n?'&name='+encodeURIComponent(n):'')}
$('fn').oninput=setLink;
function openResult(j){curId=j.id;$('rn').textContent=j.name;$('rv').src='/api/result/'+j.id;$('fn').value='';$('fn').placeholder=String(j.name).replace(/\.[^.]+$/,'').slice(0,60);setLink();show('res')}

// ---- History
async function loadHist(){let l;try{l=await(await fetch('/api/jobs')).json()}catch(e){return}
if(view!='hist')return;window._hl=l;
$('hl').innerHTML=l.length?l.map(j=>{
const i=j.steps.findIndex(x=>x.s=='run');
const stt=j.status=='done'?'✅ ပြီးပါပြီ':j.status=='err'?'❌ မအောင်မြင်ပါ':'⏳ လုပ်နေဆဲ — '+j.pct+'%'+(i>=0?' · '+STEPS[i]:'');
const bar=j.status=='run'?'<div class="bar"><i style="width:'+j.pct+'%"></i></div>':'';
const err=j.status=='err'?'<pre>'+esc(j.err||'')+'</pre>':'';
const btn=j.status=='done'?'<div class="row"><button class="btn" data-v="'+j.id+'">▶ ကြည့်မည်</button><a class="btn ol" href="/api/result/'+j.id+'?dl=1">⬇ MP4 ဒေါင်း</a></div>':j.status=='run'?'<button class="btn ol" data-w="'+j.id+'">👁 အခြေအနေကြည့်မည်</button>':'';
return '<div class="hi"><b>'+esc(j.name)+'</b><div class="note">'+stt+' · ⏱ '+mm(j.el)+'</div>'+bar+err+btn+'</div>'}).join(''):'<div class="note">ထုတ်ထားတဲ့ video မရှိသေးပါ</div>'}
$('hl').onclick=e=>{const v=e.target.closest('[data-v]'),w=e.target.closest('[data-w]');
if(v){const id=v.dataset.v;const it=(window._hl||[]).find(x=>x.id==id);openResult({id:id,name:it?it.name:'video'})}
if(w){watching=w.dataset.w;show('job');poll()}};

function newVideo(){resetSrc();af=null;af2=null;watching=null;
['vf','af','af2','lk','ck'].forEach(i=>$(i).value='');
['ap','ap2'].forEach(i=>{$(i).removeAttribute('src');$(i).classList.add('hide')});
const p=$('pv');p.pause();p.removeAttribute('src');p.load();p.classList.add('hide');$('rv').removeAttribute('src');
$('dropt').textContent='Video ဖိုင် ရွေးပါ';$('spw').classList.add('hide');$('sp').value=1;$('spv').textContent='1.00x';
document.title='Auto Movie Recap';upd();setTab(1);show('main')}
$('nx').onclick=$('nx2').onclick=$('hb8').onclick=$('newbtn').onclick=newVideo;

refreshMe();show('main');upd();
fetch('/api/jobs').then(r=>r.json()).then(l=>{if(l.some(j=>j.status=='run'))show('hist')}).catch(()=>{});
setInterval(()=>{poll();if(view=='hist')loadHist()},2000);
</script></body></html>'''

app = FastAPI()

@app.get("/")
def home(): return HTMLResponse(HTML)

@app.get("/api/me")
def me():
    r = srv(action="check", id=UID)
    ok = bool(r.get("ok"))
    return {"uid": UID, "ok": ok, "limited": bool(SERVER_URL), "rem": r.get("remaining") if ok else None, "msg": r.get("msg", "")}

@app.post("/api/new")
def new_job(name: str = Form("video")):
    r = srv(action="check", id=UID)
    if not r.get("ok"): return JSONResponse({"msg": r.get("msg", "Server ချိတ်မရပါ")}, status_code=503)
    if SERVER_URL and int(r.get("remaining") or 0) <= 0:
        return JSONResponse({"msg": f"video ထုတ်ခွင့် မကျန်တော့ပါ — Admin ကို သင့် User ID ({UID}) ပေးပြီး ထပ်ထည့်ခိုင်းပါ"}, status_code=402)
    jid = uuid.uuid4().hex[:16]
    steps = [{"s": "wait", "d": "", "f": 0} for _ in range(9)]
    steps[0] = {"s": "run", "d": "", "f": 0}
    JOBS[jid] = {"id": jid, "name": (name or "video")[:120], "created": time.time(), "fin": None, "steps": steps, "done": False, "err": None, "out": None, "upd": time.time()}
    return {"id": jid}

@app.post("/api/cookies")
def save_cookies(f: UploadFile):
    with open(COOKIES_FILE, "wb") as o: shutil.copyfileobj(f.file, o)
    return {"ok": True}

@app.post("/api/dl")
def dl_start(url: str = Form("")):
    m = re.search(r"https?://[^\s]+", url)
    if not m: return JSONResponse({"msg": "Link မမှန်ပါ"}, status_code=400)
    u = m.group(0).rstrip(".,)]}>\"'")
    did = uuid.uuid4().hex[:16]
    DL[did] = {"state": "run", "f": 0.0, "d": "", "msg": "", "path": None, "title": ""}
    threading.Thread(target=dl_worker, args=(did, u), daemon=True).start()
    return {"id": did}

@app.get("/api/dlst/{did}")
def dl_status(did: str):
    d = DL.get(did)
    if not d: return JSONResponse({"state": "err", "msg": "မတွေ့ပါ"}, status_code=404)
    return {"state": d["state"], "f": d["f"], "d": d["d"], "msg": d["msg"]}

@app.get("/api/dlv/{did}")
def dl_video(did: str):
    d = DL.get(did)
    if not d or not d.get("path") or not os.path.exists(d["path"]): return JSONResponse({"msg": "မတွေ့ပါ"}, status_code=404)
    return FileResponse(d["path"], media_type="video/mp4")

@app.post("/api/up")
def up(chunk: UploadFile, id: str = Form(""), idx: int = Form(0), n: int = Form(1)):
    if not re.fullmatch(r"[0-9a-f]{12,40}", id): return JSONResponse({"msg": "id မှားနေပါတယ်"}, status_code=400)
    with open(os.path.join(WORK, "up_" + id), "wb" if idx == 0 else "ab") as f: shutil.copyfileobj(chunk.file, f)
    return {"ok": True}

@app.post("/api/start")
def start(jid: str = Form(""), vid: str = Form(""), dlid: str = Form(""), voice: str = Form(""), voice2: str = Form(""),
          lang: str = Form("auto"), steps: int = Form(6), mode: str = Form("recap"), style: str = Form("{}"), spd: float = Form(1.0)):
    if jid not in JOBS: return JSONResponse({"msg": "job မတွေ့ပါ"}, status_code=400)
    if dlid:
        if dlid not in DL: return JSONResponse({"msg": "ဒေါင်းထားတဲ့ video မတွေ့ပါ"}, status_code=400)
        vid = ""; chk = [voice]
    else: chk = [vid, voice]
    if voice2: chk.append(voice2)
    for x in chk:
        if not re.fullmatch(r"[0-9a-f]{12,40}", x) or not os.path.exists(os.path.join(WORK, "up_" + x)):
            return JSONResponse({"msg": "ဖိုင်မတွေ့ပါ၊ ပြန်တင်ပါ"}, status_code=400)
    if not re.fullmatch(r"[a-z]{2,3}(_[a-z]{2})?|auto", lang): lang = "auto"
    mode = "dubbing" if mode == "dubbing" else "recap"
    u = srv(action="use", id=UID)      # ထုတ်ခွင့် ၁ ပုဒ် နှုတ်
    if not u.get("ok"): return JSONResponse({"msg": u.get("msg", "video ထုတ်ခွင့် မကျန်တော့ပါ")}, status_code=402)
    if u.get("ticket"): JOBS[jid]["ticket"] = u["ticket"]
    threading.Thread(target=run_job, args=(jid, vid, dlid, voice, voice2, lang, max(3, min(12, steps)), mode, style, max(0.6, min(1.6, spd))), daemon=True).start()
    return {"ok": True}

@app.post("/api/fail/{jid}")
def fail(jid: str, msg: str = Form("")):
    j = JOBS.get(jid)
    if j and not j["done"] and not j["err"]:
        j["err"] = (msg or "မအောင်မြင်ပါ")[-500:]; j["fin"] = time.time(); j["upd"] = time.time()
        for x in j["steps"]:
            if x["s"] == "run": x["s"] = "err"
        refund_job(j)
    return {"ok": True}

@app.get("/api/job/{jid}")
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j: return {"msg": "မတွေ့ပါ"}
    now = time.time()
    st = "done" if j["done"] else "err" if j["err"] else "run"
    return {"id": j["id"], "name": j["name"], "status": st, "pct": pct(j), "steps": j["steps"], "err": j["err"], "ago": now - j["upd"], "el": (j["fin"] or now) - j["created"]}

@app.get("/api/jobs")
def jobs_list():
    now, out = time.time(), []
    for j in sorted(JOBS.values(), key=lambda x: -x["created"]):
        st = "done" if j["done"] else "err" if j["err"] else "run"
        out.append({"id": j["id"], "name": j["name"], "status": st, "pct": pct(j), "steps": j["steps"], "err": j["err"], "el": (j["fin"] or now) - j["created"]})
    return out

def clean_name(s): return re.sub(r'[\\/:*?"<>|\r\n]', "", s or "").strip()[:80]

@app.get("/api/result/{jid}")
def result(jid: str, dl: int = 0, name: str = ""):
    j = JOBS.get(jid)
    if not j or not j.get("out") or not os.path.exists(j["out"]): return JSONResponse({"msg": "မတွေ့ပါ"}, status_code=404)
    if dl:
        stem = clean_name(name) or clean_name(os.path.splitext(j["name"])[0]) or "recap"
        return FileResponse(j["out"], media_type="video/mp4", filename=stem + ".mp4")
    return FileResponse(j["out"], media_type="video/mp4")

try: _SRV.should_exit = True; time.sleep(2)
except NameError: pass
_SRV = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
threading.Thread(target=_SRV.run, daemon=True).start()
time.sleep(3)
from google.colab import output
from IPython.display import display, HTML

def get_site():
    # 1) Colab ရဲ့ ကိုယ်ပိုင် proxy link
    try:
        u = output.eval_js("google.colab.kernel.proxyPort(8000, {'cache': true})", timeout_sec=20)
        if u and str(u).startswith("http"): return str(u), "colab"
    except Exception as e:
        print("Colab link မရပါ:", str(e)[:100])
    # 2) မရရင် cloudflared tunnel နဲ့ public link ယူ (ဖုန်း view မှာလည်း အလုပ်လုပ်)
    try:
        exe = "/content/cloudflared"
        if not os.path.exists(exe):
            r = requests.get("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", timeout=180)
            r.raise_for_status(); open(exe, "wb").write(r.content); os.chmod(exe, 0o755)
        lg = open("/content/cf.log", "w")
        subprocess.Popen([exe, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"], stdout=lg, stderr=lg)
        t0 = time.time()
        while time.time() - t0 < 60:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/cf.log").read())
            if m: return m.group(0), "cf"
            time.sleep(1)
    except Exception as e:
        print("tunnel link မရပါ:", str(e)[:100])
    return "", ""

site, kind = get_site()
print("✅ Website အဆင်သင့်ပါ — Colab tab ကို မပိတ်ပါနဲ့")
if site:
    print("🔗 Link:", site)
    display(HTML('<a href="%s" target="_blank" style="display:block;text-align:center;background:#e11d2e;color:#fff;font:700 18px sans-serif;padding:16px;border-radius:12px;text-decoration:none;margin:10px 0">🌐 Website ကို Tab အသစ်မှာ ဖွင့်မည်</a>' % site))
    if kind == "colab":
        try: output.serve_kernel_port_as_iframe(8000, height="900")
        except Exception as e: print("iframe မပေါ်ပါ:", str(e)[:100])
    else:
        display(HTML('<iframe src="%s" style="width:100%%;height:900px;border:1px solid #ddd;border-radius:12px;background:#fff"></iframe>' % site))
    print("💡 အောက်မှာ website မပေါ်ရင် Chrome ⋮ menu → \"Desktop site\" ကို အမှန်ခြစ်ပြီး Colab ကို refresh လုပ်ပါ (ဖုန်း view က iframe မပြတတ်ပါ)။ အပေါ်က ခလုတ်/link ကတော့ အမြဲရပါတယ်။")
else:
    print("❌ Website link မရပါ — cell ကို ပြန် run ပါ၊ မရသေးရင် output ကို screenshot ပြပါ")
