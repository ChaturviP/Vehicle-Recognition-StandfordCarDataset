"""Streamlit-only presentation: global styles, the 3D hero, motion scripts and HTML result components.

Kept separate from content.py, which the Gradio front end (app.py) also uses.
Every piece of model output that reaches HTML goes through `esc()`.

Streamlit's HTML sanitiser strips inline SVG, so icons and rings here are pure CSS.
Motion is progressive enhancement: with JavaScript, WebGL or the CDN unavailable, or with
prefers-reduced-motion set, every element is still rendered in its final, readable state.
"""
from __future__ import annotations

import base64
import io
from html import escape as esc

import numpy as np
from PIL import Image

# Status colours match the boxes drawn by inference.draw(): green, amber, red.
LEVELS = {
    "model": ("Full answer", "ok"),
    "make": ("Manufacturer only", "warn"),
    "uncertain": ("Uncertain", "bad"),
    "no_car": ("No car found", "none"),
}

# Served by Streamlit static file serving (server.enableStaticServing) from ./static.
STATIC = "app/static"
# Real output of the app on examples/tesla_model_s.jpg (normalised x1, y1, x2, y2).
HERO_BOX = (0.0547, 0.1649, 0.9323, 0.9101)
HERO_LABEL = "Tesla Model S Sedan 2012 · 99%"

CSS = """
<style>
:root {
  --bg: #07090D; --bg-2: #0B0E14;
  --glass: rgba(15, 19, 27, .72); --glass-2: rgba(21, 26, 37, .78);
  --line: #222A37; --line-2: #2F3A4B;
  --ink: #E8ECF2; --ink-2: #A3ADBE; --ink-3: #6E788B;
  --accent: #5B8CFF; --accent-2: #9BB8FF; --accent-deep: #2F5BE0; --accent-glow: rgba(91, 140, 255, .32);
  --ok: #3DDC84; --warn: #F5B43C; --bad: #FF6B5E; --none: #8A93A3;
  --mono: 'IBM Plex Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace;
  --ease: cubic-bezier(.16, 1, .3, 1);
  --r: 14px;
}
::selection { background: rgba(91, 140, 255, .35); color: #fff; }
html { scrollbar-color: #2F3A4B transparent; }
:focus-visible { outline: 2px solid var(--accent-2) !important; outline-offset: 3px; }

/* Backdrop: duotone photo, xenon glow, fading into the page colour */
.stApp {
  background:
    radial-gradient(900px 520px at 78% -6%, rgba(91, 140, 255, .20), transparent 62%),
    radial-gradient(700px 420px at 4% 30%, rgba(47, 91, 224, .10), transparent 70%),
    linear-gradient(180deg, rgba(7, 9, 13, .55) 0%, rgba(7, 9, 13, .86) 42%, var(--bg) 78%),
    url("app/static/bg_mustang.jpg") center top / cover no-repeat,
    var(--bg);
}
.stApp::before { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .5;
  background-image: linear-gradient(rgba(155, 184, 255, .045) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(155, 184, 255, .045) 1px, transparent 1px);
  background-size: 56px 56px; mask-image: radial-gradient(ellipse at 60% 0%, #000 0%, transparent 70%); }
[data-testid="stHeader"] { background: rgba(7, 9, 13, .55); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(47, 58, 75, .45); }
[data-testid="stMainBlockContainer"] { padding-top: 4.25rem; max-width: 1280px; position: relative; z-index: 1; }
[data-testid="stSidebar"] { background: rgba(10, 13, 19, .92); backdrop-filter: blur(16px); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-stretch: 110%; }
.p5-progress { position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 999991; pointer-events: none;
  background: linear-gradient(90deg, var(--accent-deep), var(--accent-2)); transform-origin: 0 50%; transform: scaleX(0);
  box-shadow: 0 0 12px var(--accent-glow); }

/* Type */
h1, h2, h3, .p5-title, .p5-name, .p5-num { font-stretch: 112%; }
[data-testid="stHeadingWithActionElements"] h3 { letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"] { color: var(--ink-2); }

/* Hero */
.p5-hero { position: relative; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.18fr); gap: 2.5rem;
  align-items: center; min-height: min(560px, 74vh); margin: 0 0 1.5rem; }
.p5-hero-copy { position: relative; z-index: 2; will-change: transform, opacity; }
.p5-title { font-size: clamp(2.4rem, 3.9vw, 3.9rem) !important; line-height: .98 !important; font-weight: 800 !important;
  letter-spacing: -0.03em; margin: 0 0 1.25rem !important; padding: 0 !important; color: var(--ink); }
.p5-title .ln { display: block; overflow: hidden; padding-bottom: .06em; }
.p5-title .ln > span { display: inline-block; animation: p5-rise 1s var(--ease) both; }
.p5-title .ln:nth-child(2) > span { color: var(--accent-2); animation-delay: .12s; }
.p5-lede { font-size: 1.075rem; line-height: 1.6; color: var(--ink-2); max-width: 46ch; margin: 0 0 1.75rem;
  animation: p5-fade .9s .3s var(--ease) both; }
.p5-lede b { color: var(--ink); font-weight: 600; }
.p5-cta { display: flex; flex-wrap: wrap; gap: .75rem; margin: 0 0 2.25rem; animation: p5-fade .9s .42s var(--ease) both; }
.p5-btn { --mx: 50%; --my: 50%; position: relative; display: inline-flex; align-items: center; gap: .55rem; min-height: 3rem;
  padding: 0 1.35rem; border-radius: 999px; font-weight: 650; font-size: .98rem; text-decoration: none !important;
  cursor: pointer; overflow: hidden; isolation: isolate; user-select: none;
  transition: transform .25s var(--ease), box-shadow .3s var(--ease), border-color .3s; }
.p5-btn.primary { color: #fff !important; background: linear-gradient(135deg, #6E9BFF 0%, var(--accent) 45%, var(--accent-deep) 100%);
  box-shadow: 0 10px 30px -10px var(--accent-glow), inset 0 1px 0 rgba(255, 255, 255, .25); }
.p5-btn.primary:hover { box-shadow: 0 16px 40px -10px rgba(91, 140, 255, .6), inset 0 1px 0 rgba(255, 255, 255, .3); }
.p5-btn.ghost { color: var(--ink) !important; background: rgba(255, 255, 255, .04); border: 1px solid var(--line-2); }
.p5-btn.ghost:hover { border-color: var(--accent); }
.p5-btn::before { content: ""; position: absolute; inset: 0; z-index: -1; opacity: 0; transition: opacity .3s;
  background: radial-gradient(140px circle at var(--mx) var(--my), rgba(255, 255, 255, .28), transparent 60%); }
.p5-btn:hover::before { opacity: 1; }
.p5-btn:active { transform: scale(.97); }
.p5-btn .arr { transition: transform .3s var(--ease); }
.p5-btn:hover .arr { transform: translateX(3px); }
.p5-stats { display: grid; grid-template-columns: repeat(3, minmax(0, auto)); justify-content: start; gap: 0 2.25rem;
  margin: 0; padding: 1.25rem 0 0 !important; list-style: none; border-top: 1px solid var(--line);
  animation: p5-fade .9s .55s var(--ease) both; }
.p5-stats li { margin: 0; padding: 0; }
.p5-num { display: block; font-size: 1.85rem; font-weight: 750; line-height: 1.1; color: var(--ink); font-variant-numeric: tabular-nums; }
.p5-stats small { display: block; font-size: .8rem; color: var(--ink-2); margin-top: .2rem; max-width: 18ch; line-height: 1.35; }

.p5-stage { position: relative; aspect-ratio: 16 / 10.5; border-radius: 22px; animation: p5-fade 1.1s .15s var(--ease) both; }
.p5-stage::before { content: ""; position: absolute; inset: 8% 4% 4%; border-radius: 50%; z-index: 0;
  background: radial-gradient(closest-side, rgba(91, 140, 255, .28), transparent); filter: blur(30px); }
.p5-stage canvas { position: absolute; inset: 0; width: 100% !important; height: 100% !important; z-index: 1;
  opacity: 0; transition: opacity .8s var(--ease); }
.p5-stage.gl-on canvas { opacity: 1; }
.p5-fallback { position: absolute; left: 5%; right: 5%; top: 12%; z-index: 1; transition: opacity .6s; }
.p5-fallback img { width: 100%; height: auto !important; aspect-ratio: 1280 / 634; display: block; border-radius: 10px; box-shadow: 0 30px 80px -30px rgba(0, 0, 0, .8); }
.p5-fallback .p5-fbox { position: absolute; border: 2px solid var(--ok); border-radius: 4px; box-shadow: 0 0 18px rgba(61, 220, 132, .35); }
.p5-stage.gl-on .p5-fallback { opacity: 0; }
.p5-tag3d { position: absolute; left: 0; top: 0; z-index: 2; pointer-events: none; white-space: nowrap;
  display: inline-flex; align-items: center; gap: .45rem; padding: .32rem .7rem .32rem .55rem; border-radius: 8px;
  background: rgba(8, 12, 10, .82); border: 1px solid rgba(61, 220, 132, .55); color: #EAFBF1;
  font-size: .8rem; font-weight: 600; box-shadow: 0 8px 24px -8px rgba(0, 0, 0, .7);
  opacity: 0; transition: opacity .5s var(--ease); }
.p5-tag3d.on { opacity: 1; }
.p5-tag3d i, .p5-dot { width: .5rem; height: .5rem; border-radius: 50%; background: var(--ok); box-shadow: 0 0 10px var(--ok); flex: none; }
.p5-stage-cap { position: absolute; left: 0; right: 0; bottom: -.25rem; z-index: 2; text-align: center; margin: 0;
  font-size: .78rem; color: var(--ink-3); }
@keyframes p5-rise { from { transform: translateY(105%); } }
@keyframes p5-fade { from { opacity: 0; transform: translateY(18px); } }
@media (max-width: 980px) {
  .p5-hero { grid-template-columns: 1fr; gap: 1.25rem; min-height: 0; }
  .p5-stage { order: 2; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top: 4.75rem; }
  .p5-title { font-size: 2.35rem !important; }
  .p5-lede { font-size: 1rem; }
  .p5-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0 1rem; }
  .p5-num { font-size: 1.35rem; }
  .p5-stats small { font-size: .74rem; }
  .p5-btn { flex: 1 1 auto; justify-content: center; }
}

/* Scroll reveals (attributes are only ever set by JS) */
[data-rv="0"] { opacity: 0; transform: translate3d(0, 26px, 0); }
[data-rv="1"] { opacity: 1; transform: none;
  transition: opacity .75s var(--ease) var(--rd, 0s), transform .9s var(--ease) var(--rd, 0s); }

/* Tabs: segmented pill bar (react-aria tabs in Streamlit 1.5x+) */
[data-testid="stTabs"] [role="tablist"] { gap: .25rem; padding: .3rem; border-radius: 999px; width: fit-content; max-width: 100%;
  background: var(--glass); border: 1px solid var(--line) !important; backdrop-filter: blur(12px); box-shadow: none; overflow-x: auto; scrollbar-width: none; }
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before { display: none; }
[data-testid="stTab"] { height: 2.5rem; padding: 0 1.15rem !important; border-radius: 999px; background: transparent; flex: none;
  transition: background-color .3s var(--ease), color .3s; }
[data-testid="stTab"] p { font-weight: 600; font-size: .95rem; color: var(--ink-2); transition: color .3s; }
[data-testid="stTab"]:hover { background: rgba(255, 255, 255, .05); }
[data-testid="stTab"]:hover p { color: var(--ink); }
[data-testid="stTab"][aria-selected="true"] { background: linear-gradient(135deg, rgba(110, 155, 255, .28), rgba(47, 91, 224, .22));
  box-shadow: inset 0 0 0 1px rgba(155, 184, 255, .35); }
[data-testid="stTab"][aria-selected="true"] p { color: #fff; }
[data-testid="stTabs"] .react-aria-SelectionIndicator { display: none; }
[data-testid="stTabPanel"] { padding-top: 1.75rem; }
@media (max-width: 640px) {
  [data-testid="stTab"] { padding: 0 .8rem !important; height: 2.35rem; }
  [data-testid="stTab"] p { font-size: .86rem; }
}
.st-key-arch [data-testid="stImage"] { max-width: 760px; }

/* Buttons (Streamlit) */
.stButton > button, .stDownloadButton > button { --mx: 50%; --my: 50%; position: relative; overflow: hidden; isolation: isolate;
  transition: transform .2s var(--ease), box-shadow .3s var(--ease), border-color .3s, background-color .3s; }
.stButton > button::before { content: ""; position: absolute; inset: 0; z-index: -1; opacity: 0; transition: opacity .3s;
  background: radial-gradient(120px circle at var(--mx) var(--my), rgba(155, 184, 255, .22), transparent 65%); }
.stButton > button:hover::before { opacity: 1; }
.stButton > button:active { transform: scale(.97); }
[data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #6E9BFF, var(--accent) 50%, var(--accent-deep)) !important;
  border: 0 !important; box-shadow: 0 8px 24px -10px var(--accent-glow); }
[data-testid="stBaseButton-secondary"] { background: rgba(255, 255, 255, .035); border-color: var(--line-2); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent); color: #fff; }

/* Upload zone */
[data-testid="stFileUploader"] > label p { font-weight: 650; font-size: 1rem; }
[data-testid="stFileUploaderDropzone"] { position: relative; min-height: 7.5rem; border-radius: var(--r);
  background: linear-gradient(180deg, rgba(91, 140, 255, .07), rgba(91, 140, 255, .02)), var(--glass);
  border: 1.5px dashed rgba(91, 140, 255, .45); backdrop-filter: blur(10px);
  transition: border-color .3s, box-shadow .4s var(--ease), background-color .3s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-2); box-shadow: 0 0 0 4px rgba(91, 140, 255, .12), 0 20px 50px -24px var(--accent-glow); }

/* Example picker: cards */
.st-key-examples [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: 1rem; }
.st-key-examples [data-testid="stColumn"] { --mx: 50%; --my: 50%; min-width: 0; padding: .55rem .55rem .7rem; border-radius: var(--r);
  background: var(--glass); border: 1px solid var(--line); backdrop-filter: blur(10px); position: relative;
  transition: transform .45s var(--ease), border-color .3s, box-shadow .45s var(--ease); }
.st-key-examples [data-testid="stColumn"]::after { content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  opacity: 0; transition: opacity .35s; background: radial-gradient(260px circle at var(--mx) var(--my), rgba(91, 140, 255, .12), transparent 60%); }
.st-key-examples [data-testid="stColumn"]:hover { transform: translateY(-4px); border-color: rgba(91, 140, 255, .55);
  box-shadow: 0 24px 50px -28px rgba(91, 140, 255, .55); }
.st-key-examples [data-testid="stColumn"]:hover::after { opacity: 1; }
.st-key-examples [data-testid="stImage"] { border-radius: 10px; overflow: hidden; }
.st-key-examples [data-testid="stImage"] img { aspect-ratio: 16 / 9; object-fit: cover; border-radius: 10px;
  transition: transform .8s var(--ease), filter .5s; filter: saturate(.85); }
.st-key-examples [data-testid="stColumn"]:hover [data-testid="stImage"] img { transform: scale(1.06); filter: saturate(1.05); }
.st-key-examples button { min-height: 2.75rem; height: auto; padding: .4rem .5rem; }
.st-key-examples button p { white-space: normal; line-height: 1.25; text-wrap: balance; font-weight: 600; }
.p5-note { font-size: .8rem; color: var(--ink-2); margin: .3rem 0 0; }
.p5-lead { font-size: .95rem; color: var(--ink-2); margin: 0; }

/* Empty state */
.p5-empty { display: flex; gap: 1rem; align-items: center; padding: 1.1rem 1.25rem; border-radius: var(--r);
  background: var(--glass); border: 1px solid var(--line); color: var(--ink-2); }
.p5-empty b { color: var(--ink); }
.p5-pulse { flex: none; width: 2.5rem; height: 2.5rem; border-radius: 50%; position: relative;
  background: radial-gradient(circle, var(--accent) 0 28%, transparent 30%); }
.p5-pulse::after { content: ""; position: absolute; inset: 0; border-radius: 50%; border: 2px solid var(--accent);
  animation: p5-ping 2.2s var(--ease) infinite; }
@keyframes p5-ping { from { transform: scale(.4); opacity: 1; } to { transform: scale(1.15); opacity: 0; } }

/* Result image: one-shot scan sweep */
.st-key-result [data-testid="stImage"] { position: relative; border-radius: var(--r); overflow: hidden;
  box-shadow: 0 30px 70px -35px rgba(0, 0, 0, .9); }
.st-key-result [data-testid="stImage"] img { border-radius: var(--r); }
.st-key-result [data-testid="stImage"]::after { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 30%;
  pointer-events: none; transform: translateY(-110%);
  background: linear-gradient(180deg, transparent, rgba(91, 140, 255, .22) 85%, rgba(155, 184, 255, .9) 99%, transparent);
  animation: p5-scan 1.5s .1s cubic-bezier(.45, 0, .2, 1) both; }
@keyframes p5-scan { to { transform: translateY(380%); } }

/* Verdict */
.p5-verdict { position: relative; border-radius: 18px; padding: 1.35rem 1.4rem 1.15rem; overflow: hidden;
  background: linear-gradient(160deg, rgba(91, 140, 255, .10), transparent 45%), var(--glass-2);
  border: 1px solid var(--line-2); backdrop-filter: blur(14px); box-shadow: 0 30px 70px -40px rgba(0, 0, 0, .9); }
.p5-verdict.ok { --c: var(--ok); } .p5-verdict.warn { --c: var(--warn); } .p5-verdict.bad { --c: var(--bad); } .p5-verdict.none { --c: var(--none); }
.p5-vtop { display: flex; gap: 1.1rem; align-items: center; }
.p5-ring { --p5p: var(--to, 0); position: relative; flex: none; width: 88px; height: 88px; border-radius: 50%;
  display: grid; place-items: center; animation: p5-ring 1.3s .15s var(--ease) both; }
.p5-ring::before { content: ""; position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p5p) * 1%), rgba(255, 255, 255, .08) 0);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 7px), #000 calc(100% - 6px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 7px), #000 calc(100% - 6px)); }
.p5-ring b { font-size: 1.3rem; font-weight: 750; font-variant-numeric: tabular-nums; font-stretch: 105%; }
.p5-ring small { position: absolute; bottom: 19px; font-size: .62rem; color: var(--ink-2); }
.p5-ring b { margin-top: -.5rem; }
@keyframes p5-ring { from { --p5p: 0; } }
.p5-chip { display: inline-flex; align-items: center; gap: .45rem; padding: .22rem .65rem .22rem .5rem; border-radius: 999px;
  font-size: .78rem; font-weight: 650; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }
.p5-chip i { width: .5rem; height: .5rem; border-radius: 50%; background: var(--c); box-shadow: 0 0 10px var(--c); }
.p5-name { font-size: 1.7rem; font-weight: 750; line-height: 1.12; letter-spacing: -0.015em; margin: .55rem 0 .25rem; color: var(--ink); text-wrap: balance; }
.p5-sub { color: var(--ink-2); margin: .8rem 0 0; line-height: 1.55; }
.p5-sub b { color: var(--ink); font-variant-numeric: tabular-nums; }
.p5-meta { display: flex; flex-wrap: wrap; gap: .35rem; margin: 1rem 0 0; padding: .9rem 0 0; border-top: 1px solid var(--line);
  list-style: none; font-size: .75rem; color: var(--ink-2); }
.p5-meta li { margin: 0; padding: .18rem .55rem; border-radius: 6px; background: rgba(255, 255, 255, .04); border: 1px solid var(--line); }
.p5-meta b { font-family: var(--mono); font-weight: 500; color: var(--ink); }

/* Ranked bars */
.p5-h { font-size: .8rem; font-weight: 650; color: var(--ink-2); margin: 1.5rem 0 .7rem; }
.p5-rank { list-style: none; margin: 0; padding: 0; }
.p5-rank li { display: grid; grid-template-columns: 1.4rem 1fr auto; gap: .2rem .5rem; align-items: baseline; margin: 0 0 .7rem; padding: 0; font-size: .92rem; }
.p5-rank .n { color: var(--ink-3); font-family: var(--mono); font-size: .75rem; }
.p5-rank .v { font-family: var(--mono); font-size: .8rem; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.p5-rank li:first-child .t { font-weight: 650; color: #fff; }
.p5-rank li:first-child .v { color: var(--ink); }
.p5-bar { grid-column: 2 / 4; height: 6px; border-radius: 3px; background: rgba(255, 255, 255, .06); overflow: hidden; }
.p5-bar > i { display: block; height: 100%; border-radius: 3px; background: #3B4A6B; transform-origin: left;
  animation: p5-grow 1s var(--ease) both; }
.p5-rank li:nth-child(2) .p5-bar > i { animation-delay: .07s; } .p5-rank li:nth-child(3) .p5-bar > i { animation-delay: .14s; }
.p5-rank li:nth-child(4) .p5-bar > i { animation-delay: .21s; } .p5-rank li:nth-child(5) .p5-bar > i { animation-delay: .28s; }
.p5-rank li:first-child .p5-bar > i { background: linear-gradient(90deg, var(--accent-deep), var(--accent-2)); box-shadow: 0 0 12px var(--accent-glow); }
@keyframes p5-grow { from { transform: scaleX(0); } }

/* Section headings inside the app */
.p5-sec { display: flex; align-items: baseline; gap: .9rem; margin: 2.75rem 0 .35rem; }
.p5-sec .k { font-family: var(--mono); font-size: .8rem; color: var(--accent-2); }
.p5-sec h3 { margin: 0 !important; padding: 0 !important; font-size: 1.45rem !important; font-weight: 750 !important; letter-spacing: -0.01em; }
.p5-secp { color: var(--ink-2); margin: 0 0 1.1rem; max-width: 75ch; }

/* Per-image branch comparison */
.p5-branches { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .92rem; border-radius: var(--r);
  overflow: hidden; background: var(--glass); border: 1px solid var(--line); }
.p5-branches th { text-align: left; font-size: .75rem; font-weight: 650; color: var(--ink-2); padding: .8rem 1rem .6rem;
  border-bottom: 1px solid var(--line); background: rgba(255, 255, 255, .02); }
.p5-branches td { padding: .85rem 1rem; border-bottom: 1px solid var(--line); vertical-align: top; }
.p5-branches tbody tr:last-child td { border-bottom: 0; }
.p5-branches td.num, .p5-branches th.num { text-align: right; font-family: var(--mono); font-size: .82rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
.p5-branches .meth { font-weight: 650; } .p5-branches .meth small { display: block; font-weight: 400; color: var(--ink-2); font-size: .8rem; }
.p5-branches .alt { color: var(--ink-2); font-size: .84rem; }
.p5-branches tr.hyb td { background: rgba(91, 140, 255, .10); }
.p5-tag { display: inline-block; margin-left: .4rem; padding: .05rem .45rem; border-radius: 5px; background: var(--accent); color: #fff;
  font-size: .68rem; font-weight: 700; vertical-align: .12em; }
@media (max-width: 640px) {
  .p5-branches thead { display: none; }
  .p5-branches tr { display: grid; grid-template-columns: 1fr auto; gap: .2rem .75rem; padding: .85rem 1rem; border-bottom: 1px solid var(--line); }
  .p5-branches td { padding: 0 !important; border: 0 !important; background: none !important; }
  .p5-branches td.meth, .p5-branches td.alt { grid-column: 1 / 3; }
  .p5-branches td.pred { grid-column: 1; } .p5-branches td.conf { grid-column: 2; }
  .p5-branches td.time { grid-column: 1 / 3; text-align: left; color: var(--ink-2); }
  .p5-branches td.time::before { content: "time "; }
  .p5-branches tr.hyb { background: rgba(91, 140, 255, .10); }
}

/* Classical pipeline stages */
.p5-stages { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1.1rem .9rem; margin: 0; padding: 0; list-style: none; }
.p5-stages li { margin: 0; padding: 0; }
.p5-stages figure { margin: 0; }
.p5-stages .im { border-radius: 10px; overflow: hidden; border: 1px solid var(--line); background: #000; }
.p5-stages img { width: 100%; aspect-ratio: 2 / 1; object-fit: cover; display: block; transition: transform .7s var(--ease); }
.p5-stages li:hover img { transform: scale(1.07); }
.p5-stages figcaption { font-size: .82rem; color: var(--ink-2); margin-top: .45rem; }
.p5-stages figcaption span { font-family: var(--mono); color: var(--accent-2); margin-right: .35rem; }
@media (max-width: 900px) { .p5-stages { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .p5-stages { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* Model comparison */
.p5-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 0 0 2rem; padding: 0; list-style: none; }
.p5-card { position: relative; margin: 0; padding: 1.25rem 1.3rem 1.15rem; border-radius: 18px; background: var(--glass);
  border: 1px solid var(--line); backdrop-filter: blur(12px); overflow: hidden;
  transition: transform .45s var(--ease), border-color .3s, box-shadow .45s var(--ease); }
.p5-card:hover { transform: translateY(-4px); border-color: var(--line-2); box-shadow: 0 26px 60px -34px rgba(91, 140, 255, .45); }
.p5-card.win { background: linear-gradient(160deg, rgba(91, 140, 255, .16), transparent 55%), var(--glass-2); border-color: rgba(91, 140, 255, .5); }
.p5-card h4 { margin: 0 0 .15rem; padding: 0; font-size: 1.05rem; font-weight: 700; font-stretch: 110%; }
.p5-card p { margin: 0 0 1rem; color: var(--ink-2); font-size: .84rem; min-height: 2.5em; }
.p5-card .big { font-size: 2.6rem; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; font-stretch: 112%; }
.p5-card .big small { font-size: .85rem; font-weight: 500; color: var(--ink-2); margin-left: .35rem; font-stretch: 100%; }
.p5-card dl { display: grid; grid-template-columns: auto 1fr; gap: .3rem .9rem; margin: 1rem 0 0; padding: .85rem 0 0 !important;
  border-top: 1px solid var(--line); font-size: .82rem; }
.p5-card dt { color: var(--ink-2); margin: 0; } .p5-card dd { margin: 0; text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.p5-card .good { color: var(--ok); }
@media (max-width: 820px) { .p5-cards { grid-template-columns: 1fr; } }
[data-testid="stMarkdownContainer"] table { font-variant-numeric: tabular-nums; font-size: .9rem; }
[data-testid="stMarkdownContainer"] th { background: rgba(255, 255, 255, .03); font-weight: 650; }
.st-key-plots [data-testid="stImage"] img, .st-key-arch [data-testid="stImage"] img { background: #fff; padding: 14px; border-radius: var(--r);
  box-shadow: 0 30px 70px -40px rgba(0, 0, 0, .9); }

/* About: pipeline flow */
.p5-flow { position: relative; display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1rem; margin: .5rem 0 2.25rem; padding: 0 !important; list-style: none; }
.p5-flow::before { content: ""; position: absolute; left: 6%; right: 6%; top: 1.45rem; height: 2px; z-index: 0;
  background: repeating-linear-gradient(90deg, var(--accent) 0 10px, transparent 10px 18px); background-size: 18px 2px;
  opacity: .55; animation: p5-flow 1.2s linear infinite; }
@keyframes p5-flow { to { background-position: 18px 0; } }
.p5-flow li { position: relative; z-index: 1; margin: 0; padding: 0; }
.p5-flow .node { width: 2.9rem; height: 2.9rem; border-radius: 50%; display: grid; place-items: center; margin-bottom: .8rem;
  background: var(--bg-2); border: 1.5px solid var(--accent); color: var(--accent-2); font-family: var(--mono); font-size: .85rem;
  box-shadow: 0 0 0 6px var(--bg), 0 0 24px -4px var(--accent-glow); }
.p5-flow li:last-child .node { background: var(--accent); color: #fff; }
.p5-flow h4 { margin: 0 0 .3rem; padding: 0; font-size: 1rem; font-weight: 700; font-stretch: 108%; }
.p5-flow p { margin: 0; color: var(--ink-2); font-size: .85rem; line-height: 1.5; }
@media (max-width: 900px) {
  .p5-flow { grid-template-columns: 1fr; gap: 1.1rem; }
  .p5-flow::before { left: 1.4rem; right: auto; top: 1rem; bottom: 1rem; width: 2px; height: auto;
    background: repeating-linear-gradient(180deg, var(--accent) 0 10px, transparent 10px 18px); background-size: 2px 18px; animation-name: p5-flowv; }
  .p5-flow li { display: grid; grid-template-columns: 2.9rem 1fr; gap: 0 1rem; }
  .p5-flow .node { grid-row: 1 / 3; margin: 0; }
}
@keyframes p5-flowv { to { background-position: 0 18px; } }

/* Sidebar legend */
.p5-stats, .p5-meta, .p5-rank, .p5-stages, .p5-legend, .p5-cards, .p5-flow { padding-left: 0 !important; margin-left: 0 !important; }
.p5-legend { list-style: none; margin: .25rem 0 0; padding: 0; font-size: .85rem; }
.p5-legend li { display: flex; align-items: center; gap: .55rem; margin: 0 0 .4rem; }
.p5-legend i { width: .8rem; height: .8rem; border-radius: 3px; border: 2px solid; flex: none; }
.p5-foot { margin: 3.5rem 0 1rem; padding-top: 1.25rem; border-top: 1px solid var(--line); color: var(--ink-3); font-size: .78rem; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-delay: 0s !important; transition-duration: .01ms !important; }
  .p5-flow::before, .p5-pulse::after { animation: none !important; }
  .st-key-result [data-testid="stImage"]::after { display: none; }
}
</style>
"""

# Motion runtime lives in static/p5fx.js: Streamlit's sanitiser drops inline <script>/<style>
# whose text contains "<", so the script is loaded by URL instead.
_JS = f'<script type="module" src="{STATIC}/p5fx.js"></script>'


def hero(facts: list[tuple[str, str, dict]]) -> str:
    """facts: (display value, label, count-up attrs such as {"to": 94.8, "dec": 1, "suf": "%"})."""
    items = []
    for value, label, count in facts:
        attrs = "".join(f' data-{k}="{esc(str(v))}"' for k, v in count.items())
        items.append(f'<li><span class="p5-num"{attrs}>{esc(value)}</span><small>{esc(label)}</small></li>')
    x1, y1, x2, y2 = HERO_BOX
    fbox = f"left:{x1:.2%};top:{y1:.2%};width:{x2 - x1:.2%};height:{y2 - y1:.2%}"
    return (
        '<section class="p5-hero" id="p5-hero">'
        '<div class="p5-hero-copy">'
        '<h1 class="p5-title"><span class="ln"><span>Name the car.</span></span>'
        '<span class="ln"><span>Down to the year.</span></span></h1>'
        '<p class="p5-lede">Upload a photo. A detector finds the car, a <b>classical vision pipeline</b> and a '
        '<b>ConvNeXt network</b> both read it, and a calibrated hybrid decides how specific it can safely be: '
        'make, model and year, the make only, or an honest “not sure”.</p>'
        '<div class="p5-cta"><a class="p5-btn primary" href="#" data-act="try">Recognise a car <span class="arr">→</span></a>'
        '<a class="p5-btn ghost" href="#" data-act="tab:About">How it works</a></div>'
        f'<ul class="p5-stats">{"".join(items)}</ul></div>'
        f'<div class="p5-stage" id="p5-stage" data-img="{STATIC}/hero_tesla.jpg" data-box="{list(HERO_BOX)}">'
        f'<div class="p5-fallback"><img src="{STATIC}/hero_tesla.jpg" alt="" width="1280" height="634">'
        f'<span class="p5-fbox" style="{fbox}"></span></div>'
        f'<span class="p5-tag3d"><i></i>{esc(HERO_LABEL)}</span>'
        '<p class="p5-stage-cap">Real output of this app on an example photo (Wikimedia Commons, CC BY-SA 4.0)</p>'
        "</div></section>" + _JS)


def legend() -> str:
    rows = [("var(--ok)", "Full answer: make, model, year"),
            ("var(--warn)", "Manufacturer only"),
            ("var(--bad)", "Uncertain: see the top-5"),
            ("#A0A0A0", "Other vehicles (ignored)")]
    return '<ul class="p5-legend">' + "".join(
        f'<li><i style="border-color:{c}"></i>{esc(t)}</li>' for c, t in rows) + "</ul>"


def section(num: str, title: str, lead: str = "") -> str:
    return (f'<div class="p5-sec"><span class="k">{esc(num)}</span><h3>{esc(title)}</h3></div>'
            + (f'<p class="p5-secp">{esc(lead)}</p>' if lead else ""))


def empty_state() -> str:
    return ('<div class="p5-empty"><span class="p5-pulse" aria-hidden="true"></span><p style="margin:0">'
            "<b>Upload a photo of a car, or pick an example above.</b> The model knows 196 cars, "
            "mostly US-market models from 1991 to 2012.</p></div>")


def _ring(p: float, label: str) -> str:
    v = round(min(max(p, 0.0), 1.0) * 100)
    return f'<div class="p5-ring" style="--to:{v}"><b>{v}%</b><small>{esc(label)}</small></div>'


def verdict(res, meta_for, tau_c: float, tau_m: float) -> str:
    label, kind = LEVELS.get(res.level, LEVELS["uncertain"])
    chip = f'<span class="p5-chip"><i></i>{esc(label)}</span>'
    if res.level == "no_car":
        return (f'<section class="p5-verdict none" aria-live="polite">{chip}'
                '<p class="p5-name">No car detected</p>'
                '<p class="p5-sub">Try a photo where one car is clearly visible and fills more of the frame.</p>'
                "</section>")

    top_name, top_p = res.top5[0]
    m = meta_for(top_name)
    ring = _ring(top_p, "confidence")
    if res.level == "model":
        name = f"{m['make']} {m['model']}" if m else top_name
        sub = (f"Model year <b>{esc(str(m['year']))}</b>. The calibrated confidence clears the full-answer "
               f"threshold ({tau_c:.2f}).") if m else ""
    elif res.level == "make":
        name = res.make
        ring = _ring(res.make_prob, "make conf.")
        sub = (f"The exact model is below its threshold, so only the manufacturer is named. "
               f"Best guess {esc(top_name)} (<b>{top_p:.0%}</b>).")
    else:
        name = "Not sure enough to name it"
        sub = (f"Best guess {esc(top_name)} at <b>{top_p:.0%}</b>. The model knows 196 cars, mostly from 1991 to 2012; "
               "newer or unseen models land here instead of getting a confident wrong answer.")

    t = res.timings
    meta = [("box", f"{res.box_score:.0%}"), ("detector", f"{t['detector_ms']:.0f} ms"),
            ("ConvNeXt INT8", f"{t['deep_ms']:.0f} ms"), ("classical", f"{t['classical_ms']:.0f} ms"),
            ("τ", f"{tau_c:.2f} / {tau_m:.2f}")]
    meta_html = "".join(f"<li>{esc(k)} <b>{esc(v)}</b></li>" for k, v in meta)
    if res.fused:
        meta_html += "<li><b>classical scores fused</b> (deep confidence below the gate)</li>"
    if res.other_boxes:
        meta_html += f"<li><b>{len(res.other_boxes)}</b> other vehicle(s) ignored</li>"
    return (f'<section class="p5-verdict {kind}" aria-live="polite"><div class="p5-vtop">{ring}<div>{chip}'
            f'<p class="p5-name">{esc(str(name))}</p></div></div>'
            + (f'<p class="p5-sub">{sub}</p>' if sub else "")
            + f'<ul class="p5-meta" aria-label="Detection and timing details">{meta_html}</ul></section>')


def ranked(items, title: str) -> str:
    rows = []
    for i, (name, p) in enumerate(items, start=1):
        p = min(max(float(p), 0.0), 1.0)
        rows.append(f'<li><span class="n">{i}</span><span class="t">{esc(name)}</span><span class="v">{p:.1%}</span>'
                    f'<span class="p5-bar" aria-hidden="true"><i style="width:{max(p * 100, 0.6):.1f}%"></i></span></li>')
    return f'<p class="p5-h">{esc(title)}</p><ol class="p5-rank">{"".join(rows)}</ol>'


def branches(res) -> str:
    rows = []
    for full, b in res.per_branch.items():
        title, _, detail = full.partition(" (")
        detail = detail.rstrip(")")
        (n1, p1), (n2, p2), (n3, p3) = b["top3"]
        hyb = title.lower().startswith("hybrid")
        cls, tag = (' class="hyb"', '<span class="p5-tag">answer</span>') if hyb else ("", "")
        rows.append(
            f"<tr{cls}><td class=\"meth\">{esc(title)}{tag}<small>{esc(detail)}</small></td>"
            f'<td class="pred">{esc(n1)}</td><td class="num conf">{p1:.0%}</td>'
            f'<td class="alt">{esc(n2)} ({p2:.0%}) · {esc(n3)} ({p3:.0%})</td>'
            f'<td class="num time">{b["ms"]:.0f} ms</td></tr>')
    return ('<table class="p5-branches"><thead><tr><th scope="col">Approach</th><th scope="col">Top-1 prediction</th>'
            '<th scope="col" class="num">Conf.</th><th scope="col">Runners-up</th><th scope="col" class="num">Time</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')


def _data_uri(arr: np.ndarray, width: int = 420) -> str:
    im = Image.fromarray(np.asarray(arr))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def stages(stage_images: dict) -> str:
    figs = []
    for key, arr in stage_images.items():
        num, _, label = key.partition(". ")
        figs.append(f'<li><figure><div class="im"><img src="{_data_uri(arr)}" alt="{esc(label or key)}" loading="lazy"></div>'
                    f"<figcaption><span>{esc(num)}</span>{esc(label or key)}</figcaption></figure></li>")
    return f'<ul class="p5-stages">{"".join(figs)}</ul>'


def compare_cards(rows: list[tuple[str, str, dict]], best_ece: str) -> str:
    """rows: (title, one-line description, metrics json). The card with the lowest ECE is highlighted."""
    cards = []
    for title, desc, m in rows:
        c = m["classification"]
        top1 = c["top1"] * 100
        win = " win" if title == best_ece else ""
        ece_cls = ' class="good"' if win else ""
        cards.append(
            f'<li class="p5-card{win}"><h4>{esc(title)}</h4><p>{esc(desc)}</p>'
            f'<div class="big"><span data-to="{top1:.1f}" data-dec="1" data-suf="%">{top1:.1f}%</span><small>top-1</small></div>'
            f'<dl><dt>Top-5</dt><dd>{c["top5"]:.1%}</dd><dt>Make top-1</dt><dd>{c["make_top1"]:.1%}</dd>'
            f'<dt>ECE (lower is better)</dt><dd{ece_cls}>{c["ece"]:.3f}</dd></dl></li>')
    return f'<ul class="p5-cards">{"".join(cards)}</ul>'


def pipeline() -> str:
    steps = [("Detect", "YOLO11s, fine-tuned on Stanford Cars, finds the most prominent car and crops it with 12% padding."),
             ("Classical CV", "CLAHE, gamma, GrabCut and Canny feed colour, LBP/GLCM texture, HOG and shape features into an RBF SVM."),
             ("Deep network", "ConvNeXt-Base at 448 px with a manufacturer head, exported to ONNX and quantised to INT8."),
             ("Calibrate and fuse", "Temperature-scaled logits; when the network is under 50% sure, the SVM's scores are fused in."),
             ("Answer policy", "Calibrated confidence picks the level: make, model and year; make only; or the top-5.")]
    lis = "".join(f'<li><span class="node">{i:02d}</span><h4>{esc(t)}</h4><p>{esc(d)}</p></li>'
                  for i, (t, d) in enumerate(steps, start=1))
    return f'<ol class="p5-flow">{lis}</ol>'


def footer() -> str:
    return ('<p class="p5-foot">Example and hero photos: Wikimedia Commons, CC BY-SA 4.0 (Peteratkins, Vauxford, '
            "Alexander Migl); the background is a tinted derivative of the Mustang photo, same licence. "
            "Trained on Stanford Cars (196 classes).</p>")
