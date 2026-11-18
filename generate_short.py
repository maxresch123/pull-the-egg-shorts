#!/usr/bin/env python3
"""
Pull The Egg - automatischer YouTube-Shorts / TikTok Renderer
=============================================================

Liest scripts.json und erzeugt pro Eintrag ein fertiges 1080x1920 / 60 fps Short:

  * Gameplay-Clip -> 9:16 (Layout "blur": ganzes Bild + unscharfer Hintergrund,
                           Layout "crop": scharfer Center-Crop)
  * Voiceover mit edge-tts (kostenlos) inkl. Wort-Timestamps (WordBoundary)
  * Video wird exakt auf die Länge des Voiceovers gebracht (Schnitt / leichtes
    Strecken / Loop)
  * Punch-Zoom alle 2,5 s (1.0x -> 1.08x)
  * Fester Hook-Banner oben + wortweise aufpoppende Untertitel (aktuelles Wort gelb)
  * Optionale Hintergrundmusik (assets/bg_music.mp3, 12 % Lautstärke)
  * Ausgabe: output/{id}_final.mp4 (H.264, AAC, 60 fps, yuv420p, faststart)

Aufruf:
  python generate_short.py                 # alle Einträge rendern
  python generate_short.py --only egg_001  # nur bestimmte IDs (kommagetrennt)
  python generate_short.py --skip-existing # vorhandene Outputs überspringen
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from proglog import ProgressBarLogger

import edge_tts
import imageio_ffmpeg
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout, audio_loop, volumex
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import VideoClip

# --------------------------------------------------------------------------- #
# Einstellungen
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
SCRIPTS_FILE = ROOT / "scripts.json"
OUTPUT_DIR = ROOT / "output"
BG_MUSIC = ROOT / "assets" / "bg_music.mp3"

W, H = 1080, 1920
FPS = 60

DEFAULT_VOICE = "en-US-BrianNeural"
DEFAULT_RATE = "+15%"
DEFAULT_LAYOUT = "blur"            # "blur" oder "crop"

MUSIC_VOLUME = 0.12                # 12 %
VOICE_VOLUME = 1.0                 # 100 %
END_TAIL = 0.35                    # Sekunden nach dem letzten Wort

ZOOM_INTERVAL = 2.5                # alle 2,5 s ein Punch-Zoom
ZOOM_MAX = 1.08                    # 1.0x -> 1.08x
ZOOM_PUNCH_IN = 0.18               # Dauer des "Reinschnappens"

MIN_STRETCH = 0.85                 # Clip darf bis 0.85x verlangsamt werden, sonst Loop

HOOK_FONT_SIZE = 100
HOOK_MIN_FONT_SIZE = 62
HOOK_TOP = 190
HOOK_BOX_COLOR = (232, 24, 40, 240)
HOOK_TEXT_COLOR = (255, 255, 255)

CAPTION_FONT_SIZE = 124
CAPTION_MAX_WORDS = 3
CAPTION_MAX_CHARS = 16
CAPTION_MAX_WIDTH = 980
CAPTION_Y = {"crop": 0.58, "blur": 0.70}   # vertikale Mitte der Untertitel (Anteil von H)
CAPTION_WHITE = (255, 255, 255)
CAPTION_YELLOW = (255, 226, 0)
CAPTION_ACTIVE_SCALE = 1.06
CAPTION_POP_TIME = 0.13

STROKE_COLOR = (0, 0, 0)

FONT_CANDIDATES = [
    ROOT / "assets" / "font.ttf",
    ROOT / "assets" / "Anton-Regular.ttf",
    Path("/usr/share/fonts/truetype/google-fonts/ofl/anton/Anton-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
]
EMOJI_FONT_CANDIDATES = [
    ROOT / "assets" / "NotoColorEmoji.ttf",
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
]

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "☀-➿"
    "⬀-⯿"
    "←-⇿"
    "⌀-⏿"
    "️‍"
    "]+"
)
RATE_RE = re.compile(r"^[+-]\d{1,3}%$")
SENTENCE_END = ".!?…"
KEEP_PUNCT = "!?"

FFMPEG = os.environ.get("IMAGEIO_FFMPEG_EXE") or imageio_ffmpeg.get_ffmpeg_exe()


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Easing
# --------------------------------------------------------------------------- #
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ease_out_cubic(x: float) -> float:
    return 1 - (1 - x) ** 3


def ease_in_out_sine(x: float) -> float:
    return -(math.cos(math.pi * x) - 1) / 2


def ease_out_back(x: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def punch_zoom(t: float) -> float:
    """Alle ZOOM_INTERVAL Sekunden: schnell auf ZOOM_MAX, danach weich zurück auf 1.0."""
    local = t % ZOOM_INTERVAL
    if local < ZOOM_PUNCH_IN:
        return 1.0 + (ZOOM_MAX - 1.0) * ease_out_cubic(local / ZOOM_PUNCH_IN)
    p = (local - ZOOM_PUNCH_IN) / (ZOOM_INTERVAL - ZOOM_PUNCH_IN)
    return ZOOM_MAX - (ZOOM_MAX - 1.0) * ease_in_out_sine(p)


# --------------------------------------------------------------------------- #
# Text-Rendering (Pillow, kein ImageMagick nötig) inkl. Farb-Emojis
# --------------------------------------------------------------------------- #
def first_existing(paths) -> Path | None:
    for p in paths:
        if Path(p).is_file():
            return Path(p)
    return None


class Typography:
    def __init__(self) -> None:
        font_path = first_existing(FONT_CANDIDATES)
        if font_path is None:
            raise SystemExit(
                "Keine Schriftart gefunden. Lege eine fette .ttf als assets/font.ttf ab."
            )
        self.font_path = str(font_path)
        log(f"[font] {self.font_path}")

        self.emoji_font = None
        emoji_path = first_existing(EMOJI_FONT_CANDIDATES)
        if emoji_path:
            try:
                # NotoColorEmoji ist ein Bitmap-Font und funktioniert nur in Größe 109
                self.emoji_font = ImageFont.truetype(str(emoji_path), 109)
                log(f"[font] Emojis: {emoji_path}")
            except OSError as exc:
                log(f"[font] Emoji-Font nicht ladbar ({exc}) - Emojis werden weggelassen")
        else:
            log("[font] Kein Emoji-Font gefunden - Emojis werden weggelassen")

        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._emoji_cache: dict[tuple[str, int], Image.Image] = {}
        self._vcrop: dict[tuple[int, int], tuple[int, int]] = {}

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def segments(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        pos = 0
        for m in EMOJI_RE.finditer(text):
            if m.start() > pos:
                out.append(("t", text[pos:m.start()]))
            if self.emoji_font is not None:
                for ch in m.group():
                    if ch not in "️‍":
                        out.append(("e", ch))
            pos = m.end()
        if pos < len(text):
            out.append(("t", text[pos:]))
        return out

    def _emoji(self, ch: str, px: int) -> Image.Image | None:
        key = (ch, px)
        if key not in self._emoji_cache:
            tmp = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((20, 20), ch, font=self.emoji_font, embedded_color=True)
            bbox = tmp.getbbox()
            if not bbox:
                self._emoji_cache[key] = None
            else:
                glyph = tmp.crop(bbox)
                scale = px / glyph.height
                self._emoji_cache[key] = glyph.resize(
                    (max(1, round(glyph.width * scale)), px), Image.Resampling.LANCZOS
                )
        return self._emoji_cache[key]

    def _emoji_px(self, size: int) -> int:
        return int(size * 0.92)

    def width(self, text: str, size: int) -> float:
        font = self.font(size)
        w = 0.0
        for kind, seg in self.segments(text):
            if kind == "t":
                w += font.getlength(seg)
            else:
                w += self._emoji_px(size) + size * 0.12
        return w

    def _vertical_crop(self, size: int, stroke: int, pad: int) -> tuple[int, int]:
        """Einheitlicher vertikaler Ausschnitt -> alle Wörter teilen dieselbe Grundlinie."""
        key = (size, stroke)
        if key not in self._vcrop:
            font = self.font(size)
            l, t, r, b = font.getbbox("ÄÖÜHgjpqy!?", stroke_width=stroke)
            self._vcrop[key] = (max(0, pad + t - 2), pad + b + 2)
        return self._vcrop[key]

    def render_line(
        self,
        text: str,
        size: int,
        fill: tuple[int, int, int],
        stroke: int,
        shadow: bool = True,
    ) -> Image.Image:
        font = self.font(size)
        pad = stroke + 12
        width = int(self.width(text, size)) + 2 * pad + 4
        height = size * 2 + 2 * pad
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        cap_l, cap_t, cap_r, cap_b = font.getbbox("H")
        x = float(pad)
        for kind, seg in self.segments(text):
            if kind == "t":
                draw.text(
                    (x, pad), seg, font=font, fill=fill + (255,),
                    stroke_width=stroke, stroke_fill=STROKE_COLOR + (255,),
                )
                x += font.getlength(seg)
            else:
                px = self._emoji_px(size)
                glyph = self._emoji(seg, px)
                x += size * 0.06
                if glyph is not None:
                    cy = pad + (cap_t + cap_b) / 2
                    img.alpha_composite(glyph, (int(x), int(cy - glyph.height / 2)))
                    x += glyph.width
                else:
                    x += px
                x += size * 0.06

        top, bottom = self._vertical_crop(size, stroke, pad)
        alpha_bbox = img.getbbox()
        left, right = (alpha_bbox[0], alpha_bbox[2]) if alpha_bbox else (0, 1)
        top = min(top, alpha_bbox[1]) if alpha_bbox else top
        bottom = max(bottom, alpha_bbox[3]) if alpha_bbox else bottom
        img = img.crop((left, top, right, bottom))

        if shadow:
            img = add_shadow(img, offset=(0, max(4, size // 18)), blur=max(4, size // 14))
        return img


def add_shadow(img: Image.Image, offset=(0, 6), blur=6, opacity=0.6) -> Image.Image:
    m = blur * 2 + max(abs(offset[0]), abs(offset[1]))
    canvas = Image.new("RGBA", (img.width + 2 * m, img.height + 2 * m), (0, 0, 0, 0))
    alpha = img.getchannel("A").point(lambda a: int(a * opacity))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 255))
    shadow.putalpha(alpha)
    canvas.alpha_composite(shadow, (m + offset[0], m + offset[1]))
    canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(img, (m, m))
    return canvas


def round_rect(size, radius, color) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=color,
        outline=(0, 0, 0, 255), width=6,
    )
    return img


def build_hook_banner(typo: Typography, text: str) -> Image.Image:
    """Fetter Titel oben: rote Box, weiße Schrift, dicke schwarze Outline, max. 2-3 Zeilen."""
    max_text_w = W - 2 * 90
    size = HOOK_FONT_SIZE
    words = text.split()

    def wrap(sz: int) -> list[str]:
        lines, cur = [], ""
        for w in words:
            test = f"{cur} {w}".strip()
            if cur and typo.width(test, sz) > max_text_w:
                lines.append(cur)
                cur = w
            else:
                cur = test
        if cur:
            lines.append(cur)
        return lines

    def ok(ls: list[str], sz: int) -> bool:
        too_wide = any(typo.width(l, sz) > max_text_w for l in ls)
        orphan = len(ls) > 1 and len(EMOJI_RE.sub("", ls[-1]).strip()) <= 3
        return len(ls) <= 2 and not too_wide and not orphan

    # größte Schrift, die in max. 2 Zeilen ohne "verwaiste" letzte Zeile passt
    lines = wrap(size)
    while size > HOOK_MIN_FONT_SIZE and not ok(lines, size):
        size -= 4
        lines = wrap(size)

    stroke = max(6, size // 11)
    rendered = [typo.render_line(l, size, HOOK_TEXT_COLOR, stroke, shadow=False) for l in lines]
    gap = -int(size * 0.08)
    text_h = sum(r.height for r in rendered) + gap * (len(rendered) - 1)
    text_w = max(r.width for r in rendered)

    pad_x, pad_y = 44, 26
    box_w = min(W - 40, text_w + 2 * pad_x)
    box_h = text_h + 2 * pad_y
    banner = round_rect((box_w, box_h), radius=36, color=HOOK_BOX_COLOR)
    y = pad_y
    for r in rendered:
        banner.alpha_composite(r, ((box_w - r.width) // 2, y))
        y += r.height + gap
    banner = banner.rotate(-2, resample=Image.Resampling.BICUBIC, expand=True)
    return add_shadow(banner, offset=(0, 10), blur=14, opacity=0.55)


# --------------------------------------------------------------------------- #
# Voiceover + Timestamps (edge-tts)
# --------------------------------------------------------------------------- #
async def _tts_stream(text: str, voice: str, rate: str, mp3_path: Path):
    words, sentences = [], []
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    except TypeError:  # ältere edge-tts Versionen kennen "boundary" nicht
        comm = edge_tts.Communicate(text, voice, rate=rate)
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            kind = chunk.get("type")
            if kind == "audio":
                f.write(chunk["data"])
            elif kind in ("WordBoundary", "SentenceBoundary"):
                start = chunk["offset"] / 1e7          # 100-ns-Ticks -> Sekunden
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                item = {"start": start, "end": end, "text": chunk["text"]}
                (words if kind == "WordBoundary" else sentences).append(item)
    return words, sentences


def voice_candidates(voice: str) -> list[str]:
    """Gewünschte Stimme zuerst; bei Störungen des Dienstes Ausweich-Stimmen."""
    cands = [voice]
    if "Multilingual" not in voice and voice.endswith("Neural"):
        cands.append(voice[: -len("Neural")] + "MultilingualNeural")
    if DEFAULT_VOICE not in cands:
        cands.append(DEFAULT_VOICE)
    return cands


def synthesize(text: str, voice: str, rate: str, mp3_path: Path, attempts: int = 3):
    """edge-tts liefert gelegentlich 'No audio was received' -> Retry + Ausweich-Stimme."""
    last_exc = None
    for cand in voice_candidates(voice):
        for attempt in range(1, attempts + 1):
            try:
                words, sentences = asyncio.run(_tts_stream(text, cand, rate, mp3_path))
                if mp3_path.stat().st_size == 0:
                    raise RuntimeError("edge-tts hat kein Audio geliefert")
                if cand != voice:
                    log(f"  [tts] WARNUNG: {voice} lieferte kein Audio - Ausweich-Stimme {cand} benutzt")
                return words, sentences, cand
            except Exception as exc:  # Netzwerk-/Dienstfehler -> erneut versuchen
                last_exc = exc
                wait = 2 * attempt
                log(f"  [tts] {cand} Versuch {attempt}/{attempts}: {exc} - warte {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"edge-tts fehlgeschlagen: {last_exc}")


def split_evenly(text: str, start: float, end: float) -> list[dict]:
    """Verteilt einen Satz proportional zur Zeichenanzahl auf seine Wörter."""
    tokens = text.split()
    if not tokens:
        return []
    total = sum(len(t) + 1 for t in tokens)
    out, cur = [], start
    for tok in tokens:
        dur = (end - start) * (len(tok) + 1) / total
        out.append({"start": cur, "end": cur + dur, "text": tok})
        cur += dur
    return out


def attach_punctuation(words: list[dict], narration: str) -> list[dict]:
    """Holt Satzzeichen aus dem Originaltext zurück (für Gruppierung und !/? in Untertiteln)."""
    low = narration.lower()
    pos = 0
    for w in words:
        idx = low.find(w["text"].lower(), pos)
        if idx < 0 or idx - pos > 40:
            w["punct"] = ""
            continue
        end = idx + len(w["text"])
        p = end
        while p < len(narration) and narration[p] in ".,!?;:…\"'":
            p += 1
        w["punct"] = narration[end:p]
        pos = p
    return words


def to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-ar", "44100", "-ac", "2", str(dst)],
        check=True,
    )


# --------------------------------------------------------------------------- #
# Untertitel
# --------------------------------------------------------------------------- #
def build_caption_groups(words: list[dict]) -> list[dict]:
    groups, cur = [], []

    def flush():
        if cur:
            groups.append({"words": list(cur)})
            cur.clear()

    for i, w in enumerate(words):
        if cur:
            chars = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
            gap = w["start"] - cur[-1]["end"]
            if len(cur) >= CAPTION_MAX_WORDS or chars > CAPTION_MAX_CHARS or gap > 0.45:
                flush()
        cur.append(w)
        if any(c in w.get("punct", "") for c in SENTENCE_END + ","):
            flush()
    flush()

    for i, g in enumerate(groups):
        g["start"] = g["words"][0]["start"]
        last_end = g["words"][-1]["end"]
        nxt = groups[i + 1]["words"][0]["start"] if i + 1 < len(groups) else None
        g["end"] = min(nxt, last_end + 0.6) if nxt is not None else last_end + 0.6
        g["end"] = max(g["end"], last_end)
    return groups


class CaptionRenderer:
    def __init__(self, typo: Typography, groups: list[dict], center_y: float):
        self.typo = typo
        self.groups = groups
        self.starts = [g["start"] for g in groups]
        self.size = CAPTION_FONT_SIZE
        self.stroke = max(8, self.size // 10)
        self.center_y = center_y
        self._scaled: dict[tuple, Image.Image] = {}
        for g in groups:
            self._layout(g)

    @staticmethod
    def display_text(w: dict) -> str:
        punct = "".join(c for c in w.get("punct", "") if c in KEEP_PUNCT)
        return (w["text"].strip(".,;:\"'") + punct).upper()

    def _layout(self, g: dict) -> None:
        white = [self.typo.render_line(self.display_text(w), self.size, CAPTION_WHITE, self.stroke)
                 for w in g["words"]]
        yellow = [self.typo.render_line(self.display_text(w), self.size, CAPTION_YELLOW, self.stroke)
                  for w in g["words"]]
        space = -int(self.size * 0.10)  # Bilder enthalten bereits Outline + Schatten-Rand

        # Zeilenumbruch
        lines, cur, cur_w = [], [], 0
        for i, img in enumerate(white):
            add = img.width + (space if cur else 0)
            if cur and cur_w + add > CAPTION_MAX_WIDTH + 60:
                lines.append(cur)
                cur, cur_w = [i], img.width
            else:
                cur.append(i)
                cur_w += add
        if cur:
            lines.append(cur)

        line_h = max(im.height for im in white)
        step = int(line_h * 0.80)
        total_h = line_h + step * (len(lines) - 1)
        top = self.center_y - total_h / 2
        centers = [None] * len(white)
        for li, idxs in enumerate(lines):
            lw = sum(white[i].width for i in idxs) + space * (len(idxs) - 1)
            x = (W - lw) / 2
            y = top + li * step
            for i in idxs:
                centers[i] = (x + white[i].width / 2, y + white[i].height / 2)
                x += white[i].width + space
        g["white"], g["yellow"], g["centers"] = white, yellow, centers

    def _scaled_img(self, gi: int, wi: int, scale: float) -> Image.Image:
        key = (gi, wi, round(scale, 2))
        if key not in self._scaled:
            base = self.groups[gi]["yellow"][wi]
            if abs(scale - 1.0) < 0.005:
                self._scaled[key] = base
            else:
                self._scaled[key] = base.resize(
                    (max(1, int(base.width * scale)), max(1, int(base.height * scale))),
                    Image.Resampling.BILINEAR,
                )
        return self._scaled[key]

    def draw(self, frame: Image.Image, t: float) -> None:
        gi = bisect.bisect_right(self.starts, t) - 1
        if gi < 0:
            return
        g = self.groups[gi]
        if t >= g["end"]:
            return
        words = g["words"]
        k = 0
        for i, w in enumerate(words):
            if w["start"] <= t:
                k = i
        # bereits gesprochene Wörter weiß
        for i in range(k):
            img = g["white"][i]
            cx, cy = g["centers"][i]
            frame.paste(img, (int(cx - img.width / 2), int(cy - img.height / 2)), img)
        # aktuelles Wort gelb mit Pop-Animation
        p = clamp01((t - words[k]["start"]) / CAPTION_POP_TIME)
        scale = (0.55 + (CAPTION_ACTIVE_SCALE - 0.55) * ease_out_back(p)) if p < 1 else CAPTION_ACTIVE_SCALE
        img = self._scaled_img(gi, k, scale)
        cx, cy = g["centers"][k]
        frame.paste(img, (int(cx - img.width / 2), int(cy - img.height / 2)), img)


# --------------------------------------------------------------------------- #
# Gameplay -> 9:16
# --------------------------------------------------------------------------- #
class Gameplay:
    def __init__(self, path: Path, duration: float, layout: str, speed: float | None,
                 start: float, focus_x: float):
        self.clip = VideoFileClip(str(path), audio=False)
        self.src_w, self.src_h = self.clip.size
        self.layout = layout
        self.focus_x = clamp01(focus_x)
        self.start = max(0.0, min(start, max(0.0, self.clip.duration - 1.0)))
        self.avail = max(0.1, self.clip.duration - self.start - 2.0 / max(self.clip.fps or 30, 1))

        if speed:
            self.speed = float(speed)
        elif self.avail >= duration:
            self.speed = 1.0
        elif self.avail >= duration * MIN_STRETCH:
            self.speed = self.avail / duration     # leicht verlangsamt, kein Loop nötig
        else:
            self.speed = 1.0                       # zu kurz -> loopen
        loops = (duration * self.speed) / self.avail
        mode = "loop" if loops > 1.0001 else "schnitt"
        log(f"  [video] {self.src_w}x{self.src_h} @ {self.clip.fps:.0f}fps, "
            f"{self.clip.duration:.1f}s -> {duration:.1f}s (speed {self.speed:.2f}x, {mode})")

        # Vordergrund-Größe für Blur-Layout: ganze Breite, ganzes UI sichtbar
        fg_h = round(W * self.src_h / self.src_w)
        if fg_h > H * 0.75:   # Hochkant-Quelle -> einfach füllen
            self.layout = "crop"
        self.fg_size = (W, fg_h - fg_h % 2)
        self.fg_y = int(H * 0.46 - self.fg_size[1] / 2)

    def src_time(self, t: float) -> float:
        return self.start + (t * self.speed) % self.avail

    def _crop_box(self, target_ratio: float, zoom: float) -> tuple[float, float, float, float]:
        sw, sh = self.src_w, self.src_h
        if sw / sh > target_ratio:
            ch = sh
            cw = sh * target_ratio
        else:
            cw = sw
            ch = sw / target_ratio
        cw /= zoom
        ch /= zoom
        cx = cw / 2 + (sw - cw) * self.focus_x
        cy = sh / 2
        return (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)

    def frame(self, t: float, zoom: float) -> Image.Image:
        src = Image.fromarray(self.clip.get_frame(self.src_time(t)))
        if self.layout == "crop":
            box = self._crop_box(W / H, zoom)
            return src.resize((W, H), Image.Resampling.BICUBIC, box=box)

        # Blur-Layout: unscharfer, abgedunkelter Hintergrund + komplettes Bild in der Mitte
        bg_box = self._crop_box(W / H, 1.0)
        small = src.resize((W // 12, H // 12), Image.Resampling.BILINEAR, box=bg_box, reducing_gap=2.0)
        small = small.filter(ImageFilter.GaussianBlur(3))
        small = Image.eval(small, lambda v: int(v * 0.55))
        canvas = small.resize((W, H), Image.Resampling.BILINEAR)

        sw, sh = self.src_w, self.src_h
        cw, ch = sw / zoom, sh / zoom
        fg_box = ((sw - cw) / 2, (sh - ch) / 2, (sw + cw) / 2, (sh + ch) / 2)
        fg = src.resize(self.fg_size, Image.Resampling.BICUBIC, box=fg_box)
        canvas.paste(fg, (0, self.fg_y))
        return canvas

    def close(self) -> None:
        self.clip.close()


class StepLogger(ProgressBarLogger):
    """Kompakter Fortschritt (alle 10 %) - hält die GitHub-Actions-Logs lesbar."""

    def __init__(self):
        super().__init__()
        self._last = -1

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar != "t" or attr != "index":
            return
        total = self.bars[bar].get("total") or 1
        pct = int(100 * value / total) // 10 * 10
        if pct != self._last:
            self._last = pct
            log(f"  [render] {pct:3d}%")


# --------------------------------------------------------------------------- #
# Audio-Mix
# --------------------------------------------------------------------------- #
def build_audio(voice_wav: Path, duration: float, music_path: Path | None, music_volume: float):
    voice = AudioFileClip(str(voice_wav))
    if VOICE_VOLUME != 1.0:
        voice = volumex(voice, VOICE_VOLUME)
    tracks = [voice]
    if music_path and music_path.is_file():
        music = AudioFileClip(str(music_path))
        if music.duration < duration:
            music = audio_loop(music, duration=duration)
        else:
            music = music.subclip(0, duration)
        music = volumex(music, music_volume)
        music = audio_fadeout(audio_fadein(music, 0.4), 0.8)
        tracks.insert(0, music)
        log(f"  [audio] Musik: {music_path.name} @ {int(music_volume * 100)}%")
    mix = CompositeAudioClip(tracks).set_duration(duration)
    mix.fps = 44100
    return mix, tracks


# --------------------------------------------------------------------------- #
# Ein Short rendern
# --------------------------------------------------------------------------- #
def safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(raw)).strip("_") or "short"


def render_short(entry: dict, typo: Typography) -> tuple[Path, dict]:
    vid = safe_id(entry["id"])
    clip_path = ROOT / entry["clip_file"]
    if not clip_path.is_file():
        raise FileNotFoundError(f"Clip nicht gefunden: {entry['clip_file']}")

    voice = entry.get("voice") or DEFAULT_VOICE
    rate = entry.get("rate") or DEFAULT_RATE
    if not RATE_RE.match(rate):
        raise ValueError(f"'rate' muss wie '+15%' oder '-5%' aussehen, nicht {rate!r}")
    layout = (entry.get("layout") or DEFAULT_LAYOUT).lower()
    if layout not in ("blur", "crop"):
        raise ValueError("'layout' muss 'blur' oder 'crop' sein")

    narration = " ".join(EMOJI_RE.sub("", entry["narration"]).split())
    hook_text = entry.get("hook_text", "").strip()
    music_path = ROOT / entry["music"] if entry.get("music") else BG_MUSIC
    music_volume = float(entry.get("music_volume", MUSIC_VOLUME))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{vid}_final.mp4"

    with tempfile.TemporaryDirectory(prefix=f"short_{vid}_") as tmp:
        tmp = Path(tmp)
        mp3, wav = tmp / "voice.mp3", tmp / "voice.wav"

        log(f"  [tts] {voice} {rate}")
        words, sentences, used_voice = synthesize(narration, voice, rate, mp3)
        to_wav(mp3, wav)
        probe = AudioFileClip(str(wav))
        audio_dur = probe.duration
        probe.close()

        if not words:
            if sentences:
                for s in sentences:
                    words += split_evenly(s["text"], s["start"], s["end"])
            else:
                words = split_evenly(narration, 0.15, max(0.5, audio_dur - 0.2))
            log("  [tts] keine Wort-Timestamps - Zeiten werden geschätzt")
        words = attach_punctuation(words, narration)
        duration = round(max(audio_dur, words[-1]["end"]) + END_TAIL, 3)
        log(f"  [tts] {len(words)} Wörter, Voiceover {audio_dur:.2f}s -> Video {duration:.2f}s")

        (OUTPUT_DIR / f"{vid}_words.json").write_text(
            json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        game = Gameplay(
            clip_path, duration, layout,
            speed=entry.get("speed"),
            start=float(entry.get("start", 0)),
            focus_x=float(entry.get("focus_x", 0.5)),
        )
        caption_y = float(entry.get("caption_y", CAPTION_Y[game.layout])) * H
        captions = CaptionRenderer(typo, build_caption_groups(words), caption_y)

        hook = build_hook_banner(typo, hook_text) if hook_text else None
        hook_cache: dict[float, Image.Image] = {}

        def make_frame(t: float) -> np.ndarray:
            frame = game.frame(t, punch_zoom(t))
            if hook is not None:
                p = clamp01(t / 0.35)
                s = round(0.6 + 0.4 * ease_out_back(p), 2) if p < 1 else 1.0
                if s not in hook_cache:
                    hook_cache[s] = hook if s == 1.0 else hook.resize(
                        (max(1, int(hook.width * s)), max(1, int(hook.height * s))),
                        Image.Resampling.BILINEAR)
                hi = hook_cache[s]
                cy = HOOK_TOP + hook.height / 2
                frame.paste(hi, (int((W - hi.width) / 2), int(cy - hi.height / 2)), hi)
            captions.draw(frame, t)
            return np.asarray(frame)

        video = VideoClip(make_frame, duration=duration)
        audio, tracks = build_audio(wav, duration, music_path, music_volume)
        video = video.set_audio(audio)

        tmp_out = tmp / "render.mp4"
        log(f"  [render] {W}x{H} @ {FPS}fps ...")
        t0 = time.time()
        video.write_videofile(
            str(tmp_out),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            audio_fps=44100,
            preset="veryfast",
            threads=os.cpu_count() or 2,
            ffmpeg_params=[
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.2",
                "-movflags", "+faststart",
            ],
            temp_audiofile=str(tmp / "temp_audio.m4a"),
            remove_temp=True,
            logger=StepLogger(),
        )
        log(f"  [render] fertig in {time.time() - t0:.0f}s")

        for tr in tracks:
            tr.close()
        game.close()
        shutil.move(str(tmp_out), out_path)

    return out_path, {"voice": used_voice, "duration": duration}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def load_entries() -> list[dict]:
    if not SCRIPTS_FILE.is_file():
        raise SystemExit(f"{SCRIPTS_FILE.name} nicht gefunden")
    data = json.loads(SCRIPTS_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("videos", [])
    entries = []
    for i, e in enumerate(data):
        missing = [k for k in ("id", "clip_file", "narration") if not e.get(k)]
        if missing:
            log(f"[warn] Eintrag #{i + 1} übersprungen - es fehlt: {', '.join(missing)}")
            continue
        entries.append(e)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull The Egg Shorts Renderer")
    ap.add_argument("--only", default=os.environ.get("ONLY_IDS", ""),
                    help="Nur diese IDs rendern (kommagetrennt)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Videos überspringen, deren Output schon existiert")
    args = ap.parse_args()

    entries = load_entries()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        entries = [e for e in entries if str(e["id"]) in only or safe_id(e["id"]) in only]
    if not entries:
        log("Nichts zu rendern.")
        return 0

    typo = Typography()
    ok, failed, skipped = [], [], []
    summary = ["| Video | Status | Länge | Stimme | Info |", "|---|---|---|---|---|"]
    for n, entry in enumerate(entries, 1):
        vid = safe_id(entry["id"])
        log(f"\n=== [{n}/{len(entries)}] {vid} ===")
        if args.skip_existing and (OUTPUT_DIR / f"{vid}_final.mp4").is_file():
            log("  existiert bereits - übersprungen")
            skipped.append(vid)
            continue
        if not (ROOT / entry["clip_file"]).is_file():
            log(f"  [skip] Clip fehlt: {entry['clip_file']}")
            skipped.append(vid)
            summary.append(f"| {vid} | ⏭️ | - | - | Clip fehlt: {entry['clip_file']} |")
            continue
        try:
            out, info = render_short(entry, typo)
            size_mb = out.stat().st_size / 1_048_576
            log(f"  -> {out.relative_to(ROOT)} ({size_mb:.1f} MB)")
            ok.append(vid)
            summary.append(f"| {vid} | ✅ | {info['duration']:.1f}s | {info['voice']} | {size_mb:.1f} MB |")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log(f"  [FEHLER] {vid}: {exc}")
            failed.append(vid)
            summary.append(f"| {vid} | ❌ | - | - | {str(exc)[:80]} |")

    log(f"\nFertig: {len(ok)} gerendert, {len(skipped)} übersprungen, {len(failed)} fehlgeschlagen")
    if failed:
        log("Fehlgeschlagen: " + ", ".join(failed))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("## Pull The Egg - gerenderte Shorts\n\n" + "\n".join(summary) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
