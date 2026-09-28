"""dynamic_rhyme_pipeline.py -- Dynamic nursery-rhyme video orchestrator.

Takes ANY rhyme/song text, decomposes it into pacing-aware scenes, locks a
master character anchor, and generates clips through the local official-Veo
daemon (server.py) chaining each scene off the previous clip's last frame
for unbroken visual flow.

Contract (all served by server.py over the official google-genai SDK):
    POST {base}/api/v1/characters/create
    POST {base}/api/v1/scenes/generate      (reference_image_path = seed frame)
    GET  {base}/api/v1/tasks/{task_id}      (poll until COMPLETED / FAILED)

Usage:
    python dynamic_rhyme_pipeline.py --preset english
    python dynamic_rhyme_pipeline.py --preset hindi
    python dynamic_rhyme_pipeline.py --input lyrics.txt
    python dynamic_rhyme_pipeline.py --text "Ek do teen char, aao gine mere yaar"
    python dynamic_rhyme_pipeline.py --text "..." --plan-only   # no dispatch

Requires: pip install httpx   (and a running `python server.py` for dispatch)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

log = logging.getLogger("rhyme_pipeline")

API_BASE_DEFAULT = "http://127.0.0.1:8080"
SCENE_TIMEOUT_SECONDS = 300
POLL_BACKOFF_SECONDS = (3, 5, 8)  # then stays at 8

STYLE_BLOCK = (
    "High-end 3D Pixar-Disney animation, vibrant studio lighting, soft rounded "
    "toy-like rendering, vertical 9:16 framing. Strictly no captions, subtitles "
    "or written lyrics on screen; any numeral shown must be clean English "
    "typography only."
)

DEFAULT_CHARACTER_NAME = "The Numberlings"
DEFAULT_CHARACTER_ANCHOR = (
    "Five anthropomorphic 3D number characters shaped like the English numerals "
    "1 to 5: glossy candy-red, amber, teal, violet and lime bodies with soft "
    "rounded toy-like proportions, oversized sparkling amber eyes, tiny white "
    "gloves, and matching midnight-blue satin jackets with golden buttons. "
    "Identical design, colors, outfits, eye style and proportions in every "
    "single shot; they never change appearance between scenes."
)

PRESETS: dict[str, str] = {
    "english": (
        "One, two, buckle my shoe;\n"
        "Three, four, shut the door;\n"
        "Five, six, pick up sticks;\n"
        "Seven, eight, lay them straight;\n"
        "Nine, ten, a big fat hen!"
    ),
    "hindi": (
        "Ek do teen char, aao gine mere yaar!\n"
        "Paanch chhe saat aath, ginti karein saath saath!\n"
        "Nau das poora hua, sab ne milkar dhamaal machaaya!"
    ),
}

# Number words for counting-rhyme detection (English + romanized Hindi).
NUMBER_WORDS: dict[str, str] = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4",
    "paanch": "5", "pach": "5", "chhe": "6", "chhah": "6", "saat": "7",
    "aath": "8", "nau": "9", "das": "10",
}
ACTION_RE = re.compile(
    r"\b(jump|run|roll|spin|dance|clap|fly|swim|wag|splash|buckle|shut|pick|"
    r"lay|count|hide|skip|hop|bounce|march|stomp|nach|kud|bhaag|uda|macha)\w*",
    re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\b(10|[1-9])\b")


# --------------------------------------------------------------------------
# Scene analysis & decomposition
# --------------------------------------------------------------------------


@dataclass
class ScenePlan:
    index: int
    source_line: str
    title: str
    prompt: str
    duration_seconds: int
    pacing_note: str
    numerals: list[str]


def split_into_units(text: str) -> list[str]:
    """Stanza -> line -> comma-beats for over-long lines."""
    units: list[str] = []
    for stanza in re.split(r"\n\s*\n", text.strip()):
        for line in stanza.splitlines():
            line = line.strip().rstrip(";,.!") or ""
            if not line:
                continue
            if len(line) > 90:
                units.extend(b.strip() for b in line.split(",") if b.strip())
            else:
                units.append(line)
    return units


def detect_numerals(line: str) -> list[str]:
    """Return unique numerals referenced by a line, in spoken order."""
    found: list[str] = []
    for token in re.findall(r"[A-Za-z]+|\d+", line.lower()):
        numeral = NUMBER_WORDS.get(token) or (
            token if DIGIT_RE.fullmatch(token) else None
        )
        if numeral and numeral not in found:
            found.append(numeral)
    return found


def classify_pacing(line: str, numerals: list[str]) -> tuple[int, str]:
    """Heuristic beat analysis -> (duration in 4..10s, human pacing note)."""
    word_count = len(line.split())
    has_action = bool(ACTION_RE.search(line))
    exclam = "!" in line
    if word_count <= 6 and (numerals or exclam):
        return (4 if word_count <= 4 else 5), "short rhythmic counting cut"
    if has_action and word_count >= 10:
        return 10, "action-heavy narrative beat"
    if has_action or word_count >= 12:
        return 8, "action/narrative beat"
    return 6, "steady narrative beat"


def build_prompt(line: str, numerals: list[str]) -> str:
    numeral_clause = (
        " Featured numeral characters: "
        + ", ".join(numerals)
        + ", rendered as clean English typography on their bodies and glowing "
        "softly in the background."
        if numerals else ""
    )
    return (
        f"{STYLE_BLOCK} Visual beat inspired by the verse \u201c{line}\u201d: "
        "the characters act its rhythm and meaning out through gesture, motion "
        "and dance, with a clear beginning, action and payoff inside one shot. "
        "Continue seamlessly from the previous frame when one is provided."
        f"{numeral_clause}"
    )


def analyze(text: str) -> list[ScenePlan]:
    plans: list[ScenePlan] = []
    for index, line in enumerate(split_into_units(text), start=1):
        numerals = detect_numerals(line)
        duration, note = classify_pacing(line, numerals)
        duration = max(4, min(10, duration))
        plans.append(
            ScenePlan(
                index=index,
                source_line=line,
                title=f"Scene {index}: {line[:48]}{'...' if len(line) > 48 else ''}",
                prompt=build_prompt(line, numerals),
                duration_seconds=duration,
                pacing_note=note,
                numerals=numerals,
            )
        )
    if not plans:
        raise ValueError("no scene units could be parsed from the input text")
    return plans


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------


class DaemonClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str) -> None:
        self.http = http
        self.base = base_url.rstrip("/")

    async def preflight(self) -> dict:
        r = await self.http.get(f"{self.base}/health")
        r.raise_for_status()
        health = r.json()
        if not health.get("gemini_api_key_configured"):
            raise RuntimeError(
                "daemon is up but GEMINI_API_KEY is not configured on it"
            )
        return health

    async def create_character(self, name: str, anchor_prompt: str) -> str:
        r = await self.http.post(
            f"{self.base}/api/v1/characters/create",
            json={"name": name, "anchor_prompt": anchor_prompt},
        )
        r.raise_for_status()
        character_id = r.json()["character_id"]
        log.info("character anchor LOCKED: %s (%s)", name, character_id[:8])
        return character_id

    async def submit_scene(
        self,
        prompt: str,
        duration_seconds: int,
        title: str,
        character_id: str,
        reference_image_path: str | None,
    ) -> str:
        r = await self.http.post(
            f"{self.base}/api/v1/scenes/generate",
            json={
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "title": title,
                "character_id": character_id,
                "reference_image_path": reference_image_path,
            },
        )
        if r.status_code != 202:
            raise RuntimeError(f"scene submission rejected ({r.status_code}): {r.text}")
        return r.json()["task_id"]

    async def poll_task(self, task_id: str, label: str) -> dict:
        """Poll with 3s -> 5s -> 8s backoff until terminal state or timeout."""
        deadline = time.monotonic() + SCENE_TIMEOUT_SECONDS
        delay_index = 0
        started = time.monotonic()
        while True:
            r = await self.http.get(f"{self.base}/api/v1/tasks/{task_id}")
            r.raise_for_status()
            task = r.json()
            if task["status"] in ("COMPLETED", "FAILED"):
                return task
            if time.monotonic() > deadline:
                return {
                    "status": "FAILED",
                    "error": f"timed out after {SCENE_TIMEOUT_SECONDS}s",
                }
            delay = (
                POLL_BACKOFF_SECONDS[delay_index]
                if delay_index < len(POLL_BACKOFF_SECONDS)
                else POLL_BACKOFF_SECONDS[-1]
            )
            delay_index += 1
            log.info(
                "%s rendering... %.0fs elapsed (next poll in %ds)",
                label, time.monotonic() - started, delay,
            )
            await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


async def run_pipeline(args: argparse.Namespace) -> int:
    if args.text:
        source_text = args.text
    elif args.input:
        source_text = Path(args.input).read_text(encoding="utf-8")
    else:
        source_text = PRESETS[args.preset]
    log.info("input: %s", args.text or args.input or f"preset:{args.preset}")

    plans = analyze(source_text)
    log.info(
        "decomposed into %d scene(s); total runtime ~%ds",
        len(plans), sum(p.duration_seconds for p in plans),
    )
    for p in plans:
        log.info(
            "  [scene %d] %2ds | %-28s | %s",
            p.index, p.duration_seconds, p.pacing_note, p.source_line,
        )
    if args.plan_only:
        return 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:
        client = DaemonClient(http, args.api_base)
        try:
            health = await client.preflight()
        except (httpx.HTTPError, RuntimeError) as exc:
            log.error("daemon preflight failed at %s: %s", args.api_base, exc)
            return 2
        log.info(
            "daemon healthy (model=%s, ffmpeg=%s)",
            health.get("model"), health.get("ffmpeg_available"),
        )

        character_id = await client.create_character(
            args.character_name, DEFAULT_CHARACTER_ANCHOR
        )

        results: list[dict] = []
        last_frame_path: str | None = None
        for plan in plans:
            log.info("=" * 68)
            log.info(
                "[scene %d/%d] dispatching | %ds | %s | anchor=%s | seed=%s",
                plan.index, len(plans), plan.duration_seconds, plan.pacing_note,
                args.character_name,
                Path(last_frame_path).name if last_frame_path else "none",
            )
            record = asdict(plan) | {"task_id": None, "status": None,
                                     "video_path": None, "error": None}
            try:
                task_id = await client.submit_scene(
                    prompt=plan.prompt,
                    duration_seconds=plan.duration_seconds,
                    title=plan.title,
                    character_id=character_id,
                    reference_image_path=last_frame_path,
                )
                record["task_id"] = task_id
                task = await client.poll_task(task_id, f"[scene {plan.index}]")
                record["status"] = task["status"]
                if task["status"] == "COMPLETED":
                    output = task.get("output") or {}
                    record["video_path"] = output.get("local_path")
                    log.info(
                        "[scene %d/%d] SAVED -> %s (download: %s)",
                        plan.index, len(plans),
                        output.get("local_path"), output.get("download_url"),
                    )
                    next_frame = task.get("last_frame_path")
                    if next_frame:
                        last_frame_path = next_frame
                        log.info("[scene %d] next scene seeds from %s",
                                 plan.index, next_frame)
                    else:
                        last_frame_path = None
                        log.warning(
                            "[scene %d] no last frame available; next scene "
                            "continues from text only", plan.index,
                        )
                else:
                    record["error"] = task.get("error")
                    log.error("[scene %d/%d] FAILED: %s",
                              plan.index, len(plans), task.get("error"))
                    last_frame_path = None
                    if args.stop_on_error:
                        log.error("aborting (--stop-on-error)")
                        results.append(record)
                        break
            except httpx.HTTPError as exc:
                record["status"] = "FAILED"
                record["error"] = f"network error: {exc}"
                log.error("[scene %d] network error: %s", plan.index, exc)
                if args.stop_on_error:
                    results.append(record)
                    break
            results.append(record)

    completed = sum(1 for r in results if r["status"] == "COMPLETED")
    log.info("=" * 68)
    log.info("SUMMARY: %d/%d scenes completed", completed, len(plans))
    for r in results:
        log.info("  scene %d [%s] %s", r["index"], r["status"],
                 r["video_path"] or r["error"] or "")

    manifest_dir = Path("./output/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"run_{int(time.time())}.json"
    manifest.write_text(
        json.dumps(
            {"character_name": args.character_name, "scenes": results},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("manifest written: %s", manifest.resolve())
    return 0 if completed == len(plans) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic nursery rhyme video generation pipeline"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="rhyme text inline")
    source.add_argument("--input", help="path to a lyrics/story file")
    source.add_argument("--preset", choices=sorted(PRESETS),
                        help=f"built-in rhyme ({', '.join(sorted(PRESETS))})")
    parser.add_argument("--api-base", default=API_BASE_DEFAULT,
                        help=f"daemon base URL (default {API_BASE_DEFAULT})")
    parser.add_argument("--character-name", default=DEFAULT_CHARACTER_NAME,
                        help="display name for the character anchor")
    parser.add_argument("--plan-only", action="store_true",
                        help="analyze and print the scene plan without dispatching")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="abort the run on the first failed scene")
    parser.set_defaults(preset=None)
    args = parser.parse_args(argv)
    if args.text is None and args.input is None and args.preset is None:
        args.preset = "english"
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    raise SystemExit(asyncio.run(run_pipeline(parse_args())))
