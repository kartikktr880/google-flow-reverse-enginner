"""rhyme_generator.py -- Nursery rhyme video production pipeline.

Generates Pixar-style 3D animated nursery rhyme scenes with Google's
official Veo API via the google-genai SDK, sequentially (concurrency = 1),
with retry + exponential backoff, structured logging, and per-scene
error boundaries.

Setup:
    pip install google-genai
    set GEMINI_API_KEY=your_key_here        (Windows, current shell)
    setx GEMINI_API_KEY "your_key_here"     (Windows, persistent)

Run:
    python rhyme_generator.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import errors, types

log = logging.getLogger("rhyme_generator")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL = "veo-3.1-fast-generate-preview"  # swap for e.g. "veo-3.0-fast-generate-001"
ASPECT_RATIO = "9:16"
OUTPUT_DIR = Path("./output/scenes")
MAX_RETRIES = 3            # attempts per scene after the first failure
BACKOFF_BASE_SECONDS = 5.0  # 5s, 10s, 20s ...
POLL_INTERVAL_SECONDS = 10.0
FAIL_FAST = False          # False: one bad scene logs and the run continues

# Shared visual style applied to every scene prompt.
STYLE = (
    "Vibrant 3D Pixar-Disney animation style: soft rounded shapes, candy-bright "
    "colors, big expressive glossy eyes, gentle cinematic key lighting, shallow "
    "depth of field. Vertical 9:16 framing. Any on-screen numerals must be "
    "clean English typography and nothing else."
)


@dataclass(frozen=True)
class Scene:
    """One nursery-rhyme shot submitted as a single generation job."""

    id: str
    title: str
    prompt: str
    duration_seconds: int = 8  # schema: 5..10; Gemini API keys serve fixed 8s
    aspect_ratio: str = ASPECT_RATIO  # "9:16" or "16:9"
    reference_image_path: str | None = None  # i2v seed frame for continuity


SCENES: list[Scene] = [
    Scene(
        id="intro",
        title="Opening -- Number Friends Hello",
        prompt=(
            "A cheerful crew of cute anthropomorphic 3D number characters -- 1, 2, 3, "
            "4 and 5 -- bounce and wave hello together on a sunny meadow stage with "
            "rainbow bunting; the big friendly numeral 5 sparkles front and center."
        ),
        duration_seconds=5,
    ),
    Scene(
        id="five_little_ducks",
        title="Five Little Ducks",
        prompt=(
            "Five fluffy 3D ducklings wearing tiny numbered bibs reading 1 to 5 waddle "
            "in a line behind Mama Duck across a wooden bridge; a carved wooden sign "
            "shows the numeral 5; the ducklings quack and waggle their tails with "
            "oversized happy eyes."
        ),
    ),
    Scene(
        id="ten_in_the_bed",
        title="Ten in the Bed",
        prompt=(
            "Ten cuddly pajama-wearing 3D number characters, labeled 10 down to 1, "
            "snuggle in one giant cozy bed under starry nursery wallpaper; they roll "
            "over one by one and the little numeral 1 giggles goodnight."
        ),
    ),
    Scene(
        id="buckle_my_shoe",
        title="One, Two, Buckle My Shoe",
        prompt=(
            "A cute 3D numeral 1 and numeral 2 character team up to buckle a giant "
            "shiny red shoe, then numerals 3 and 4 push a big nursery door shut; "
            "playful bouncy musical motion."
        ),
    ),
    Scene(
        id="hickory_dickory",
        title="Hickory Dickory Dock",
        prompt=(
            "A smiling grandfather clock strikes as the numeral 1 character, wearing "
            "tiny mouse ears, rides the pendulum down and scurries across the nursery "
            "floor; the clock face shows one bold English numeral 1."
        ),
    ),
    Scene(
        id="ten_green_bottles",
        title="Ten Green Bottles",
        prompt=(
            "Ten glossy green bottles stand on a nursery wall, each wearing a big "
            "white English numeral from 1 to 10; one wobbles cartoonishly and tips "
            "off with a soft comedic bounce while the others gasp with wide eyes."
        ),
    ),
]


class SceneGenerationError(RuntimeError):
    """A single scene failed after retries (or for a non-retryable reason)."""


def _is_retryable(exc: Exception) -> bool:
    """Transient failures worth retrying: rate limits, server errors, network."""
    if isinstance(exc, errors.APIError):
        return exc.code in {429, 500, 502, 503, 504} or exc.code is None
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return "timeout" in str(exc).lower() or "temporarily" in str(exc).lower()


class RhymeOrchestrator:
    """Sequential scene-by-scene video generation against the official Veo API."""

    def __init__(
        self,
        client: genai.Client,
        output_dir: Path = OUTPUT_DIR,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = BACKOFF_BASE_SECONDS,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def run(self, scenes: list[Scene]) -> list[tuple[Scene, Path]]:
        """Generate every scene in order; return (scene, saved_path) successes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: list[tuple[Scene, Path]] = []
        total = len(scenes)

        for index, scene in enumerate(scenes, start=1):
            log.info("=" * 64)
            log.info("[%d/%d] START scene %s -- %s", index, total, scene.id, scene.title)
            started = time.monotonic()
            try:
                path = await self.generate_scene(scene, index)
            except SceneGenerationError as exc:
                if FAIL_FAST:
                    raise
                log.error("[ %s ] FAILED after retries: %s", scene.id, exc)
                continue
            elapsed = time.monotonic() - started
            log.info(
                "[%d/%d] DONE scene %s in %.1fs -> %s",
                index, total, scene.id, elapsed, path.resolve(),
            )
            results.append((scene, path))
        return results

    async def generate_scene(
        self, scene: Scene, index: int, output_name: str | None = None
    ) -> Path:
        """Generate one scene with retry/backoff; return the saved MP4 path.

        output_name overrides the default scene_{index:02d}.mp4 filename
        (server.py names files by task id).
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # The SDK is blocking; keep the asyncio loop responsive.
                return await asyncio.to_thread(
                    self._generate_scene_blocking, scene, index, output_name
                )
            except Exception as exc:  # noqa: BLE001 - boundary, classified below
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise SceneGenerationError(
                        f"scene {scene.id} failed on attempt {attempt}: {exc!r}"
                    ) from exc
                backoff = self.backoff_base * (2 ** (attempt - 1))
                log.warning(
                    "[ %s ] attempt %d/%d failed (%r); retrying in %.0fs",
                    scene.id, attempt, self.max_retries, exc, backoff,
                )
                await asyncio.sleep(backoff)
        raise SceneGenerationError(f"scene {scene.id} exhausted retries: {last_exc!r}")

    # ------------------------------------------------------------------ #

    def _generate_scene_blocking(
        self, scene: Scene, index: int, output_name: str | None = None
    ) -> Path:
        """Submit, poll to completion, download, and save one video."""
        prompt = f"{STYLE} Scene: {scene.prompt}"
        config = types.GenerateVideosConfig(
            aspect_ratio=scene.aspect_ratio,
            number_of_videos=1,
            duration_seconds=scene.duration_seconds,
        )
        seed_image: types.Image | None = None
        if scene.reference_image_path:
            ref = Path(scene.reference_image_path)
            if ref.is_file():
                suffix = ref.suffix.lower().lstrip(".")
                mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
                seed_image = types.Image(image_bytes=ref.read_bytes(), mime_type=mime)
                log.info("[ %s ] using seed frame: %s", scene.id, ref.name)
            else:
                log.warning(
                    "[ %s ] reference image not found (%s); generating without it",
                    scene.id, scene.reference_image_path,
                )
        try:
            operation = self.client.models.generate_videos(
                model=MODEL, prompt=prompt, config=config, image=seed_image
            )
        except errors.APIError as exc:
            # The Gemini API-key path serves fixed-length clips and rejects
            # explicit durations; fall back once without the field.
            if config.duration_seconds is not None and "duration" in str(exc).lower():
                log.info(
                    "[ %s ] duration not accepted; retrying with default length", scene.id
                )
                config.duration_seconds = None
                operation = self.client.models.generate_videos(
                    model=MODEL, prompt=prompt, config=config, image=seed_image
                )
            else:
                raise

        started = time.monotonic()
        while not operation.done:
            time.sleep(self.poll_interval)
            operation = self.client.operations.get(operation)
            log.info(
                "[ %s ] rendering... %.0fs elapsed", scene.id, time.monotonic() - started
            )

        videos = (operation.response.generated_videos or []) if operation.response else []
        if not videos:
            raise SceneGenerationError(
                f"scene {scene.id}: operation finished without a video "
                f"(error={getattr(operation, 'error', None)!r})"
            )

        out_path = self.output_dir / (output_name or f"scene_{index:02d}.mp4")
        download = self.client.models.download(video=videos[0])
        if hasattr(download, "save"):  # newer SDK versions
            download.save(str(out_path))
        else:
            out_path.write_bytes(download.content)
        return out_path


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error(
            "GEMINI_API_KEY is not set. Get a key at "
            "https://aistudio.google.com/apikey and export it first."
        )
        return 2

    run_started = time.monotonic()
    orchestrator = RhymeOrchestrator(client=genai.Client(api_key=api_key))
    try:
        results = await orchestrator.run(SCENES)
    except SceneGenerationError as exc:
        log.error("Run aborted (FAIL_FAST): %s", exc)
        return 1

    log.info("=" * 64)
    log.info(
        "SUMMARY: %d/%d scenes succeeded in %.1fs; output dir: %s",
        len(results), len(SCENES), time.monotonic() - run_started,
        OUTPUT_DIR.resolve(),
    )
    for scene, path in results:
        log.info("  %-18s -> %s", scene.id, path)
    failed = len(SCENES) - len(results)
    if failed:
        log.warning("  %d scene(s) failed and were skipped", failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
