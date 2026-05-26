#!/usr/bin/env python3
import asyncio
import json
import sys
from pathlib import Path


def parse_shazam_track(recognition):
    track = (recognition or {}).get("track") or {}
    if not track:
        return {}
    sections = track.get("sections") or []
    meta_section = next((s for s in sections if s.get("type") == "SONG"), {})
    metadata_pairs = meta_section.get("metadata") or []
    metadata_map = {
        (item.get("title") or "").strip().lower(): (item.get("text") or "").strip()
        for item in metadata_pairs
    }
    images = track.get("images") or {}
    return {
        "title": track.get("title") or metadata_map.get("title"),
        "artist": track.get("subtitle"),
        "album": metadata_map.get("album"),
        "genre": (track.get("genres") or {}).get("primary"),
        "shazam_url": track.get("url"),
        "cover_url": images.get("coverarthq") or images.get("coverart"),
    }


async def recognize(path_str):
    from shazamio import Shazam

    client = Shazam()
    return await client.recognize(path_str)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Usage: shazam_worker.py <audio_path>"}))
        return 2

    audio_path = Path(sys.argv[1])
    if not audio_path.is_file():
        print(json.dumps({"ok": False, "error": f"Audio file not found: {audio_path}"}))
        return 2

    try:
        recognized = asyncio.run(recognize(str(audio_path)))
        metadata = parse_shazam_track(recognized)
        if not metadata:
            print(json.dumps({"ok": False, "error": "Shazam did not return track metadata"}))
            return 0
        print(json.dumps({"ok": True, "metadata": metadata}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"Shazam lookup failed: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
