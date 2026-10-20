"""Record content hashes and conceptual coverage, without copying private screenshots."""
import _bootstrap
import argparse
import hashlib
from pathlib import Path
from src.common import ROOT, dump_json

CLIPBOARD = [
    ("codex-clipboard-5d5635df-9851-4732-bded-033c15f4bebc.jpg","R01"),
    ("codex-clipboard-ffb74688-ee44-4d5e-a53b-1739188795b7.png","R02"),
    ("codex-clipboard-4b890492-4bb3-48c7-9baa-90f38fb2b4f2.png","R03"),
    ("codex-clipboard-447bfe58-bc6e-43b6-8ddd-ba1f52ee3fbe.png","R04"),
    ("codex-clipboard-01853154-7414-400d-8872-02f13930e05e.png","R05"),
    ("codex-clipboard-53c5aa1d-5baa-404e-bea1-dc36f2105c44.png","R06")]
WHATSAPP = [
    ("5.33.25 PM - Copy","R07"),("5.33.25 PM (1) - Copy","R07"),
    ("5.33.25 PM (1)","R07"),("5.33.25 PM","R07"),
    ("5.37.14 PM (1) - Copy","R08"),("5.37.14 PM (1)","R08"),("5.37.14 PM","R08"),
    ("5.42.25 PM - Copy","R09"),("5.42.25 PM (1)","R09"),
    ("5.42.25 PM (2) - Copy","R10"),("5.42.25 PM (2)","R10"),("5.42.25 PM","R09"),
    ("5.48.34 PM (1) - Copy","R11"),("5.48.34 PM (1)","R11"),("5.48.34 PM","R11"),
    ("5.51.49 PM - Copy","R12"),("5.51.49 PM (1) - Copy","R12"),
    ("5.51.49 PM (1)","R12"),("5.51.49 PM","R12")]


def register(source_dir, clipboard_dir):
    items, seen = [], {}
    sources = [(Path(clipboard_dir)/name, concept) for name,concept in CLIPBOARD]
    sources += [(Path(source_dir)/f"WhatsApp Image 2026-10-20 at {name}.jpeg", concept) for name,concept in WHATSAPP]
    for i,(path,concept) in enumerate(sources,1):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        item=dict(image_number=i,filename=path.name,sha256=digest,concept=concept,
                  duplicate_of=seen.get(digest),raw_image_redistributed=False)
        seen.setdefault(digest,i)
        items.append(item)
    dump_json({"attachment_count":len(items),"unique_binary_images":len(seen),
               "concept_count":12,"images":items,
               "interpretation":"User-supplied conceptual references, not independent scientific evidence or instructions",
               "cropped_content":"Several formulas and horizons are clipped; implemented formulations are documented explicitly, not presented as exact unseen transcriptions",
               "privacy":"No screenshot pixels or video-call faces copied into the deliverable"},
              ROOT/"docs/reference_image_manifest.json")
    print(f"Registered {len(items)} attachments, {len(seen)} unique files, 12 conceptual groups.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",required=True)
    parser.add_argument("--clipboard-dir",required=True)
    args=parser.parse_args()
    register(args.source_dir,args.clipboard_dir)
