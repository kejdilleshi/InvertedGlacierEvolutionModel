"""Insert the slides of one pptxgenjs deck into another deck built with the same layouts.

    python merge_slides.py DECK.pptx NEW_SLIDES.pptx OUT.pptx [--remove 12] [--at 2 13]

--remove: positions (1-based, in DECK) of slides to drop first; --at: final position of each new slide
(default: appended). Dropped slides' parts are deleted with the pptx skill's clean.py (PPTX_SKILL).

Used for the joint-inversion slides (joint_slides.js): the main deck has been edited by hand since
build.js wrote it, so it is extended instead of rebuilt. Slides keep their pictures and speaker
notes; their layout is mapped by name (CONTENT, TITLE) onto the main deck's layouts.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CT_SLIDE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
CT_NOTES = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"


def layouts_by_name(root):
    out = {}
    for f in sorted((root / "ppt/slideLayouts").glob("slideLayout*.xml")):
        m = re.search(r'<p:cSld name="([^"]*)"', f.read_text(encoding="utf-8"))
        if m:
            out[m.group(1)] = f.name
    return out


def main(deck, new, out, remove=(), at=()):
    tmp = Path(tempfile.mkdtemp())
    old, nw = tmp / "old", tmp / "new"
    zipfile.ZipFile(deck).extractall(old)
    zipfile.ZipFile(new).extractall(nw)
    lay_old, lay_new = layouts_by_name(old), layouts_by_name(nw)
    lay_map = {f: lay_old[n] for n, f in lay_new.items()}
    n_old = max(int(re.findall(r"\d+", f.name)[0]) for f in (old / "ppt/slides").glob("slide*.xml"))
    notes_old = max([int(re.findall(r"\d+", f.name)[0]) for f in (old / "ppt/notesSlides").glob("notesSlide*.xml")] or [0])
    pres = (old / "ppt/presentation.xml").read_text(encoding="utf-8")
    prels = (old / "ppt/_rels/presentation.xml.rels").read_text(encoding="utf-8")
    ct = (old / "[Content_Types].xml").read_text(encoding="utf-8")
    next_id = max(int(i) for i in re.findall(r'<p:sldId id="(\d+)"', pres)) + 1
    next_rid = max(int(i) for i in re.findall(r'Id="rId(\d+)"', prels)) + 1
    order = re.findall(r'<p:sldId [^>]*r:id="(rId\d+)"', (nw / "ppt/presentation.xml").read_text(encoding="utf-8"))
    nrels = (nw / "ppt/_rels/presentation.xml.rels").read_text(encoding="utf-8")
    new_slides = [re.search(rf'Id="{r}"[^>]*Target="slides/(slide\d+\.xml)"', nrels).group(1) for r in order]

    sld_ids, added = [], []
    for k, name in enumerate(new_slides, start=1):
        sn, nn = f"slide{n_old + k}.xml", f"notesSlide{notes_old + k}.xml"
        rels = (nw / f"ppt/slides/_rels/{name}.rels").read_text(encoding="utf-8")
        for tgt in re.findall(r'Target="\.\./media/([^"]+)"', rels):
            dst = f"joint-{tgt}"
            shutil.copy(nw / "ppt/media" / tgt, old / "ppt/media" / dst)
            rels = rels.replace(f"../media/{tgt}", f"../media/{dst}")
        rels = re.sub(r"\.\./slideLayouts/(slideLayout\d+\.xml)", lambda m: f"../slideLayouts/{lay_map[m.group(1)]}", rels)
        m = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', rels)
        if m:
            note_src = m.group(1)
            rels = rels.replace(f"../notesSlides/{note_src}", f"../notesSlides/{nn}")
            nrel = (nw / f"ppt/notesSlides/_rels/{note_src}.rels").read_text(encoding="utf-8")
            nrel = nrel.replace(f"../slides/{name}", f"../slides/{sn}")
            shutil.copy(nw / "ppt/notesSlides" / note_src, old / "ppt/notesSlides" / nn)
            (old / f"ppt/notesSlides/_rels/{nn}.rels").write_text(nrel, encoding="utf-8")
            ct = ct.replace("</Types>", f'<Override PartName="/ppt/notesSlides/{nn}" ContentType="{CT_NOTES}"/></Types>')
        shutil.copy(nw / "ppt/slides" / name, old / "ppt/slides" / sn)
        (old / f"ppt/slides/_rels/{sn}.rels").write_text(rels, encoding="utf-8")
        ct = ct.replace("</Types>", f'<Override PartName="/ppt/slides/{sn}" ContentType="{CT_SLIDE}"/></Types>')
        prels = prels.replace("</Relationships>",
                              f'<Relationship Id="rId{next_rid}" Type="{REL}/slide" Target="slides/{sn}"/></Relationships>')
        sld_ids.append(f'<p:sldId id="{next_id}" r:id="rId{next_rid}"/>')
        added.append(sn)
        next_id += 1; next_rid += 1
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S)
    items = re.findall(r"<p:sldId [^>]*/>", lst.group(1))
    items = [it for k, it in enumerate(items, start=1) if k not in set(remove)]
    pos = list(at) or [len(items) + k for k in range(1, len(sld_ids) + 1)]
    for p_, it in sorted(zip(pos, sld_ids)):
        items.insert(p_ - 1, it)
    pres = pres[: lst.start(1)] + "".join(items) + pres[lst.end(1):]
    (old / "ppt/presentation.xml").write_text(pres, encoding="utf-8")
    (old / "ppt/_rels/presentation.xml.rels").write_text(prels, encoding="utf-8")
    (old / "[Content_Types].xml").write_text(ct, encoding="utf-8")
    if remove:
        subprocess.run(["python3", os.path.join(os.environ["PPTX_SKILL"], "scripts/clean.py"), str(old)], check=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(old / "[Content_Types].xml", "[Content_Types].xml")
        for f in sorted(old.rglob("*")):
            if f.is_file() and f.name != "[Content_Types].xml":
                z.write(f, f.relative_to(old).as_posix())
    shutil.rmtree(tmp)
    print("added", added, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("deck"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--remove", type=int, nargs="*", default=[])
    ap.add_argument("--at", type=int, nargs="*", default=[])
    a = ap.parse_args()
    main(a.deck, a.new, a.out, a.remove, a.at)
