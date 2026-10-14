from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(r"C:\Users\imgw\Documents\Codex\NMR模拟\docs\nmr_t2_workflow_ppt_rebuild")
FINAL = ROOT / "final"
QA = FINAL / "qa"
PPTX = FINAL / "nmr_t2_workflow_editable.pptx"
ASSETS = FINAL / "independent-assets"
SOURCES = Path(r"C:\Users\imgw\Documents\Codex\NMR模拟\docs\nmr_t2_workflow_figures")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_pptx() -> dict:
    forbidden = ["TODO", "REPLACE", "prompt", "generated_images", "AppData\\Local\\Temp", "codex-clipboard"]
    hits = []
    text_runs = pictures = charts = shapes = 0
    slide_names = []
    with zipfile.ZipFile(PPTX) as archive:
        names = archive.namelist()
        slide_names = sorted(
            (n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"\d+", n).group()),
        )
        charts = len([n for n in names if n.startswith("ppt/charts/chart") and n.endswith(".xml")])
        for name in names:
            if not name.endswith((".xml", ".rels")):
                continue
            text = archive.read(name).decode("utf-8", errors="ignore")
            text_runs += text.count("<a:t>")
            pictures += text.count("<p:pic>")
            shapes += text.count("<p:sp>")
            for token in forbidden:
                if token.lower() in text.lower():
                    hits.append({"file": name, "token": token})
    return {
        "pptx": str(PPTX),
        "slide_count": len(slide_names),
        "slide_xml": slide_names,
        "editable_text_runs": text_runs,
        "embedded_pictures": pictures,
        "editable_charts": charts,
        "editable_shapes": shapes,
        "forbidden_internal_artifact_hits": hits,
        "rendered_slide_count": len(list((ROOT / "work" / "rendered").glob("slide-*.png"))),
        "required_qa_files_present": {
            name: (QA / name).exists()
            for name in ["preview.png", "overlay.png", "difference.png", "source-preview-contact-sheet.png", "visual-qa.json"]
        },
        "status": "pass" if len(slide_names) == 5 and not hits else "fail",
    }


def write_provenance() -> None:
    assets = [
        {"file": "reconstructed/ct_stack.png", "method": "imagegen", "status": "used", "role": "schematic grayscale CT stack"},
        {"file": "reconstructed/segmented_slice.png", "method": "imagegen", "status": "used", "role": "three-phase segmented representative slice"},
        {"file": "reconstructed/fem_mesh.png", "method": "imagegen", "status": "used", "role": "boundary-refined triangular FEM mesh"},
        {"file": "reconstructed/magnetization_field.png", "method": "imagegen", "status": "used", "role": "normalized magnetization field schematic"},
        {"file": "reconstructed/porosity_groups.png", "method": "imagegen", "status": "used", "role": "porosity-group representative-slice schematic"},
        {"file": "reconstructed/interface_generated_attempt.png", "method": "imagegen", "status": "rejected", "reason": "solid-fluid and water-air boundary encodings were not cleanly separable"},
        {
            "file": "reconstructed/interface_authoritative_fallback.png",
            "method": "post-attempt source crop",
            "status": "used",
            "source": str(SOURCES / "03_ct_to_fem_16x9.png"),
            "crop_pixels": [875, 280, 330, 340],
            "reason": "preserve the source-authoritative boundary classification after the generated attempt failed scientific audit",
        },
    ]
    for i, crop in enumerate(([45, 270, 305, 380], [405, 270, 305, 380], [765, 270, 305, 380])):
        assets.append({
            "file": f"reconstructed/mag_t{i}_authoritative_fallback.png",
            "method": "post-attempt source crop",
            "status": "used",
            "source": str(SOURCES / "04_magnetization_to_decay_16x9.png"),
            "crop_pixels": crop,
            "reason": "preserve the exact t0-t2 topology and temporal progression after a generated magnetization-field attempt",
        })
    payload = {
        "scope": "independent scientific visual assets and formula provenance for a five-slide editable rebuild",
        "assets": assets,
        "formula_policy": "equations and scientific labels were transcribed from the five user-provided source figures; no new calibrated parameter values were introduced",
        "model_dimension_statement": "2D representative-slice approximation, explicitly not a full 3D solve",
    }
    (QA / "scientific-provenance.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(validation: dict) -> None:
    inputs = []
    for source in sorted(SOURCES.glob("*_16x9.png")):
        inputs.append({"path": str(source), "sha256": sha256(source)})
    outputs = []
    for file in sorted(FINAL.rglob("*")):
        if file.is_file() and file.name != "manifest.json":
            outputs.append({"path": str(file.relative_to(FINAL)), "bytes": file.stat().st_size, "sha256": sha256(file)})
    payload = {
        "deliverable": "five-slide 16:9 editable PowerPoint rebuild",
        "inputs": inputs,
        "primary_output": str(PPTX),
        "renderer": "WPS Presentation 12.1 COM slide.Export at 1600x900",
        "validation_status": validation["status"],
        "outputs": outputs,
    }
    (QA / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def package_assets() -> None:
    target = FINAL / "independent-assets.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(ASSETS.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(FINAL))


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    package_assets()
    write_provenance()
    validation = inspect_pptx()
    (FINAL / "validation-clean.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(validation)
    print(json.dumps({"status": validation["status"], "slide_count": validation["slide_count"]}))


if __name__ == "__main__":
    main()
