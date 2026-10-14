from pathlib import Path
import shutil
import win32com.client


PPTX = Path(r"C:\Users\imgw\Documents\Codex\NMR模拟\docs\nmr_t2_workflow_ppt_rebuild\final\nmr_t2_workflow_editable.pptx")
OUT = Path(r"C:\Users\imgw\Documents\Codex\NMR模拟\docs\nmr_t2_workflow_ppt_rebuild\work\rendered")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("slide-*.png"):
        old.unlink()

    app = win32com.client.Dispatch("KWPP.Application")
    app.Visible = True
    pres = app.Presentations.Open(str(PPTX), False, True, False)
    try:
        for index in range(1, pres.Slides.Count + 1):
            target = OUT / f"slide-{index}.png"
            pres.Slides(index).Export(str(target), "PNG", 1600, 900)
            print(target.name)
    finally:
        pres.Close()
        app.Quit()


if __name__ == "__main__":
    main()
