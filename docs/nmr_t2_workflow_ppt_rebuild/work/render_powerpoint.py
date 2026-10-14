from pathlib import Path
import sys

import win32com.client


pptx_path = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

app = win32com.client.Dispatch("PowerPoint.Application")
presentation = app.Presentations.Open(str(pptx_path), WithWindow=False)
try:
    for index, slide in enumerate(presentation.Slides, start=1):
        slide.Export(str(out_dir / f"slide-{index}.png"), "PNG", 1672, 941)
finally:
    presentation.Close()
    app.Quit()

print(f"rendered {index} slides to {out_dir}")
