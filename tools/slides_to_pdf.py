"""Make slides/slides.pdf from the slide sources in slides/deck_source/.

Usage: python tools/slides_to_pdf.py
Joins the slide files (in the order of deck.json) into slides/slides.html,
with the uploaded plots replaced by the local files in results/, then prints
that page to PDF with Microsoft Edge in headless mode. Speaker notes are left out.
"""
import json
import os
import re
import subprocess

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
IMAGES = {  # uploaded asset -> local file
    "/_blob/d5459bbb8b0324afcc1eff33af9586d9": "../results/test_acc.png",
    "/_blob/ec537612e7431efb26fc15a2cb5e3072": "../results/budget.png",
}
CSS = """
@page { size: 1920px 1080px; margin: 0; }
* { margin: 0; box-sizing: border-box; }
body { background: #1B2A3A; }
section { width: 1920px; height: 1080px; position: relative; overflow: hidden; break-after: page; }
ul { padding-left: 40px; }
table { border-collapse: collapse; }
th, td { border: 1px solid #CFCCC2; padding: 0.35em 0.6em; text-align: left; }
th { font-weight: 600; }
"""

deck = json.load(open("slides/deck_source/deck.json", encoding="utf-8"))
fonts = "".join('<link rel="stylesheet" href="%s">' % f["href"] for f in deck["faces"].values())
slides = []
for slide_id in deck["order"]:
    html = open(f"slides/deck_source/slides/{slide_id}.html", encoding="utf-8").read()
    html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)  # drop speaker notes
    for blob, path in IMAGES.items():
        html = html.replace(blob, path)
    slides.append(html)

page = '<!doctype html><html><head><meta charset="utf-8"><title>%s</title>%s<style>%s</style></head><body>%s</body></html>' % (
    deck["title"], fonts, CSS, "\n".join(slides))
with open("slides/slides.html", "w", encoding="utf-8") as f:
    f.write(page)

html_url = "file:///" + os.path.abspath("slides/slides.html").replace("\\", "/")
subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                "--virtual-time-budget=10000",  # give the web fonts time to load
                "--print-to-pdf=" + os.path.abspath("slides/slides.pdf"), html_url], check=True)
print("wrote slides/slides.pdf")
