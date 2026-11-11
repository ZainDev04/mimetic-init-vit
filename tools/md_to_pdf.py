"""Convert the simple Markdown notes in this project to PDF.

Usage: python tools/md_to_pdf.py report/report.md [more files...]
Each PDF is written next to its .md file.

Handles only what our notes use: headings, paragraphs, bullet and numbered
lists, tables, **bold**, `code`, code blocks between ``` lines, and images (a line `![caption](path)`, path
relative to the .md file).
"""
import html
import re
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import fitz  # PyMuPDF

CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; }
h1 { font-size: 17pt; color: #1f3b63; margin-bottom: 6pt; }
h2 { font-size: 13pt; color: #2e5c8a; margin-top: 12pt; margin-bottom: 4pt; }
h3 { font-size: 11.5pt; color: #333333; margin-top: 8pt; margin-bottom: 3pt; }
p { margin-top: 3pt; margin-bottom: 5pt; }
li { margin-bottom: 2pt; }
code { font-family: monospace; font-size: 9.5pt; background-color: #eeeeee; }
table { border-collapse: collapse; margin-top: 4pt; margin-bottom: 8pt; }
th, td { border: 1px solid #999999; padding: 3pt 6pt; font-size: 10pt; }
th { font-weight: bold; }
p.fig { text-align: center; font-size: 9.5pt; }
pre { font-family: monospace; font-size: 8pt; white-space: pre-wrap; background-color: #f2f2ee; padding: 5pt; margin-top: 3pt; margin-bottom: 6pt; }
"""


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def md_to_html(md):
    out, para, lines = [], [], md.splitlines()
    list_tag = None

    def flush_para():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(para)))
            para.clear()

    def close_list():
        nonlocal list_tag
        if list_tag:
            out.append("</%s>" % list_tag)
            list_tag = None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        bullet = re.match(r"^\s*[-*] (.*)", line)
        number = re.match(r"^\s*\d+\. (.*)", line)
        if line.startswith("```"):
            # code block: everything up to the closing ``` is shown as it is
            flush_para(); close_list()
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append("<pre>%s</pre>" % html.escape("\n".join(code), quote=False))
        elif not line.strip():
            flush_para(); close_list()
        elif re.match(r"^!\[.*\]\(.*\)$", line):
            flush_para(); close_list()
            caption, src = re.match(r"^!\[(.*)\]\((.*)\)$", line).groups()
            out.append('<p class="fig"><img src="%s" width="290"/><br/>%s</p>' % (src, inline(caption)))
        elif line.startswith("#"):
            flush_para(); close_list()
            level = len(line) - len(line.lstrip("#"))
            out.append("<h%d>%s</h%d>" % (level, inline(line[level:].strip()), level))
        elif line.startswith("|"):
            flush_para(); close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append("<table>")
            for r, cells in enumerate(rows):
                tag = "th" if r == 0 else "td"
                out.append("<tr>" + "".join("<%s>%s</%s>" % (tag, inline(c), tag) for c in cells) + "</tr>")
            out.append("</table>")
            continue
        elif bullet or number:
            flush_para()
            tag = "ul" if bullet else "ol"
            if list_tag != tag:
                close_list()
                out.append("<%s>" % tag)
                list_tag = tag
            out.append("<li>%s</li>" % inline((bullet or number).group(1)))
        elif list_tag and line.startswith("   "):
            # continuation line of a list item
            out[-1] = out[-1][:-5] + " " + inline(line.strip()) + "</li>"
        else:
            close_list()
            para.append(line.strip())
        i += 1
    flush_para(); close_list()
    return "\n".join(out)


PAGE_BREAK = "<!-- pagebreak -->"


def convert(md_path):
    pdf_path = md_path[:-3] + ".pdf"
    with open(md_path, encoding="utf-8") as f:
        # a line containing only PAGE_BREAK starts a new page in the PDF
        chunks = f.read().split("\n" + PAGE_BREAK + "\n")
    writer = fitz.DocumentWriter(pdf_path)
    page = fitz.paper_rect("a4")
    area = page + (54, 54, -54, -54)
    # put every image the file uses into the archive, under the name used in the .md
    folder = os.path.dirname(os.path.abspath(md_path))
    archive = fitz.Archive()
    for chunk in chunks:
        for src in re.findall(r"^!\[.*\]\((.*)\)$", chunk, flags=re.M):
            with open(os.path.join(folder, src), "rb") as img:
                archive.add((img.read(), src))
    for chunk in chunks:
        story = fitz.Story(html=md_to_html(chunk), user_css=CSS, archive=archive)
        more = True
        while more:
            dev = writer.begin_page(page)
            more, _ = story.place(area)
            story.draw(dev)
            writer.end_page()
    writer.close()
    print("wrote", pdf_path)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        convert(path)
