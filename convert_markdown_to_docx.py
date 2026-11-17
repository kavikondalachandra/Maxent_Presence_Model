# -*- coding: utf-8 -*-
"""
Convert MaxEnt Studio Pro Publication Guide Markdown to a professional Word (.docx) document.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}>'
                      f'<w:top w:w="{top}" w:type="dxa"/>'
                      f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
                      f'<w:left w:w="{left}" w:type="dxa"/>'
                      f'<w:right w:w="{right}" w:type="dxa"/>'
                      f'</w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def make_callout(doc, text, color_hex="1E3A8A", bg_hex="F0F4F8"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{color_hex}"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Georgia"
    run.font.size = Pt(9.5)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(4)

def parse_markdown_to_docx(md_path, docx_path):
    doc = docx.Document()
    
    # Set page margins to 0.75 inch (balanced)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
        # Add Header
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("MaxEnt Studio Pro — Research Article Guide & Publication Walkthrough")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        
        # Add Footer
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Author: K. S. Chandra | Independent Researcher, Spatial Geoinformatics | 2026")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "".join(code_lines)
                code_lines = []
                
                # Render code box as a single-cell shaded table
                tbl = doc.add_table(rows=1, cols=1)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                tbl.autofit = False
                cell = tbl.cell(0, 0)
                cell.width = Inches(6.5)
                set_cell_background(cell, "F8FAFC")
                set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
                set_table_borders(tbl, color="CBD5E1", sz="6", val="single")
                
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_before = Pt(2)
                cp.paragraph_format.space_after = Pt(2)
                cp.paragraph_format.line_spacing = 1.05
                crun = cp.add_run(code_text.rstrip())
                crun.font.name = "Consolas"
                crun.font.size = Pt(8.5)
                crun.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                
                doc.add_paragraph().paragraph_format.space_after = Pt(3)
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Tables
        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
            table_lines.append(stripped)
            # Peek if next line is table
            if i + 1 < len(lines) and ("|" in lines[i+1].strip()):
                i += 1
                continue
            else:
                # Render table
                if len(table_lines) >= 2:
                    rows_data = []
                    for tline in table_lines:
                        # Skip delimiter row e.g. | :--- | :---: |
                        if re.match(r"^\|(\s*:?-+:?\s*\|)+$", tline):
                            continue
                        cells = [c.strip() for c in tline.strip("|").split("|")]
                        rows_data.append(cells)
                    
                    if rows_data:
                        num_rows = len(rows_data)
                        num_cols = max(len(r) for r in rows_data)
                        # Pad rows
                        for r in rows_data:
                            while len(r) < num_cols:
                                r.append("")
                                
                        table = doc.add_table(rows=num_rows, cols=num_cols)
                        table.alignment = WD_TABLE_ALIGNMENT.CENTER
                        set_table_borders(table, color="CBD5E1", sz="6", val="single")
                        
                        for row_idx, r_data in enumerate(rows_data):
                            row = table.rows[row_idx]
                            is_header = (row_idx == 0)
                            for col_idx, text in enumerate(r_data):
                                cell = row.cells[col_idx]
                                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                                set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
                                
                                p = cell.paragraphs[0]
                                p.paragraph_format.space_before = Pt(2)
                                p.paragraph_format.space_after = Pt(2)
                                p.paragraph_format.line_spacing = 1.05
                                
                                # Clean markdown bold/math from text
                                clean_text = text.replace("**", "").replace("$", "").replace("<br>", " ")
                                
                                run = p.add_run(clean_text)
                                run.font.name = "Calibri"
                                
                                if is_header:
                                    set_cell_background(cell, "1E3A8A")
                                    run.font.size = Pt(9.5)
                                    run.font.bold = True
                                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                else:
                                    if row_idx % 2 == 1:
                                        set_cell_background(cell, "FFFFFF")
                                    else:
                                        set_cell_background(cell, "F8FAFC")
                                    run.font.size = Pt(8.5)
                                    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                        
                        doc.add_paragraph().paragraph_format.space_after = Pt(4)
                table_lines = []
                i += 1
                continue

        # Skip empty lines
        if not stripped:
            i += 1
            continue
            
        # Horizontal rules
        if stripped in ("---", "***", "___"):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            i += 1
            continue
            
        # Document Main Title (# Title)
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(title_text)
            run.font.name = "Georgia"
            run.font.size = Pt(22)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
            i += 1
            continue
            
        # Subtitle (### Subtitle)
        if stripped.startswith("### "):
            sub_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(sub_text)
            run.font.name = "Calibri"
            run.font.size = Pt(12)
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
            i += 1
            continue
            
        # Heading 1 (## 1. Title)
        if stripped.startswith("## "):
            h_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Georgia"
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
            i += 1
            continue

        # Heading 2 (### 1.1 Subtitle or #### Subtitle)
        if stripped.startswith("### ") or stripped.startswith("#### "):
            h_text = stripped.lstrip("#").strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = "Georgia"
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
            i += 1
            continue

        # Blockquote (> Quote)
        if stripped.startswith(">"):
            q_text = stripped[1:].strip()
            make_callout(doc, q_text, color_hex="2563EB", bg_hex="EFF6FF")
            i += 1
            continue

        # Bullet list items (- item or * item)
        if stripped.startswith(("- ", "* ")) and not stripped.startswith(("***", "---")):
            b_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            
            # Parse bold parts
            parts = re.split(r"(\*\*.*?\*\*)", b_text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    run.font.bold = True
                else:
                    run = p.add_run(part.replace("$", ""))
                run.font.name = "Calibri"
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            i += 1
            continue

        # Numbered list items (1. item)
        m_num = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m_num:
            num_str = m_num.group(1)
            n_text = m_num.group(2).strip()
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            
            parts = re.split(r"(\*\*.*?\*\*)", n_text)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    run = p.add_run(part[2:-2])
                    run.font.bold = True
                else:
                    run = p.add_run(part.replace("$", ""))
                run.font.name = "Calibri"
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            i += 1
            continue

        # Regular Body Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        
        parts = re.split(r"(\*\*.*?\*\*)", stripped)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                run = p.add_run(part[2:-2])
                run.font.bold = True
            elif part.startswith("`") and part.endswith("`"):
                run = p.add_run(part[1:-1])
                run.font.name = "Consolas"
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
            else:
                run = p.add_run(part.replace("$", ""))
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            
        i += 1

    doc.save(docx_path)
    print(f"Successfully generated Word document: {docx_path}")

if __name__ == "__main__":
    md_file = r"c:\Users\DuDu-PC\Downloads\DigiLib\MaxEnt_Studio_Pro_Research_Article_Guide_and_Publication_Walkthrough.md"
    docx_file = r"c:\Users\DuDu-PC\Downloads\DigiLib\MaxEnt_Studio_Pro_Research_Article_Guide_and_Publication_Walkthrough.docx"
    parse_markdown_to_docx(md_file, docx_file)
