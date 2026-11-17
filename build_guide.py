# -*- coding: utf-8 -*-
"""
Build Script for MaxEnt Studio Pro Research Article Guide & Publication Walkthrough
Generates:
1. Full Markdown Document (.md) in c:\Users\DuDu-PC\Downloads\DigiLib
2. Publication-grade Word Document (.docx) in c:\Users\DuDu-PC\Downloads\DigiLib
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

print("Initializing publication guide builder...")
