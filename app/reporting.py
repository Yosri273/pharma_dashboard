"""
PDF Reporting Module for Pharma Dashboard. (Version 4 - Stable)

This module uses FPDF2 (fpdf) to generate professional PDF reports.

V4 Update: Fixed critical bug in _add_filters_section. The previous version used
pdf.cell() for the filter key, which does not wrap text. If a filter key was too long,
it would overflow its boundary, leaving no horizontal space for the filter value,
crashing pdf.multi_cell().

This version rewrites _add_filters_section to use a robust, two-column layout 
where BOTH key and value are rendered with multi_cell(), ensuring text wraps correctly.
This incorporates all previous fixes (KPI boxes, dynamic table fonts, and table headers).
"""

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any

# --- Constants for PDF Layout ---
A4_WIDTH_MM = 210
MARGIN_MM = 15
EFFECTIVE_WIDTH_MM = A4_WIDTH_MM - (2 * MARGIN_MM)


class PDFReport(FPDF):
    """
    Custom FPDF subclass to create a consistent branded header and footer.
    """
    
    def __init__(self, *args, report_title="Pharma Dashboard Report", **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = report_title
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(MARGIN_MM, MARGIN_MM, MARGIN_MM)

    def header(self):
        """Generates the report header on each page."""
        if self.page_no() == 1:
            self.set_font("Arial", 'B', 16)
            self.cell(0, 10, self.report_title, ln=True, align="C")
            self.set_font("Arial", '', 10)
            timestamp = datetime.now().strftime("Report Generated: %Y-%m-%d %H:%M:%S")
            self.cell(0, 8, timestamp, ln=True, align="C")
            self.ln(10) 
        else:
            self.set_font("Arial", 'I', 10)
            self.cell(0, 10, self.report_title, align="L")
            self.ln(5)

    def footer(self):
        """Generates the report footer on each page."""
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")


# --- Helper Functions for Report Sections ---

def _add_kpi_section(pdf: FPDF, kpi_data: Dict[str, str]):
    """
    Helper to render the pre-calculated KPIs in boxed cards.
    """
    if not kpi_data:
        return

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Key Performance Indicators", ln=True)
    pdf.set_font("Arial", '', 10)
    
    kpi_width = EFFECTIVE_WIDTH_MM / 2 - 5  # Two columns with 5mm gap
    line_height = 8
    box_height = line_height * 2.5
    count = 0

    for title, value in kpi_data.items():
        clean_title = title.replace("_", " ").title()
        
        start_x = pdf.get_x()
        start_y = pdf.get_y()
        
        pdf.set_fill_color(245, 245, 245) # Light gray bg
        pdf.multi_cell(kpi_width, box_height, "", border=1, fill=True)
        
        pdf.set_xy(start_x + 3, start_y + 3) # Padding
        
        pdf.set_font("Arial", 'B', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(kpi_width - 6, 6, clean_title)
        
        pdf.set_xy(start_x + 3, start_y + 9)
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(kpi_width - 6, 8, value)

        count += 1
        if count % 2 == 0:
            pdf.ln(box_height + 5) # Move to next row + 5mm gap
        else:
            pdf.set_xy(start_x + kpi_width + 5, start_y)

    if count % 2 != 0:
         pdf.ln(box_height + 5)
    else:
         pdf.ln(5) 
    
    pdf.set_text_color(0, 0, 0) # Reset text color


def _add_filters_section(pdf: FPDF, filters_dict: Dict[str, Any]):
    """
    Helper to write the 'filters applied' section to the PDF.
    
    V4 FIX: This function now uses multi_cell for BOTH key and value in a 
    manual two-column layout to prevent text-wrapping crashes from long keys.
    """
    if not filters_dict:
        return

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Report Filters Applied", ln=True)
    pdf.ln(2)

    key_col_width = 45  # Fixed width for the filter title column
    val_col_width = EFFECTIVE_WIDTH_MM - key_col_width # Remaining width for the value
    line_height = 6 # Use a smaller line height for filters

    for key, value in filters_dict.items():
        # 1. Format Key and Value strings
        key_title = f"{key.replace('_', ' ').title()}:"
        if isinstance(value, list):
            value_str = ", ".join(map(str, value)) if value else "None"
        elif value is None:
            value_str = "None"
        else:
            value_str = str(value)

        # 2. Get current Y pos; this anchors the row
        start_y = pdf.get_y()
        start_x = pdf.get_x()

        # 3. Render the Key column (with wrapping)
        pdf.set_font("Arial", 'B', 9)
        pdf.multi_cell(key_col_width, line_height, key_title, border=0, align='R')
        
        # 4. Get height of the cell we just drew (in case key wrapped to 2+ lines)
        key_render_height = pdf.get_y() - start_y

        # 5. Reset position to render the Value column
        # Move cursor to the right of the key column, at the same starting Y
        pdf.set_xy(start_x + key_col_width + 2, start_y)
        
        # 6. Render the Value column (with wrapping)
        pdf.set_font("Arial", '', 9)
        pdf.multi_cell(val_col_width - 2, line_height, value_str, border=0, align='L')
        
        # 7. Get height of the value cell (in case value wrapped to 5+ lines)
        val_render_height = pdf.get_y() - start_y

        # 8. Move cursor to the next line, clear of the tallest of the two columns
        final_y = start_y + max(key_render_height, val_render_height) + 1 # +1mm padding
        pdf.set_y(final_y)

    pdf.ln(5) # Space after the section


def _add_dataframe_table(pdf: FPDF, df: pd.DataFrame, title="Detailed Data Table"):
    """
    Helper to render a Pandas DataFrame as a formatted table.
    
    V3 FIX: Uses dynamic font sizing and multi_cell() for headers to prevent crashes.
    """
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, title, ln=True)
    
    if df.empty:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "No data available for this selection.", ln=True)
        return

    num_cols = len(df.columns)
    if num_cols == 0: return # Safety check
    
    col_width = EFFECTIVE_WIDTH_MM / num_cols

    # --- DYNAMIC FONT AND HEIGHT LOGIC ---
    if col_width < 10:  
        font_size = 6
        line_height = 5
    elif col_width < 15: 
        font_size = 7
        line_height = 6
    else: 
        font_size = 9
        line_height = 8
    # --- END DYNAMIC LOGIC ---

    # --- Table Header (Uses multi_cell to wrap long headers) ---
    pdf.set_font("Arial", 'B', font_size) 
    pdf.set_fill_color(220, 230, 240) 
    
    for col_name in df.columns:
        pdf.multi_cell(col_width, line_height, str(col_name), border=1, fill=True, align='C',
                       new_x="RIGHT", new_y="TOP", max_line_height=line_height)
    pdf.ln(line_height) 

    # --- Table Data Rows ---
    pdf.set_font("Arial", '', font_size) 
    pdf.set_fill_color(255, 255, 255)
    fill_row = False 

    for _, row in df.iterrows():
        
        # --- Page Break Header Check ---
        if pdf.get_y() + line_height > (pdf.page_break_trigger - 5): 
            pdf.add_page()
            pdf.set_font("Arial", 'B', font_size) 
            pdf.set_fill_color(220, 230, 240)
            
            for col_name in df.columns:
                pdf.multi_cell(col_width, line_height, str(col_name), border=1, fill=True, align='C',
                               new_x="RIGHT", new_y="TOP", max_line_height=line_height)
            pdf.ln(line_height) 
            
            pdf.set_font("Arial", '', font_size) 
            fill_row = False 
        
        pdf.set_fill_color(245, 245, 245) if fill_row else pdf.set_fill_color(255, 255, 255)
            
        for item in row:
            pdf.multi_cell(col_width, line_height, str(item), border=1, fill=True, align='L', 
                           new_x="RIGHT", new_y="TOP", max_line_height=line_height)
        
        pdf.ln(line_height) 
        fill_row = not fill_row

    pdf.ln(5)


def _add_figures_section(pdf: FPDF, figures_list: List[go.Figure]):
    """Helper to embed Plotly charts (as PNG images) into the PDF."""
    if not figures_list:
        return

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Charts and Visualizations", ln=True)
    pdf.ln(5)

    for i, fig in enumerate(figures_list):
        fig_title = f"Figure {i+1}"
        if fig.layout.title and fig.layout.title.text:
            fig_title = fig.layout.title.text
        
        try:
            img_bytes = fig.to_image(format="png", scale=2, width=800, height=450)
            img_file = BytesIO(img_bytes)

            required_space = 85 
            if pdf.get_y() + required_space > pdf.page_break_trigger:
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "Charts (Continued)", ln=True)
                pdf.ln(5)

            pdf.set_font("Arial", 'BI', 11)
            pdf.cell(0, 8, fig_title, ln=True, align='C')

            pdf.image(img_file, x=MARGIN_MM, w=EFFECTIVE_WIDTH_MM)
            pdf.ln(5)

        except Exception as e:
            pdf.set_font("Arial", 'I', 10)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Error rendering chart '{fig_title}': {str(e)}", ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(5)


# --- Main Public Function ---

def generate_pdf_report(
    kpi_data: Dict[str, str],
    filters_dict: Dict[str, Any],
    main_dataframe: pd.DataFrame,
    figures_list: List[go.Figure],
    report_title: str = "Pharma Dashboard Report",
    table_title: str = "Detailed Data Table"
) -> BytesIO:
    """
    Generates a professional PDF report from the dashboard state. (V4)

    Args:
        kpi_data: Dict of pre-calculated KPIs {title: value_str}.
        filters_dict: Dict of the filter settings currently applied.
        main_dataframe: A pandas DataFrame to be rendered as the primary table.
        figures_list: A list of Plotly Figure objects to embed as PNG images.
        report_title: The main title for the PDF report.
        table_title: The title to print above the main data table.

    Returns:
        A BytesIO object containing the complete PDF file.
    """
    
    # 1. Initialize the PDF document
    pdf = PDFReport(report_title=report_title)
    pdf.add_page()

    # 2. Add KPI Section
    _add_kpi_section(pdf, kpi_data)

    # 3. Add Filters Section (Now debugged)
    _add_filters_section(pdf, filters_dict)
    
    # 4. Add Main Data Table Section (Now debugged)
    _add_dataframe_table(pdf, main_dataframe, title=table_title)

    # 5. Add Figures Section
    if figures_list:
        pdf.add_page()
        _add_figures_section(pdf, figures_list)

    # 6. Finalize and return the BytesIO buffer
    output_buffer = BytesIO()
    pdf.output(output_buffer)
    output_buffer.seek(0)
    
    return output_buffer