from fpdf import FPDF


def create_pdf(period, reuma_input, nero_input, episkeves_input=0.0, repair_name="Repairs"):
    # Μαθηματικοί υπολογισμοί
    r_a1 = reuma_input * 0.2901
    r_b1 = reuma_input * 0.2472
    r_b2 = reuma_input * 0.4627

    n_share = nero_input / 3
    e_share = episkeves_input / 3

    total_a1 = r_a1 + 6.70 + 10.33 + n_share + e_share
    total_b1 = r_b1 + 6.70 + 11.50 + n_share + e_share
    total_b2 = r_b2 + 6.70 + 11.50 + n_share + e_share
    grand_total = reuma_input + 20.00 + 33.33 + nero_input + episkeves_input

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # --- TOP ELEGANT HEADER BANNER ---
    pdf.set_fill_color(15, 23, 42)  # Dark Slate Blue / Modern Tech Color
    pdf.rect(0, 0, 210, 38, "F")

    # Title inside Banner
    pdf.set_y(10)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(180, 8, "UTILITY STATEMENT", 0, 1, "C")

    # Address & Period Info Sub-header
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(203, 213, 225)  # Soft Light Gray text
    pdf.cell(180, 6, "Filikon 35, Peristeri  |  Building Management", 0, 1, "C")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(180, 6, f"PERIOD: {period}", 0, 1, "C")

    pdf.set_y(48)  # Space below header banner

    # --- TABLE CONFIGURATION ---
    col_widths = [50, 30, 33, 33, 34]
    row_h = 9

    # --- TABLE HEADER ---
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 41, 59)  # Slate Gray Header
    pdf.set_text_color(255, 255, 255)
    pdf.set_draw_color(226, 232, 240)  # Light border color

    pdf.cell(col_widths[0], row_h, "  Expense Category", 1, 0, "L", True)
    pdf.cell(col_widths[1], row_h, "Total (EUR)", 1, 0, "C", True)
    pdf.cell(col_widths[2], row_h, "Apt. A1", 1, 0, "C", True)
    pdf.cell(col_widths[3], row_h, "Apt. B1", 1, 0, "C", True)
    pdf.cell(col_widths[4], row_h, "Apt. B2", 1, 1, "C", True)

    # --- TABLE DATA ROWS (Zebra Striping) ---
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 41, 59)

    data = [
        ("Electricity", reuma_input, r_a1, r_b1, r_b2),
        ("Cleaning Service", 20.00, 6.70, 6.70, 6.70),
        ("Elevator Maintenance", 33.33, 10.33, 11.50, 11.50),
        ("Water Utility", nero_input, n_share, n_share, n_share),
    ]

    # Dynamic repair row title
    repair_title = f"Repairs ({repair_name})" if repair_name and repair_name != "Repairs" else "Repairs"
    data.append((repair_title[:24], episkeves_input, e_share, e_share, e_share))

    # Render rows with alternating row colors
    for idx, (desc, tot, a1, b1, b2) in enumerate(data):
        fill = (idx % 2 == 1)
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)

        pdf.cell(col_widths[0], row_h, f"  {desc}", 1, 0, "L", fill)
        pdf.cell(col_widths[1], row_h, f"{tot:.2f} ", 1, 0, "R", fill)
        pdf.cell(col_widths[2], row_h, f"{a1:.2f} ", 1, 0, "R", fill)
        pdf.cell(col_widths[3], row_h, f"{b1:.2f} ", 1, 0, "R", fill)
        pdf.cell(col_widths[4], row_h, f"{b2:.2f} ", 1, 1, "R", fill)

    # --- GRAND TOTAL ROW ---
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(224, 231, 255)  # Light Accent Blue
    pdf.set_text_color(15, 23, 42)

    pdf.cell(col_widths[0], row_h + 1, "  TOTAL DUE", 1, 0, "L", True)
    pdf.cell(col_widths[1], row_h + 1, f"{grand_total:.2f} ", 1, 0, "R", True)
    pdf.cell(col_widths[2], row_h + 1, f"{total_a1:.2f} ", 1, 0, "R", True)
    pdf.cell(col_widths[3], row_h + 1, f"{total_b1:.2f} ", 1, 0, "R", True)
    pdf.cell(col_widths[4], row_h + 1, f"{total_b2:.2f} ", 1, 1, "R", True)

    # --- FOOTER ---
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(180, 5, "Generated automatically via Filikon Koinoxrista System", 0, 1, "C")

    return bytes(pdf.output())