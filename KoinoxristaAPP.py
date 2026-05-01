import streamlit as st
from fpdf import FPDF


def create_pdf(period, reuma_input, nero_input):
    # Μαθηματικοί υπολογισμοί βάσει του προτύπου[cite: 1, 2]
    # Ρεύμα (Electricity) - Κατανομή βάσει ποσοστών[cite: 1, 2]
    r_a1 = reuma_input * 0.2901
    r_b1 = reuma_input * 0.2472
    r_b2 = reuma_input * 0.4627

    # Νερό (Water) - Το μοιράζουμε ισόποσα διά του 3 (ή βάσει αναγκών)
    n_share = nero_input / 3

    # Σταθερά ποσά ανά διαμέρισμα
    # Καθαριότητα: 6.70 | Ασανσέρ: 10.33 για Α1, 11.50 για Β1/Β2
    total_a1 = r_a1 + 6.70 + 10.33 + n_share
    total_b1 = r_b1 + 6.70 + 11.50 + n_share
    total_b2 = r_b2 + 6.70 + 11.50 + n_share
    grand_total = reuma_input + 20.00 + 33.33 + nero_input

    pdf = FPDF()
    pdf.add_page()

    # Τίτλος
    pdf.set_font("Helvetica", 'B', 16)
    pdf.set_text_color(0, 51, 102)  # Σκούρο Μπλε
    pdf.cell(190, 10, txt="KOINOXRISTA FILIKON", ln=True, align='C')
    pdf.set_font("Helvetica", '', 11)
    pdf.cell(190, 10, txt=f"PERIOD: {period}", ln=True, align='C')
    pdf.ln(5)

    # Στήλες: Description | Total | A1 | B1 | B2[cite: 2]
    col_widths = [45, 30, 38, 38, 38]
    h = 10

    # Κεφαλίδα με Μπλε Χρώμα[cite: 2]
    pdf.set_font("Helvetica", 'B', 9)
    pdf.set_fill_color(0, 102, 204)  # Μπλε
    pdf.set_text_color(255, 255, 255)  # Λευκά γράμματα
    pdf.cell(col_widths[0], h, "Description", 1, 0, 'C', True)
    pdf.cell(col_widths[1], h, "Total (Euro)", 1, 0, 'C', True)
    pdf.cell(col_widths[2], h, "Apartment A1", 1, 0, 'C', True)
    pdf.cell(col_widths[3], h, "Apartment B1", 1, 0, 'C', True)
    pdf.cell(col_widths[4], h, "Apartment B2", 1, 1, 'C', True)

    # Επαναφορά χρωμάτων για το περιεχόμενο
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", '', 10)

    # Γραμμή Electricity[cite: 2]
    pdf.cell(col_widths[0], h, "Electricity", 1)
    pdf.cell(col_widths[1], h, f"{reuma_input:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[2], h, f"{r_a1:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[3], h, f"{r_b1:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[4], h, f"{r_b2:.2f}", 1, 1, 'R')

    # Γραμμή Cleaning[cite: 2]
    pdf.cell(col_widths[0], h, "Cleaning", 1)
    pdf.cell(col_widths[1], h, "20.00", 1, 0, 'R')
    pdf.cell(col_widths[2], h, "6.70", 1, 0, 'R')
    pdf.cell(col_widths[3], h, "6.70", 1, 0, 'R')
    pdf.cell(col_widths[4], h, "6.70", 1, 1, 'R')

    # Γραμμή Elevator[cite: 2]
    pdf.cell(col_widths[0], h, "Elevator Maint.", 1)
    pdf.cell(col_widths[1], h, "33.33", 1, 0, 'R')
    pdf.cell(col_widths[2], h, "10.33", 1, 0, 'R')
    pdf.cell(col_widths[3], h, "11.50", 1, 0, 'R')
    pdf.cell(col_widths[4], h, "11.50", 1, 1, 'R')

    # Γραμμή Water (ΝΕΟ)[cite: 2]
    pdf.cell(col_widths[0], h, "Water", 1)
    pdf.cell(col_widths[1], h, f"{nero_input:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[2], h, f"{n_share:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[3], h, f"{n_share:.2f}", 1, 0, 'R')
    pdf.cell(col_widths[4], h, f"{n_share:.2f}", 1, 1, 'R')

    # Γραμμή Σύνολο (Grand Total) με απαλό μπλε[cite: 2]
    pdf.set_font("Helvetica", 'B', 10)
    pdf.set_fill_color(204, 229, 255)
    pdf.cell(col_widths[0], h, "GRAND TOTAL", 1, 0, 'L', True)
    pdf.cell(col_widths[1], h, f"{grand_total:.2f}", 1, 0, 'R', True)
    pdf.cell(col_widths[2], h, f"{total_a1:.2f}", 1, 0, 'R', True)
    pdf.cell(col_widths[3], h, f"{total_b1:.2f}", 1, 0, 'R', True)
    pdf.cell(col_widths[4], h, f"{total_b2:.2f}", 1, 1, 'R', True)

    return pdf.output()


# --- STREAMLIT UI ---
st.title("🏢 Building Fees Generator")

with st.form("data_form"):
    period = st.text_input("Period", "16/02/26 - 15/03/26")
    reuma = st.number_input("Electricity Total (€)", min_value=0.0, format="%.2f")
    nero = st.number_input("Water Total (€)", min_value=0.0, format="%.2f")
    submit_button = st.form_submit_button("Calculate")

if submit_button:
    st.info("Review the totals and download the blue-themed PDF below.")
    pdf_bytes = create_pdf(period, reuma, nero)

    st.download_button(
        label="📥 Download Blue PDF Report",
        data=bytes(pdf_bytes),
        file_name=f"koinoxrista_{period}.pdf",
        mime="application/pdf"
    )