import os
import pandas as pd
import pdfplumber

PDF_PATH = "/Users/alli-borland/Desktop/Outseer_Fraud_and_Payments_Report_Q3_2021.pdf" 

# Extract full text
with pdfplumber.open(PDF_PATH) as pdf:
    all_text = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()

# Export as CSV
article_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
df = pd.DataFrame([{"article_name": article_name, "text": all_text}])
csv_name = f"{article_name}.csv"
df.to_csv(csv_name, index=False)

print("Extracted text saved to:", csv_name)