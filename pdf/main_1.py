import sys
import io
import os
import Quartz
import Vision
from Cocoa import NSData
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# --- The OCR Engine (Same as before) ---
class AppleVisionOCR:
    def _pil_to_cgimage(self, pil_image):
        img_byte_arr = io.BytesIO()
        # Use JPEG for in-memory transfer, sometimes faster than PNG for photos
        pil_image.save(img_byte_arr, format='JPEG', quality=100)
        img_bytes = img_byte_arr.getvalue()
        ns_data = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        img_source = Quartz.CGImageSourceCreateWithData(ns_data, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(img_source, 0, None)
        return cg_image

    def recognize_text(self, cg_image, custom_words=None):
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        if custom_words: request.setCustomWords_(custom_words)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        success, error = handler.performRequests_error_([request], None)
        if not success: print(f"OCR Error: {error}"); return []

        result = []
        for obs in request.results():
            candidate = obs.topCandidates_(1)[0]
            # Vision BBox: Normalized (0.0-1.0), Origin Bottom-Left
            bbox = obs.boundingBox() 
            results.append({
                "text": candidate.string(),
                "bbox_norm": [bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height]
            })
        return results

# --- The Annotation Logic ---
def annotate_pdf(input_path, output_path, custom_words=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    ocr_engine = AppleVisionOCR()
    
    # 1. Open PDF with PyMuPDF for drawing
    doc = fitz.open(input_path)
    print(f"Opened PDF: {input_path} ({len(doc)} pages)")

    # 2. Rasterize pages for OCR engine (using pdf2image)
    # Using 300 DPI for good OCR accuracy
    print("Rasterizing pages for OCR scan...")
    page_images = convert_from_path(input_path, dpi=300)

    if len(doc) != len(page_images):
        print("Error: Page count mismatch found between PDF vs Image conversion.")
        doc.close()
        return

    # 3. Process each page
    for i, pdf_page in enumerate(doc):
        print(f"Processing and Annotating Page {i+1}...")
        
        # Get PDF page dimensions in points (e.g., 612x792)
        pdf_w = pdf_page.rect.width
        pdf_h = pdf_page.rect.height

        # Run OCR on the corresponding image version
        cg_img = ocr_engine._pil_to_cgimage(page_images[i])
        ocr_results = ocr_engine.recognize_text(cg_img, custom_words)

        # Draw annotations onto the PDF page
        for item in ocr_results:
            text = item['text']
            # [x, y, w, h] normalized, bottom-left origin
            vx, vy, vw, vh = item['bbox_norm']

            # --- COORDINATE CONVERSION ---
            # 1. Scale normalized width/height to PDF points
            box_w = vw * pdf_w
            box_h = vh * pdf_h
            
            # 2. Calculate X (simple scaling)
            x_left = vx * pdf_w
            
            # 3. Calculate Y (Crucial step: flip the axis)
            # Vision Y is distance from BOTTOM. PDF Y is distance from TOP.
            # The top of the Vision box is at (vy + vh).
            # In PDF coords, that is: TotalHeight - (Top of Vision Box scaled)
            y_top = pdf_h * (1.0 - (vy + vh))
            
            # Define rectangle for PyMuPDF (left, top, right, bottom)
            rect_coords = fitz.Rect(x_left, y_top, x_left + box_w, y_top + box_h)

            # --- DRAWING ---
            # Draw Red Rectangle (thin line width 0.5)
            pdf_page.draw_rect(rect_coords, color=(1, 0, 0), width=0.5)

            # Draw Text Overlay
            # Position text slightly above the top-left of the box
            text_pos = fitz.Point(x_left, y_top - 2)
            # Safety check: if box is at the very top, draw text inside box instead
            if text_pos.y < 5: text_pos.y = y_top + 8

            # Insert tiny red text using a standard PDF font
            pdf_page.insert_text(
                text_pos, 
                text, 
                fontsize=6, # Very small font so it doesn't overlap too much
                fontname="helv", # Standard Helvetica
                color=(1, 0, 0)
            )

    # 4. Save final output
    doc.save(output_path)
    doc.close()
    print(f"\nSuccessfully saved annotated PDF to: {output_path}")


# --- Execution ---
if __name__ == "__main__":
    # Input PDF (Replace with your actual resume filename)
    input_pdf = "resume.pdf"
    
    # Output PDF filename
    output_pdf = "resume_sample_ANNOTATED.pdf"


    # Create a dummy file for testing if you don't have one
    if not os.path.exists(input_pdf):
        print(f"Creating dummy PDF '{input_pdf}' for testing...")
        doc_test = fitz.open()
        page_test = doc_test.new_page()
        page_test.insert_text((50, 100), "John Doe\nSenior Software Engineer\nExperience with Python, AWS, and Kubernetes.", fontsize=12)
        doc_test.save(input_pdf)
        doc_test.close()

    # Run the annotation process
    annotate_pdf(input_pdf, output_pdf, custom_words=resume_keywords)