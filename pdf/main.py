import sys
import io
import os
import Quartz
import Vision
from Cocoa import NSData
from pdf2image import convert_from_path
import fitz  # PyMuPDF

class AppleVisionOCR:
    def _pil_to_cgimage(self, pil_image):
        """Converts a PIL Image to a Quartz CGImage."""
        img_byte_arr = io.BytesIO()
        # Use JPEG for speed; Ensure quality is high for OCR
        pil_image.save(img_byte_arr, format='JPEG', quality=100)
        img_bytes = img_byte_arr.getvalue()
        
        ns_data = NSData.dataWithBytes_length_(img_bytes, len(img_bytes))
        img_source = Quartz.CGImageSourceCreateWithData(ns_data, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(img_source, 0, None)
        return cg_image

    def recognize_text(self, cg_image, custom_words=None):
        """Runs Apple Vision OCR on a CGImage."""
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        if custom_words: 
            request.setCustomWords_(custom_words)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        
        success, error = handler.performRequests_error_([request], None)
        if not success: 
            print(f"OCR Error: {error}")
            return []

        ocr_output = [] 
        
        # --- FIX: Use .results() (plural) instead of .result() ---
        if not request.results():
            return []

        for obs in request.results():
            candidate = obs.topCandidates_(1)[0]
            
            # Vision BBox: Normalized (0.0-1.0), Origin Bottom-Left
            bbox = obs.boundingBox() 
            
            ocr_output.append({
                "text": candidate.string(),
                # Store as list for easy unpacking
                "bbox_norm": [bbox.origin.x, bbox.origin.y, bbox.size.width, bbox.size.height]
            })
        return ocr_output

def annotate_pdf(input_path, output_path, custom_words=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    try:
        ocr_engine = AppleVisionOCR()
        
        # 1. Open PDF with PyMuPDF
        doc = fitz.open(input_path)
        print(f"Processing: {os.path.basename(input_path)} ({len(doc)} pages)")

        # 2. Rasterize pages (Requires Poppler installed)
        try:
            # DPI 300 is optimal for OCR accuracy vs speed
            page_images = convert_from_path(input_path, dpi=300)
        except Exception as e:
            print(f"Error converting PDF to image: {e}")
            print("Ensure 'poppler' is installed (brew install poppler).")
            return

        # Handle page count mismatches safely
        limit = min(len(doc), len(page_images))
        if len(doc) != len(page_images):
            print("Warning: Page count mismatch. Annotating available matching pages only.")

        # 3. Process each page
        for i in range(limit):
            pdf_page = doc[i]
            pil_image = page_images[i]
            
            # Get PDF page dimensions
            pdf_w = pdf_page.rect.width
            pdf_h = pdf_page.rect.height

            # Run OCR
            cg_img = ocr_engine._pil_to_cgimage(pil_image)
            ocr_results = ocr_engine.recognize_text(cg_img, custom_words)

            # Draw annotations
            for item in ocr_results:
                text = item['text']
                vx, vy, vw, vh = item['bbox_norm']

                # --- COORDINATE CONVERSION ---
                # Vision uses Bottom-Left origin (normalized 0-1)
                # PDF uses Top-Left origin (points)
                
                box_w = vw * pdf_w
                box_h = vh * pdf_h
                x_left = vx * pdf_w
                
                # Flip Y axis: 
                # Vision Top = (vy + vh). PDF Top = Height - (Vision Top * Height)
                y_top = pdf_h * (1.0 - (vy + vh))
                
                rect_coords = fitz.Rect(x_left, y_top, x_left + box_w, y_top + box_h)

                # Draw Red Box
                pdf_page.draw_rect(rect_coords, color=(1, 0, 0), width=0.5)

                # Draw Text Overlay
                # Ensure text stays on page (y > 5)
                text_pos_y = y_top - 2
                if text_pos_y < 5: text_pos_y = y_top + box_h + 5
                
                text_pos = fitz.Point(x_left, text_pos_y)

                pdf_page.insert_text(
                    text_pos, 
                    text, 
                    fontsize=6, 
                    fontname="helv", 
                    color=(1, 0, 0)
                )

        doc.save(output_path)
        doc.close()
        print(f"Saved -> {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Critical error processing {input_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform != "darwin":
        print("Error: This script requires macOS (Darwin) for Apple Vision framework.")
        sys.exit(1)

    current_dir = os.getcwd()
    path_to_explore = os.path.join(current_dir, 'resumes')
    output_dir = os.path.join(current_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_to_explore):
        print(f"Directory not found: {path_to_explore}")
        sys.exit(1)

    contents = os.listdir(path_to_explore)

    for i in contents:
        if not i.lower().endswith('.pdf') or i.startswith('.'): 
            continue 

        file_path = os.path.join(path_to_explore, i)
        
        # Clean output filename: name_annotated.pdf
        base_name = os.path.splitext(i)[0]
        output_filename = f"{base_name}_annotated.pdf"
        output_path = os.path.join(output_dir, output_filename)

        annotate_pdf(file_path, output_path)

    print("\n--- Batch Processing Complete ---")