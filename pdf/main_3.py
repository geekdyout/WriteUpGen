import os
import Quartz
import Vision
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# --- Configuration ---
INPUT_PDF = "resume.pdf"
OUTPUT_PDF = "resume_annotated_final.pdf"

class GeometryUtils:
    @staticmethod
    def merge_boxes(boxes):
        if not boxes: return [0,0,0,0]
        return [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)
        ]

class ColumnDetector:
    @staticmethod
    def find_split_x(boxes, page_width, page_height):
        # 1. Setup Histogram
        bins = 400
        density_map = np.zeros(bins)
        
        # 2. Smart Filtering
        # We ignore boxes that are likely headers or full-width elements
        # to prevent them from "hiding" the column gap.
        valid_boxes = []
        for box in boxes:
            # Filter A: Ignore Top 15% (likely Name/Title headers)
            if box[1] < (page_height * 0.15): continue 
            
            # Filter B: Ignore Full Width Elements (>85% of page width)
            box_w = box[2] - box[0]
            if box_w > (page_width * 0.85): continue
                
            valid_boxes.append(box)

        # Fallback: If filtering removed everything, use original boxes
        if not valid_boxes: valid_boxes = boxes

        # 3. Build Histogram
        for box in valid_boxes:
            start = max(0, int((box[0] / page_width) * bins))
            end = min(bins, int((box[2] / page_width) * bins))
            density_map[start:end] = 1

        # 4. Search for Gaps (Middle 20% - 80% of page width)
        search_start = int(bins * 0.20)
        search_end = int(bins * 0.80)
        
        gaps = []
        current_gap_start = -1
        
        for i in range(search_start, search_end):
            if density_map[i] == 0:
                if current_gap_start == -1: current_gap_start = i
            else:
                if current_gap_start != -1:
                    gaps.append((current_gap_start, i))
                    current_gap_start = -1
        
        if current_gap_start != -1:
             gaps.append((current_gap_start, search_end))

        if not gaps: return None

        # 5. Widest Gap Threshold
        # Gap must be > 2% of page width to be considered a column split
        widest_gap = max(gaps, key=lambda g: g[1] - g[0])
        gap_width_bins = widest_gap[1] - widest_gap[0]

        if gap_width_bins > (bins * 0.02):
            center_bin = widest_gap[0] + (gap_width_bins / 2)
            return (center_bin / bins) * page_width
            
        return None

class LayoutEngine:
    def process_page(self, raw_items, page_width, page_height):
        if not raw_items: return [], None
        
        # 1. Detect Split Line
        raw_boxes = [item['box'] for item in raw_items]
        split_x = ColumnDetector.find_split_x(raw_boxes, page_width, page_height)
        
        # 2. Form Lines (Horizontal Clustering)
        raw_items.sort(key=lambda x: x['box'][1])
        lines = []
        current_line = []
        
        for item in raw_items:
            if not current_line:
                current_line.append(item)
                continue
            
            last = current_line[-1]
            
            # Metrics
            min_y = max(last['box'][1], item['box'][1])
            max_y = min(last['box'][3], item['box'][3])
            overlap_h = max(0, max_y - min_y)
            min_h = min(last['box'][3]-last['box'][1], item['box'][3]-item['box'][1])
            overlap_ratio = overlap_h / min_h if min_h > 0 else 0
            h_gap = item['box'][0] - last['box'][2]
            
            # STRICT BARRIER CHECK (Horizontal)
            # Prevent merging words across the split line
            crosses_barrier = False
            if split_x:
                buff = 5 # Small buffer for noise
                # Check if Left Box is merging with Right Box
                is_left_last = last['box'][2] < (split_x + buff)
                is_right_item = item['box'][0] > (split_x - buff)
                
                is_right_last = last['box'][0] > (split_x - buff)
                is_left_item = item['box'][2] < (split_x + buff)
                
                if (is_left_last and is_right_item) or (is_right_last and is_left_item):
                    crosses_barrier = True

            # Thresholds: High overlap, small horizontal gap
            if overlap_ratio > 0.4 and h_gap < 50 and not crosses_barrier:
                current_line.append(item)
            else:
                lines.append(self._merge_items(current_line))
                current_line = [item]
        
        if current_line: lines.append(self._merge_items(current_line))
            
        # 3. Form Blocks (Vertical Clustering)
        final_blocks = self._cluster_vertical(lines, split_x)
        return final_blocks, split_x

    def _cluster_vertical(self, lines, split_x):
        if not lines: return []
        lines.sort(key=lambda x: x['box'][1])
        
        blocks = []
        current_block_lines = []
        
        # Calculate Global Average Height (as a fallback baseline)
        global_avg_h = sum((l['box'][3]-l['box'][1]) for l in lines) / len(lines) if lines else 20
        
        for line in lines:
            if not current_block_lines:
                current_block_lines.append(line)
                continue
                
            last_line = current_block_lines[-1]
            v_gap = line['box'][1] - last_line['box'][3]
            
            # --- ADAPTIVE THRESHOLD LOGIC ---
            # Use the MAX height of the pair (Previous vs Current)
            h_prev = last_line['box'][3] - last_line['box'][1]
            h_curr = line['box'][3] - line['box'][1]
            dominant_h = max(h_prev, h_curr)
            
            # Multiplier: Tighter (1.5x) for small text, Looser (2.0x) for headers
            if dominant_h < global_avg_h:
                multiplier = 1.5 
            else:
                multiplier = 2.0
                
            threshold = max(dominant_h * multiplier, 5.0) # Floor of 5px
            
            is_close = v_gap < threshold
            
            # Relaxed Alignment (80px) to handle indentation/bullets
            left_diff = abs(line['box'][0] - last_line['box'][0])
            right_diff = abs(line['box'][2] - last_line['box'][2])
            is_aligned = (left_diff < 80) or (right_diff < 80)

            # STRICT BARRIER CHECK (Vertical)
            # Prevent merging a Left Column Block with a Right Column Line
            crosses_barrier = False
            if split_x:
                block_box = GeometryUtils.merge_boxes([l['box'] for l in current_block_lines])
                
                # Check Center Points
                block_center = block_box[0] + (block_box[2]-block_box[0])/2
                line_center = line['box'][0] + (line['box'][2]-line['box'][0])/2
                
                block_side = "L" if block_center < split_x else "R"
                line_side = "L" if line_center < split_x else "R"
                
                if block_side != line_side:
                    crosses_barrier = True

            if is_close and is_aligned and not crosses_barrier:
                current_block_lines.append(line)
            else:
                blocks.append(self._merge_items(current_block_lines, join_char="\n"))
                current_block_lines = [line]
                
        if current_block_lines: blocks.append(self._merge_items(current_block_lines, join_char="\n"))
        return blocks

    def _merge_items(self, items, join_char=" "):
        box = GeometryUtils.merge_boxes([i['box'] for i in items])
        items.sort(key=lambda x: x['box'][0])
        text = join_char.join([i['text'] for i in items])
        return {'text': text, 'box': box}

def perform_ocr(image_path):
    url = Quartz.NSURL.fileURLWithPath_(image_path)
    img_ref = Quartz.CGImageSourceCreateImageAtIndex(Quartz.CGImageSourceCreateWithURL(url, None), 0, None)
    w = Quartz.CGImageGetWidth(img_ref)
    h = Quartz.CGImageGetHeight(img_ref)
    
    req = Vision.VNRecognizeTextRequest.alloc().init()
    # Level 1 = Accurate (Best for small fonts like addresses)
    req.setRecognitionLevel_(1) 
    req.setUsesLanguageCorrection_(True)
    
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img_ref, {})
    success, _ = handler.performRequests_error_([req], None)
    
    raw_items = []
    if success:
        for obs in req.results():
            bbox = obs.boundingBox()
            # Vision (Bottom-Left) -> PIL (Top-Left) Conversion
            x = int(bbox.origin.x * w)
            y = int((1.0 - bbox.origin.y - bbox.size.height) * h)
            w_box = int(bbox.size.width * w)
            h_box = int(bbox.size.height * h)
            raw_items.append({'text': str(obs.text()), 'box': [x, y, x+w_box, y+h_box]})
            
    return raw_items, w, h

def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: {INPUT_PDF} not found.")
        return

    print(f"Processing {INPUT_PDF}...")
    # Rasterize PDF to images (300 DPI for clarity)
    images = convert_from_path(INPUT_PDF, dpi=300)
    annotated_images = []
    engine = LayoutEngine()
    
    for i, img in enumerate(images):
        print(f"  > Analyzing Page {i+1}...")
        temp_path = f"temp_page_{i}.png"
        img.save(temp_path)
        
        # 1. OCR & Layout Analysis
        raw_items, w, h = perform_ocr(temp_path)
        blocks, split_x = engine.process_page(raw_items, w, h)
        
        # 2. Annotation
        # Convert to RGB to draw colors
        draw_img = img.copy().convert("RGB")
        draw = ImageDraw.Draw(draw_img)
        
        # Draw Split Line (Magenta)
        if split_x:
            draw.line([(split_x, 0), (split_x, h)], fill="magenta", width=4)
        
        # Draw Text Blocks (Bounding Boxes)
        for b in blocks:
            box = b['box']
            color = "red" # Default (Header / Spanning)
            
            if split_x:
                center_x = box[0] + (box[2] - box[0])/2
                if center_x < split_x: color = "blue"   # Left Column
                elif center_x > split_x: color = "green" # Right Column
            
            # Outline only (Clean look)
            draw.rectangle(box, outline=color, width=3)
        
        annotated_images.append(draw_img)
        os.remove(temp_path)

    # Save final Multipage PDF
    if annotated_images:
        print(f"Saving to {OUTPUT_PDF}...")
        annotated_images[0].save(
            OUTPUT_PDF, "PDF", 
            resolution=100.0, 
            save_all=True, 
            append_images=annotated_images[1:]
        )
        print("Done.")

if __name__ == "__main__":
    main()