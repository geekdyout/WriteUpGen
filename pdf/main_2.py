import sys
import Quartz
import Vision
import AppKit
from Cocoa import NSURL
from CoreFoundation import CFURLCreateWithFileSystemPath, kCFURLPOSIXPathStyle

# =================CONFIGURATION=================
RENDER_SCALE = 3.5        # Higher = better detection for small fonts (2.0 is std, 3.0 is high res)
ANNOTATION_LINE_WIDTH = 1.0 # Thinner lines so we don't block the text we are highlighting
# ===============================================

def render_page_to_image(page, scale):
    """
    Renders a PDFPage to a generic CGImage for Vision processing.
    """
    page_rect = page.boundsForBox_(Quartz.kPDFDisplayBoxMediaBox)
    width = int(page_rect.size.width * scale)
    height = int(page_rect.size.height * scale)
    
    # Create a Bitmap Representation
    img_rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, width, height, 8, 4, True, False, AppKit.NSDeviceRGBColorSpace, 0, 0
    )
    
    # Setup Drawing Context
    ns_ctx = AppKit.NSGraphicsContext.graphicsContextWithBitmapImageRep_(img_rep)
    AppKit.NSGraphicsContext.setCurrentContext_(ns_ctx)
    cg_ctx = ns_ctx.CGContext()
    
    # Scale the Context (Zoom in)
    Quartz.CGContextScaleCTM(cg_ctx, scale, scale)
    
    # Draw White Background
    AppKit.NSColor.whiteColor().set()
    AppKit.NSRectFill(page_rect)
    
    # Draw the PDF Page
    page.drawWithBox_toContext_(Quartz.kPDFDisplayBoxMediaBox, cg_ctx)
    
    return img_rep.CGImage()

def get_text_observations(cg_image):
    """
    Runs Apple Vision Text Recognition on a CGImage.
    """
    req = Vision.VNRecognizeTextRequest.alloc().init()
    # 1 = VNRequestRecognitionLevelAccurate (Best for small/bad fonts)
    req.setRecognitionLevel_(0) 
    req.setUsesLanguageCorrection_(True) # Fixes typos based on context
    
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    success, error = handler.performRequests_error_([req], None)
    
    if not success:
        print(f"[Error] Vision Request failed: {error}")
        return []
    
    return req.results()

def calculate_split_x(observations):
    """
    Detects the visual 'gutter' (white space) between columns.
    Returns normalized X coordinate (0.0 - 1.0) or None if single column.
    """
    if not observations: return None

    # 1. Build Histogram (0-100 buckets)
    density_map = [0] * 100
    for obs in observations:
        bbox = obs.boundingBox()
        # Clamp coordinates
        start = max(0, int(bbox.origin.x * 100))
        end = min(100, int((bbox.origin.x + bbox.size.width) * 100))
        
        for i in range(start, end):
            if i < 100: density_map[i] += 1

    # 2. Find Gaps in the "Middle" of the page (20% - 80%)
    search_start, search_end = 20, 80
    longest_gap_start = -1
    current_gap_start = -1
    max_gap_len = 0
    
    for i in range(search_start, search_end):
        if density_map[i] == 0:
            if current_gap_start == -1: current_gap_start = i
        else:
            if current_gap_start != -1:
                gap_len = i - current_gap_start
                if gap_len > max_gap_len:
                    max_gap_len = gap_len
                    longest_gap_start = current_gap_start
                current_gap_start = -1
    
    # Check if gap ended at the search boundary
    if current_gap_start != -1:
        gap_len = search_end - current_gap_start
        if gap_len > max_gap_len:
            max_gap_len = gap_len
            longest_gap_start = current_gap_start

    # 3. Validate Gap (Must be > 2% of page width to be a real column split)
    if max_gap_len > 2:
        split_center = longest_gap_start + (max_gap_len / 2)
        return split_center / 100.0
        
    return None

def process_pdf(input_path, output_path):
    # Load Input PDF
    in_url = CFURLCreateWithFileSystemPath(None, input_path, kCFURLPOSIXPathStyle, False)
    pdf_doc = Quartz.PDFDocument.alloc().initWithURL_(in_url)
    
    if not pdf_doc:
        print(f"Error: Could not open {input_path}")
        return

    # Create Output PDF Context (Destination)
    out_url = CFURLCreateWithFileSystemPath(None, output_path, kCFURLPOSIXPathStyle, False)
    
    # We need the first page's rect to initialize the context, 
    # but we'll update the media box per page in the loop.
    first_page = pdf_doc.pageAtIndex_(0)
    first_rect = first_page.boundsForBox_(Quartz.kPDFDisplayBoxMediaBox)
    
    write_ctx = Quartz.CGPDFContextCreateWithURL(out_url, first_rect, None)
    
    total_pages = pdf_doc.pageCount()
    print(f"Processing {total_pages} page(s)...")

    for i in range(total_pages):
        print(f"  > Analyzing Page {i+1}...")
        page = pdf_doc.pageAtIndex_(i)
        page_rect = page.boundsForBox_(Quartz.kPDFDisplayBoxMediaBox)
        
        # 1. Render & Recognize
        cg_image = render_page_to_image(page, RENDER_SCALE)
        observations = get_text_observations(cg_image)
        
        # 2. Calculate Layout
        split_x = calculate_split_x(observations)
        
        # 3. Start Writing Page
        Quartz.CGPDFContextBeginPage(write_ctx, None) # Uses default media box from creation? 
        # Better: Pass the specific box for this page to handle varying page sizes
        # Note: PyObjC binding for BeginPage takes a dictionary for page info, usually None is fine if sizes match.
        # If pages differ in size, we rely on the context to inherit current transformation or crop.
        # For strict sizing, we draw the original page into the context rect.
        
        # Draw Original Content
        Quartz.CGContextSaveGState(write_ctx)
        page.drawWithBox_toContext_(Quartz.kPDFDisplayBoxMediaBox, write_ctx)
        Quartz.CGContextRestoreGState(write_ctx)
        
        # 4. Draw Debug Annotations
        page_w = page_rect.size.width
        page_h = page_rect.size.height

        # A. Draw Split Line
        if split_x:
            Quartz.CGContextSetRGBStrokeColor(write_ctx, 1.0, 0.0, 1.0, 1.0) # Magenta
            Quartz.CGContextSetLineWidth(write_ctx, 2.0)
            Quartz.CGContextMoveToPoint(write_ctx, split_x * page_w, 0)
            Quartz.CGContextAddLineToPoint(write_ctx, split_x * page_w, page_h)
            Quartz.CGContextStrokePath(write_ctx)

        # B. Draw Text Boxes
        Quartz.CGContextSetLineWidth(write_ctx, ANNOTATION_LINE_WIDTH)
        
        for obs in observations:
            bbox = obs.boundingBox()
            # Vision Coords: (0,0) is Bottom-Left. PDF Coords: (0,0) is Bottom-Left.
            # Direct mapping works.
            rect = Quartz.CGRectMake(
                bbox.origin.x * page_w,
                bbox.origin.y * page_h,
                bbox.size.width * page_w,
                bbox.size.height * page_h
            )
            
            x_min = bbox.origin.x
            x_max = bbox.origin.x + bbox.size.width
            
            # Determine Color based on Layout
            if split_x:
                # Header (Red): Spans across the split line (+/- 5% tolerance)
                if x_min < (split_x - 0.05) and x_max > (split_x + 0.05):
                    Quartz.CGContextSetRGBStrokeColor(write_ctx, 1.0, 0.0, 0.0, 1.0) # Red
                # Left Column (Blue)
                elif x_min < split_x:
                    Quartz.CGContextSetRGBStrokeColor(write_ctx, 0.0, 0.0, 1.0, 1.0) # Blue
                # Right Column (Green)
                else:
                    Quartz.CGContextSetRGBStrokeColor(write_ctx, 0.0, 0.6, 0.0, 1.0) # Green
            else:
                # Single Column (Red)
                Quartz.CGContextSetRGBStrokeColor(write_ctx, 1.0, 0.0, 0.0, 1.0) 

            Quartz.CGContextStrokeRect(write_ctx, rect)

        Quartz.CGPDFContextEndPage(write_ctx)

    Quartz.CGPDFContextClose(write_ctx)
    print(f"Done. Annotated PDF saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_resume_advanced.py <input.pdf>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = "debug_annotated.pdf"
    
    process_pdf(input_file, output_file)