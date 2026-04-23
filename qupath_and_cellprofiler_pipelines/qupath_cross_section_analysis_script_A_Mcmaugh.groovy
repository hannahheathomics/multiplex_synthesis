/**
 * Cellpose Detection Script — IBU_C Updated Histology
 * Fixed: auto-selects full image annotation so batch running works
 *        without needing manual selection each time
 *
 * Cleanup approach: CLASSIFY rather than DELETE
 * ─────────────────────────────────────────────
 * Flagged fibers are tagged with a named classification instead of
 * being removed. This preserves the full dataset for auditing and
 * allows sensitivity analyses (with vs. without flagged fibers),
 * which is strongly preferred for publication.
 *
 * QuPath will colour-code each class differently in the viewer,
 * making it easy to visually inspect what was flagged and why.
 *
 * Classes assigned:
 *   "Fiber - Validated"   → passes all QC thresholds (use for CSA)
 *   "Fiber - Too Small"   → likely segmentation fragment or debris
 *   "Fiber - Too Large"   → likely merged / unsegmented fibers
 *   "Fiber - Irregular"   → low circularity, edge artifact or damage
 *   "Fiber - No Measure"  → measurements missing, cannot assess
 */

import qupath.ext.biop.cellpose.Cellpose2D

// ── Redirect temp directory to external SanDisk SSD ──────────────────────────
// This fixes the "No usable temporary directory found" Cellpose crash
// caused by a full internal disk. Update the drive name below if your
// SanDisk appears under a different name in Finder.
def tmpDir = new File("/Volumes/AM Drive/cellpose_tmp")
tmpDir.mkdirs()
System.setProperty("java.io.tmpdir", tmpDir.absolutePath)
println "✓ Temp directory set to: ${tmpDir.absolutePath}"

// ── Create full image annotation if none exists ───────────────────────────────
createFullImageAnnotation(true)

def annotations = getAnnotationObjects()
if (annotations == null || annotations.isEmpty()) {
    Dialogs.showErrorMessage("Cellpose", "No annotations found in this image!")
    return
}
selectObjects(annotations)

// ── Cellpose builder ──────────────────────────────────────────────────────────
def pathModel = 'cyto3'
def cellpose = Cellpose2D.builder(pathModel)
        .pixelSize(0.75488)                // True pixel size from acquisition software (µm/px)
        .channels('Green')                 // Laminin channel for fiber detection
//      .cellprobThreshold(0.0)            // Tune if over/under detecting
//      .flowThreshold(0.4)                // Tune if shapes look wrong
//      .diameter(15)                      // Set if fibers are consistent size
        .measureShape()                    // Adds area, perimeter, circularity etc.
        .createAnnotations()               // Creates annotations (not detections)
        .excludeEdges()                    // Excludes fibers touching the image boundary (cut tissue)
        .build()

// ── Run detection ─────────────────────────────────────────────────────────────
def imageData = getCurrentImageData()
def pathObjects = getSelectedObjects() ?: []

if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("Cellpose", "Please select a parent object!")
    return
}

cellpose.detectObjects(imageData, pathObjects)
println '✓ Cellpose detection done'

// ── CSA Cleanup — CLASSIFY approach ──────────────────────────────────────────
// Thresholds — set these BEFORE looking at group-level results.
// Validate on a blinded random subset and document in your Methods.
def MIN_AREA_UM2    = 200.0    // µm² — below this = likely debris (~<16 µm diameter)
def MAX_AREA_UM2    = 20000.0  // µm² — above this = likely merged (~>160 µm diameter)
def MIN_CIRCULARITY = 0.3      // 0 (line) → 1 (perfect circle)

// ── Define named classifications ──────────────────────────────────────────────
// getPathClass() is the scripting-context shortcut available in all QuPath
// versions including 0.7.0 — avoids direct API class resolution entirely
def classValidated  = getPathClass("Fiber - Validated")
def classTooSmall   = getPathClass("Fiber - Too Small")
def classTooLarge   = getPathClass("Fiber - Too Large")
def classIrregular  = getPathClass("Fiber - Irregular")
def classNoMeasure  = getPathClass("Fiber - No Measure")

// ── Counters for the summary log ──────────────────────────────────────────────
int nValidated  = 0
int nTooSmall   = 0
int nTooLarge   = 0
int nIrregular  = 0
int nNoMeasure  = 0

// ── Classify every detected annotation ───────────────────────────────────────
// NOTE: We re-fetch annotations AFTER Cellpose detection so we get the
// newly created fiber annotations, not just the parent full-image one.
def fiberAnnotations = getAnnotationObjects().findAll { ann ->
    // Skip the full-image parent annotation (it won't have fiber measurements)
    ann.getROI() != null && !ann.isRootObject()
}

fiberAnnotations.each { annotation ->
    def ml          = annotation.getMeasurementList()
    // getMeasurementValue returns Double.NaN (not null) when missing in QuPath v0.4+
    double area        = ml.get("Area µm^2")
    double circularity = ml.get("Circularity")

    if (Double.isNaN(area) || Double.isNaN(circularity)) {
        annotation.setPathClass(classNoMeasure)
        nNoMeasure++
    } else if (area < MIN_AREA_UM2) {
        annotation.setPathClass(classTooSmall)
        nTooSmall++
    } else if (area > MAX_AREA_UM2) {
        annotation.setPathClass(classTooLarge)
        nTooLarge++
    } else if (circularity < MIN_CIRCULARITY) {
        annotation.setPathClass(classIrregular)
        nIrregular++
    } else {
        annotation.setPathClass(classValidated)
        nValidated++
    }
}

fireHierarchyUpdate()

// ── Summary log ───────────────────────────────────────────────────────────────
int nTotal   = fiberAnnotations.size()
int nFlagged = nTooSmall + nTooLarge + nIrregular + nNoMeasure
double pctFlagged = nTotal > 0 ? (nFlagged / nTotal * 100).round(1) : 0

println ""
println "══════════════════════════════════════════"
println "  CSA Classification Summary"
println "══════════════════════════════════════════"
println "  Total fibers detected : ${nTotal}"
println "  ✅ Validated (use for CSA): ${nValidated}"
println "  ── Flagged (${pctFlagged}% of total) ──"
println "     Too small  (<${MIN_AREA_UM2} µm²)    : ${nTooSmall}"
println "     Too large  (>${MAX_AREA_UM2} µm²)  : ${nTooLarge}"
println "     Irregular  (circ <${MIN_CIRCULARITY})     : ${nIrregular}"
println "     No measure                  : ${nNoMeasure}"
println "══════════════════════════════════════════"
println ""
println "💡 Tip: Run your CSA analysis TWICE —"
println "   (1) All 'Fiber - Validated' annotations only"
println "   (2) All annotations combined"
println "   If conclusions are consistent, your cleanup is defensible."
println ""
println "💡 Tip: Use Measure > Export measurements in QuPath to export"
println "   per-fiber area and class to CSV for statistical analysis."
