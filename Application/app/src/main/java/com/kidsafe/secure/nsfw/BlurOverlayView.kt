package com.kidsafe.secure.nsfw

import android.content.Context
import android.graphics.*
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
import android.util.Log
import android.view.View

class BlurOverlayView(context: Context) : View(context) {

    companion object {
        private const val TAG = "BlurOverlayView"
        private const val BLUR_RADIUS = 22f
        private const val BLUR_SAMPLE_FACTOR = 0.35f
    }

    private val blurPaint = Paint(Paint.ANTI_ALIAS_FLAG)

    // Opaque solid fill paint
    private val boundsPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL  // Changed to FILL for solid block
        color = Color.argb(255, 255, 0, 0) // Opaque RED
    }

    // Border paint for debugging (optional)
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.argb(255, 255, 255, 0) // Yellow border
        strokeWidth = 3f
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        typeface = Typeface.DEFAULT_BOLD
    }

    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        alpha = 255
    }

    private var predictions: List<Prediction> = emptyList()
    private var blurredRegions = mutableMapOf<String, BlurredRegion>()
    private var renderScript: RenderScript? = null
    private var previousBitmap: Bitmap? = null

    // View dimensions for proper scaling
    private var viewWidth = 0
    private var viewHeight = 0
    private var imageWidth = 0
    private var imageHeight = 0

    // Solid fill mode by default
    private var showBoundingBoxes = true

    data class BlurredRegion(
        val bitmap: Bitmap,
        val rect: RectF,
        val confidence: Float
    )

    init {
        setLayerType(LAYER_TYPE_SOFTWARE, null)

        try {
            renderScript = RenderScript.create(context)
            Log.d(TAG, "RenderScript initialized")
        } catch (e: Exception) {
            Log.e(TAG, "RenderScript unavailable. Using bounding boxes only.", e)
            showBoundingBoxes = true
        }
    }

    fun updatePredictions(newPredictions: List<Prediction>, sourceBitmap: Bitmap?) {
        predictions = newPredictions

        Log.d(TAG, "updatePredictions - predictions: ${predictions.size}, bitmap: ${sourceBitmap != null}")

        if (sourceBitmap != null) {
            imageWidth = sourceBitmap.width
            imageHeight = sourceBitmap.height
            Log.d(TAG, "Source bitmap size: ${imageWidth}x${imageHeight}")
        }

        previousBitmap?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }

        if (sourceBitmap == null || predictions.isEmpty()) {
            clearBlurRegions()
            previousBitmap = null
            invalidate()
            return
        }

        previousBitmap = sourceBitmap.copy(sourceBitmap.config ?: Bitmap.Config.ARGB_8888, false)

        if (!showBoundingBoxes) {
            generateBlurRegions(sourceBitmap)
        }

        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        viewWidth = w
        viewHeight = h
        Log.d(TAG, "View size changed: ${viewWidth}x${viewHeight}")
    }

    private fun clearBlurRegions() {
        blurredRegions.values.forEach {
            try {
                if (!it.bitmap.isRecycled) {
                    it.bitmap.recycle()
                }
            } catch (_: Exception) {}
        }
        blurredRegions.clear()
    }

    private fun generateBlurRegions(sourceBitmap: Bitmap) {
        clearBlurRegions()
        val newMap = mutableMapOf<String, BlurredRegion>()

        for (prediction in predictions) {
            try {
                val rect = prediction.boundingBox
                val padding = 20f

                val padded = RectF(
                    (rect.left - padding).coerceAtLeast(0f),
                    (rect.top - padding).coerceAtLeast(0f),
                    (rect.right + padding).coerceAtMost(sourceBitmap.width.toFloat()),
                    (rect.bottom + padding).coerceAtMost(sourceBitmap.height.toFloat())
                )

                if (padded.width() > 0 && padded.height() > 0) {
                    val regionBitmap = Bitmap.createBitmap(
                        sourceBitmap,
                        padded.left.toInt(),
                        padded.top.toInt(),
                        padded.width().toInt(),
                        padded.height().toInt()
                    )

                    val smallWidth = (regionBitmap.width * BLUR_SAMPLE_FACTOR).toInt().coerceAtLeast(1)
                    val smallHeight = (regionBitmap.height * BLUR_SAMPLE_FACTOR).toInt().coerceAtLeast(1)
                    val small = Bitmap.createScaledBitmap(regionBitmap, smallWidth, smallHeight, false)

                    val blurred = blurBitmap(small)

                    small.recycle()
                    regionBitmap.recycle()

                    if (blurred != null) {
                        val key = "${padded.left},${padded.top}"
                        newMap[key] = BlurredRegion(blurred, padded, prediction.confidence)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Blur generation failed for prediction", e)
            }
        }

        blurredRegions = newMap
    }

    private fun blurBitmap(bitmap: Bitmap): Bitmap? {
        return try {
            val rs = renderScript ?: return null
            val input = Allocation.createFromBitmap(rs, bitmap)
            val output = Allocation.createTyped(rs, input.type)
            val blur = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))

            blur.setRadius(BLUR_RADIUS)
            blur.setInput(input)
            blur.forEach(output)

            val blurred = Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config ?: Bitmap.Config.ARGB_8888)
            output.copyTo(blurred)

            input.destroy()
            output.destroy()
            blur.destroy()
            blurred
        } catch (e: Exception) {
            Log.e(TAG, "Blur failed", e)
            null
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (predictions.isEmpty()) return

        if (showBoundingBoxes || blurredRegions.isEmpty()) {
            drawBoundingBoxes(canvas)
        } else {
            drawBlur(canvas)
        }
    }

    private fun drawBoundingBoxes(canvas: Canvas) {
        Log.d(TAG, "Drawing ${predictions.size} bounding boxes")
        Log.d(TAG, "View size: ${viewWidth}x${viewHeight}, Image size: ${imageWidth}x${imageHeight}")

        // Calculate scale factors if view and image dimensions differ
        val scaleX = if (imageWidth > 0) viewWidth.toFloat() / imageWidth else 1f
        val scaleY = if (imageHeight > 0) viewHeight.toFloat() / imageHeight else 1f

        Log.d(TAG, "Scale factors: scaleX=$scaleX, scaleY=$scaleY")

        for ((index, p) in predictions.withIndex()) {
            // Get the bounding box in image coordinates
            val imageRect = p.boundingBox

            // Scale to view coordinates
            val viewRect = RectF(
                imageRect.left * scaleX,
                imageRect.top * scaleY,
                imageRect.right * scaleX,
                imageRect.bottom * scaleY
            )

            Log.d(TAG, "Box $index: image=(${imageRect.left}, ${imageRect.top}, ${imageRect.right}, ${imageRect.bottom}) -> view=(${viewRect.left}, ${viewRect.top}, ${viewRect.right}, ${viewRect.bottom})")

            // Draw solid opaque block
            canvas.drawRect(viewRect, boundsPaint)

            // Optional: Draw border for debugging
            canvas.drawRect(viewRect, borderPaint)

            // Draw label
            val label = "NSFW ${(p.confidence * 100).toInt()}%"
            val textWidth = labelPaint.measureText(label)
            val textHeight = labelPaint.textSize

            val labelRect = RectF(
                viewRect.left,
                viewRect.top - textHeight - 12f,
                viewRect.left + textWidth + 20f,
                viewRect.top
            )

            // Ensure label stays within view bounds
            if (labelRect.top < 0) {
                labelRect.offset(0f, -labelRect.top + 5f)
            }

            canvas.drawRect(labelRect, labelBackgroundPaint)
            canvas.drawText(label, labelRect.left + 10f, labelRect.bottom - 5f, labelPaint)
        }
    }

    private fun drawBlur(canvas: Canvas) {
        // Calculate scale factors if view and image dimensions differ
        val scaleX = if (imageWidth > 0) viewWidth.toFloat() / imageWidth else 1f
        val scaleY = if (imageHeight > 0) viewHeight.toFloat() / imageHeight else 1f

        blurredRegions.values.forEach { region ->
            // Scale the region rect to view coordinates
            val viewRect = RectF(
                region.rect.left * scaleX,
                region.rect.top * scaleY,
                region.rect.right * scaleX,
                region.rect.bottom * scaleY
            )
            canvas.drawBitmap(region.bitmap, null, viewRect, blurPaint)
        }
    }

    fun cleanup() {
        clearBlurRegions()
        previousBitmap?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }
        previousBitmap = null
        try {
            renderScript?.destroy()
        } catch (_: Exception) {}
        renderScript = null
    }
}