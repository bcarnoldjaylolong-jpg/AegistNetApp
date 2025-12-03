package com.kidsafe.secure.nsfw

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import kotlin.math.max
import kotlin.math.min

/**
 * Data class representing a detection prediction
 */
data class Prediction(
    val x: Float,      // Center X coordinate
    val y: Float,      // Center Y coordinate
    val w: Float,      // Width
    val h: Float,      // Height
    val confidence: Float,
    val className: String = "not_safe",
    val classId: Int = 0
) {
    // Optimized: Create RectF once to avoid allocation during NMS loops
    // Convert from center format (x, y, w, h) to corner format (left, top, right, bottom)
    val boundingBox: RectF by lazy {
        RectF(
            x - w / 2f,  // left
            y - h / 2f,  // top
            x + w / 2f,  // right
            y + h / 2f   // bottom
        )
    }

    override fun toString(): String {
        return "Prediction(center=($x, $y), size=($w, $h), conf=$confidence, box=${boundingBox.toShortString()})"
    }
}

/**
 * Content detector for NSFW images using TensorFlow Lite model
 * Detects inappropriate content in any part of the screen/image
 */
class RoboflowContentDetector(private val context: Context) {

    companion object {
        private const val TAG = "RoboflowDetector"
        private const val MODEL_FILE = "best_float32.tflite"
        private const val INPUT_SIZE = 512
    }

    private val CONF_THRESH = 0.10f
    private val IOU_THRESH = 0.45f

    private var interpreter: Interpreter? = null
    private var imageProcessor: ImageProcessor? = null

    // Pre-allocated buffers to avoid Garbage Collection churn
    private var outputBuffer: Array<FloatArray>? = null
    private var outputWrapper: Array<Array<FloatArray>>? = null
    private var isOutputRotated = false
    private var numAnchors = 0

    init {
        try {
            val model = FileUtil.loadMappedFile(context, MODEL_FILE)

            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(false)
            }

            interpreter = Interpreter(model, options)

            val inputShape = interpreter?.getInputTensor(0)?.shape()
            val outputTensor = interpreter?.getOutputTensor(0)
            val outputShape = outputTensor?.shape()
            Log.d(TAG, "Model loaded - Input: ${inputShape?.contentToString()}, Output: ${outputShape?.contentToString()}")

            // Pre-calculate output logic
            if (outputShape != null) {
                val dim1 = outputShape[1]
                val dim2 = outputShape[2]

                isOutputRotated = dim1 == 5 && dim2 > 100
                numAnchors = if (isOutputRotated) dim2 else dim1

                // Allocate buffer ONCE
                outputBuffer = Array(dim1) { FloatArray(dim2) }
                outputWrapper = arrayOf(outputBuffer!!)
            }

            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(CastOp(DataType.FLOAT32))
                .add(NormalizeOp(0f, 255f))
                .build()

        } catch (e: Exception) {
            Log.e(TAG, "Model load failed", e)
        }
    }

    /**
     * Detects objects in the provided bitmap.
     * Optimization: Reuses buffers and objects to minimize GC pauses.
     */
    suspend fun detect(bitmap: Bitmap): List<Prediction> {
        return withContext(Dispatchers.Default) {
            val tfl = interpreter ?: return@withContext emptyList()
            val processor = imageProcessor ?: return@withContext emptyList()
            val outWrapper = outputWrapper ?: return@withContext emptyList()
            val outBuffer = outputBuffer ?: return@withContext emptyList()

            val originalWidth = bitmap.width
            val originalHeight = bitmap.height

            Log.d(TAG, "=== DETECTION START ===")
            Log.d(TAG, "Original image size: ${originalWidth}x${originalHeight}")

            // 1. Load Bitmap into TensorImage
            var tensorImage = TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)
            tensorImage = processor.process(tensorImage)

            // 2. Run Inference using pre-allocated buffer
            tfl.run(tensorImage.buffer, outWrapper)

            // 3. Parse Results with correct coordinate interpretation
            val raw = parsePredictions(outBuffer, isOutputRotated, originalWidth, originalHeight)

            Log.d(TAG, "Raw detections: ${raw.size}")
            raw.take(3).forEach { pred ->
                Log.d(TAG, "Raw prediction: $pred")
            }

            // 4. NMS
            val nms = nonMaxSuppression(raw)

            Log.d(TAG, "After NMS: ${nms.size} detections")
            nms.forEachIndexed { index, pred ->
                Log.d(TAG, "Final #$index: center=(${pred.x}, ${pred.y}), size=(${pred.w}x${pred.h}), box=${pred.boundingBox.toShortString()}, conf=${pred.confidence}")
            }
            Log.d(TAG, "=== DETECTION END ===")

            return@withContext nms
        }
    }

    private fun parsePredictions(
        output: Array<FloatArray>,
        isRotated: Boolean,
        originalWidth: Int,
        originalHeight: Int
    ): List<Prediction> {
        val list = ArrayList<Prediction>(20)

        val anchors = numAnchors

        Log.d(TAG, "parsePredictions - isRotated: $isRotated, anchors: $anchors")
        Log.d(TAG, "Original dimensions: ${originalWidth}x${originalHeight}")

        // Log sample raw values to understand the model output format
        if (output.isNotEmpty() && output[0].isNotEmpty()) {
            if (isRotated) {
                Log.d(TAG, "Sample raw values (rotated): x[0]=${output[0][0]}, y[0]=${output[1][0]}, w[0]=${output[2][0]}, h[0]=${output[3][0]}, conf[0]=${output[4][0]}")
            } else {
                Log.d(TAG, "Sample raw values (non-rotated): [${output[0].take(6).joinToString(", ")}]")
            }
        }

        if (isRotated) {
            // Format [5, N] -> [feature][i]
            val xs = output[0]
            val ys = output[1]
            val ws = output[2]
            val hs = output[3]
            val confs = output[4]

            var detectedCount = 0
            for (i in 0 until anchors) {
                val conf = confs[i]
                if (conf < CONF_THRESH) continue

                detectedCount++

                // YOLO format: coordinates are normalized [0,1] OR already in pixels
                // We need to check if values are > 1 to determine the format
                val isNormalized = xs[i] <= 1.0f && ys[i] <= 1.0f && ws[i] <= 1.0f && hs[i] <= 1.0f

                val centerX: Float
                val centerY: Float
                val width: Float
                val height: Float

                if (isNormalized) {
                    // Values are normalized [0,1], scale to image size
                    centerX = xs[i] * originalWidth
                    centerY = ys[i] * originalHeight
                    width = ws[i] * originalWidth
                    height = hs[i] * originalHeight
                } else {
                    // Values are already in pixels relative to INPUT_SIZE (512)
                    // Scale them to original image size
                    val scaleX = originalWidth.toFloat() / INPUT_SIZE
                    val scaleY = originalHeight.toFloat() / INPUT_SIZE
                    centerX = xs[i] * scaleX
                    centerY = ys[i] * scaleY
                    width = ws[i] * scaleX
                    height = hs[i] * scaleY
                }

                if (detectedCount <= 3) {
                    Log.d(TAG, "Detection $detectedCount: raw(${xs[i]}, ${ys[i]}, ${ws[i]}, ${hs[i]}) -> pixel($centerX, $centerY, $width, $height), conf=$conf, normalized=$isNormalized")
                }

                list.add(
                    Prediction(
                        x = centerX,
                        y = centerY,
                        w = width,
                        h = height,
                        confidence = conf
                    )
                )
            }
            Log.d(TAG, "Total detections (rotated): $detectedCount")
        } else {
            // Format [N, 6] -> [i][feature]
            var detectedCount = 0
            for (i in 0 until anchors) {
                val row = output[i]
                val conf = row[4]

                if (conf < CONF_THRESH) continue

                detectedCount++

                // Check if coordinates are normalized [0,1] or in pixels
                val isNormalized = row[0] <= 1.0f && row[1] <= 1.0f && row[2] <= 1.0f && row[3] <= 1.0f

                val centerX: Float
                val centerY: Float
                val width: Float
                val height: Float

                if (isNormalized) {
                    // Values are normalized [0,1], scale to image size
                    centerX = row[0] * originalWidth
                    centerY = row[1] * originalHeight
                    width = row[2] * originalWidth
                    height = row[3] * originalHeight
                } else {
                    // Values are already in pixels relative to INPUT_SIZE (512)
                    // Scale them to original image size
                    val scaleX = originalWidth.toFloat() / INPUT_SIZE
                    val scaleY = originalHeight.toFloat() / INPUT_SIZE
                    centerX = row[0] * scaleX
                    centerY = row[1] * scaleY
                    width = row[2] * scaleX
                    height = row[3] * scaleY
                }

                if (detectedCount <= 3) {
                    Log.d(TAG, "Detection $detectedCount: raw(${row[0]}, ${row[1]}, ${row[2]}, ${row[3]}) -> pixel($centerX, $centerY, $width, $height), conf=$conf, normalized=$isNormalized")
                }

                list.add(
                    Prediction(
                        x = centerX,
                        y = centerY,
                        w = width,
                        h = height,
                        confidence = conf
                    )
                )
            }
            Log.d(TAG, "Total detections (non-rotated): $detectedCount")
        }

        return list
    }

    private fun nonMaxSuppression(preds: List<Prediction>): List<Prediction> {
        if (preds.isEmpty()) return emptyList()

        val sorted = preds.sortedByDescending { it.confidence }.toMutableList()
        val keep = ArrayList<Prediction>(preds.size)

        while (sorted.isNotEmpty()) {
            val top = sorted.removeAt(0)
            keep.add(top)

            val topBox = top.boundingBox

            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                val next = iterator.next()
                if (iou(topBox, next.boundingBox) > IOU_THRESH) {
                    iterator.remove()
                }
            }
        }

        return keep
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val w = right - left
        val h = bottom - top

        if (w <= 0f || h <= 0f) return 0f

        val inter = w * h
        val union = a.width() * a.height() + b.width() * b.height() - inter

        return if (union <= 0f) 0f else inter / union
    }

    fun close() {
        interpreter?.close()
        interpreter = null
        outputBuffer = null
        outputWrapper = null
    }
}