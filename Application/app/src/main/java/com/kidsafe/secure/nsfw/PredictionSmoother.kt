package com.kidsafe.secure.nsfw

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

/**
 * Stabilizes detections by smoothing coordinates and persisting missing predictions.
 */
class PredictionSmoother {

    // Configuration
    private val IOU_THRESHOLD = 0.4f      // Intersection over Union to consider it the "same" object
    private val SMOOTHING_FACTOR = 0.6f   // 0.0 = frozen, 1.0 = no smoothing (instant). 0.6 is balanced.
    private val MAX_MISSED_FRAMES = 4     // How many frames to keep the box if detection fails

    private data class SmoothedPrediction(
        var prediction: Prediction,
        var missedFrames: Int = 0
    )

    private var activePredictions = mutableListOf<SmoothedPrediction>()

    fun process(newPredictions: List<Prediction>): List<Prediction> {
        val currentFramePredictions = newPredictions.toMutableList()
        val nextActivePredictions = mutableListOf<SmoothedPrediction>()

        // 1. Match existing tracks with new detections
        val iterator = activePredictions.iterator()
        while (iterator.hasNext()) {
            val tracked = iterator.next()

            // Find the best matching prediction in the new frame
            val bestMatchIndex = findBestMatchIndex(tracked.prediction, currentFramePredictions)

            if (bestMatchIndex != -1) {
                // MATCH FOUND: Smooth the coordinates
                val newPred = currentFramePredictions[bestMatchIndex]

                val smoothedPred = Prediction(
                    x = lerp(tracked.prediction.x, newPred.x, SMOOTHING_FACTOR),
                    y = lerp(tracked.prediction.y, newPred.y, SMOOTHING_FACTOR),
                    w = lerp(tracked.prediction.w, newPred.w, SMOOTHING_FACTOR),
                    h = lerp(tracked.prediction.h, newPred.h, SMOOTHING_FACTOR),
                    confidence = newPred.confidence,
                    className = newPred.className,
                    classId = newPred.classId
                )

                // Reset missed frames count
                nextActivePredictions.add(SmoothedPrediction(smoothedPred, 0))

                // Remove this prediction so it doesn't get matched again
                currentFramePredictions.removeAt(bestMatchIndex)
            } else {
                // NO MATCH: Keep the old one for a few frames (Anti-Flicker)
                tracked.missedFrames++
                if (tracked.missedFrames <= MAX_MISSED_FRAMES) {
                    nextActivePredictions.add(tracked)
                }
            }
        }

        // 2. Add any completely new detections that weren't matched
        for (newPred in currentFramePredictions) {
            nextActivePredictions.add(SmoothedPrediction(newPred, 0))
        }

        // Update state
        activePredictions = nextActivePredictions

        return activePredictions.map { it.prediction }
    }

    private fun findBestMatchIndex(target: Prediction, candidates: List<Prediction>): Int {
        var bestIndex = -1
        var bestIou = 0f

        for (i in candidates.indices) {
            val iou = iou(target.boundingBox, candidates[i].boundingBox)
            // Check if it overlaps enough and is the best match so far
            if (iou > IOU_THRESHOLD && iou > bestIou) {
                bestIou = iou
                bestIndex = i
            }
        }
        return bestIndex
    }

    // Linear interpolation for smooth movement
    private fun lerp(start: Float, end: Float, fraction: Float): Float {
        return start + (end - start) * fraction
    }

    // Calculate intersection over union
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
}
