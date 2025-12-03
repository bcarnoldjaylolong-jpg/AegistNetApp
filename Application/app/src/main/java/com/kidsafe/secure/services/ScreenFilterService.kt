package com.kidsafe.secure.services

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.res.Resources
import android.graphics.*
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.*
import android.util.Log
import android.view.*
import androidx.core.app.NotificationCompat
import com.mansourappdevelopment.androidapp.kidsafe.R
import com.kidsafe.secure.nsfw.BlurOverlayView
import com.kidsafe.secure.nsfw.RoboflowContentDetector
import kotlinx.coroutines.*

class ScreenFilterService : Service() {

    companion object {
        private const val TAG = "AegistNet-ScreenFilter"
        private const val NOTIFICATION_ID = 1001
        private const val PROCESS_INTERVAL_MS = 300L // Process every 300ms
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_DATA = "data"
        const val EXTRA_THRESHOLD = "threshold"
    }

    private var mediaProjection: MediaProjection? = null
    private var imageReader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null

    private var detector: RoboflowContentDetector? = null
    private var blurOverlay: BlurOverlayView? = null

    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0

    // Detection threshold (0.0 - 1.0)
    private var detectionThreshold = 0.10f

    // Optimization: Reusable bitmaps to avoid allocation churn
    private var paddedBitmap: Bitmap? = null
    private var cleanBitmap: Bitmap? = null
    private var cleanCanvas: Canvas? = null

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var lastProcessTime = 0L
    private var isProcessing = false

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service onCreate")
        startForegroundNotification()

        try {
            detector = RoboflowContentDetector(this)
            Log.d(TAG, "Detector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Detector initialization failed", e)
            stopSelf()
            return
        }

        setupOverlay()
    }

    private fun setupOverlay() {
        try {
            blurOverlay = BlurOverlayView(this)

            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                else WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            )

            val windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
            windowManager.addView(blurOverlay, params)

            // Initially hidden until detection occurs
            blurOverlay?.visibility = View.GONE

            Log.d(TAG, "Overlay view added successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Overlay setup failed", e)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val rCode = intent?.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val data = intent?.getParcelableExtra<Intent>(EXTRA_DATA)
        detectionThreshold = intent?.getFloatExtra(EXTRA_THRESHOLD, 0.10f) ?: 0.10f

        Log.d(TAG, "Service started with threshold: $detectionThreshold")

        if (rCode == Activity.RESULT_OK && data != null) {
            initProjection(rCode, data)
        } else {
            Log.e(TAG, "Invalid result code or data")
        }

        return START_STICKY
    }

    private fun initProjection(resultCode: Int, data: Intent) {
        try {
            val pm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            mediaProjection = pm.getMediaProjection(resultCode, data)

            mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    super.onStop()
                    Log.d(TAG, "MediaProjection stopped")
                    stopSelf()
                }
            }, Handler(Looper.getMainLooper()))

            val metrics = Resources.getSystem().displayMetrics
            screenWidth = metrics.widthPixels
            screenHeight = metrics.heightPixels
            screenDensity = metrics.densityDpi

            Log.d(TAG, "Screen metrics - Width: $screenWidth, Height: $screenHeight, Density: $screenDensity")

            imageReader = ImageReader.newInstance(
                screenWidth,
                screenHeight,
                PixelFormat.RGBA_8888,
                2
            )

            virtualDisplay = mediaProjection?.createVirtualDisplay(
                "AegistNet-ScreenCapture",
                screenWidth,
                screenHeight,
                screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface,
                null,
                null
            )

            imageReader?.setOnImageAvailableListener(
                { reader -> processFrame(reader) },
                Handler(Looper.getMainLooper())
            )

            Log.d(TAG, "MediaProjection initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Projection initialization failed", e)
            stopSelf()
        }
    }

    private fun processFrame(reader: ImageReader) {
        val now = System.currentTimeMillis()

        // Throttle processing to avoid overwhelming the system
        if (isProcessing || now - lastProcessTime < PROCESS_INTERVAL_MS) {
            reader.acquireLatestImage()?.close()
            return
        }

        val image = reader.acquireLatestImage() ?: return
        isProcessing = true
        lastProcessTime = now

        scope.launch(Dispatchers.Default) {
            var bitmap: Bitmap? = null
            try {
                bitmap = imageToBitmap(image)
                image.close()

                if (bitmap == null) {
                    Log.w(TAG, "Failed to convert image to bitmap")
                    return@launch
                }

                // Perform detection (detect is already thread-safe internally)
                val predictions = detector?.detect(bitmap) ?: emptyList()

                // Filter by threshold
                val filteredPredictions = predictions.filter { it.confidence >= detectionThreshold }

                Log.d(TAG, "Raw detections: ${predictions.size}, Filtered: ${filteredPredictions.size}")

                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    try {
                        if (filteredPredictions.isNotEmpty()) {
                            blurOverlay?.updatePredictions(filteredPredictions, bitmap)
                            blurOverlay?.visibility = View.VISIBLE

                            Log.w(TAG, "⚠️ NSFW content detected! Count: ${filteredPredictions.size}")
                            filteredPredictions.forEachIndexed { index, pred ->
                                Log.d(TAG, "Detection #$index: center=(${pred.x}, ${pred.y}), size=(${pred.w}x${pred.h}), conf=${pred.confidence}")
                            }
                        } else {
                            // Clear predictions and hide overlay
                            blurOverlay?.updatePredictions(emptyList(), null)
                            blurOverlay?.visibility = View.GONE
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "UI update error", e)
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Frame processing error", e)
            } finally {
                isProcessing = false
            }
        }
    }

    /**
     * Optimized conversion that reuses Bitmaps to prevent Garbage Collection stutter.
     */
    private fun imageToBitmap(image: Image): Bitmap? {
        try {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * image.width

            val paddedWidth = image.width + rowPadding / pixelStride

            // Reuse padded bitmap if dimensions match
            if (paddedBitmap == null ||
                paddedBitmap?.width != paddedWidth ||
                paddedBitmap?.height != image.height) {
                paddedBitmap?.recycle()
                paddedBitmap = Bitmap.createBitmap(
                    paddedWidth,
                    image.height,
                    Bitmap.Config.ARGB_8888
                )
                Log.d(TAG, "Created padded bitmap: ${paddedWidth}x${image.height}")
            }

            buffer.rewind()
            paddedBitmap?.copyPixelsFromBuffer(buffer)

            // Reuse clean bitmap if dimensions match
            if (cleanBitmap == null ||
                cleanBitmap?.width != image.width ||
                cleanBitmap?.height != image.height) {
                cleanBitmap?.recycle()
                cleanBitmap = Bitmap.createBitmap(
                    image.width,
                    image.height,
                    Bitmap.Config.ARGB_8888
                )
                cleanCanvas = Canvas(cleanBitmap!!)
                Log.d(TAG, "Created clean bitmap: ${image.width}x${image.height}")
            }

            // Copy without padding
            val srcRect = Rect(0, 0, image.width, image.height)
            val dstRect = Rect(0, 0, image.width, image.height)
            cleanCanvas?.drawBitmap(paddedBitmap!!, srcRect, dstRect, null)

            return cleanBitmap

        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            return null
        }
    }

    private fun startForegroundNotification() {
        val channelId = "kidsafe_nsfw_filter"

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "NSFW Content Filter",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Monitors screen for inappropriate content"
                setShowBadge(false)
            }
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("AegistNet Protection Active")
            .setContentText("Monitoring screen for inappropriate content")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service destroying")

        // Cancel all coroutines
        scope.cancel()

        // Stop media projection
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()

        // Clean up bitmaps
        paddedBitmap?.recycle()
        cleanBitmap?.recycle()
        paddedBitmap = null
        cleanBitmap = null
        cleanCanvas = null

        // Close detector (no need for synchronization - already on single thread)
        try {
            detector?.close()
            detector = null
        } catch (e: Exception) {
            Log.e(TAG, "Detector cleanup error", e)
        }

        // Remove overlay
        blurOverlay?.let { overlay ->
            try {
                overlay.cleanup()
                (getSystemService(WINDOW_SERVICE) as WindowManager).removeView(overlay)
                Log.d(TAG, "Overlay removed successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Overlay removal error", e)
            }
        }
        blurOverlay = null

        Log.d(TAG, "Service destroyed successfully")
    }

    override fun onBind(intent: Intent?) = null
}