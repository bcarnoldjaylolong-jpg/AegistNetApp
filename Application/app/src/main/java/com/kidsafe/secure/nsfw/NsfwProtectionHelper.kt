package com.kidsafe.secure.nsfw

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.provider.Settings
import androidx.core.content.ContextCompat
import com.kidsafe.secure.services.ScreenFilterService

/**
 * Helper class to manage NSFW protection features
 */
object NsfwProtectionHelper {

    const val MEDIA_PROJECTION_REQUEST_CODE = 9001
    const val OVERLAY_PERMISSION_REQUEST_CODE = 9002

    /**
     * Check if all required permissions are granted
     */
    fun hasRequiredPermissions(context: Context): Boolean {
        // Check overlay permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(context)) {
                return false
            }
        }
        return true
    }

    /**
     * Request overlay permission
     */
    fun requestOverlayPermission(activity: Activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                android.net.Uri.parse("package:${activity.packageName}")
            )
            activity.startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
        }
    }

    /**
     * Start media projection request
     */
    fun requestMediaProjection(activity: Activity) {
        val projectionManager = activity.getSystemService(Context.MEDIA_PROJECTION_SERVICE) 
            as MediaProjectionManager
        val captureIntent = projectionManager.createScreenCaptureIntent()
        activity.startActivityForResult(captureIntent, MEDIA_PROJECTION_REQUEST_CODE)
    }

    /**
     * Start the screen filter service with media projection data
     */
    fun startScreenFilterService(
        context: Context,
        resultCode: Int,
        data: Intent,
        threshold: Float = 0.6f
    ) {
        val serviceIntent = Intent(context, ScreenFilterService::class.java).apply {
            putExtra(ScreenFilterService.EXTRA_RESULT_CODE, resultCode)
            putExtra(ScreenFilterService.EXTRA_DATA, data)
            putExtra(ScreenFilterService.EXTRA_THRESHOLD, threshold)
        }
        ContextCompat.startForegroundService(context, serviceIntent)
    }

    /**
     * Stop the screen filter service
     */
    fun stopScreenFilterService(context: Context) {
        val serviceIntent = Intent(context, ScreenFilterService::class.java)
        context.stopService(serviceIntent)
    }

    /**
     * Check if screen filter service is running
     */
    fun isServiceRunning(context: Context): Boolean {
        val manager = context.getSystemService(Context.ACTIVITY_SERVICE) 
            as android.app.ActivityManager
        @Suppress("DEPRECATION")
        for (service in manager.getRunningServices(Integer.MAX_VALUE)) {
            if (ScreenFilterService::class.java.name == service.service.className) {
                return true
            }
        }
        return false
    }
}
