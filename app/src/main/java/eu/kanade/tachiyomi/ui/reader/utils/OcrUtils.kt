package eu.kanade.tachiyomi.ui.reader.utils

import android.graphics.Bitmap
import android.graphics.Rect
import java.nio.FloatBuffer

object OcrUtils {

    /**
     * Shrinks massive images so the ONNX models don't crash the phone's RAM.
     */
    fun downscaleImageForDetection(bitmap: Bitmap, maxLength: Int = 960): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val maxSide = maxOf(width, height)
        
        if (maxSide <= maxLength) return bitmap

        val scale = maxLength.toFloat() / maxSide
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    /**
     * Converts standard Android image pixels into a mathematical FloatBuffer 
     * normalized between 0.0 and 1.0 (The format ONNX models read).
     */
    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val floatBuffer = FloatBuffer.allocate(1 * 3 * height * width)
        
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            floatBuffer.put(r)
            floatBuffer.put(g)
            floatBuffer.put(b)
        }
        floatBuffer.rewind()
        return floatBuffer
    }

    /**
     * Physically crops the original high-res image using the math coordinates.
     */
    fun cropBubble(originalBitmap: Bitmap, box: Rect, scaleX: Float, scaleY: Float): Bitmap {
        // Map the small detection box coordinates back to the original massive image
        val left = (box.left * scaleX).toInt().coerceAtLeast(0)
        val top = (box.top * scaleY).toInt().coerceAtLeast(0)
        val right = (box.right * scaleX).toInt().coerceAtMost(originalBitmap.width)
        val bottom = (box.bottom * scaleY).toInt().coerceAtMost(originalBitmap.height)

        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <= 0) return originalBitmap // Failsafe

        return Bitmap.createBitmap(originalBitmap, left, top, width, height)
    }
}

