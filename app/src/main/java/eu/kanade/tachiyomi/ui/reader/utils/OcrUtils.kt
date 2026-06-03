package eu.kanade.tachiyomi.ui.reader.utils

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
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
     * Pads the image to the nearest multiple of 32 independently on both
     * the width and height without warping the original text.
     */
    fun padToMultipleOf32(bitmap: Bitmap): Bitmap {
        val w = bitmap.width
        val h = bitmap.height

        val targetW = (Math.ceil(w / 32.0) * 32).toInt()
        val targetH = (Math.ceil(h / 32.0) * 32).toInt()

        if (w == targetW && h == targetH) return bitmap

        val paddedBitmap = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(bitmap, 0f, 0f, null)

        return paddedBitmap
    }

    /**
     * Converts standard Android image pixels into a mathematical FloatBuffer
     * normalized between 0.0 and 1.0 (The format ONNX models read).
     *
     * UPGRADED: Includes a High-Contrast Thresholding layer. It mathematically sharpens
     * faded anti-aliased text strokes to pure black and washes out noise to pure white.
     */
    fun bitmapToFloatBuffer(bitmap: Bitmap, applyContrastBoost: Boolean = true): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val floatBuffer = FloatBuffer.allocate(1 * 3 * height * width)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val totalPixels = width * height

        // Pass 1: Red Channel
        for (i in 0 until totalPixels) {
            val pixel = pixels[i]
            var r = ((pixel shr 16) and 0xFF) / 255.0f
            if (applyContrastBoost) {
                // High-Contrast S-Curve: Deepens dark text strokes, brightens whites
                r = if (r < 0.5f) r * 0.4f else minOf(1.0f, r * 1.4f)
            }
            floatBuffer.put(r)
        }

        // Pass 2: Green Channel
        for (i in 0 until totalPixels) {
            val pixel = pixels[i]
            var g = ((pixel shr 8) and 0xFF) / 255.0f
            if (applyContrastBoost) {
                g = if (g < 0.5f) g * 0.4f else minOf(1.0f, g * 1.4f)
            }
            floatBuffer.put(g)
        }

        // Pass 3: Blue Channel
        for (i in 0 until totalPixels) {
            val pixel = pixels[i]
            var b = (pixel and 0xFF) / 255.0f
            if (applyContrastBoost) {
                b = if (b < 0.5f) b * 0.4f else minOf(1.0f, b * 1.4f)
            }
            floatBuffer.put(b)
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    /**
     * Physically crops the original high-res image using the math coordinates.
     */
    fun cropBubble(originalBitmap: Bitmap, box: Rect, scaleX: Float, scaleY: Float): Bitmap {
        val left = (box.left * scaleX).toInt().coerceAtLeast(0)
        val top = (box.top * scaleY).toInt().coerceAtLeast(0)
        val right = (box.right * scaleX).toInt().coerceAtMost(originalBitmap.width)
        val bottom = (box.bottom * scaleY).toInt().coerceAtMost(originalBitmap.height)

        val width = right - left
        val height = bottom - top

        if (width <= 0 || height <= 0) return originalBitmap

        return Bitmap.createBitmap(originalBitmap, left, top, width, height)
    }
}
