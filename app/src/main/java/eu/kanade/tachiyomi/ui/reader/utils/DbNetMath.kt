package eu.kanade.tachiyomi.ui.reader.utils

import android.graphics.Rect

object DbNetMath {

    /**
     * Takes the massive grid of raw probabilities from the ONNX detection model and
     * mathematically draws bounding boxes around areas with a high text probability.
     */
    fun extractBoundingBoxes(
        probabilityGrid: FloatArray,
        width: Int,
        height: Int,
        threshold: Float = 0.18f,
    ): List<Rect> {
        val boxes = mutableListOf<Rect>()
        val visited = BooleanArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x

                // If we haven't checked this pixel, and the AI thinks it's text
                if (!visited[index] && probabilityGrid[index] > threshold) {
                    val box = traceContourBFS(probabilityGrid, visited, x, y, width, height, threshold)

                    // 🔥 GARBAGE DEFENSE LAYER 1: MINIMUM SIZE FILTER 🔥
                    if (box.width() > 16 && box.height() > 16) {
                        boxes.add(box)
                    }
                }
            }
        }
        return boxes
    }

    /**
     * Breadth-First Search (BFS) algorithm to expand outwards from a text pixel
     * to find the outer limits (left, top, right, bottom) of the whole text bubble.
     *
     * OPTIMIZED: Avoids allocating thousands of temporary Pair objects on the heap.
     * Packing coordinates (X, Y) into a single primitive 32-bit integer keeps GC cycles at zero.
     */
    private fun traceContourBFS(
        grid: FloatArray,
        visited: BooleanArray,
        startX: Int,
        startY: Int,
        width: Int,
        height: Int,
        threshold: Float,
    ): Rect {
        var minX = startX
        var maxX = startX
        var minY = startY
        var maxY = startY

        val queue = ArrayDeque<Int>()
        // Pack (startX, startY) into a single Int: High 16 bits = X, Low 16 bits = Y
        queue.add((startX shl 16) or startY)
        visited[startY * width + startX] = true

        val dx = intArrayOf(-1, 1, 0, 0, -1, -1, 1, 1)
        val dy = intArrayOf(0, 0, -1, 1, -1, 1, -1, 1)

        while (queue.isNotEmpty()) {
            val packed = queue.removeFirst()
            val cx = packed ushr 16
            val cy = packed and 0xFFFF

            for (i in 0..7) {
                val nx = cx + dx[i]
                val ny = cy + dy[i]

                if (nx in 0 until width && ny in 0 until height) {
                    val nIndex = ny * width + nx
                    if (!visited[nIndex] && grid[nIndex] > threshold) {
                        visited[nIndex] = true
                        queue.add((nx shl 16) or ny)

                        if (nx < minX) minX = nx
                        if (nx > maxX) maxX = nx
                        if (ny < minY) minY = ny
                        if (ny > maxY) maxY = ny
                    }
                }
            }
        }

        val padding = 5
        return Rect(
            (minX - padding).coerceAtLeast(0),
            (minY - padding).coerceAtLeast(0),
            (maxX + padding).coerceAtMost(width - 1),
            (maxY + padding).coerceAtMost(height - 1),
        )
    }
}
