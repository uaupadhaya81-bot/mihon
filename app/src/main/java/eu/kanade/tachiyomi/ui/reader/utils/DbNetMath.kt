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
                    // Filter out tiny dots that are just noise
                    if (box.width() > 5 && box.height() > 5) {
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

        val queue = ArrayDeque<Pair<Int, Int>>()
        queue.add(Pair(startX, startY))
        visited[startY * width + startX] = true

        // Check neighboring pixels
        val dx = intArrayOf(-1, 1, 0, 0, -1, -1, 1, 1)
        val dy = intArrayOf(0, 0, -1, 1, -1, 1, -1, 1)

        while (queue.isNotEmpty()) {
            val (cx, cy) = queue.removeFirst()

            for (i in 0..7) {
                val nx = cx + dx[i]
                val ny = cy + dy[i]

                if (nx in 0 until width && ny in 0 until height) {
                    val nIndex = ny * width + nx
                    if (!visited[nIndex] && grid[nIndex] > threshold) {
                        visited[nIndex] = true
                        queue.add(Pair(nx, ny))

                        // Push the boundaries of our box outwards
                        if (nx < minX) minX = nx
                        if (nx > maxX) maxX = nx
                        if (ny < minY) minY = ny
                        if (ny > maxY) maxY = ny
                    }
                }
            }
        }

        // Add a small padding to ensure we don't cut off character edges
        val padding = 5
        return Rect(
            (minX - padding).coerceAtLeast(0),
            (minY - padding).coerceAtLeast(0),
            (maxX + padding).coerceAtMost(width - 1),
            (maxY + padding).coerceAtMost(height - 1),
        )
    }
}
