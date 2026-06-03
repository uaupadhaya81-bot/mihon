package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import eu.kanade.tachiyomi.ui.reader.utils.DbNetMath
import eu.kanade.tachiyomi.ui.reader.utils.OcrUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.Collections

class MangaOcrEngine(
    private val context: Context,
    private val apiKey: String,
) {
    private var ortEnv: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var dictionary: List<String> = emptyList()
    private var initError: String? = null

    var detModelSizeMb by mutableStateOf(0f)
        private set
    var recModelSizeMb by mutableStateOf(0f)
        private set

    data class TranslationResult(val translatedBlocks: List<String>)

    data class ParsedBox(
        val x: Int,
        val y: Int,
        val w: Int,
        val h: Int,
        val text: String,
        val localCenterY: Int,
    )

    init {
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            fun getAssetFilePath(assetName: String): String {
                val file = File(context.cacheDir, assetName)
                if (file.exists()) {
                    file.delete()
                }
                context.assets.open(assetName).use { inputStream ->
                    file.outputStream().use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
                return file.absolutePath
            }

            val detModelPath = getAssetFilePath("ch_PP-OCRv5_det_infer.onnx")
            detModelSizeMb = File(detModelPath).length() / 1024f / 1024f
            detSession = ortEnv?.createSession(detModelPath, OrtSession.SessionOptions())

            val recModelPath = getAssetFilePath("ch_PP-OCRv5_rec_infer.onnx")
            recModelSizeMb = File(recModelPath).length() / 1024f / 1024f
            recSession = ortEnv?.createSession(recModelPath, OrtSession.SessionOptions())

            dictionary = context.assets.open("ppocrv5_dict.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            val details = buildString {
                appendLine("MangaOcrEngine init failed")
                appendLine("ORT env: ${ortEnv != null}")
                appendLine("Det session: ${detSession != null}")
                appendLine("Rec session: ${recSession != null}")
                appendLine(e.stackTraceToString())
            }
            initError = details
            Log.e(TAG, details, e)
            Toast.makeText(context, details, Toast.LENGTH_LONG).show()
        }
    }

    @Composable
    fun ShowModelSizes(modifier: Modifier = Modifier) {
        Text(
            text = "DET model: %.2f MB\nREC model: %.2f MB".format(detModelSizeMb, recModelSizeMb),
            modifier = modifier,
        )
    }

    suspend fun processSingleImage(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        try {
            val env = ortEnv ?: return@withContext "Engine Error: ORT env not init"
            val det = detSession ?: return@withContext "Engine Error: Det session not init"
            val rec = recSession ?: return@withContext "Engine Error: Rec session not init"

            val scaledBitmap = OcrUtils.downscaleImageForDetection(bitmap)
            val processedBitmap = OcrUtils.padToMultipleOf32(scaledBitmap)

            val floatBuffer = OcrUtils.bitmapToFloatBuffer(processedBitmap)
            val shape = longArrayOf(1, 3, processedBitmap.height.toLong(), processedBitmap.width.toLong())

            var detW = processedBitmap.width
            var detH = processedBitmap.height
            var flatProbabilities = FloatArray(0)

            OnnxTensor.createTensor(env, floatBuffer, shape).use { tensor ->
                val inputName = det.inputNames.firstOrNull()
                    ?: return@withContext "Engine Error: Det model input name missing"

                val detMap = Collections.singletonMap(inputName, tensor)
                det.run(detMap).use { results ->
                    val detOutputTensor = results.iterator().next().value as? OnnxTensor
                    if (detOutputTensor != null) {
                        @Suppress("UNCHECKED_CAST")
                        val rawDetArray = detOutputTensor.value as? Array<Array<Array<FloatArray>>>
                        if (rawDetArray != null && rawDetArray.isNotEmpty()) {
                            val batch = rawDetArray[0]
                            if (batch.isNotEmpty()) {
                                val channel = batch[0]
                                if (channel.isNotEmpty()) {
                                    detH = channel.size
                                    detW = channel[0].size
                                    flatProbabilities = FloatArray(detW * detH)
                                    var idx = 0
                                    for (y in 0 until detH) {
                                        val row = channel[y]
                                        for (x in 0 until detW) {
                                            flatProbabilities[idx++] = row[x]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (flatProbabilities.isEmpty()) return@withContext "Failed to parse detection output."
            val scaleX = bitmap.width.toFloat() / scaledBitmap.width
            val scaleY = bitmap.height.toFloat() / scaledBitmap.height

            val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
            if (boxes.isEmpty()) return@withContext "No text found in this image."
            val japaneseTextBlocks = mutableListOf<String>()
            val recInputName = rec.inputNames.firstOrNull()
                ?: return@withContext "Engine Error: Rec model input name missing"

            for (box in boxes) {
                val croppedBubble = OcrUtils.cropBubble(bitmap, box, scaleX, scaleY)
                if (croppedBubble.width <= 0 || croppedBubble.height <= 0) continue

                val recHeight = 48
                val ratio = croppedBubble.width.toFloat() / croppedBubble.height
                val recWidth = (ratio * recHeight).toInt().coerceAtLeast(1)
                val recBitmap = Bitmap.createScaledBitmap(croppedBubble, recWidth, recHeight, true)

                try {
                    val recBufferIn = OcrUtils.bitmapToFloatBuffer(recBitmap)
                    val recShapeIn = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())

                    OnnxTensor.createTensor(env, recBufferIn, recShapeIn).use { recTensor ->
                        val recMap = Collections.singletonMap(recInputName, recTensor)
                        rec.run(recMap).use { recRes ->
                            val recOutputTensor = recRes.iterator().next().value as? OnnxTensor
                            if (recOutputTensor != null) {
                                @Suppress("UNCHECKED_CAST")
                                val rawRecArray = recOutputTensor.value as? Array<Array<FloatArray>>
                                if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                    val batch = rawRecArray[0]
                                    if (batch.isNotEmpty()) {
                                        val (decodedText, confidence) = decodeRecognitionArray(batch)

                                        // 🔥 GARBAGE DEFENSE LAYER 2: CONFIDENCE SCORE FILTER 🔥
                                        if (decodedText.isNotBlank() && confidence > 0.60f) {
                                            // 🔥 GARBAGE DEFENSE LAYER 3: TEXT DENSITY VOID FILTER 🔥
                                            val w = ((box.right - box.left) * scaleX).toInt()
                                            val h = ((box.bottom - box.top) * scaleY).toInt()
                                            val areaPerChar = (w * h) / decodedText.length
                                            if (decodedText.length <= 2 && areaPerChar > 3500) {
                                                continue
                                            }
                                            japaneseTextBlocks.add(decodedText)
                                        }
                                    }
                                }
                            }
                        }
                    }
                } finally {
                    recBitmap.recycle()
                }
            }

            if (japaneseTextBlocks.isEmpty()) return@withContext "Failed to extract text."
            val prompt = buildMegaPrompt(japaneseTextBlocks)
            return@withContext sendToGemini(prompt)
        } catch (e: Exception) {
            val details = buildString {
                appendLine("processSingleImage failed")
                appendLine("bitmap=${bitmap.width}x${bitmap.height}")
                appendLine(e.stackTraceToString())
            }
            Log.e(TAG, details, e)
            return@withContext details
        }
    }

    suspend fun runLocalOcrTest(
        context: Context,
        uris: List<Uri>,
    ): String = withContext(Dispatchers.IO) {
        if (initError != null) return@withContext initError!!
        val sb = StringBuilder()
        var globalYOffset = 0
        val globalBoxes = mutableListOf<ParsedBox>()
        var carryOver: Bitmap? = null

        try {
            val env = ortEnv ?: return@withContext "Engine Error: ORT env not init"
            val det = detSession ?: return@withContext "Engine Error: Det session not init"
            val rec = recSession ?: return@withContext "Engine Error: Rec session not init"

            uris.forEachIndexed { index, uri ->
                val rawBitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                } ?: return@forEachIndexed

                val hasCarryOver = carryOver != null
                val carryOverH = if (hasCarryOver) carryOver!!.height else 0

                // UNIFIED LAYER: If previous page cut text, physically stitch it here
                val activeBitmap = if (hasCarryOver) {
                    val stitched = Bitmap.createBitmap(
                        rawBitmap.width,
                        rawBitmap.height + carryOverH,
                        Bitmap.Config.ARGB_8888,
                    )
                    val canvas = Canvas(stitched)
                    canvas.drawBitmap(carryOver!!, 0f, 0f, null)
                    canvas.drawBitmap(rawBitmap, 0f, carryOverH.toFloat(), null)
                    carryOver!!.recycle()
                    carryOver = null
                    stitched
                } else {
                    rawBitmap
                }

                val windowHeight = 1024
                val overlap = 180
                var localYOffset = 0

                while (localYOffset < activeBitmap.height) {
                    // Save remaining pixel slice if window oversteps the bounds
                    if (localYOffset + windowHeight > activeBitmap.height) {
                        val remaining = activeBitmap.height - localYOffset
                        if (index < uris.size - 1) {
                            carryOver = Bitmap.createBitmap(
                                activeBitmap,
                                0,
                                localYOffset,
                                activeBitmap.width,
                                remaining,
                            )
                            break
                        }
                    }

                    val chunkH = minOf(windowHeight, activeBitmap.height - localYOffset)
                    val slice = Bitmap.createBitmap(activeBitmap, 0, localYOffset, activeBitmap.width, chunkH)

                    val scaledSlice = OcrUtils.downscaleImageForDetection(slice)
                    val processedSlice = OcrUtils.padToMultipleOf32(scaledSlice)

                    val floatBuffer = OcrUtils.bitmapToFloatBuffer(processedSlice)
                    val shape = longArrayOf(1, 3, processedSlice.height.toLong(), processedSlice.width.toLong())

                    var detW = processedSlice.width
                    var detH = processedSlice.height
                    var flatProbabilities = FloatArray(0)

                    OnnxTensor.createTensor(env, floatBuffer, shape).use { tensor ->
                        val inputName = det.inputNames.firstOrNull() ?: return@use
                        val detMap = Collections.singletonMap(inputName, tensor)
                        det.run(detMap).use { results ->
                            val detOutputTensor = results.iterator().next().value as? OnnxTensor

                            @Suppress("UNCHECKED_CAST")
                            val rawDetArray = detOutputTensor?.value as? Array<Array<Array<FloatArray>>>
                            if (rawDetArray != null && rawDetArray.isNotEmpty()) {
                                val batch = rawDetArray[0]
                                if (batch.isNotEmpty()) {
                                    val channel = batch[0]
                                    if (channel.isNotEmpty()) {
                                        detH = channel.size
                                        detW = channel[0].size
                                        flatProbabilities = FloatArray(detW * detH)
                                        var idx = 0
                                        for (y in 0 until detH) {
                                            val row = channel[y]
                                            for (x in 0 until detW) {
                                                flatProbabilities[idx++] = row[x]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (flatProbabilities.isNotEmpty()) {
                        val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
                        val scaleX = slice.width.toFloat() / scaledSlice.width
                        val scaleY = slice.height.toFloat() / scaledSlice.height
                        val recInputName = rec.inputNames.firstOrNull()

                        if (recInputName != null && boxes.isNotEmpty()) {
                            for (box in boxes) {
                                val croppedBubble = OcrUtils.cropBubble(slice, box, scaleX, scaleY)
                                if (croppedBubble.width <= 0 || croppedBubble.height <= 0) continue

                                val recHeight = 48
                                val ratio = croppedBubble.width.toFloat() / croppedBubble.height
                                val recWidth = (ratio * recHeight).toInt().coerceAtLeast(1)

                                val recBitmap = Bitmap.createScaledBitmap(
                                    croppedBubble,
                                    recWidth,
                                    recHeight,
                                    true,
                                )

                                try {
                                    val recBufferIn = OcrUtils.bitmapToFloatBuffer(recBitmap)
                                    val recShapeIn = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())

                                    OnnxTensor.createTensor(env, recBufferIn, recShapeIn).use { recTensor ->
                                        val recMap = Collections.singletonMap(recInputName, recTensor)
                                        rec.run(recMap).use { recRes ->
                                            val recOut = recRes.iterator().next().value as? OnnxTensor

                                            @Suppress("UNCHECKED_CAST")
                                            val rawRecArray = recOut?.value as? Array<Array<FloatArray>>

                                            if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                                val batch = rawRecArray[0]
                                                if (batch.isNotEmpty()) {
                                                    val (decodedText, confidence) = decodeRecognitionArray(batch)

                                                    // 🔥 GARBAGE DEFENSE LAYER 2: CONFIDENCE SCORE FILTER 🔥
                                                    if (decodedText.isNotBlank() && confidence > 0.60f) {
                                                        val absX = (box.left * scaleX).toInt()
                                                        val w = ((box.right - box.left) * scaleX).toInt()
                                                        val h = ((box.bottom - box.top) * scaleY).toInt()

                                                        // 🔥 GARBAGE DEFENSE LAYER 3: TEXT DENSITY VOID FILTER 🔥
                                                        val areaPerChar = (w * h) / decodedText.length
                                                        if (decodedText.length <= 2 && areaPerChar > 3500) {
                                                            continue
                                                        }

                                                        // GLOBAL REMAP MATH: Accounts for pre-stitched canvas shift
                                                        val absY = (box.top * scaleY).toInt() +
                                                            localYOffset + globalYOffset - carryOverH

                                                        val localCenterY = (box.top + box.bottom) / 2

                                                        globalBoxes.add(
                                                            ParsedBox(
                                                                absX,
                                                                absY,
                                                                w,
                                                                h,
                                                                decodedText,
                                                                localCenterY,
                                                            ),
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } finally {
                                    recBitmap.recycle()
                                }
                            }
                        }
                    }

                    slice.recycle()
                    localYOffset += (windowHeight - overlap)
                }

                globalYOffset += rawBitmap.height
                if (activeBitmap != rawBitmap) activeBitmap.recycle()
                rawBitmap.recycle()
            }

            // CLEANUP: Enforce horizontal column lock, text quality, and your repeat tracking rule
            val deduplicated = deduplicateBoxes(globalBoxes)
            val cleanedBoxes = removeRepeatingWatermarks(deduplicated)

            // 🔥 NEW: MERGE THE LINES INTO FULL BUBBLES 🔥
            val finalMergedBubbles = groupTextBubbles(cleanedBoxes)

            appendDebug(sb, "========================")
            appendDebug(sb, "MANGA CHAPTER PROCESSING COMPLETE")
            appendDebug(sb, "Total Continuous Height processed: $globalYOffset px")
            appendDebug(sb, "Total Unique Text Bubbles: ${finalMergedBubbles.size}")
            appendDebug(sb, "========================\n")

            finalMergedBubbles.forEachIndexed { i, box ->
                appendDebug(sb, "[BUBBLE:${i + 1}] {x: ${box.x}, y: ${box.y}, w: ${box.w}, h: ${box.h}}")
                appendDebug(sb, box.text)
                appendDebug(sb, "")
            }
        } catch (e: Exception) {
            Log.e(TAG, "runLocalOcrTest crashed", e)
            sb.append("\n\nCRASH ERROR: ${e.message}")
        }

        return@withContext sb.toString()
    }

    // --- Deduplication & Overlap Handling ---

    private fun deduplicateBoxes(boxes: List<ParsedBox>): List<ParsedBox> {
        val result = mutableListOf<ParsedBox>()
        val xTolerance = 15

        for (box in boxes) {
            var isDuplicate = false
            for (i in result.indices) {
                val existing = result[i]

                val xAligned = Math.abs(box.x - existing.x) <= xTolerance
                val wAligned = Math.abs(box.w - existing.w) <= xTolerance

                if (xAligned && wAligned && calculateIoU(box, existing) > 0.3f) {
                    isDuplicate = true

                    val existingDist = Math.abs(existing.localCenterY - 512)
                    val newDist = Math.abs(box.localCenterY - 512)
                    if (newDist < existingDist) {
                        result[i] = box
                    }
                    break
                }
            }
            if (!isDuplicate) {
                result.add(box)
            }
        }

        return result
    }

    /**
     * LOGO DEFENSE PIPELINE: URL FAST TRACK + FUZZY PATTERN LOCATOR
     * Tracks elements separated horizontally with substantial vertical gaps.
     */
    private fun removeRepeatingWatermarks(boxes: List<ParsedBox>): List<ParsedBox> {
        val bannedWatermarks = mutableSetOf<String>()
        val minimumYDistance = 800
        val xTolerance = 30

        // Logo Defense Layer 1: URL Fast Strip + Brand Hard Filter
        for (box in boxes) {
            val txt = box.text
            if (txt.contains(".com", true) ||
                txt.contains(".org", true) ||
                txt.contains("www.", true) ||
                txt.contains("菠萝包", true) // 🔥 Hard-bans typo variants of the main watermark brand
            ) {
                bannedWatermarks.add(txt)
            }
        }

        val uniqueTexts = boxes.map { it.text }.distinct()
        val watermarkClusters = mutableSetOf<String>()

        for (text in uniqueTexts) {
            if (watermarkClusters.contains(text) || bannedWatermarks.contains(text)) continue

            // Logo Defense Layer 3 (Upgrade): Group text strings within an 80% error tolerance limit
            val instances = boxes.filter {
                it.text == text || calculateSimilarity(it.text, text) >= 0.80f
            }.sortedBy { it.y }

            // Logo Defense Layer 2: Spatial Layout Mapping Pattern Rule
            if (instances.size >= 3) {
                // Find a subset of instances where at least 3 items align closely on X
                val alignedInstances = instances.filter { target ->
                    instances.count { Math.abs(it.x - target.x) <= xTolerance } >= 3
                }

                // If we found a valid, vertically repeating column layout
                if (alignedInstances.size >= 3) {
                    var regularPattern = true
                    for (i in 1 until alignedInstances.size) {
                        val gap = alignedInstances[i].y - alignedInstances[i - 1].y
                        if (gap < minimumYDistance) {
                            regularPattern = false
                            break
                        }
                    }
                    if (regularPattern) {
                        // Ban all slight variations of this logo stamp across the chapter stream
                        alignedInstances.forEach { bannedWatermarks.add(it.text) }
                        watermarkClusters.add(text)
                    }
                }
            }
        }

        return boxes.filter { !bannedWatermarks.contains(it.text) }.sortedBy { it.y }
    }

    private fun calculateIoU(b1: ParsedBox, b2: ParsedBox): Float {
        val left = maxOf(b1.x, b2.x)
        val top = maxOf(b1.y, b2.y)
        val right = minOf(b1.x + b1.w, b2.x + b2.w)
        val bottom = minOf(b1.y + b1.h, b2.y + b2.h)

        if (left < right && top < bottom) {
            val intersection = (right - left) * (bottom - top)
            val union = (b1.w * b1.h) + (b2.w * b2.h) - intersection
            return intersection.toFloat() / union.toFloat()
        }
        return 0f
    }

    /**
     * Self-contained edit distance helper to implement string similarity limits
     * without introducing external dependency naming conflicts.
     */
    private fun calculateSimilarity(s1: String, s2: String): Float {
        if (s1 == s2) return 1.0f
        val len1 = s1.length
        val len2 = s2.length
        if (len1 == 0 || len2 == 0) return 0.0f

        val dp = Array(len1 + 1) { IntArray(len2 + 1) }
        for (i in 0..len1) dp[i][0] = i
        for (j in 0..len2) dp[0][j] = j

        for (i in 1..len1) {
            for (j in 1..len2) {
                val cost = if (s1[i - 1] == s2[j - 1]) 0 else 1
                dp[i][j] = minOf(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
            }
        }
        return 1.0f - (dp[len1][len2].toFloat() / maxOf(len1, len2))
    }

    // --- Auxiliary Tools ---

    /**
     * BUBBLE MERGING LOGIC
     * Glues individual lines of text back together into full speech bubbles
     * based on their physical distance from each other on the page.
     */
    private fun groupTextBubbles(boxes: List<ParsedBox>): List<ParsedBox> {
        if (boxes.isEmpty()) return emptyList()

        val groups = mutableListOf<MutableList<ParsedBox>>()

        // Distance Thresholds: How close boxes must be to merge (in pixels)
        val expandX = 60
        val expandY = 70

        for (box in boxes) {
            var foundGroup = false

            // Create a magnetic zone around the current text box
            val boxLeft = box.x - expandX
            val boxTop = box.y - expandY
            val boxRight = box.x + box.w + expandX
            val boxBottom = box.y + box.h + expandY

            for (group in groups) {
                // Check if this box's magnetic zone overlaps with any box already in the group
                val intersects = group.any { gBox ->
                    val gLeft = gBox.x
                    val gTop = gBox.y
                    val gRight = gBox.x + gBox.w
                    val gBottom = gBox.y + gBox.h

                    // Simple rectangle intersection math
                    boxLeft < gRight && boxRight > gLeft && boxTop < gBottom && boxBottom > gTop
                }

                if (intersects) {
                    group.add(box)
                    foundGroup = true
                    break
                }
            }

            if (!foundGroup) {
                groups.add(mutableListOf(box))
            }
        }

        // Combine the grouped lines into single massive paragraph boxes!
        val mergedBoxes = mutableListOf<ParsedBox>()
        for (group in groups) {
            // Sort lines top-to-bottom so the sentence reads correctly
            val sortedGroup = group.sortedBy { it.y }

            val minX = sortedGroup.minOf { it.x }
            val minY = sortedGroup.minOf { it.y }
            val maxX = sortedGroup.maxOf { it.x + it.w }
            val maxY = sortedGroup.maxOf { it.y + it.h }

            // Glue the text strings together
            val mergedText = sortedGroup.joinToString("") { it.text }
            val center = (minY + maxY) / 2

            mergedBoxes.add(ParsedBox(minX, minY, maxX - minX, maxY - minY, mergedText, center))
        }

        return mergedBoxes.sortedBy { it.y }
    }

    /**
     * Decodes the raw float array into a String AND calculates the AI's average confidence.
     * Returns a Pair: (Decoded Text, Average Confidence Score 0.0 to 1.0)
     */
    private fun decodeRecognitionArray(sequence: Array<FloatArray>): Pair<String, Float> {
        val sb = StringBuilder()
        var lastIndex = -1
        var totalProb = 0f
        var validChars = 0

        for (timeStep in sequence) {
            var maxProb = -1f
            var maxIdx = -1

            for (i in timeStep.indices) {
                if (timeStep[i] > maxProb) {
                    maxProb = timeStep[i]
                    maxIdx = i
                }
            }

            if (maxIdx > 0 && maxIdx != lastIndex && maxIdx <= dictionary.size) {
                sb.append(dictionary[maxIdx - 1])
                totalProb += maxProb
                validChars++
            }

            lastIndex = maxIdx
        }

        val avgConfidence = if (validChars > 0) totalProb / validChars else 0f
        return Pair(sb.toString(), avgConfidence)
    }

    private fun buildMegaPrompt(japaneseBlocks: List<String>): String {
        val sb = StringBuilder()
        sb.append(
            "You are an elite manga translator. Translate the following Japanese " +
                "text blocks to English. Keep each block separated:\n\n",
        )

        japaneseBlocks.forEachIndexed { index, text ->
            sb.append("Block ${index + 1}: $text\n")
        }

        return sb.toString()
    }

    private fun sendToGemini(prompt: String): String {
        return try {
            val urlString = "https://generativelanguage.googleapis.com/v1beta/" +
                "models/gemini-3.1-flash-lite:generateContent?key=$apiKey"
            val url = URL(urlString)
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.doOutput = true

            val cleanPrompt = prompt.replace("\n", "\\n").replace("\"", "\\\"")
            val jsonPayload = "{\"contents\": [{\"parts\": [{\"text\": \"$cleanPrompt\"}]}]}"

            connection.outputStream.use { os ->
                os.write(jsonPayload.toByteArray(Charsets.UTF_8))
            }

            if (connection.responseCode == 200) {
                val responseBody = connection.inputStream.bufferedReader().readText()
                val jsonObject = JSONObject(responseBody)
                val candidates = jsonObject.getJSONArray("candidates")

                if (candidates.length() > 0) {
                    val content = candidates.getJSONObject(0).getJSONObject("content")
                    val parts = content.getJSONArray("parts")

                    if (parts.length() > 0) {
                        return parts.getJSONObject(0).getString("text").trim()
                    }
                }
            }

            "API Error: ${connection.responseCode}"
        } catch (e: Exception) {
            Log.e(TAG, "sendToGemini failed", e)
            "Connection Failed: ${e.message}"
        }
    }

    suspend fun processDownloadedChapter(
        chapterDir: File,
    ): Map<Int, TranslationResult> = withContext(Dispatchers.IO) {
        val resultMap = mutableMapOf<Int, TranslationResult>()
        resultMap[0] = TranslationResult(listOf("Chapter Mode Ready!"))
        return@withContext resultMap
    }

    companion object {
        private const val TAG = "MangaOcrEngine"

        suspend fun testGeminiAPI(
            testKey: String,
            message: String,
        ): String = withContext(Dispatchers.IO) {
            try {
                val urlString = "https://generativelanguage.googleapis.com/v1beta/" +
                    "models/gemini-3.1-flash-lite:generateContent?key=$testKey"
                val url = URL(urlString)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true

                val cleanPrompt = message.replace("\n", "\\n").replace("\"", "\\\"")
                val jsonPayload = "{\"contents\": [{\"parts\": [{\"text\": \"$cleanPrompt\"}]}]}"

                connection.outputStream.use { os ->
                    os.write(jsonPayload.toByteArray(Charsets.UTF_8))
                }

                if (connection.responseCode == 200) {
                    val responseBody = connection.inputStream.bufferedReader().readText()
                    val jsonObject = JSONObject(responseBody)
                    val candidates = jsonObject.getJSONArray("candidates")

                    if (candidates.length() > 0) {
                        val content = candidates.getJSONObject(0).getJSONObject("content")
                        val parts = content.getJSONArray("parts")

                        if (parts.length() > 0) {
                            return@withContext parts.getJSONObject(0).getString("text").trim()
                        }
                    }
                }

                "API Error: ${connection.responseCode}"
            } catch (e: Exception) {
                Log.e(TAG, "testGeminiAPI failed", e)
                "Error: ${e.message}"
            }
        }
    }

    private fun appendDebug(sb: StringBuilder, message: String) {
        sb.append(message).append('\n')
        Log.d(TAG, message)
    }
}
