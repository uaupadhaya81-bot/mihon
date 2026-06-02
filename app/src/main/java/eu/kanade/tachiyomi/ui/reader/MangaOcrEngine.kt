package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
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

    data class TranslationResult(val translatedBlocks: List<String>)

    init {
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            val detModelBytes = context.assets.open("ch_PP-OCRv5_det_infer.onnx").use { it.readBytes() }
            detSession = ortEnv?.createSession(detModelBytes, OrtSession.SessionOptions())

            val recModelBytes = context.assets.open("ch_PP-OCRv5_rec_infer.onnx").use { it.readBytes() }
            recSession = ortEnv?.createSession(recModelBytes, OrtSession.SessionOptions())

            dictionary = context.assets.open("ppocrv5_dict.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            initError = e.stackTraceToString()
        }
    }

    suspend fun processSingleImage(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        try {
            val env = ortEnv ?: return@withContext "Engine Error: ORT environment not initialized"
            val det = detSession ?: return@withContext "Engine Error: Detection session not initialized"
            val rec = recSession ?: return@withContext "Engine Error: Recognition session not initialized"

            val scaledBitmap = OcrUtils.downscaleImageForDetection(bitmap)
            val floatBuffer = OcrUtils.bitmapToFloatBuffer(scaledBitmap)
            val shape = longArrayOf(
                1,
                3,
                scaledBitmap.height.toLong(),
                scaledBitmap.width.toLong(),
            )

            var detW = scaledBitmap.width
            var detH = scaledBitmap.height
            var flatProbabilities = FloatArray(0)

            OnnxTensor.createTensor(env, floatBuffer, shape).use { tensor ->
                val inputName = det.inputNames.firstOrNull()
                    ?: return@withContext "Engine Error: Detection model input name missing"

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

            val scaleX = bitmap.width.toFloat() / detW
            val scaleY = bitmap.height.toFloat() / detH
            val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
            if (boxes.isEmpty()) return@withContext "No text found in this image."

            val japaneseTextBlocks = mutableListOf<String>()
            val recInputName = rec.inputNames.firstOrNull()
                ?: return@withContext "Engine Error: Recognition model input name missing"

            for (box in boxes) {
                val croppedBubble = OcrUtils.cropBubble(bitmap, box, scaleX, scaleY)
                if (croppedBubble.width <= 0 || croppedBubble.height <= 0) {
                    continue
                }

                val recHeight = 48
                val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight)
                    .toInt()
                    .coerceAtLeast(1)

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
                            val recOutputTensor = recRes.iterator().next().value as? OnnxTensor
                            if (recOutputTensor != null) {
                                @Suppress("UNCHECKED_CAST")
                                val rawRecArray = recOutputTensor.value as? Array<Array<FloatArray>>
                                if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                    val batch = rawRecArray[0]
                                    if (batch.isNotEmpty()) {
                                        val decodedText = decodeRecognitionArray(batch)
                                        if (decodedText.isNotBlank()) {
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
            Log.e(TAG, "processSingleImage failed", e)
            return@withContext "Engine Error: ${e.message}"
        }
    }

    suspend fun runLocalOcrTest(
        context: Context,
        uris: List<Uri>,
    ): String = withContext(Dispatchers.IO) {
        if (initError != null) {
            return@withContext initError!!
        }

        val sb = StringBuilder()

        try {
            val env = ortEnv ?: return@withContext "Engine Error: ORT environment not initialized"
            val det = detSession ?: return@withContext "Engine Error: Detection session not initialized"
            val rec = recSession ?: return@withContext "Engine Error: Recognition session not initialized"

            uris.forEachIndexed { index, uri ->
                val pageName = "PAGE_${index + 1}"
                appendDebug(sb, "========================")
                appendDebug(sb, "[$pageName]")
                appendDebug(sb, "========================")

                val bitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }

                if (bitmap == null) {
                    appendDebug(sb, "Error: Failed to load picture into memory.")
                    appendDebug(sb, "")
                    return@forEachIndexed
                }

                appendDebug(sb, "Loaded bitmap: ${bitmap.width}x${bitmap.height}")

                val sliceMaxHeight = 2048
                var yOffset = 0
                var blockCounter = 1

                while (yOffset < bitmap.height) {
                    val currentHeight = minOf(sliceMaxHeight, bitmap.height - yOffset)
                    val slice = Bitmap.createBitmap(
                        bitmap,
                        0,
                        yOffset,
                        bitmap.width,
                        currentHeight,
                    )

                    try {
                        appendDebug(
                            sb,
                            "Slice: yOffset=$yOffset height=$currentHeight size=${slice.width}x${slice.height}",
                        )

                        val scaledSlice = OcrUtils.downscaleImageForDetection(slice)
                        appendDebug(sb, "Scaled slice for detection: ${scaledSlice.width}x${scaledSlice.height}")

                        val floatBuffer = OcrUtils.bitmapToFloatBuffer(scaledSlice)
                        val shape = longArrayOf(
                            1,
                            3,
                            scaledSlice.height.toLong(),
                            scaledSlice.width.toLong(),
                        )

                        var detW = scaledSlice.width
                        var detH = scaledSlice.height
                        var flatProbabilities = FloatArray(0)

                        OnnxTensor.createTensor(env, floatBuffer, shape).use { tensor ->
                            val inputName = det.inputNames.firstOrNull()
                            if (inputName == null) {
                                appendDebug(sb, "Detection input name missing.")
                                return@use
                            }

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
                                                appendDebug(
                                                    sb,
                                                    "Detection parsed: detW=$detW detH=$detH probs=${flatProbabilities.size}",
                                                )
                                            } else {
                                                appendDebug(sb, "Detection parsed but channel was empty.")
                                            }
                                        } else {
                                            appendDebug(sb, "Detection parsed but batch was empty.")
                                        }
                                    } else {
                                        appendDebug(sb, "Detection output shape was unexpected.")
                                    }
                                } else {
                                    appendDebug(sb, "Detection output tensor was null.")
                                }
                            }
                        }

                        if (flatProbabilities.isEmpty()) {
                            appendDebug(sb, "Failed to parse detection output.")
                            appendDebug(sb, "")
                            yOffset += sliceMaxHeight
                            continue
                        }

                        val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
                        appendDebug(sb, "Boxes found: ${boxes.size}")

                        if (boxes.isEmpty()) {
                            appendDebug(sb, "No text found in this image.")
                            appendDebug(sb, "")
                            yOffset += sliceMaxHeight
                            continue
                        }

                        val scaleX = slice.width.toFloat() / detW
                        val scaleY = slice.height.toFloat() / detH
                        val recInputName = rec.inputNames.firstOrNull()

                        if (recInputName == null) {
                            appendDebug(sb, "Recognition input name missing.")
                            appendDebug(sb, "")
                            yOffset += sliceMaxHeight
                            continue
                        }

                        for ((boxIndex, box) in boxes.withIndex()) {
                            appendDebug(sb, "Box $boxIndex: $box")

                            val croppedBubble = OcrUtils.cropBubble(slice, box, scaleX, scaleY)
                            appendDebug(sb, "Crop size: ${croppedBubble.width}x${croppedBubble.height}")

                            if (croppedBubble.width <= 0 || croppedBubble.height <= 0) {
                                appendDebug(sb, "Skipping invalid crop.")
                                continue
                            }

                            val recHeight = 48
                            val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight)
                                .toInt()
                                .coerceAtLeast(1)

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
                                        if (recOut != null) {
                                            @Suppress("UNCHECKED_CAST")
                                            val rawRecArray = recOut.value as? Array<Array<FloatArray>>
                                            if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                                val batch = rawRecArray[0]
                                                if (batch.isNotEmpty()) {
                                                    val decodedText = decodeRecognitionArray(batch)
                                                    appendDebug(sb, "Decoded: '$decodedText'")

                                                    if (decodedText.isNotBlank()) {
                                                        val absY = (box.top * scaleY).toInt() + yOffset
                                                        val absX = (box.left * scaleX).toInt()
                                                        val w = ((box.right - box.left) * scaleX).toInt()
                                                        val h = ((box.bottom - box.top) * scaleY).toInt()

                                                        appendDebug(
                                                            sb,
                                                            "[BLOCK:$blockCounter] {x: $absX, y: $absY, w: $w, h: $h}",
                                                        )
                                                        appendDebug(sb, decodedText)
                                                        appendDebug(sb, "")
                                                        blockCounter++
                                                    } else {
                                                        appendDebug(sb, "Recognition decoded blank text.")
                                                        appendDebug(sb, "")
                                                    }
                                                } else {
                                                    appendDebug(sb, "Recognition batch was empty.")
                                                    appendDebug(sb, "")
                                                }
                                            } else {
                                                appendDebug(sb, "Recognition output shape was unexpected.")
                                                appendDebug(sb, "")
                                            }
                                        } else {
                                            appendDebug(sb, "Recognition output tensor was null.")
                                            appendDebug(sb, "")
                                        }
                                    }
                                }
                            } finally {
                                recBitmap.recycle()
                            }
                        }
                    } catch (e: Exception) {
                        appendDebug(sb, "Slice error: ${e.message}")
                        appendDebug(sb, "")
                        Log.e(TAG, "Slice processing failed", e)
                    } finally {
                        slice.recycle()
                    }

                    yOffset += sliceMaxHeight
                }

                bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.e(TAG, "runLocalOcrTest crashed", e)
            sb.append("\n\nCRASH ERROR: ${e.message}")
        }

        return@withContext sb.toString()
    }

    private fun decodeRecognitionArray(sequence: Array<FloatArray>): String {
        val sb = StringBuilder()
        var lastIndex = -1

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
            }

            lastIndex = maxIdx
        }

        return sb.toString()
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
