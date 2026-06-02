package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
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

    data class TranslationResult(val translatedBlocks: List<String>)

    init {
        try {
            ortEnv = OrtEnvironment.getEnvironment()

            val detModelBytes = context.assets.open("ch_PP-OCRv5_det_infer.onnx").readBytes()
            detSession = ortEnv?.createSession(detModelBytes, OrtSession.SessionOptions())

            val recModelBytes = context.assets.open("ch_PP-OCRv5_rec_infer.onnx").readBytes()
            recSession = ortEnv?.createSession(recModelBytes, OrtSession.SessionOptions())

            dictionary = context.assets.open("ppocrv5_dict.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    suspend fun processSingleImage(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        try {
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

            OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { tensor ->
                val inputName = detSession?.inputNames?.iterator()?.next()
                val detMap = Collections.singletonMap(inputName, tensor)
                val detResults = detSession?.run(detMap)

                detResults?.use { results ->
                    val detOutputTensor = results.iterator().next().value as? OnnxTensor
                    if (detOutputTensor != null) {
                        @Suppress("UNCHECKED_CAST")
                        val rawDetArray = detOutputTensor.value as?
                            Array<Array<Array<FloatArray>>>

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
            val recInputName = recSession?.inputNames?.iterator()?.next()

            for (box in boxes) {
                val croppedBubble = OcrUtils.cropBubble(bitmap, box, scaleX, scaleY)
                val recHeight = 48
                val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight)
                    .toInt().coerceAtLeast(1)

                val recBitmap = Bitmap.createScaledBitmap(
                    croppedBubble,
                    recWidth,
                    recHeight,
                    true,
                )
                val recBufferIn = OcrUtils.bitmapToFloatBuffer(recBitmap)
                val recShapeIn = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())

                OnnxTensor.createTensor(ortEnv, recBufferIn, recShapeIn).use { recTensor ->
                    val recMap = Collections.singletonMap(recInputName, recTensor)
                    val recResults = recSession?.run(recMap)

                    recResults?.use { recRes ->
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
            }

            if (japaneseTextBlocks.isEmpty()) return@withContext "Failed to extract text."
            val prompt = buildMegaPrompt(japaneseTextBlocks)
            return@withContext sendToGemini(prompt)
        } catch (e: Exception) {
            return@withContext "Engine Error: ${e.message}"
        }
    }

    suspend fun runLocalOcrTest(
        context: Context,
        uris: List<Uri>,
    ): String = withContext(Dispatchers.IO) {
        val sb = StringBuilder()

        try {
            uris.forEachIndexed { index, uri ->
                val pageName = "PAGE_${index + 1}"
                sb.append("========================\n[$pageName]\n========================\n")

                val inputStream = context.contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap == null) {
                    sb.append("Error: Failed to load picture into memory.\n\n")
                    return@forEachIndexed
                }

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

                    val scaledSlice = OcrUtils.downscaleImageForDetection(slice)
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

                    OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { tensor ->
                        val inputName = detSession?.inputNames?.iterator()?.next()
                        val detMap = Collections.singletonMap(inputName, tensor)
                        val detResults = detSession?.run(detMap)

                        detResults?.use { results ->
                            val detOutputTensor = results.iterator().next().value as? OnnxTensor
                            if (detOutputTensor != null) {
                                @Suppress("UNCHECKED_CAST")
                                val rawDetArray = detOutputTensor.value as?
                                    Array<Array<Array<FloatArray>>>

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

                    if (flatProbabilities.isNotEmpty()) {
                        val scaleX = slice.width.toFloat() / detW
                        val scaleY = slice.height.toFloat() / detH
                        val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
                        val recInputName = recSession?.inputNames?.iterator()?.next()

                        for (box in boxes) {
                            val croppedBubble = OcrUtils.cropBubble(slice, box, scaleX, scaleY)
                            val recHeight = 48
                            val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight)
                                .toInt().coerceAtLeast(1)

                            val recBitmap = Bitmap.createScaledBitmap(
                                croppedBubble,
                                recWidth,
                                recHeight,
                                true,
                            )
                            val recBufferIn = OcrUtils.bitmapToFloatBuffer(recBitmap)
                            val recShapeIn = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())

                            OnnxTensor.createTensor(ortEnv, recBufferIn, recShapeIn).use { recTensor ->
                                val recMap = Collections.singletonMap(recInputName, recTensor)
                                val recResults = recSession?.run(recMap)

                                recResults?.use { recRes ->
                                    val recOut = recRes.iterator().next().value as? OnnxTensor
                                    if (recOut != null) {
                                        @Suppress("UNCHECKED_CAST")
                                        val rawRecArray = recOut.value as? Array<Array<FloatArray>>
                                        if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                            val batch = rawRecArray[0]
                                            if (batch.isNotEmpty()) {
                                                val decodedText = decodeRecognitionArray(batch)
                                                if (decodedText.isNotBlank()) {
                                                    val absY = (box.top * scaleY).toInt() + yOffset
                                                    val absX = (box.left * scaleX).toInt()
                                                    val w = ((box.right - box.left) * scaleX).toInt()
                                                    val h = ((box.bottom - box.top) * scaleY).toInt()

                                                    sb.append("[BLOCK:$blockCounter] ")
                                                    sb.append("{x: $absX, y: $absY, w: $w, h: $h}\n")
                                                    sb.append("$decodedText\n\n")
                                                    blockCounter++
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            recBitmap.recycle()
                        }
                    }
                    if (slice != bitmap) slice.recycle()
                    yOffset += sliceMaxHeight
                }
                bitmap.recycle()
            }
        } catch (e: Exception) {
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
        try {
            val urlString = "https://generativelanguage.googleapis.com/v1beta/" +
                "models/gemini-3.1-flash-lite:generateContent?key=$apiKey"
            val url = URL(urlString)
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.doOutput = true

            val cleanPrompt = prompt.replace("\n", "\\n").replace("\"", "\\\"")
            val jsonPayload =
                "{\"contents\": [{\"parts\": [{\"text\": \"$cleanPrompt\"}]}]}"

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
            return "API Error: ${connection.responseCode}"
        } catch (e: Exception) {
            return "Connection Failed: ${e.message}"
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
                val jsonPayload =
                    "{\"contents\": [{\"parts\": [{\"text\": \"$cleanPrompt\"}]}]}"

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
                return@withContext "API Error: ${connection.responseCode}"
            } catch (e: Exception) {
                return@withContext "Error: ${e.message}"
            }
        }
    }
}
