package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import eu.kanade.tachiyomi.ui.reader.utils.DbNetMath
import eu.kanade.tachiyomi.ui.reader.utils.OcrUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.FloatBuffer
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

            // Load the dictionary file
            dictionary = context.assets.open("ppocr_keys_v1.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * 📸 Process a single image from the Gallery (The Real Deal)
     */
    suspend fun processSingleImage(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        try {
            // 1. RAM PROTECTION (Shrink)
            val scaledBitmap = OcrUtils.downscaleImageForDetection(bitmap)
            val scaleX = bitmap.width.toFloat() / scaledBitmap.width
            val scaleY = bitmap.height.toFloat() / scaledBitmap.height

            // 2. MATHEMATICAL CONVERSION
            val floatBuffer = OcrUtils.bitmapToFloatBuffer(scaledBitmap)
            val shape = longArrayOf(1, 3, scaledBitmap.height.toLong(), scaledBitmap.width.toLong())
            val tensor = OnnxTensor.createTensor(ortEnv, floatBuffer, shape)

            // 3. RUN THE DETECT MODEL
            val inputName = detSession?.inputNames?.iterator()?.next()
            val detResults = detSession?.run(Collections.singletonMap(inputName, tensor))
            val rawOutput = detResults?.get(0)?.value as Array<Array<Array<FloatArray>>>

            // Flatten the weird multi-dimensional array into a simple 1D array for our math
            val flatProbabilities = FloatArray(scaledBitmap.width * scaledBitmap.height)
            var index = 0
            for (y in 0 until scaledBitmap.height) {
                for (x in 0 until scaledBitmap.width) {
                    flatProbabilities[index++] = rawOutput[0][0][y][x]
                }
            }

            // 4. THE MAGIC MATH (Draw the boxes)
            val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, scaledBitmap.width, scaledBitmap.height)
            if (boxes.isEmpty()) return@withContext "No text found in this image."

            // 5. CROP & RECOGNIZE
            val japaneseTextBlocks = mutableListOf<String>()
            val recInputName = recSession?.inputNames?.iterator()?.next()

            for (box in boxes) {
                // Crop the high-res image
                val croppedBubble = OcrUtils.cropBubble(bitmap, box, scaleX, scaleY)

                // PP-OCR expects recognition images to be exactly 48 pixels high
                val recHeight = 48
                val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight).toInt().coerceAtLeast(
                    1,
                )
                val recBitmap = Bitmap.createScaledBitmap(croppedBubble, recWidth, recHeight, true)

                // Run the recognition model
                val recBuffer = OcrUtils.bitmapToFloatBuffer(recBitmap)
                val recShape = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())
                val recTensor = OnnxTensor.createTensor(ortEnv, recBuffer, recShape)

                val recResults = recSession?.run(Collections.singletonMap(recInputName, recTensor))

                // Decode the Kanji using our dictionary
                val recOutput = recResults?.get(0)?.value as Array<Array<FloatArray>>
                val decodedText = decodeRecognitionOutput(recOutput[0])

                if (decodedText.isNotBlank()) {
                    japaneseTextBlocks.add(decodedText)
                }
            }

            if (japaneseTextBlocks.isEmpty()) return@withContext "Failed to read the characters."

            // 6. THE CLOUD LEAP
            val prompt = buildMegaPrompt(japaneseTextBlocks)
            return@withContext sendToGemini(prompt)
        } catch (e: Exception) {
            return@withContext "Engine Error: ${e.message}"
        }
    }

    /**
     * 📖 Translates the AI's output numbers back into actual Japanese characters
     */
    private fun decodeRecognitionOutput(outputGrid: Array<FloatArray>): String {
        val sb = java.lang.StringBuilder()
        var lastIndex = -1

        for (timeStep in outputGrid) {
            var maxProb = 0f
            var maxIdx = -1
            for (i in timeStep.indices) {
                if (timeStep[i] > maxProb) {
                    maxProb = timeStep[i]
                    maxIdx = i
                }
            }
            // Ignore blanks (usually index 0) and repeating characters (CTC decoding)
            if (maxIdx > 0 && maxIdx != lastIndex && maxIdx <= dictionary.size) {
                sb.append(dictionary[maxIdx - 1])
            }
            lastIndex = maxIdx
        }
        return sb.toString()
    }

    /**
     * 🧠 Structures the layout parameters for Gemini
     */
    private fun buildMegaPrompt(japaneseBlocks: List<String>): String {
        val sb = java.lang.StringBuilder()
        sb.append(
            "You are an elite manga translator. Translate the following Japanese text blocks to English. Keep each block separated:\n\n",
        )
        japaneseBlocks.forEachIndexed { index, text ->
            sb.append("Block ${index + 1}: $text\n")
        }
        return sb.toString()
    }

    /**
     * ☁️ The HTTP connection to Gemini
     */
    private fun sendToGemini(prompt: String): String {
        try {
            val url =
                URL(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$apiKey",
                )
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
            return "API Error: ${connection.responseCode}"
        } catch (e: Exception) {
            return "Connection Failed: ${e.message}"
        }
    }

    // Retaining this so the rest of your app doesn't break
    suspend fun processDownloadedChapter(chapterDir: File): Map<Int, TranslationResult> = withContext(Dispatchers.IO) {
        val resultMap = mutableMapOf<Int, TranslationResult>()
        resultMap[0] = TranslationResult(listOf("Chapter Mode Coming Next!"))
        return@withContext resultMap
    }
}
