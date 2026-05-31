package eu.kanade.tachiyomi.ui.reader

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
            
            dictionary = context.assets.open("ppocr_keys_v1.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * 📸 Process a single image (Bulletproof Array Unpacking Version)
     */
    suspend fun processSingleImage(bitmap: Bitmap): String = withContext(Dispatchers.IO) {
        try {
            // 1. RAM PROTECTION (Shrink)
            val scaledBitmap = OcrUtils.downscaleImageForDetection(bitmap)

            // 2. TENSOR CREATION
            val floatBuffer = OcrUtils.bitmapToFloatBuffer(scaledBitmap)
            val shape = longArrayOf(1, 3, scaledBitmap.height.toLong(), scaledBitmap.width.toLong())
            
            var detW = scaledBitmap.width
            var detH = scaledBitmap.height
            var flatProbabilities = FloatArray(0)

            OnnxTensor.createTensor(ortEnv, floatBuffer, shape).use { tensor ->
                val inputName = detSession?.inputNames?.iterator()?.next()
                val detResults = detSession?.run(Collections.singletonMap(inputName, tensor))
                
                detResults?.use { results ->
                    // Safely extract the raw value without triggering getShape()
                    val detOutputTensor = results.iterator().next().value as? OnnxTensor
                    if (detOutputTensor != null) {
                        // Recursively unpack the raw Java Object to bypass Kotlin Cast Exceptions
                        val rawDetArray = detOutputTensor.value as? Array<*>
                        if (rawDetArray != null && rawDetArray.isNotEmpty()) {
                            val batch = rawDetArray[0] as? Array<*>
                            if (batch != null && batch.isNotEmpty()) {
                                val channel = batch[0] as? Array<*>
                                if (channel != null && channel.isNotEmpty()) {
                                    detH = channel.size
                                    detW = (channel[0] as? FloatArray)?.size ?: 0
                                    
                                    flatProbabilities = FloatArray(detW * detH)
                                    var idx = 0
                                    for (y in 0 until detH) {
                                        val row = channel[y] as? FloatArray
                                        if (row != null) {
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
            }

            if (flatProbabilities.isEmpty()) return@withContext "Failed to parse detection output."

            val scaleX = bitmap.width.toFloat() / detW
            val scaleY = bitmap.height.toFloat() / detH

            // 3. DRAW BOXES
            val boxes = DbNetMath.extractBoundingBoxes(flatProbabilities, detW, detH)
            if (boxes.isEmpty()) return@withContext "No text found in this image."

            // 4. CROP & RECOGNIZE
            val japaneseTextBlocks = mutableListOf<String>()
            val recInputName = recSession?.inputNames?.iterator()?.next()

            for (box in boxes) {
                val croppedBubble = OcrUtils.cropBubble(bitmap, box, scaleX, scaleY)
                
                val recHeight = 48
                val recWidth = (croppedBubble.width.toFloat() / croppedBubble.height * recHeight).toInt().coerceAtLeast(1)
                val recBitmap = Bitmap.createScaledBitmap(croppedBubble, recWidth, recHeight, true)

                val recBufferIn = OcrUtils.bitmapToFloatBuffer(recBitmap)
                val recShapeIn = longArrayOf(1, 3, recHeight.toLong(), recWidth.toLong())
                
                OnnxTensor.createTensor(ortEnv, recBufferIn, recShapeIn).use { recTensor ->
                    val recResults = recSession?.run(Collections.singletonMap(recInputName, recTensor))
                    
                    recResults?.use { recRes ->
                        val recOutputTensor = recRes.iterator().next().value as? OnnxTensor
                        if (recOutputTensor != null) {
                            val rawRecArray = recOutputTensor.value as? Array<*>
                            if (rawRecArray != null && rawRecArray.isNotEmpty()) {
                                val batch = rawRecArray[0] as? Array<*>
                                if (batch != null && batch.isNotEmpty()) {
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

            if (japaneseTextBlocks.isEmpty()) return@withContext "Failed to extract text from detected regions."

            // 5. THE CLOUD LEAP
            val prompt = buildMegaPrompt(japaneseTextBlocks)
            return@withContext sendToGemini(prompt)

        } catch (e: Exception) {
            return@withContext "Engine Error: ${e.message}"
        }
    }

    /**
     * 📖 Translates raw sequential float arrays into Kanji characters
     */
    private fun decodeRecognitionArray(sequence: Array<*>): String {
        val sb = java.lang.StringBuilder()
        var lastIndex = -1

        for (timeStepObj in sequence) {
            val timeStep = timeStepObj as? FloatArray ?: continue
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

    /**
     * 🧠 Structures the layout parameters for Gemini
     */
    private fun buildMegaPrompt(japaneseBlocks: List<String>): String {
        val sb = java.lang.StringBuilder()
        sb.append("You are an elite manga translator. Translate the following Japanese text blocks to English. Keep each block separated:\n\n")
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
            val url = URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$apiKey")
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

    suspend fun processDownloadedChapter(chapterDir: File): Map<Int, TranslationResult> = withContext(Dispatchers.IO) {
        val resultMap = mutableMapOf<Int, TranslationResult>()
        resultMap[0] = TranslationResult(listOf("Chapter Mode Ready!"))
        return@withContext resultMap
    }
}
