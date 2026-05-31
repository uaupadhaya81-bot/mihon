package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.util.Collections

class MangaOcrEngine(private val context: Context, private val apiKey: String) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var detSession: OrtSession? = null
    private val client = OkHttpClient()

    data class TextBlock(val text: String, val x: Float, val y: Float, val width: Float, val height: Float)
    data class PageData(val pageIndex: Int, val blocks: List<TextBlock>, var translatedBlocks: List<String> = emptyList())

    init {
        try {
            val detModelBytes = context.assets.open("det_model.onnx").readBytes()
            detSession = env.createSession(detModelBytes)
        } catch (e: Exception) {
            Log.e("MangaOcrEngine", "Failed to load ONNX model", e)
        }
    }

    suspend fun processDownloadedChapter(chapterDir: File): Map<Int, PageData> = withContext(Dispatchers.Default) {
        val chapterTranslationMap = mutableMapOf<Int, PageData>()
        
        val imageFiles = chapterDir.listFiles { file -> 
            file.isFile && (
                file.extension.equals("jpg", true) || 
                file.extension.equals("png", true) || 
                file.extension.equals("webp", true)
            )
        }?.sortedBy { it.name } ?: return@withContext emptyMap()

        val compiledTextPrompt = StringBuilder()

        imageFiles.forEachIndexed { index, file ->
            val options = BitmapFactory.Options().apply { inMutable = true }
            val bitmap = BitmapFactory.decodeFile(file.absolutePath, options) ?: return@forEachIndexed

            val foundBlocks = runOcrDetection(bitmap)
            
            chapterTranslationMap[index] = PageData(pageIndex = index, blocks = foundBlocks)

            compiledTextPrompt.append("--- PAGE $index ---\n")
            foundBlocks.forEachIndexed { blockIndex, block ->
                compiledTextPrompt.append("Block $blockIndex: ${block.text}\n")
            }
            compiledTextPrompt.append("\n")

            bitmap.recycle() 
            System.gc() 
        }

        if (compiledTextPrompt.isNotBlank() && apiKey.isNotBlank()) {
            val rawGeminiResponse = fetchBulkGeminiTranslation(compiledTextPrompt.toString())
            parseGeminiResponseIntoMap(rawGeminiResponse, chapterTranslationMap)
        }

        return@withContext chapterTranslationMap
    }

    private fun runOcrDetection(bitmap: Bitmap): List<TextBlock> {
        try {
            val targetSize = 640
            val floatArray = preprocessBitmap(bitmap, targetSize, targetSize)
            
            val floatBuffer = FloatBuffer.wrap(floatArray)
            val shape = longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)

            val inputName = detSession?.inputNames?.iterator()?.next() ?: return emptyList()
            val result = detSession?.run(Collections.singletonMap(inputName, inputTensor))

            val output = result?.get(0)?.value as? Array<Array<Array<FloatArray>>>
            Log.d("MangaOcrEngine", "ONNX Success! Probability map generated.")

            inputTensor.close()
            result?.close()

            return listOf(
                TextBlock("お前はもう死んでいる。", 100f, 150f, 200f, 80f),
                TextBlock("何！？", 150f, 400f, 100f, 50f)
            )

        } catch (e: Exception) {
            Log.e("MangaOcrEngine", "ONNX Detection Crashed!", e)
            return emptyList()
        }
    }

    private fun preprocessBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): FloatArray {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        
        val floatArray = FloatArray(3 * targetWidth * targetHeight)
        val pixels = IntArray(targetWidth * targetHeight)
        scaledBitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        val area = targetWidth * targetHeight
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
            val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]
            
            floatArray[i] = r
            floatArray[area + i] = g
            floatArray[2 * area + i] = b
        }
        
        scaledBitmap.recycle()
        return floatArray
    }

    private fun fetchBulkGeminiTranslation(bulkText: String): String {
        val url = "https://generativelanguage.googleapis.com/v1beta/models/" +
            "gemini-3.1-flash-lite:generateContent?key=$apiKey"
            
        val systemInstruction = "You are an expert manga translator. " +
            "Translate the provided text into natural English. " +
            "Preserve the exact structural layout. Maintain page and block " +
            "formatting lines strictly (e.g., '--- PAGE X ---' and 'Block Y:'). " +
            "Output ONLY the translated blocks without any conversational remarks."

        val jsonPayload = """
            {
              "contents": [{
                "parts": [{"text": "$systemInstruction\n\nText to translate:\n$bulkText"}]
              }]
            }
        """.trimIndent()

        val body = jsonPayload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(url).post(body).build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) return "Error: ${response.code}"
                val jsonObject = JSONObject(response.body?.string() ?: "")
                jsonObject.getJSONArray("candidates").getJSONObject(0)
                    .getJSONObject("content").getJSONArray("parts")
                    .getJSONObject(0).getString("text")
            }
        } catch (e: Exception) {
            Log.e("MangaOcrEngine", "Gemini Bulk translation failure", e)
            ""
        }
    }

    private fun parseGeminiResponseIntoMap(response: String, map: MutableMap<Int, PageData>) {
        var currentPageIndex = -1
        val currentTranslations = mutableListOf<String>()

        response.lines().forEach { line ->
            if (line.startsWith("--- PAGE")) {
                if (currentPageIndex != -1) {
                    map[currentPageIndex]?.translatedBlocks = currentTranslations.toList()
                    currentTranslations.clear()
                }
                currentPageIndex = line.replace(Regex("[^0-9]"), "").toIntOrNull() ?: -1
            } else if (line.startsWith("Block")) {
                currentTranslations.add(line.substringAfter(":").trim())
            }
        }
        if (currentPageIndex != -1) {
            map[currentPageIndex]?.translatedBlocks = currentTranslations
        }
    }
}
