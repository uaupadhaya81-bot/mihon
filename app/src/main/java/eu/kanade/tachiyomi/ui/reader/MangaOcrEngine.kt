package eu.kanade.tachiyomi.ui.reader

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
            file.isFile && (file.extension.equals("jpg", true) || file.extension.equals("png", true) || file.extension.equals("webp", true)) 
        }?.sortedBy { it.name } ?: return@withContext emptyMap()

        val compiledTextPrompt = StringBuilder()

        imageFiles.forEachIndexed { index, file ->
            val options = BitmapFactory.Options().apply { inMutable = true }
            val bitmap = BitmapFactory.decodeFile(file.absolutePath, options) ?: return@forEachIndexed

            // Simulated ONNX detection for now
            val foundBlocks = runOcrDetection(bitmap)
            
            chapterTranslationMap[index] = PageData(pageIndex = index, blocks = foundBlocks)

            compiledTextPrompt.append("--- PAGE $index ---\n")
            foundBlocks.forEachIndexed { blockIndex, block ->
                compiledTextPrompt.append("Block $blockIndex: ${block.text}\n")
            }
            compiledTextPrompt.append("\n")

            bitmap.recycle() // Instantly clears RAM!
            System.gc() 
        }

        if (compiledTextPrompt.isNotBlank() && apiKey.isNotBlank()) {
            val rawGeminiResponse = fetchBulkGeminiTranslation(compiledTextPrompt.toString())
            parseGeminiResponseIntoMap(rawGeminiResponse, chapterTranslationMap)
        }

        return@withContext chapterTranslationMap
    }

    private fun runOcrDetection(bitmap: Bitmap): List<TextBlock> {
        // Dummy data to test the Gemini connection before we write the complex ONNX math
        return listOf(
            TextBlock("お前はもう死んでいる。", 100f, 150f, 200f, 80f),
            TextBlock("何！？", 150f, 400f, 100f, 50f)
        )
    }

    private fun fetchBulkGeminiTranslation(bulkText: String): String {
        val url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key=$apiKey"
        val systemInstruction = "You are an expert manga translator. Translate the provided text into natural English. Preserve the exact structural layout. Maintain page and block formatting lines strictly (e.g., '--- PAGE X ---' and 'Block Y:'). Output ONLY the translated blocks without any conversational remarks or notes."

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
                jsonObject.getJSONArray("candidates").getJSONObject(0).getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text")
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
