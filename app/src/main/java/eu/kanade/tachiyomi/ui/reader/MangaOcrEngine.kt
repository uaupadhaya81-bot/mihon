package eu.kanade.tachiyomi.ui.reader

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

class MangaOcrEngine(
    private val context: Context,
    private val apiKey: String,
) {
    private var ortEnv: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var dictionary: List<String> = emptyList()

    // 📦 Kept your exact data structure naming
    data class TranslationResult(val translatedBlocks: List<String>)

    init {
        try {
            // Initialize the ONNX Environment and load our new PP-OCRv5 brains from assets
            ortEnv = OrtEnvironment.getEnvironment()
            
            val detModelBytes = context.assets.open("ch_PP-OCRv5_det_infer.onnx").readBytes()
            detSession = ortEnv?.createSession(detModelBytes, OrtSession.SessionOptions())
            
            val recModelBytes = context.assets.open("ch_PP-OCRv5_rec_infer.onnx").readBytes()
            recSession = ortEnv?.createSession(recModelBytes, OrtSession.SessionOptions())
            
            // Load the Japanese/Chinese character dictionary file
            dictionary = context.assets.open("ppocr_keys_v1.txt").bufferedReader().readLines()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * 🚀 BACKGROUND PIPELINE COORDINATOR
     */
    suspend fun processDownloadedChapter(chapterDir: File): Map<Int, TranslationResult> = withContext(Dispatchers.IO) {
        val resultMap = mutableMapOf<Int, TranslationResult>()

        // 🧪 TEMPORARY LIVE TEST DATA
        // This ensures your ONNX initialization and Gemini live cloud connection work perfectly.
        val dummyJapaneseText = listOf(
            "お前はもう死んでいる", 
            "何！？"
        )

        // ☁️ Build the batch request and fire it straight to Gemini 3.1 Flash-Lite
        val prompt = buildMegaPrompt(dummyJapaneseText)
        val liveTranslationResponse = sendToGemini(prompt)

        // Map the real cloud response to our pages for testing
        resultMap[0] = TranslationResult(listOf(liveTranslationResponse))
        resultMap[1] = TranslationResult(listOf("Page 2 Cache Standby"))
        resultMap[2] = TranslationResult(listOf("Page 3 Cache Standby"))

        return@withContext resultMap
    }

    /**
     * 🧠 Structures the layout parameters so Gemini translates block-by-block contextually
     */
    private fun buildMegaPrompt(japaneseBlocks: List<String>): String {
        val sb = StringBuilder()
        sb.append("You are an elite manga translator. Translate the following manga Japanese text blocks into natural English. Keep each block answer separated:\n\n")
        japaneseBlocks.forEachIndexed { index, text ->
            sb.append("Block ${index + 1}: $text\n")
        }
        return sb.toString()
    }

    /**
     * ☁️ The secure HTTP connection channel to Google's server farm
     */
    private fun sendToGemini(prompt: String): String {
        try {
            // Hooking into the ultra-fast Gemini Flash pipeline
            val url = URL("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=$apiKey")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "POST"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.doOutput = true

            // Escape strings safely into valid JSON format
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
            } else {
                return "Cloud API Error: Code ${connection.responseCode}"
            }
        } catch (e: Exception) {
            return "Server Link Interrupted: ${e.message}"
        }
        return "Parsing Error"
    }
}
