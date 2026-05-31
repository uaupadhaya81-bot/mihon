package eu.kanade.tachiyomi.ui.reader

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class MangaOcrEngine(private val context: Context, private val apiKey: String) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var detSession: OrtSession? = null
    
    // Mihon natively ships with OkHttp, so we use it for our API calls
    private val client = OkHttpClient()

    init {
        try {
            val detModelBytes = context.assets.open("det_model.onnx").readBytes()
            detSession = env.createSession(detModelBytes)
            Log.d("MangaOcrEngine", "ONNX Model Loaded Successfully!")
        } catch (e: Exception) {
            Log.e("MangaOcrEngine", "Failed to load ONNX model", e)
        }
    }

    suspend fun processAndTranslate(bitmap: Bitmap): String = withContext(Dispatchers.Default) {
        // TODO: The ONNX C++ Math goes here to extract bounding boxes.
        // For testing the API connection, we will simulate the ONNX model 
        // successfully extracting a string of Japanese text from a speech bubble:
        val extractedJapaneseText = "こんにちは、世界！"

        // Send it to Google's Gemini server
        val translatedText = fetchGeminiTranslation(extractedJapaneseText)
        
        return@withContext translatedText
    }

    private fun fetchGeminiTranslation(sourceText: String): String {
        // Failsafe if the user forgot to enter their API Key in the settings
        if (apiKey.isBlank()) return "Error: Gemini API Key is missing."

        // The official 2026 Gemini 3.1 Flash-Lite endpoint
        val url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key=$apiKey"
        
        // Build the strict prompt required to prevent Gemini from being too chatty
        val jsonPayload = """
            {
              "contents": [{
                "parts": [{"text": "You are a professional manga translator. Translate the following text to English. Reply ONLY with the English translation, without quotes, formatting, or extra conversational text. Text: $sourceText"}]
              }]
            }
        """.trimIndent()

        val body = jsonPayload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e("MangaOcrEngine", "Gemini API Error: ${response.code}")
                    return "API Error: ${response.code}"
                }
                
                // Extract the JSON body returned by Google's servers
                val responseData = response.body?.string() ?: return "Empty API Response"
                val jsonObject = JSONObject(responseData)
                
                // Drill down into the JSON tree to find the raw translated text
                val candidates = jsonObject.getJSONArray("candidates")
                val content = candidates.getJSONObject(0).getJSONObject("content")
                val parts = content.getJSONArray("parts")
                val translatedString = parts.getJSONObject(0).getString("text").trim()
                
                Log.d("MangaOcrEngine", "Successfully Translated: $translatedString")
                return translatedString
            }
        } catch (e: IOException) {
            Log.e("MangaOcrEngine", "Network failure while contacting Gemini", e)
            return "Network Timeout"
        } catch (e: Exception) {
            Log.e("MangaOcrEngine", "Failed to parse Gemini JSON", e)
            return "Parsing Error"
        }
    }
}
