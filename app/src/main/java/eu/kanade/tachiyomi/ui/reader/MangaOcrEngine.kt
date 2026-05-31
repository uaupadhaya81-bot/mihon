package eu.kanade.tachiyomi.ui.reader

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class MangaOcrEngine(
    private val context: Context,
    private val apiKey: String,
) {
    data class TranslationResult(val translatedBlocks: List<String>)

    suspend fun processDownloadedChapter(chapterDir: File): Map<Int, TranslationResult> {
        return withContext(Dispatchers.IO) {
            val resultMap = mutableMapOf<Int, TranslationResult>()

            // Our test dummy data
            resultMap[0] = TranslationResult(listOf("You are already dead.", "What!?"))
            resultMap[1] = TranslationResult(listOf("You are already dead.", "What!?"))
            resultMap[2] = TranslationResult(listOf("You are already dead.", "What!?"))

            resultMap
        }
    }
}
