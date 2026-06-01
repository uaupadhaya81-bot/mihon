package eu.kanade.tachiyomi.data.translation

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import eu.kanade.tachiyomi.ui.reader.MangaOcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat
import android.content.Context
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf

fun enqueueChapterTranslation(context: Context, chapterId: Long) {
    val request = OneTimeWorkRequestBuilder<ChapterTranslationWorker>()
        .setInputData(workDataOf(ChapterTranslationWorker.KEY_CHAPTER_ID to chapterId))
        .build()

    WorkManager.getInstance(context).enqueue(request)
}

class ChapterTranslationWorker(
    private val context: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(context, workerParams) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val chapterId = inputData.getLong(KEY_CHAPTER_ID, -1L)
        if (chapterId == -1L) return@withContext Result.failure()

        try {
            // 1. Fetch your Gemini API key from Preferences
            val prefs = context.getSharedPreferences("OcrPrefs", Context.MODE_PRIVATE)
            val apiKey = prefs.getString("gemini_key", "") ?: ""
            if (apiKey.isEmpty()) {
                logcat(LogPriority.ERROR) { "TranslationWorker: Missing API Key" }
                return@withContext Result.failure()
            }

            // 2. Initialize the Engine ONCE for the whole chapter
            val engine = MangaOcrEngine(context, apiKey)

            // 3. TODO: Get the downloaded chapter directory using Mihon's DownloadManager
            // val downloadDir = downloadManager.findChapterDir(chapterId)
            // val imageFiles = downloadDir.listFiles()

            val pageTranslations = mutableMapOf<String, PageTranslation>()

            /* * 4. THE PROCESSING LOOP
             * Loop through imageFiles here. For each file:
             * - Load the Bitmap.
             * - Pass it to engine.processSingleImage(bitmap).
             * - Map the results into your TranslatedBlock data class.
             * - Add it to pageTranslations map.
             * - IMPORTANT: bitmap.recycle() after every loop to prevent OOM!
             */

            // 5. Serialize and save to disk
            val finalTranslation = ChapterTranslation(
                chapterId = chapterId,
                pages = pageTranslations
            )
            
            // TODO: Use kotlinx.serialization to write finalTranslation to a "translation.json" 
            // file inside the chapter's download directory.

            Result.success()
        } catch (e: Exception) {
            logcat(LogPriority.ERROR, e) { "TranslationWorker failed" }
            Result.failure()
        }
    }

    companion object {
        const val KEY_CHAPTER_ID = "chapter_id"
    }
}
