package eu.kanade.tachiyomi.data.translation

import android.content.Context
import android.graphics.BitmapFactory
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import eu.kanade.tachiyomi.data.download.DownloadManager
import eu.kanade.tachiyomi.ui.reader.MangaOcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import logcat.LogPriority
import tachiyomi.core.common.util.system.logcat
import tachiyomi.domain.chapter.interactor.GetChapter
import tachiyomi.domain.manga.interactor.GetManga
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get

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

            // 3. Inject Mihon's Database & Download Manager to find the physical folder
            val getChapter: GetChapter = Injekt.get()
            val getManga: GetManga = Injekt.get()
            val downloadManager: DownloadManager = Injekt.get()

            val chapter = getChapter.await(chapterId) ?: return@withContext Result.failure()
            val manga = getManga.await(chapter.mangaId) ?: return@withContext Result.failure()
            
            val chapterDir = downloadManager.getProvider().findChapterDir(chapter.name, chapter.scanlator, manga.title, manga.source)
            if (chapterDir == null || !chapterDir.exists()) {
                logcat(LogPriority.ERROR) { "TranslationWorker: Chapter not downloaded" }
                return@withContext Result.failure()
            }

            val pageTranslations = mutableMapOf<String, PageTranslation>()

            // 4. THE PROCESSING LOOP (Find all .jpg, .png, .webp files)
            val files = chapterDir.listFiles()?.filter { 
                it.name?.endsWith(".jpg", true) == true || 
                it.name?.endsWith(".png", true) == true || 
                it.name?.endsWith(".webp", true) == true 
            }?.sortedBy { it.name } ?: emptyList()

            for (file in files) {
                val fileName = file.name ?: continue
                
                // Load Bitmap safely
                val inputStream = file.openInputStream()
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream.close()
                
                if (bitmap != null) {
                    // Send to OCR & Gemini
                    val translatedText = engine.processSingleImage(bitmap)
                    
                    // 🔥 CRITICAL: Clear from RAM immediately to prevent crashes 🔥
                    bitmap.recycle()
                    
                    // Store the result. 
                    // (Note: Currently saving the whole text block at X:0, Y:0. 
                    // We will refine exact coordinates in a later step).
                    pageTranslations[fileName] = PageTranslation(
                        blocks = listOf(TranslatedBlock(englishText = translatedText, x = 0f, y = 0f, width = 0f, height = 0f))
                    )
                }
            }

            // 5. Serialize and save to translation.json inside the chapter folder
            val finalTranslation = ChapterTranslation(
                chapterId = chapterId,
                pages = pageTranslations
            )
            
            val jsonString = Json.encodeToString(finalTranslation)
            val translationFile = chapterDir.createFile("translation.json")
            
            translationFile?.openOutputStream()?.use { os ->
                os.write(jsonString.toByteArray(Charsets.UTF_8))
            }

            logcat(LogPriority.INFO) { "TranslationWorker: Chapter $chapterId translated successfully!" }
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
