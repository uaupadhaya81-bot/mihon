package eu.kanade.tachiyomi.data.translation

import android.content.Context
import android.graphics.BitmapFactory
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.hippo.unifile.UniFile
import eu.kanade.tachiyomi.data.download.DownloadProvider
import eu.kanade.tachiyomi.ui.reader.MangaOcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import logcat.LogPriority
import org.json.JSONArray
import org.json.JSONObject
import tachiyomi.core.common.util.system.logcat
import tachiyomi.domain.chapter.interactor.GetChapter
import tachiyomi.domain.manga.interactor.GetManga
import tachiyomi.domain.source.service.SourceManager // <-- Fixed Import
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get
import java.io.OutputStream

class ChapterTranslationWorker(
    private val context: Context,
    workerParams: WorkerParameters,
) : CoroutineWorker(context, workerParams) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val chapterId = inputData.getLong(KEY_CHAPTER_ID, -1L)
        if (chapterId == -1L) return@withContext Result.failure()

        try {
            // Read the key securely from the app's internal SharedPreferences
            val prefs = context.getSharedPreferences("OcrPrefs", Context.MODE_PRIVATE)
            val apiKey = prefs.getString("gemini_key", "") ?: ""

            if (apiKey.isEmpty()) {
                logcat(LogPriority.ERROR) { "TranslationWorker: Missing API Key" }
                return@withContext Result.failure()
            }

            // 2. Initialize the Engine ONCE for the whole chapter
            val engine = MangaOcrEngine(context, apiKey)

            // 3. Inject Mihon's Database, Download Provider & Source Manager
            val getChapter: GetChapter = Injekt.get()
            val getManga: GetManga = Injekt.get()
            val downloadProvider: DownloadProvider = Injekt.get()
            val sourceManager: SourceManager = Injekt.get()

            val chapter = getChapter.await(chapterId) ?: return@withContext Result.failure()
            val manga = getManga.await(chapter.mangaId) ?: return@withContext Result.failure()

            // Fix 1: Convert the Long ID into the actual Source object
            val source = sourceManager.get(manga.source) ?: return@withContext Result.failure()

            // Fix 2: Pass all 5 parameters (added chapterUrl = chapter.url)
            val chapterDir: UniFile? = downloadProvider.findChapterDir(
                chapterName = chapter.name,
                chapterScanlator = chapter.scanlator,
                chapterUrl = chapter.url,
                mangaTitle = manga.title,
                source = source,
            )

            // Fix 3: Handle nullability explicitly
            if (chapterDir == null || !chapterDir.exists()) {
                logcat(LogPriority.ERROR) { "TranslationWorker: Chapter not downloaded" }
                return@withContext Result.failure()
            }
            // Lock the folder into a non-null variable so Kotlin stops whining
            val safeDir = chapterDir!!

            val pageTranslations = mutableMapOf<String, PageTranslation>()

            // 4. THE PROCESSING LOOP (Find all .jpg, .png, .webp files)
            val files = safeDir.listFiles()?.filter {
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

                    pageTranslations[fileName] = PageTranslation(
                        blocks = listOf(
                            TranslatedBlock(englishText = translatedText, x = 0f, y = 0f, width = 0f, height = 0f),
                        ),
                    )
                }
            }

            // 5. Serialize and save to translation.json inside the chapter folder
            val rootJson = JSONObject()
            rootJson.put("chapterId", chapterId)

            val pagesJson = JSONObject()
            for ((fileName, pageData) in pageTranslations) {
                val blocksArray = JSONArray()
                for (block in pageData.blocks) {
                    val blockJson = JSONObject()
                    blockJson.put("englishText", block.englishText)
                    blockJson.put("x", block.x.toDouble())
                    blockJson.put("y", block.y.toDouble())
                    blockJson.put("width", block.width.toDouble())
                    blockJson.put("height", block.height.toDouble())
                    blocksArray.put(blockJson)
                }
                val pageObj = JSONObject()
                pageObj.put("blocks", blocksArray)
                pagesJson.put(fileName, pageObj)
            }
            rootJson.put("pages", pagesJson)

            // Convert to string (The '4' makes the JSON perfectly formatted and readable)
            val jsonString = rootJson.toString(4)

            // Fix 4: Use safeDir to write the file
            val translationFile = safeDir.createFile("translation.json")
            val outputStream: OutputStream? = translationFile?.openOutputStream()
            if (outputStream != null) {
                try {
                    outputStream.write(jsonString.toByteArray(Charsets.UTF_8))
                } finally {
                    outputStream.close()
                }
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
