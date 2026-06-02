package eu.kanade.tachiyomi.data.translation

import kotlinx.serialization.Serializable

/**
 * Represents the translation data for an entire chapter.
 */
@Serializable
data class ChapterTranslation(
    val chapterId: Long,
    // Maps the image filename (e.g., "01.jpg") to its translated page data
    val pages: Map<String, PageTranslation>,
)

/**
 * Holds all translated text blocks for a single page.
 */
@Serializable
data class PageTranslation(
    val blocks: List<TranslatedBlock>,
)

/**
 * A single translated text bubble and its mathematical coordinates on the image.
 */
@Serializable
data class TranslatedBlock(
    val englishText: String,
    val originalText: String = "", // Optional: Keep the Japanese text for reference
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
)
