package eu.kanade.tachiyomi.ui.reader.viewer.pager

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.text.method.ScrollingMovementMethod
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.widget.FrameLayout
import android.widget.TextView
import androidx.core.view.isVisible
import eu.kanade.presentation.util.formattedMessage
import eu.kanade.tachiyomi.data.download.DownloadProvider
import eu.kanade.tachiyomi.databinding.ReaderErrorBinding
import eu.kanade.tachiyomi.source.model.Page
import eu.kanade.tachiyomi.ui.reader.model.InsertPage
import eu.kanade.tachiyomi.ui.reader.model.ReaderPage
import eu.kanade.tachiyomi.ui.reader.viewer.ReaderPageImageView
import eu.kanade.tachiyomi.ui.reader.viewer.ReaderProgressIndicator
import eu.kanade.tachiyomi.ui.webview.WebViewActivity
import eu.kanade.tachiyomi.widget.ViewPagerAdapter
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.supervisorScope
import logcat.LogPriority
import okio.Buffer
import okio.BufferedSource
import tachiyomi.core.common.i18n.stringResource
import tachiyomi.core.common.util.lang.launchIO
import tachiyomi.core.common.util.lang.withIOContext
import tachiyomi.core.common.util.lang.withUIContext
import tachiyomi.core.common.util.system.ImageUtil
import tachiyomi.core.common.util.system.logcat
import tachiyomi.domain.source.service.SourceManager
import tachiyomi.i18n.MR
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get

@SuppressLint("ViewConstructor")
class PagerPageHolder(
    readerThemedContext: Context,
    val viewer: PagerViewer,
    val page: ReaderPage,
) : ReaderPageImageView(readerThemedContext), ViewPagerAdapter.PositionableView {

    override val item
        get() = page

    private var progressIndicator: ReaderProgressIndicator? = null
    private var errorLayout: ReaderErrorBinding? = null
    private val scope = MainScope()
    private var loadJob: Job? = null

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        loadJob = scope.launch { loadPageAndProcessStatus() }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        loadJob?.cancel()
        loadJob = null
    }

    private fun initProgressIndicator() {
        if (progressIndicator == null) {
            progressIndicator = ReaderProgressIndicator(context)
            addView(progressIndicator)
        }
    }

    private suspend fun loadPageAndProcessStatus() {
        val loader = page.chapter.pageLoader ?: return
        supervisorScope {
            launchIO {
                loader.loadPage(page)
            }
            page.statusFlow.collectLatest { state ->
                when (state) {
                    Page.State.Queue -> setQueued()
                    Page.State.LoadPage -> setLoading()
                    Page.State.DownloadImage -> {
                        setDownloading()
                        page.progressFlow.collectLatest { value ->
                            progressIndicator?.setProgress(value)
                        }
                    }
                    Page.State.Ready -> setImage()
                    is Page.State.Error -> setError(state.error)
                }
            }
        }
    }

    private fun setQueued() {
        initProgressIndicator()
        progressIndicator?.show()
        removeErrorLayout()
    }

    private fun setLoading() {
        initProgressIndicator()
        progressIndicator?.show()
        removeErrorLayout()
    }

    private fun setDownloading() {
        initProgressIndicator()
        progressIndicator?.show()
        removeErrorLayout()
    }

    private suspend fun setImage() {
        progressIndicator?.setProgress(0)
        val streamFn = page.stream ?: return

        try {
            val (source, isAnimated, background) = withIOContext {
                val source = streamFn().use { process(item, Buffer().readFrom(it)) }
                val isAnimated = ImageUtil.isAnimatedAndSupported(source)
                val background = if (!isAnimated && viewer.config.automaticBackground) {
                    ImageUtil.chooseBackground(context, source.peek().inputStream())
                } else {
                    null
                }
                Triple(source, isAnimated, background)
            }
            withUIContext {
                setImage(
                    source,
                    isAnimated,
                    Config(
                        zoomDuration = viewer.config.doubleTapAnimDuration,
                        minimumScaleType = viewer.config.imageScaleType,
                        cropBorders = viewer.config.imageCropBorders,
                        zoomStartPosition = viewer.config.imageZoomType,
                        landscapeZoom = viewer.config.landscapeZoom,
                    ),
                )
                if (!isAnimated) {
                    pageBackground = background
                }
                removeErrorLayout()

                // 🔥 OUR NEW CHORE: Load and inject translation script layer if available! 🔥
                loadTranslationOverlay()
            }
        } catch (e: Throwable) {
            logcat(LogPriority.ERROR, e)
            withUIContext {
                setError(e)
            }
        }
    }

    /**
     * Reads translation.json and dynamically drops a floating toggle pill and text card on screen
     */
    private fun loadTranslationOverlay() {
        // Clear any leftover views from layout recycling to prevent duplicate layering artifacts
        val oldView = findViewWithTag<View>("ai_translation_layer")
        if (oldView != null) removeView(oldView)

        try {
            val downloadProvider: DownloadProvider = Injekt.get()
            val sourceManager: SourceManager = Injekt.get()
            val manga = viewer.activity.viewModel.manga ?: return
            val source = sourceManager.get(manga.source) ?: return

            val chapterDir = downloadProvider.findChapterDir(
                chapterName = page.chapter.chapter.name,
                chapterScanlator = page.chapter.chapter.scanlator,
                chapterUrl = page.chapter.chapter.url,
                mangaTitle = manga.title,
                source = source,
            ) ?: return

            val translationFile = chapterDir.findFile("translation.json") ?: return
            if (!translationFile.exists()) return

            val jsonString = translationFile.openInputStream().bufferedReader().use { it.readText() }
            val rootJson = org.json.JSONObject(jsonString)
            val pagesJson = rootJson.optJSONObject("pages") ?: return

            // Sort files using the exact sequence index structure of our background loop
            val files = chapterDir.listFiles()?.filter {
                it.name?.endsWith(".jpg", true) == true ||
                    it.name?.endsWith(".png", true) == true ||
                    it.name?.endsWith(".webp", true) == true
            }?.sortedBy { it.name } ?: emptyList()

            if (page.number !in files.indices) return
            val fileName = files[page.number].name
            val pageObj = pagesJson.optJSONObject(fileName) ?: return
            val blocksArray = pageObj.optJSONArray("blocks") ?: return
            if (blocksArray.length() == 0) return

            val englishText = blocksArray.getJSONObject(0).optString("englishText", "")
            if (englishText.isBlank()) return

            // 🏗️ Construct Layer Container
            val overlayLayout = FrameLayout(context).apply {
                tag = "ai_translation_layer"
            }

            // 🧊 Construct Floating Toggle Pill Button
            val toggleButton = TextView(context).apply {
                text = "AI Eng"
                setTextColor(Color.WHITE)
                textSize = 12f
                setPadding(24, 12, 24, 12)
                background = GradientDrawable().apply {
                    setColor(Color.parseColor("#99000000")) // 60% semi-transparent black
                    cornerRadius = 30f
                }
            }

            // 📜 Construct Bottom Scrollable Text Display Sheet
            val translationTextView = TextView(context).apply {
                text = englishText
                setTextColor(Color.WHITE)
                textSize = 15f
                setPadding(40, 40, 40, 40)
                setBackgroundColor(Color.parseColor("#E6121214"))
                visibility = View.GONE
                movementMethod = ScrollingMovementMethod()
            }

            // Handle clean UI toggle action state bounds
            toggleButton.setOnClickListener {
                if (translationTextView.visibility == View.VISIBLE) {
                    translationTextView.visibility = View.GONE
                    toggleButton.text = "AI Eng"
                } else {
                    translationTextView.visibility = View.VISIBLE
                    toggleButton.text = "Hide Text"
                }
            }

            // Layout Params for text script box (Covers bottom 35% of page context comfortably)
            val textParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                (resources.displayMetrics.heightPixels * 0.35).toInt(),
                Gravity.BOTTOM,
            )
            overlayLayout.addView(translationTextView, textParams)

            // Layout Params for the floating toggle pill (Top right, safe below system toolbar bounds)
            val buttonParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.END,
            ).apply {
                topMargin = 140
                rightMargin = 40
            }
            overlayLayout.addView(toggleButton, buttonParams)

            addView(
                overlayLayout,
                FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT),
            )
        } catch (e: Exception) {
            logcat(LogPriority.ERROR, e) { "Failed to inject translation overlay" }
        }
    }

    private fun process(page: ReaderPage, imageSource: BufferedSource): BufferedSource {
        if (viewer.config.dualPageRotateToFit) {
            return rotateDualPage(imageSource)
        }
        if (!viewer.config.dualPageSplit) {
            return imageSource
        }
        if (page is InsertPage) {
            return splitInHalf(imageSource)
        }
        val isDoublePage = ImageUtil.isWideImage(imageSource)
        if (!isDoublePage) {
            return imageSource
        }
        onPageSplit(page)
        return splitInHalf(imageSource)
    }

    private fun rotateDualPage(imageSource: BufferedSource): BufferedSource {
        val isDoublePage = ImageUtil.isWideImage(imageSource)
        return if (isDoublePage) {
            val rotation = if (viewer.config.dualPageRotateToFitInvert) -90f else 90f
            ImageUtil.rotateImage(imageSource, rotation)
        } else {
            imageSource
        }
    }

    private fun splitInHalf(imageSource: BufferedSource): BufferedSource {
        var side = when {
            viewer is L2RPagerViewer && page is InsertPage -> ImageUtil.Side.RIGHT
            viewer !is L2RPagerViewer && page is InsertPage -> ImageUtil.Side.LEFT
            viewer is L2RPagerViewer && page !is InsertPage -> ImageUtil.Side.LEFT
            viewer !is L2RPagerViewer && page !is InsertPage -> ImageUtil.Side.RIGHT
            else -> error("We should choose a side!")
        }

        if (viewer.config.dualPageInvert) {
            side = when (side) {
                ImageUtil.Side.RIGHT -> ImageUtil.Side.LEFT
                ImageUtil.Side.LEFT -> ImageUtil.Side.RIGHT
            }
        }
        return ImageUtil.splitInHalf(imageSource, side)
    }

    private fun onPageSplit(page: ReaderPage) {
        val newPage = InsertPage(page)
        viewer.onPageSplit(page, newPage)
    }

    private fun setError(error: Throwable?) {
        progressIndicator?.hide()
        showErrorLayout(error)
    }

    override fun onImageLoaded() {
        super.onImageLoaded()
        progressIndicator?.hide()
    }

    override fun onImageLoadError(error: Throwable?) {
        super.onImageLoadError(error)
        setError(error)
    }

    override fun onScaleChanged(newScale: Float) {
        super.onScaleChanged(newScale)
        viewer.activity.hideMenu()
    }

    private fun showErrorLayout(error: Throwable?): ReaderErrorBinding {
        if (errorLayout == null) {
            errorLayout = ReaderErrorBinding.inflate(LayoutInflater.from(context), this, true)
            errorLayout?.actionRetry?.viewer = viewer
            errorLayout?.actionRetry?.setOnClickListener {
                page.chapter.pageLoader?.retryPage(page)
            }
        }

        val imageUrl = page.imageUrl
        errorLayout?.actionOpenInWebView?.isVisible = imageUrl != null
        if (imageUrl != null) {
            if (imageUrl.startsWith("http", true)) {
                errorLayout?.actionOpenInWebView?.viewer = viewer
                errorLayout?.actionOpenInWebView?.setOnClickListener {
                    val sourceId = viewer.activity.viewModel.manga?.source
                    val intent = WebViewActivity.newIntent(context, imageUrl, sourceId)
                    context.startActivity(intent)
                }
            }
        }

        errorLayout?.errorMessage?.text = with(context) { error?.formattedMessage }
            ?: context.stringResource(MR.strings.decode_image_error)

        errorLayout?.root?.isVisible = true
        return errorLayout!!
    }

    private fun removeErrorLayout() {
        errorLayout?.root?.isVisible = false
        errorLayout = null
    }
}
