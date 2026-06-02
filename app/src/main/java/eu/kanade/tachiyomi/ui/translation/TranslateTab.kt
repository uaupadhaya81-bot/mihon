package eu.kanade.tachiyomi.ui.translation

import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Pending
import androidx.compose.material.icons.outlined.Translate
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.rememberVectorPainter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import cafe.adriel.voyager.navigator.Navigator
import cafe.adriel.voyager.navigator.tab.TabOptions
import eu.kanade.presentation.util.Tab
import eu.kanade.tachiyomi.data.download.DownloadProvider
import eu.kanade.tachiyomi.ui.reader.MangaOcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uy.kohesive.injekt.Injekt
import uy.kohesive.injekt.api.get

data class DownloadedChapterInfo(
    val mangaTitle: String,
    val chapterTitle: String,
    val isTranslated: Boolean,
)

object TranslateTab : Tab {

    override val options: TabOptions
        @Composable
        get() {
            val title = "Translate"
            val icon = rememberVectorPainter(Icons.Outlined.Translate)
            return remember {
                TabOptions(
                    index = 5u,
                    title = title,
                    icon = icon,
                )
            }
        }

    /**
     * Called when the active translation tab is clicked a second time while already focused
     */
    override suspend fun onReselect(navigator: Navigator) {
        // Optional: Can be used to scroll list to top or trigger an inventory re-scan
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    override fun Content() {
        val context = LocalContext.current
        val prefs = remember { context.getSharedPreferences("OcrPrefs", Context.MODE_PRIVATE) }
        val scope = rememberCoroutineScope()
        
        var showApiKeyDialog by remember { mutableStateOf(false) }
        var apiKeyInput by remember { mutableStateOf(prefs.getString("gemini_key", "") ?: "") }
        var testMessageInput by remember { mutableStateOf("") }
        var testResponse by remember { mutableStateOf("") }
        var isTesting by remember { mutableStateOf(false) }

        var chapters by remember { mutableStateOf<List<DownloadedChapterInfo>>(emptyList()) }
        var isLoading by remember { mutableStateOf(true) }

        // Scan local storage for downloaded chapters and check for translation.json
        LaunchedEffect(Unit) {
            withContext(Dispatchers.IO) {
                val downloadProvider: DownloadProvider = Injekt.get()
                val list = mutableListOf<DownloadedChapterInfo>()
                
                val sourceDirs = downloadProvider.downloadsDir?.listFiles() ?: arrayOf()
                for (sourceDir in sourceDirs) {
                    if (sourceDir.isFile) continue
                    val mangaDirs = sourceDir.listFiles() ?: arrayOf()
                    for (mangaDir in mangaDirs) {
                        if (mangaDir.isFile) continue
                        val chapDirs = mangaDir.listFiles() ?: arrayOf()
                        for (chapDir in chapDirs) {
                            if (chapDir.isFile) continue
                            val hasTranslation = chapDir.findFile("translation.json")?.exists() == true
                            list.add(
                                DownloadedChapterInfo(
                                    mangaTitle = mangaDir.name ?: "Unknown Manga",
                                    chapterTitle = chapDir.name ?: "Unknown Chapter",
                                    isTranslated = hasTranslation
                                )
                            )
                        }
                    }
                }
                chapters = list
                isLoading = false
            }
        }

        Scaffold(
            topBar = { TopAppBar(title = { Text("AI Translations") }) },
            bottomBar = {
                BottomAppBar {
                    Button(
                        onClick = { showApiKeyDialog = true },
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
                    ) {
                        Text("Enter & Test API Key")
                    }
                }
            }
        ) { paddingValues ->
            if (isLoading) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            } else if (chapters.isEmpty()) {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No downloaded chapters found.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize().padding(paddingValues),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(chapters) { chap ->
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(16.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(text = chap.mangaTitle, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                    Text(text = chap.chapterTitle, style = MaterialTheme.typography.bodySmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                }
                                Spacer(modifier = Modifier.width(16.dp))
                                // Sign-only translation status indicator
                                if (chap.isTranslated) {
                                    Icon(Icons.Filled.CheckCircle, contentDescription = "Translated", tint = MaterialTheme.colorScheme.primary)
                                } else {
                                    Icon(Icons.Filled.Pending, contentDescription = "Not Translated", tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f))
                                }
                            }
                        }
                    }
                }
            }
        }

        // The API Key & Gemini Test Dialog
        if (showApiKeyDialog) {
            AlertDialog(
                onDismissRequest = { showApiKeyDialog = false },
                title = { Text("Gemini API Setup & Test") },
                text = {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = apiKeyInput,
                            onValueChange = { apiKeyInput = it },
                            label = { Text("Gemini API Key") },
                            singleLine = true
                        )
                        OutlinedTextField(
                            value = testMessageInput,
                            onValueChange = { testMessageInput = it },
                            label = { Text("Direct Test Message") }
                        )
                        if (isTesting) {
                            CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 8.dp))
                        } else if (testResponse.isNotEmpty()) {
                            Text(
                                text = "Reply: $testResponse",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.padding(top = 8.dp)
                            )
                        }
                    }
                },
                confirmButton = {
                    TextButton(onClick = {
                        prefs.edit().putString("gemini_key", apiKeyInput).apply()
                        showApiKeyDialog = false
                    }) { Text("Save Key") }
                },
                dismissButton = {
                    TextButton(onClick = {
                        if (apiKeyInput.isNotBlank() && testMessageInput.isNotBlank()) {
                            isTesting = true
                            scope.launch {
                                // Calls the static tester without loading heavy ONNX models
                                testResponse = MangaOcrEngine.testGeminiAPI(apiKeyInput, testMessageInput)
                                isTesting = false
                            }
                        }
                    }) { Text("Test API") }
                }
            )
        }
    }
}
