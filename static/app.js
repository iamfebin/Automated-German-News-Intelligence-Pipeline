document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const queryInput = document.getElementById("query-input");
    const searchBtn = document.getElementById("search-submit-btn");
    const searchResults = document.getElementById("search-results-list");
    const resultsStatus = document.getElementById("results-status");
    
    const topKSlider = document.getElementById("top-k-slider");
    const topKVal = document.getElementById("top-k-val");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    
    const syncBtn = document.getElementById("sync-btn");
    const toast = document.getElementById("toast");
    
    // Sidebar Status elements
    const hfRepoVal = document.getElementById("hf-repo-val");
    const totalArticlesVal = document.getElementById("total-articles-val");
    const sourceBreakdownList = document.getElementById("source-breakdown-list");
    
    // Health metrics elements
    const psiVal = document.getElementById("psi-val");
    const wassersteinVal = document.getElementById("wasserstein-val");
    const pipelineStatusVal = document.getElementById("pipeline-status-val");
    const sampleBreakdownVal = document.getElementById("sample-breakdown-val");
    const driftTimestampVal = document.getElementById("drift-timestamp-val");
    const iframeContainer = document.getElementById("report-iframe-container");

    // Initialize state
    let driftReportLoaded = false;

    // 1. Tab Switching Logic
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    navButtons.forEach(button => {
        button.addEventListener("click", () => {
            const targetTabId = button.getAttribute("data-tab");
            
            // Toggle nav buttons active class
            navButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            
            // Toggle tab contents active class
            tabContents.forEach(tab => tab.classList.remove("active"));
            const targetTab = document.getElementById(targetTabId);
            targetTab.classList.add("active");
            
            // Special initialization on entering health tab
            if (targetTabId === "health-tab" && !driftReportLoaded) {
                loadDriftReportIframe();
            }
        });
    });

    // 2. Slider Value Updates
    topKSlider.addEventListener("input", (e) => {
        topKVal.textContent = e.target.value;
    });

    thresholdSlider.addEventListener("input", (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // 3. Search submit logic
    searchBtn.addEventListener("click", performSearch);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            performSearch();
        }
    });

    // 4. Fetch Platform Status
    async function fetchPlatformStatus() {
        try {
            const response = await fetch("/api/status");
            if (!response.ok) throw new Error("Failed to fetch status");
            const data = await response.json();
            
            // Update Sidebar values
            hfRepoVal.textContent = data.hf_repo_id;
            totalArticlesVal.textContent = data.article_count;
            
            // Render source counts
            sourceBreakdownList.innerHTML = "";
            Object.entries(data.source_breakdown).forEach(([source, count]) => {
                const li = document.createElement("li");
                li.innerHTML = `<span>${source}</span><span>${count}</span>`;
                sourceBreakdownList.appendChild(li);
            });

            // Update Health tab metrics if loaded
            if (data.drift_metrics && data.drift_metrics.embedding_drift) {
                const drift = data.drift_metrics.embedding_drift;
                psiVal.textContent = parseFloat(drift.population_stability_index).toFixed(4);
                wassersteinVal.textContent = parseFloat(drift.wasserstein_distance).toFixed(4);
                
                pipelineStatusVal.textContent = drift.status || "Unknown";
                pipelineStatusVal.className = "metric-value";
                if (drift.status === "Significant Drift") {
                    pipelineStatusVal.classList.add("status-drift");
                } else if (drift.status === "Moderate Drift") {
                    pipelineStatusVal.classList.add("status-moderate");
                } else {
                    pipelineStatusVal.classList.add("status-normal");
                }

                sampleBreakdownVal.textContent = `${data.drift_metrics.current_count || 0} / ${data.drift_metrics.reference_count || 0}`;
                driftTimestampVal.textContent = `Drift report last updated: ${data.drift_metrics.timestamp || "Unknown"}`;
            }
        } catch (error) {
            console.error("Error loading status:", error);
            showToast("Failed to connect to database backend", "error");
        }
    }

    // 5. Load Drift Report Iframe
    function loadDriftReportIframe() {
        iframeContainer.innerHTML = '<iframe src="/data/text_drift_report.html"></iframe>';
        driftReportLoaded = true;
    }

    // 6. Perform Vector Search Query
    async function performSearch() {
        const query = queryInput.value.trim();
        if (!query) {
            showToast("Please enter a search query", "error");
            return;
        }

        // Show Loader
        searchResults.innerHTML = '<div class="loader"></div>';
        resultsStatus.innerHTML = "";

        const topK = parseInt(topKSlider.value);
        const threshold = parseFloat(thresholdSlider.value);

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, top_k: topK, threshold })
            });

            if (!response.ok) throw new Error("Search request failed");
            const data = await response.json();
            const results = data.results || [];

            if (results.length === 0) {
                // If there was no match, find out if we received a database empty or threshold block
                searchResults.innerHTML = `
                    <div class="placeholder-state">
                        <span class="placeholder-icon">⚠️</span>
                        <p>No matching articles found. Try lowering the Relevance Threshold or using different search terms.</p>
                    </div>
                `;
                resultsStatus.innerHTML = '<div class="warning-box">No articles matched your similarity threshold. Try adjusting the query settings.</div>';
                return;
            }

            // Render Results
            resultsStatus.innerHTML = `<p class="match-count">Found ${results.length} contextually relevant articles:</p>`;
            searchResults.innerHTML = "";

            results.forEach(item => {
                const card = document.createElement("div");
                card.className = "news-card";

                // Group and format entities
                let tagsHtml = '<p style="font-size:0.85rem; color:var(--text-muted);">No named entities found.</p>';
                if (item.entities && item.entities.length > 0) {
                    const grouped = {};
                    item.entities.forEach(ent => {
                        const word = ent.word.trim();
                        const etype = ent.entity.toUpperCase();
                        
                        if (word.length <= 1) return;
                        
                        const category_map = {
                            "LOC": "Location",
                            "PER": "Person",
                            "ORG": "Organization",
                            "LOCDERIV": "Location Derivative",
                            "PERDERIV": "Person Derivative"
                        };
                        const expanded = category_map[etype] || etype;
                        
                        if (!grouped[expanded]) grouped[expanded] = new Set();
                        grouped[expanded].add(word);
                    });

                    if (Object.keys(grouped).length > 0) {
                        tagsHtml = "";
                        Object.entries(grouped).sort().forEach(([category, words]) => {
                            let bClass = "badge-org";
                            if (category === "Person" || category === "Person Derivative") {
                                bClass = "badge-per";
                            } else if (category === "Location" || category === "Location Derivative") {
                                bClass = "badge-loc";
                            }
                            
                            const badges = Array.from(words).sort().map(w => `<span class="badge ${bClass}">${w}</span>`).join(" ");
                            tagsHtml += `
                                <div class="tag-category-group">
                                    <p class="tag-category-title">${category}</p>
                                    <div class="badge-container">${badges}</div>
                                </div>
                            `;
                        });
                    }
                }

                card.innerHTML = `
                    <div class="news-header">
                        <span class="news-source">${item.source}</span>
                        <span class="news-date">${item.timestamp} | Match Score: ${parseFloat(item.similarity_score).toFixed(4)}</span>
                    </div>
                    <h3 class="news-title">${item.title_de}</h3>
                    <div class="summary-box">
                        <strong>Bilingual English Summary:</strong>
                        <p class="summary-text">${item.summary_en}</p>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <a href="${item.url}" target="_blank" class="news-url-link">Read original article page ↗</a>
                    </div>
                    
                    <!-- Collapsible tags expander -->
                    <details>
                        <summary>Show Tags</summary>
                        <div class="details-content">
                            ${tagsHtml}
                        </div>
                    </details>

                    <!-- Collapsible German body text expander -->
                    <details>
                        <summary>Show Original German Text</summary>
                        <div class="details-content">
                            <p style="white-space: pre-wrap; font-size: 0.95rem; line-height:1.6;">${item.body_de}</p>
                        </div>
                    </details>
                `;
                searchResults.appendChild(card);
            });

        } catch (error) {
            console.error("Search error:", error);
            searchResults.innerHTML = `
                <div class="placeholder-state">
                    <span class="placeholder-icon">❌</span>
                    <p>An error occurred during search. Please check your backend connection.</p>
                </div>
            `;
            showToast("Search failed. Verify network connection.", "error");
        }
    }

    // 7. Manual Database Sync trigger
    syncBtn.addEventListener("click", async () => {
        syncBtn.disabled = true;
        syncBtn.innerHTML = '<span class="btn-icon loader" style="width:14px; height:14px; border-width:2px; margin:0; display:inline-block;"></span> Syncing...';
        showToast("Initiating database sync from Hugging Face...", "info");

        try {
            const response = await fetch("/api/sync", { method: "POST" });
            if (!response.ok) throw new Error("Sync failed");
            const data = await response.json();
            
            showToast(`Sync complete! Database now contains ${data.article_count} articles.`, "success");
            await fetchPlatformStatus();
        } catch (error) {
            console.error("Sync error:", error);
            showToast("Sync failed. Check API log files.", "error");
        } finally {
            syncBtn.disabled = false;
            syncBtn.innerHTML = '<span class="btn-icon">🔄</span> Sync & Refresh DB';
        }
    });

    // 8. Custom Toast Notification Banner
    function showToast(message, type = "success") {
        toast.textContent = message;
        toast.className = `toast show ${type}`;
        
        setTimeout(() => {
            toast.classList.remove("show");
        }, 4000);
    }

    // Initial Loading
    fetchPlatformStatus();
});
