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
    const sortOrderSelect = document.getElementById("sort-order-select");

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

    // Theme Toggle Logic (Default: Light Mode)
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIcon = document.getElementById("theme-icon");

    const savedTheme = localStorage.getItem("news-intel-theme");
    if (savedTheme === "dark") {
        document.body.classList.remove("light-theme");
        if (themeIcon) themeIcon.textContent = "light_mode";
    } else {
        document.body.classList.add("light-theme");
        if (themeIcon) themeIcon.textContent = "dark_mode";
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            document.body.classList.toggle("light-theme");
            const isLight = document.body.classList.contains("light-theme");
            localStorage.setItem("news-intel-theme", isLight ? "light" : "dark");
            if (themeIcon) {
                themeIcon.textContent = isLight ? "dark_mode" : "light_mode";
            }
            syncIframeTheme();
        });
    }

    // Floating Stats Popover Logic with Auto-Close
    const statsToggleBtn = document.getElementById("stats-toggle-btn");
    const statsPopover = document.getElementById("stats-popover");
    const statsCloseBtn = document.getElementById("stats-close-btn");
    let autoCloseTimer = null;

    function openStatsPopover() {
        if (!statsPopover) return;
        statsPopover.classList.add("open");
        resetAutoCloseTimer();
    }

    function closeStatsPopover() {
        if (!statsPopover) return;
        statsPopover.classList.remove("open");
        if (autoCloseTimer) {
            clearTimeout(autoCloseTimer);
            autoCloseTimer = null;
        }
    }

    function resetAutoCloseTimer() {
        if (autoCloseTimer) clearTimeout(autoCloseTimer);
        autoCloseTimer = setTimeout(() => {
            closeStatsPopover();
        }, 6000);
    }

    if (statsToggleBtn) {
        statsToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            if (statsPopover.classList.contains("open")) {
                closeStatsPopover();
            } else {
                openStatsPopover();
            }
        });
    }

    if (statsCloseBtn) {
        statsCloseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            closeStatsPopover();
        });
    }

    if (statsPopover) {
        statsPopover.addEventListener("mousemove", resetAutoCloseTimer);
        statsPopover.addEventListener("click", (e) => e.stopPropagation());
    }

    document.addEventListener("click", (e) => {
        if (statsPopover && statsPopover.classList.contains("open")) {
            if (!statsPopover.contains(e.target) && !statsToggleBtn.contains(e.target)) {
                closeStatsPopover();
            }
        }
    });

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
            if (targetTabId === "health-tab") {
                fetchPlatformStatus();
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
    if (sortOrderSelect) {
        sortOrderSelect.addEventListener("change", () => {
            if (queryInput.value.trim()) {
                performSearch();
            }
        });
    }
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            performSearch();
        }
    });

    // Suggested Searches click handling
    const suggestedBtns = document.querySelectorAll("#suggested-searches-list .chip-btn");
    suggestedBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            if (query) {
                queryInput.value = query;
                performSearch();
            }
        });
    });

    // Fetch and render Trending Topics
    async function fetchTrendingTopics() {
        try {
            const response = await fetch("/api/trending");
            if (!response.ok) return;
            const data = await response.json();

            const trendingContainer = document.getElementById("trending-container");
            const trendingList = document.getElementById("trending-topics-list");

            if (data.trending && data.trending.length > 0) {
                trendingList.innerHTML = "";
                data.trending.forEach(item => {
                    const btn = document.createElement("button");
                    btn.className = "chip-btn";
                    btn.innerHTML = `${item.icon} ${item.name} <span style="opacity:0.7; font-size:0.75rem;">(${item.count})</span>`;
                    btn.addEventListener("click", () => {
                        queryInput.value = item.name;
                        performSearch();
                    });
                    trendingList.appendChild(btn);
                });
                trendingContainer.style.display = "flex";
            }
        } catch (error) {
            console.error("Error fetching trending topics:", error);
        }
    }

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

            // Update KPI metric cards if drift_metrics loaded
            if (data.drift_metrics) {
                const metrics = data.drift_metrics;
                if (metrics.embedding_drift) {
                    const drift = metrics.embedding_drift;
                    if (psiVal) psiVal.textContent = parseFloat(drift.population_stability_index || 0).toFixed(4);
                    if (wassersteinVal) wassersteinVal.textContent = parseFloat(drift.wasserstein_distance || 0).toFixed(4);

                    if (pipelineStatusVal) {
                        pipelineStatusVal.textContent = drift.status || "Unknown";
                        pipelineStatusVal.className = "metric-value";
                        if (drift.status === "Significant Drift") {
                            pipelineStatusVal.classList.add("status-drift");
                        } else if (drift.status === "Moderate Drift") {
                            pipelineStatusVal.classList.add("status-moderate");
                        } else if (drift.status === "Insufficient Baseline Data") {
                            pipelineStatusVal.classList.add("status-insufficient");
                        } else {
                            pipelineStatusVal.classList.add("status-normal");
                        }
                    }
                }

                if (sampleBreakdownVal) {
                    sampleBreakdownVal.textContent = `${metrics.current_count || 0} / ${metrics.reference_count || 0}`;
                }
                if (driftTimestampVal) {
                    driftTimestampVal.textContent = `Drift report last updated: ${metrics.timestamp ? new Date(metrics.timestamp).toLocaleString() : "Unknown"}`;
                }

                // Render native text drift dashboard
                renderNativeDriftDashboard(metrics);
            }
        } catch (error) {
            console.error("Error loading status:", error);
            showToast("Failed to connect to database backend", "error");
        }
    }

    // Render Native Evidently AI Text Drift Diagnostics Dashboard
    function renderNativeDriftDashboard(driftMetrics) {
        const tableBody = document.getElementById("drift-table-body");
        const distGrid = document.getElementById("distribution-cards-grid");

        if (!driftMetrics || !driftMetrics.text_drift_details || driftMetrics.text_drift_details.length === 0) {
            if (tableBody) tableBody.innerHTML = '<tr><td colspan="5" class="table-loading-text">No text drift data available</td></tr>';
            if (distGrid) distGrid.innerHTML = '<div class="table-loading-text">No distribution data available</div>';
            return;
        }

        const details = driftMetrics.text_drift_details;

        // 1. Render Table Rows
        if (tableBody) {
            tableBody.innerHTML = "";
            details.forEach(item => {
                const tr = document.createElement("tr");
                const badgeClass = item.drift_detected ? "badge-drifted" : "badge-stable";
                const statusText = item.drift_detected ? "Drift Detected" : "Stable";

                tr.innerHTML = `
                    <td style="font-weight: 600;">${item.name}</td>
                    <td>${item.ref_mean} <span style="font-size:0.75rem; color:var(--text-muted);">${item.unit}</span></td>
                    <td>${item.cur_mean} <span style="font-size:0.75rem; color:var(--text-muted);">${item.unit}</span></td>
                    <td><strong>${item.drift_score}</strong></td>
                    <td><span class="drift-badge ${badgeClass}">${statusText}</span></td>
                `;
                tableBody.appendChild(tr);
            });
        }

        // 2. Render Distribution Bins Cards
        if (distGrid) {
            distGrid.innerHTML = "";
            details.forEach(item => {
                if (!item.histogram || item.histogram.length === 0) return;

                const card = document.createElement("div");
                card.className = "dist-card";

                let barsHtml = "";
                item.histogram.forEach(b => {
                    barsHtml += `
                        <div class="dist-bar-item">
                            <div class="dist-bar-label">
                                <span>${b.bin}</span>
                                <span>Ref ${b.ref_pct}% / Val ${b.cur_pct}%</span>
                            </div>
                            <div class="dist-bar-tracks">
                                <div class="dist-track"><div class="dist-fill-ref" style="width: ${Math.min(b.ref_pct, 100)}%;"></div></div>
                                <div class="dist-track"><div class="dist-fill-cur" style="width: ${Math.min(b.cur_pct, 100)}%;"></div></div>
                            </div>
                        </div>
                    `;
                });

                card.innerHTML = `
                    <div class="dist-card-header">
                        <span class="dist-card-title">${item.name} (${item.unit})</span>
                        <div class="dist-legend">
                            <span class="legend-item"><span class="legend-dot dot-ref"></span> Ref</span>
                            <span class="legend-item"><span class="legend-dot dot-cur"></span> Val</span>
                        </div>
                    </div>
                    <div class="dist-bar-list">
                        ${barsHtml}
                    </div>
                `;
                distGrid.appendChild(card);
            });
        }
    }

    // Diagnostics Refresh Button
    const reportRefreshBtn = document.getElementById("report-refresh-btn");
    if (reportRefreshBtn) {
        reportRefreshBtn.addEventListener("click", () => {
            showToast("Refreshing data drift status...", "info");
            fetchPlatformStatus();
        });
    }


    // 6. Perform Vector Search Query
    async function performSearch() {
        const query = queryInput.value.trim();
        if (!query) {
            showToast("Please enter a search query", "error");
            return;
        }

        // Show Loader
        searchResults.innerHTML = `
            <div class="loading-container">
                <div class="loader loader-lg"></div>
                <span class="loading-title">Performing Semantic Search...</span>
                <span class="loading-subtitle">Computing vector embeddings & scanning article similarities across the news corpus</span>
            </div>
        `;
        resultsStatus.innerHTML = "";

        const topK = parseInt(topKSlider.value);
        const threshold = parseFloat(thresholdSlider.value);
        const sortBy = sortOrderSelect ? sortOrderSelect.value : "relevance";

        try {
            const response = await fetch("/api/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, top_k: topK, threshold, sort_by: sortBy })
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

            // Sort results if "newest" is selected
            const sortMode = sortOrderSelect ? sortOrderSelect.value : "relevance";
            if (sortMode === "newest") {
                results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            }

            // Render Results
            resultsStatus.innerHTML = `<p class="match-count">Found ${results.length} contextually relevant articles (${sortMode === "newest" ? "Sorted by Newest First" : "Sorted by Relevance"}):</p>`;
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
                        <summary>Show NER Tags</summary>
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
        syncBtn.innerHTML = '<span class="btn-icon loader loader-sm"></span> Syncing...';
        showToast("Initiating database sync from Hugging Face...", "info");

        try {
            const response = await fetch("/api/sync", { method: "POST" });
            if (!response.ok) throw new Error("Sync failed");
            const data = await response.json();

            showToast(`Sync complete! Database now contains ${data.article_count} articles.`, "success");
            await fetchPlatformStatus();
            await fetchTrendingTopics();
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
    fetchTrendingTopics();
});
