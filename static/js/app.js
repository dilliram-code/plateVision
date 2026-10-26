/**
 * PlateVision Nepal - Frontend Application Engine
 * Handles modern glassmorphic UI interactions, live drag & drop,
 * camera streaming, instant sample testing, Chart.js analytics, and registry management.
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // 1. NAVIGATION TAB SWITCHING
    // =========================================================================
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    navTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const targetTabId = tab.getAttribute("data-tab");

            navTabs.forEach((t) => t.classList.remove("active"));
            tabContents.forEach((c) => c.classList.remove("active"));

            tab.classList.add("active");
            const targetContent = document.getElementById(targetTabId);
            if (targetContent) {
                targetContent.classList.add("active");
            }

            // Trigger specific tab handlers
            if (targetTabId === "registry-tab") {
                loadRegistryData();
            } else if (targetTabId === "analytics-tab") {
                loadAnalyticsCharts();
            }
        });
    });

    // =========================================================================
    // 2. IMAGE UPLOAD & DRAG-AND-DROP SCANNER
    // =========================================================================
    const dropzone = document.getElementById("image-dropzone");
    const fileInput = document.getElementById("file-input");
    const btnBrowse = document.getElementById("btn-browse-file");
    const dropzoneLoader = document.getElementById("dropzone-loader");
    const dropzoneContent = document.querySelector(".dropzone-content");

    const placeholderState = document.getElementById("placeholder-state");
    const detectionDisplay = document.getElementById("detection-display");
    const annotatedImgPreview = document.getElementById("annotated-img-preview");
    const platesList = document.getElementById("plates-list");
    const processingTimeBadge = document.getElementById("processing-time-badge");
    const overlayTagCount = document.getElementById("overlay-tag-count");

    // Click to browse
    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadAndProcessImage(e.target.files[0]);
        }
    });

    // Drag-and-drop events
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadAndProcessImage(e.dataTransfer.files[0]);
        }
    });

    /**
     * Uploads image file to FastAPI backend for ANPR processing
     */
    async function uploadAndProcessImage(file) {
        showDropzoneLoading(true);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("source_name", file.name);

        try {
            const response = await fetch("/api/detect/image", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            if (data.success) {
                renderDetectionResults(data);
                refreshStats();
                showToast("success", `Processed ${data.total_detected} plate(s) in ${data.processing_time_ms}ms`);
            } else {
                showToast("error", data.error || "Plate detection failed.");
            }
        } catch (err) {
            console.error("Upload error:", err);
            showToast("error", "Network connection or server error.");
        } finally {
            showDropzoneLoading(false);
            fileInput.value = "";
        }
    }

    // =========================================================================
    // 3. INSTANT DEMO SAMPLE TESTING
    // =========================================================================
    const sampleChips = document.querySelectorAll(".sample-chip");
    sampleChips.forEach((chip) => {
        chip.addEventListener("click", async () => {
            const sampleName = chip.getAttribute("data-sample");
            showDropzoneLoading(true);

            const formData = new FormData();
            formData.append("sample_name", sampleName);

            try {
                const response = await fetch("/api/detect/sample", {
                    method: "POST",
                    body: formData,
                });

                const data = await response.json();
                if (data.success) {
                    renderDetectionResults(data);
                    refreshStats();
                    showToast("success", `Analyzed demo sample (${sampleName})`);
                } else {
                    showToast("error", data.error || "Demo detection failed.");
                }
            } catch (err) {
                console.error("Sample test error:", err);
                showToast("error", "Error loading sample asset.");
            } finally {
                showDropzoneLoading(false);
            }
        });
    });

    function showDropzoneLoading(isLoading) {
        if (isLoading) {
            dropzoneContent.style.display = "none";
            dropzoneLoader.style.display = "block";
        } else {
            dropzoneContent.style.display = "block";
            dropzoneLoader.style.display = "none";
        }
    }

    /**
     * Renders detection output, bounding box overlays, and breakdown cards
     */
    function renderDetectionResults(data) {
        placeholderState.style.display = "none";
        detectionDisplay.style.display = "block";

        // Set annotated frame
        if (data.annotated_image) {
            annotatedImgPreview.src = data.annotated_image;
        }

        processingTimeBadge.textContent = `${data.processing_time_ms} ms`;
        overlayTagCount.textContent = `${data.total_detected} Plate(s) Recognized`;

        // Render plate cards
        platesList.innerHTML = "";

        if (!data.detections || data.detections.length === 0) {
            platesList.innerHTML = `
                <div class="plate-result-card" style="border-left-color: var(--amber);">
                    <div style="color: var(--amber); font-weight: 600; display: flex; align-items: center; gap: 8px;">
                        <i class="fa-solid fa-triangle-exclamation"></i> No License Plate Confirmed
                    </div>
                    <p style="font-size: 13px; color: var(--text-secondary);">
                        Vehicle was scanned, but license plate digits could not be verified with high confidence. Ensure the plate is clearly illuminated and unoccluded.
                    </p>
                </div>
            `;
            return;
        }

        data.detections.forEach((det, index) => {
            const card = document.createElement("div");
            card.className = "plate-result-card";

            const confPercent = Math.round(det.confidence * 100);
            const cropSrc = det.plate_crop_path || "/static/crops/placeholder.jpg";

            card.innerHTML = `
                <div class="plate-result-header">
                    <div class="plate-main-text-group">
                        <img src="${cropSrc}" alt="Plate Crop" class="plate-crop-thumb" 
                             onclick="openCropModal('${cropSrc}', '${escapeHtml(det.standardized_plate)}', '${escapeHtml(det.devanagari_display)}', '${escapeHtml(det.vehicle_category)}', '${escapeHtml(det.zone_or_province)}', '${confPercent}%')">
                        <div class="plate-numbers">
                            <span class="plate-devanagari-number">${det.devanagari_display || det.plate_number}</span>
                            <span class="plate-standard-number">${det.standardized_plate || det.plate_number}</span>
                        </div>
                    </div>
                    <span class="badge-tag confidence">
                        <i class="fa-solid fa-check"></i> ${confPercent}%
                    </span>
                </div>

                <div class="plate-badges-row">
                    <span class="badge-tag vehicle">
                        <i class="fa-solid ${getVehicleIcon(det.vehicle_type)}"></i> ${det.vehicle_category || det.vehicle_type}
                    </span>
                    <span class="badge-tag script">
                        <i class="fa-solid fa-language"></i> ${det.script}
                    </span>
                    <span class="badge-tag zone">
                        <i class="fa-solid fa-map-pin"></i> ${det.zone_or_province}
                    </span>
                    <span class="badge-tag" style="background: rgba(255,255,255,0.06); color: var(--text-secondary);">
                        ${det.plate_type}
                    </span>
                </div>

                <div class="confidence-bar-wrapper">
                    <span>Confidence: ${confPercent}%</span>
                    <div class="confidence-progress">
                        <div class="confidence-fill" style="width: ${confPercent}%;"></div>
                    </div>
                </div>
            `;
            platesList.appendChild(card);
        });
    }

    function getVehicleIcon(type) {
        if (!type) return "fa-car";
        const t = type.toLowerCase();
        if (t.includes("motorcycle") || t.includes("bike")) return "fa-motorcycle";
        if (t.includes("bus")) return "fa-bus";
        if (t.includes("truck")) return "fa-truck";
        return "fa-car";
    }

    // =========================================================================
    // 4. LIVE WEBCAM & VIDEO STREAMING
    // =========================================================================
    const btnToggleCam = document.getElementById("btn-toggle-cam");
    const btnSnapCam = document.getElementById("btn-snap-cam");
    const videoElem = document.getElementById("webcam-video");
    const canvasElem = document.getElementById("webcam-canvas");
    const camOfflineMsg = document.getElementById("camera-offline-msg");
    const streamAnnotatedResult = document.getElementById("stream-annotated-result");
    const liveStreamRecords = document.getElementById("live-stream-records");

    let isCameraStreaming = false;
    let cameraMediaStream = null;
    let liveDetectionInterval = null;

    btnToggleCam.addEventListener("click", () => {
        if (!isCameraStreaming) {
            startCamera();
        } else {
            stopCamera();
        }
    });

    async function startCamera() {
        try {
            cameraMediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
                audio: false,
            });

            videoElem.srcObject = cameraMediaStream;
            videoElem.style.display = "block";
            camOfflineMsg.style.display = "none";
            isCameraStreaming = true;

            btnToggleCam.innerHTML = `<i class="fa-solid fa-stop"></i> Stop Camera`;
            btnToggleCam.classList.remove("primary");
            btnToggleCam.classList.add("danger");
            btnSnapCam.disabled = false;

            showToast("info", "Camera feed initialized.");

            // Start live periodic frame scanning every 1.5 seconds
            liveDetectionInterval = setInterval(captureAndSendLiveFrame, 1600);
        } catch (err) {
            console.error("Camera access error:", err);
            showToast("error", "Unable to access camera. Please grant browser permissions.");
        }
    }

    function stopCamera() {
        if (cameraMediaStream) {
            cameraMediaStream.getTracks().forEach((track) => track.stop());
            cameraMediaStream = null;
        }
        if (liveDetectionInterval) {
            clearInterval(liveDetectionInterval);
            liveDetectionInterval = null;
        }

        videoElem.style.display = "none";
        camOfflineMsg.style.display = "block";
        streamAnnotatedResult.style.display = "none";
        isCameraStreaming = false;

        btnToggleCam.innerHTML = `<i class="fa-solid fa-play"></i> Start Camera`;
        btnToggleCam.classList.remove("danger");
        btnToggleCam.classList.add("primary");
        btnSnapCam.disabled = true;

        showToast("info", "Camera feed stopped.");
    }

    async function captureAndSendLiveFrame() {
        if (!isCameraStreaming || !videoElem.videoWidth) return;

        const ctx = canvasElem.getContext("2d");
        canvasElem.width = videoElem.videoWidth;
        canvasElem.height = videoElem.videoHeight;
        ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);

        const base64Data = canvasElem.toDataURL("image/jpeg", 0.75);

        try {
            const response = await fetch("/api/detect/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: base64Data, source_name: "live_camera" }),
            });

            const data = await response.json();
            if (data.success && data.total_detected > 0) {
                if (data.annotated_image) {
                    streamAnnotatedResult.src = data.annotated_image;
                    streamAnnotatedResult.style.display = "block";
                }

                data.detections.forEach((det) => {
                    addLiveStreamRecord(det);
                });
                refreshStats();
            }
        } catch (err) {
            console.error("Live frame processing error:", err);
        }
    }

    // Manual snapshot & scan
    btnSnapCam.addEventListener("click", () => {
        captureAndSendLiveFrame();
        showToast("info", "Snapshot captured and processed.");
    });

    function addLiveStreamRecord(det) {
        const existingHint = liveStreamRecords.querySelector(".empty-feed-hint");
        if (existingHint) existingHint.remove();

        const item = document.createElement("div");
        item.className = "plate-result-card";
        item.style.padding = "10px 14px";
        item.style.fontSize = "12px";

        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="devanagari-font" style="font-size: 16px; font-weight: 700; color: #fff;">
                    ${det.devanagari_display || det.plate_number}
                </span>
                <span class="badge-tag confidence" style="font-size: 10px;">
                    ${Math.round(det.confidence * 100)}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; color: var(--text-secondary); margin-top: 4px;">
                <span>${det.standardized_plate}</span>
                <span>${det.vehicle_type}</span>
            </div>
        `;

        liveStreamRecords.prepend(item);
        // Limit list to 15 items
        while (liveStreamRecords.children.length > 15) {
            liveStreamRecords.removeChild(liveStreamRecords.lastChild);
        }
    }

    // =========================================================================
    // 5. REGISTRY LOGS TABLE
    // =========================================================================
    const registryTbody = document.getElementById("registry-tbody");
    const searchInput = document.getElementById("registry-search-input");
    const filterVehicle = document.getElementById("filter-vehicle-type");
    const filterScript = document.getElementById("filter-script-type");
    const btnRefreshRegistry = document.getElementById("btn-refresh-registry");
    const btnClearDb = document.getElementById("btn-clear-db");
    const registryCountLabel = document.getElementById("registry-count-label");

    let searchDebounceTimeout = null;
    searchInput.addEventListener("input", () => {
        clearTimeout(searchDebounceTimeout);
        searchDebounceTimeout = setTimeout(loadRegistryData, 350);
    });

    filterVehicle.addEventListener("change", loadRegistryData);
    filterScript.addEventListener("change", loadRegistryData);
    btnRefreshRegistry.addEventListener("click", loadRegistryData);

    btnClearDb.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all recognized plate records?")) return;
        try {
            const res = await fetch("/api/detections/clear", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                showToast("info", `Cleared ${data.deleted_count} detection record(s).`);
                loadRegistryData();
                refreshStats();
            }
        } catch (e) {
            showToast("error", "Failed to clear registry.");
        }
    });

    async function loadRegistryData() {
        const query = searchInput.value.trim();
        const vType = filterVehicle.value;
        const scriptType = filterScript.value;

        const params = new URLSearchParams({
            limit: "50",
            search: query,
            vehicle_type: vType,
            script: scriptType,
        });

        try {
            const res = await fetch(`/api/history?${params.toString()}`);
            const data = await res.json();
            if (data.success) {
                renderRegistryTable(data.detections);
                registryCountLabel.textContent = `Displaying ${data.detections.length} recorded plate(s)`;
            }
        } catch (err) {
            console.error("Registry fetch error:", err);
        }
    }

    function renderRegistryTable(records) {
        registryTbody.innerHTML = "";
        if (!records || records.length === 0) {
            registryTbody.innerHTML = `
                <tr>
                    <td colspan="10" style="text-align: center; padding: 35px; color: var(--text-muted);">
                        <i class="fa-solid fa-folder-open" style="font-size: 24px; margin-bottom: 8px; display: block;"></i>
                        No matching vehicle plate records found.
                    </td>
                </tr>
            `;
            return;
        }

        records.forEach((r) => {
            const tr = document.createElement("tr");
            const confPercent = Math.round(r.confidence * 100);
            const cropSrc = r.plate_crop_path || "/static/crops/placeholder.jpg";
            const dateStr = r.timestamp ? r.timestamp.replace("T", " ") : "N/A";

            tr.innerHTML = `
                <td>
                    <img src="${cropSrc}" alt="Plate" class="table-crop-img"
                         onclick="openCropModal('${cropSrc}', '${escapeHtml(r.standardized_plate)}', '${escapeHtml(r.devanagari_display)}', '${escapeHtml(r.vehicle_category)}', '${escapeHtml(r.zone_or_province)}', '${confPercent}%')">
                </td>
                <td style="font-weight: 600;">${escapeHtml(r.plate_number)}</td>
                <td style="color: var(--cyan); font-weight: 600;">${escapeHtml(r.standardized_plate)}</td>
                <td class="devanagari-font" style="font-size: 15px; font-weight: 700;">${escapeHtml(r.devanagari_display)}</td>
                <td><span class="badge-tag script">${escapeHtml(r.script || "N/A")}</span></td>
                <td><span class="badge-tag vehicle"><i class="fa-solid ${getVehicleIcon(r.vehicle_type)}"></i> ${escapeHtml(r.vehicle_type || "N/A")}</span></td>
                <td>${escapeHtml(r.zone_or_province || "N/A")}</td>
                <td><span class="badge-tag confidence">${confPercent}%</span></td>
                <td style="font-size: 11px; color: var(--text-secondary);">${dateStr}</td>
                <td>
                    <button class="glass-btn danger" style="padding: 4px 8px; font-size: 11px;" onclick="deleteRecord(${r.id})">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </td>
            `;
            registryTbody.appendChild(tr);
        });
    }

    window.deleteRecord = async (id) => {
        try {
            const res = await fetch(`/api/detections/${id}`, { method: "DELETE" });
            const data = await res.json();
            if (data.success) {
                showToast("info", "Record removed.");
                loadRegistryData();
                refreshStats();
            }
        } catch (err) {
            showToast("error", "Error deleting record.");
        }
    };

    // =========================================================================
    // 6. ANALYTICS & CHART.JS INTEGRATION
    // =========================================================================
    let vehicleChart = null;
    let scriptChart = null;
    let trafficChart = null;

    async function loadAnalyticsCharts() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            if (!data.success) return;

            const stats = data.stats;

            // 1. Vehicle Distribution Chart
            const vCtx = document.getElementById("vehicleDistributionChart").getContext("2d");
            const vLabels = Object.keys(stats.vehicle_distribution);
            const vCounts = Object.values(stats.vehicle_distribution);

            if (vehicleChart) vehicleChart.destroy();
            vehicleChart = new Chart(vCtx, {
                type: "doughnut",
                data: {
                    labels: vLabels.length ? vLabels.map((l) => l.toUpperCase()) : ["No Data"],
                    datasets: [{
                        data: vCounts.length ? vCounts : [1],
                        backgroundColor: ["#00f2fe", "#10b981", "#8b5cf6", "#f59e0b", "#64748b"],
                        borderWidth: 0,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "right", labels: { color: "#94a3b8", font: { family: "Inter" } } },
                    },
                },
            });

            // 2. Script Ratio Chart
            const sCtx = document.getElementById("scriptRatioChart").getContext("2d");
            const sLabels = Object.keys(stats.script_distribution);
            const sCounts = Object.values(stats.script_distribution);

            if (scriptChart) scriptChart.destroy();
            scriptChart = new Chart(sCtx, {
                type: "pie",
                data: {
                    labels: sLabels.length ? sLabels : ["Devanagari (नेपाली)", "Latin (Embossed)"],
                    datasets: [{
                        data: sCounts.length ? sCounts : [50, 50],
                        backgroundColor: ["#ef4444", "#3b82f6", "#10b981"],
                        borderWidth: 0,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "right", labels: { color: "#94a3b8", font: { family: "Inter" } } },
                    },
                },
            });

            // 3. Hourly Traffic Trend Chart
            const tCtx = document.getElementById("trafficFlowChart").getContext("2d");
            const hours = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, "0"));
            const trafficCounts = hours.map((h) => stats.hourly_flow[h] || 0);

            if (trafficChart) trafficChart.destroy();
            trafficChart = new Chart(tCtx, {
                type: "bar",
                data: {
                    labels: hours.map((h) => `${h}:00`),
                    datasets: [{
                        label: "Plates Scanned",
                        data: trafficCounts,
                        backgroundColor: "rgba(0, 242, 254, 0.45)",
                        borderColor: "#00f2fe",
                        borderWidth: 1.5,
                        borderRadius: 6,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: "#64748b" }, grid: { color: "rgba(255,255,255,0.03)" } },
                        y: { ticks: { color: "#64748b", stepSize: 1 }, grid: { color: "rgba(255,255,255,0.05)" } },
                    },
                    plugins: {
                        legend: { display: false },
                    },
                },
            });
        } catch (err) {
            console.error("Analytics chart render error:", err);
        }
    }

    async function refreshStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            if (data.success) {
                const s = data.stats;
                document.getElementById("kpi-total-scans").textContent = s.total_detections;
                document.getElementById("kpi-today-scans").textContent = s.today_detections;
                document.getElementById("kpi-avg-conf").textContent = `${Math.round(s.avg_confidence * 100)}%`;
            }
        } catch (e) {
            console.error("Stats refresh error:", e);
        }
    }

    // =========================================================================
    // 7. CROP INSPECTION LIGHTBOX MODAL
    // =========================================================================
    const cropModal = document.getElementById("crop-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const modalCropImg = document.getElementById("modal-crop-img");
    const modalDetailsGrid = document.getElementById("modal-details-grid");
    const modalPlateTitle = document.getElementById("modal-plate-title");

    window.openCropModal = (imgSrc, stdPlate, devPlate, category, zone, conf) => {
        modalCropImg.src = imgSrc;
        modalPlateTitle.textContent = `${stdPlate} (${devPlate})`;
        modalDetailsGrid.innerHTML = `
            <div class="detail-item">
                <span class="detail-label">Standardized Plate</span>
                <span class="detail-val" style="color: var(--cyan);">${stdPlate}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Devanagari Script</span>
                <span class="detail-val devanagari-font">${devPlate}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Vehicle Category</span>
                <span class="detail-val">${category}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Zone / Province</span>
                <span class="detail-val">${zone}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Confidence Score</span>
                <span class="detail-val" style="color: var(--emerald);">${conf}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Verification State</span>
                <span class="detail-val" style="color: #6ee7b7;"><i class="fa-solid fa-circle-check"></i> OCR Confirmed</span>
            </div>
        `;
        cropModal.style.display = "flex";
    };

    btnCloseModal.addEventListener("click", () => {
        cropModal.style.display = "none";
    });

    cropModal.addEventListener("click", (e) => {
        if (e.target === cropModal) {
            cropModal.style.display = "none";
        }
    });

    // =========================================================================
    // 8. TOAST NOTIFICATIONS UTILITY
    // =========================================================================
    const toastContainer = document.getElementById("toast-container");

    function showToast(type, message) {
        const toast = document.createElement("div");
        toast.className = "toast";

        let icon = "fa-circle-info";
        let color = "var(--cyan)";

        if (type === "success") {
            icon = "fa-circle-check";
            color = "var(--emerald)";
        } else if (type === "error") {
            icon = "fa-triangle-exclamation";
            color = "var(--crimson)";
        }

        toast.innerHTML = `
            <i class="fa-solid ${icon}" style="color: ${color}; font-size: 16px;"></i>
            <span>${escapeHtml(message)}</span>
        `;

        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial load
    refreshStats();
});
