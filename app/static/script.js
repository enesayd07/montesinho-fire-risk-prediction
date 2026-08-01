document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const resultBox = document.getElementById("result-box");
    const initialState = document.getElementById("initial-state");
    const dynamicResult = document.getElementById("dynamic-result");
    const areaVal = document.getElementById("area-val");
    const riskLabel = document.getElementById("risk-label");
    const marginVal = document.getElementById("margin-val"); 
    const analyzeBtn = document.getElementById("analyze-btn");

    const modePredict = document.getElementById("mode-predict");
    const modeReport = document.getElementById("mode-report");
    const areaGroup = document.getElementById("area-group");
    let currentMode = "predict"; // Varsayılan mod

    modeReport.addEventListener("click", () => {
        currentMode = "report";
        document.body.className = "";
        modeReport.style.background = "#dc2626"; // Buton kırmızı olur
        modePredict.style.background = "rgba(255,255,255,0.2)";
        areaGroup.style.display = "block"; // Gizli input görünür
        analyzeBtn.innerHTML = "VERİYİ SİSTEME KAYDET";
    });

    modePredict.addEventListener("click", () => {
        currentMode = "predict";
        modePredict.style.background = "#3b82f6"; // Buton tekrar mavi olur
        modeReport.style.background = "rgba(255,255,255,0.2)";
        areaGroup.style.display = "none"; // Alan inputu gizlenir
        analyzeBtn.innerHTML = "ANALİZİ BAŞLAT";
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault(); 
        
        const originalBtnText = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = "İŞLENİYOR...";
        analyzeBtn.style.opacity = "0.7";
        analyzeBtn.style.pointerEvents = "none";

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const targetUrl = currentMode === "predict" ? "/predict" : "/api/report";
            
            const response = await fetch(targetUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok && result.success) {
                if (initialState) initialState.style.display = "none";
                dynamicResult.style.display = "block";

                                if (currentMode === "predict") {
                    areaVal.innerHTML = `${result.area_ha} <span class="unit">ha</span>`;
                    
                    if (marginVal) {
                        marginVal.style.display = "block";
                        marginVal.innerHTML = `(± %90 Güvenle: ${result.lower_bound} - ${result.upper_bound} ha)`;
                    }

                    riskLabel.textContent = result.risk_level;
                    riskLabel.className = "result-label";
                    riskLabel.classList.add(result.css_class);
                    document.body.className = result.css_class;
                } else {
                    
                    areaVal.innerHTML = `<span style="color: #22c55e; font-size: 0.7em;">BAŞARILI</span>`;
                    if (marginVal) marginVal.style.display = "none"; // Raporlama modunda margin gizlensin
                    riskLabel.textContent = "YENİ VERİ EKLENDİ";
                    riskLabel.className = "result-label risk-low"; 
                }
            } else {
                alert("Sistemden hata döndü: " + (result.error || "Bilinmeyen hata"));
            }
        } catch (error) {
            console.error("İstek başarısız:", error);
            alert("Sunucuya bağlanılamadı. API açık mı?");
        } finally {
            analyzeBtn.innerHTML = originalBtnText;
            analyzeBtn.style.opacity = "1";
            analyzeBtn.style.pointerEvents = "auto";
        }
    });
});
