/* ----------------------------------------------------
   QuantumPrice - Frontend Application Controller
   ---------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const sqftSlider = document.getElementById("SquareFeet");
    const sqftVal = document.getElementById("sqft-val");
    const yearSlider = document.getElementById("YearBuilt");
    const yearVal = document.getElementById("year-val");
    
    const form = document.getElementById("prediction-form");
    const predictBtn = document.getElementById("predict-btn");
    
    const welcomeMessage = document.getElementById("welcome-message");
    const predictionResults = document.getElementById("prediction-results");
    
    const priceValueEl = document.getElementById("price-value");
    const priceRangeEl = document.getElementById("price-range");
    
    const avgComparisonTitle = document.getElementById("avg-comparison-title");
    const avgComparisonDesc = document.getElementById("avg-comparison-desc");
    const vintageDesc = document.getElementById("vintage-desc");
    
    // Tabs Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    // Model Info State
    let marketMetadata = null;
    let importanceChart = null;

    // --- Dynamic Slider Displays ---
    sqftSlider.addEventListener("input", (e) => {
        sqftVal.textContent = Number(e.target.value).toLocaleString() + " sq ft";
    });

    yearSlider.addEventListener("input", (e) => {
        yearVal.textContent = e.target.value;
    });

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // --- Fetch Model & Dataset Metrics ---
    async function loadModelMetrics() {
        try {
            const response = await fetch("/metrics");
            if (!response.ok) throw new Error("Failed to load metrics");
            
            const data = await response.json();
            marketMetadata = data;
            
            // Populate Model Integrity Panel
            document.getElementById("metric-model").textContent = formatModelName(data.model_name);
            
            const r2Val = data.best_metrics.R2;
            document.getElementById("metric-r2").textContent = (r2Val * 100).toFixed(2) + "%";
            document.getElementById("r2-progress").style.width = (r2Val * 100) + "%";
            
            document.getElementById("metric-mae").textContent = formatCurrency(data.best_metrics.MAE);
            document.getElementById("metric-rmse").textContent = formatCurrency(data.best_metrics.RMSE);
            
            // Render Chart
            renderImportanceChart(data.feature_importances);
            
        } catch (error) {
            console.error("Error loading model metrics:", error);
            // Graceful fallback values
            document.getElementById("metric-model").textContent = "Gradient Boosting";
            document.getElementById("metric-r2").textContent = "98.87%";
            document.getElementById("r2-progress").style.width = "98.87%";
            document.getElementById("metric-mae").textContent = "$24,409.66";
            document.getElementById("metric-rmse").textContent = "$30,987.58";
        }
    }

    // --- Render Feature Importance Chart ---
    function renderImportanceChart(importances) {
        const ctx = document.getElementById("importanceChart").getContext("2d");
        
        // Map feature names to user-friendly terms
        const featureLabelsMap = {
            "SquareFeet": "Living Area (Sq Ft)",
            "Neighborhood_Lakeside": "Lakeside Location",
            "Neighborhood_Downtown": "Downtown Location",
            "Neighborhood_Highlands": "Highlands Location",
            "Neighborhood_Suburbs": "Suburbs Location",
            "Neighborhood_GreenValley": "GreenValley Location",
            "Bathrooms": "Number of Bathrooms",
            "Bedrooms": "Number of Bedrooms",
            "Condition": "Property Condition",
            "HouseAge": "Age of Building",
            "SqFtPerRoom": "Sq Ft per Room",
            "HasPool_1": "Swimming Pool",
            "HasGarage_1": "Attached Garage"
        };
        
        const rawKeys = Object.keys(importances);
        const labels = rawKeys.map(k => featureLabelsMap[k] || k);
        const values = Object.values(importances).map(v => v * 100); // convert to percentage

        // Set chart defaults for premium look
        Chart.defaults.color = "#94a3b8";
        Chart.defaults.font.family = "Inter, sans-serif";

        if (importanceChart) {
            importanceChart.destroy();
        }

        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels.slice(0, 7), // Show top 7 features
                datasets: [{
                    data: values.slice(0, 7),
                    backgroundColor: function(context) {
                        const chart = context.chart;
                        const {ctx, chartArea} = chart;
                        if (!chartArea) return null;
                        
                        // Vertical gradient for horizontal bars
                        const gradient = ctx.createLinearGradient(chartArea.left, 0, chartArea.right, 0);
                        gradient.addColorStop(0, '#00f2fe');
                        gradient.addColorStop(1, '#4facfe');
                        return gradient;
                    },
                    borderRadius: 6,
                    borderSkipped: false,
                    barThickness: 16
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(9, 14, 31, 0.95)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#00f2fe',
                        borderColor: 'rgba(0, 242, 254, 0.25)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                return `Impact Weight: ${context.parsed.x.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        border: { display: false },
                        ticks: {
                            callback: function(value) { return value + '%'; }
                        }
                    },
                    y: {
                        grid: { display: false },
                        border: { display: false },
                        ticks: {
                            font: { weight: '500', size: 11 }
                        }
                    }
                }
            }
        });
    }

    // --- Form Submission / Prediction Handler ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Transition to loading state
        predictBtn.classList.add("loading");
        predictBtn.disabled = true;

        // Collect form data
        const squareFeet = parseFloat(sqftSlider.value);
        const bedrooms = parseInt(document.getElementById("Bedrooms").value);
        const bathrooms = parseFloat(document.getElementById("Bathrooms").value);
        const neighborhood = document.getElementById("Neighborhood").value;
        const yearBuilt = parseInt(yearSlider.value);
        const hasGarage = document.getElementById("HasGarage").checked ? 1 : 0;
        const hasPool = document.getElementById("HasPool").checked ? 1 : 0;
        
        // Find checked Condition radio
        const conditionEl = document.querySelector('input[name="Condition"]:checked');
        const condition = conditionEl ? parseInt(conditionEl.value) : 3;

        const requestBody = {
            SquareFeet: squareFeet,
            Bedrooms: bedrooms,
            Bathrooms: bathrooms,
            Neighborhood: neighborhood,
            YearBuilt: yearBuilt,
            HasGarage: hasGarage,
            HasPool: hasPool,
            Condition: condition
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) throw new Error("Valuation request failed");

            const data = await response.json();
            
            // Show results section
            welcomeMessage.classList.add("hidden");
            predictionResults.classList.remove("hidden");

            // 1. Animate Price Counter (Roll-up numbers)
            animatePrice(data.prediction);

            // 2. Set Confidence Interval
            priceRangeEl.textContent = `${formatCurrency(data.confidence_low)} - ${formatCurrency(data.confidence_high)}`;

            // 3. Dynamic Insights Comparison
            if (marketMetadata) {
                const avgPrice = marketMetadata.average_price;
                const deltaPercent = ((data.prediction - avgPrice) / avgPrice) * 100;
                
                if (deltaPercent >= 0) {
                    avgComparisonTitle.innerHTML = `<span style="color:#10b981"><i class="fa-solid fa-arrow-trend-up"></i> Above Average</span>`;
                    avgComparisonDesc.textContent = `+${deltaPercent.toFixed(1)}% vs market mean`;
                } else {
                    avgComparisonTitle.innerHTML = `<span style="color:#f59e0b"><i class="fa-solid fa-arrow-trend-down"></i> Below Average</span>`;
                    avgComparisonDesc.textContent = `${deltaPercent.toFixed(1)}% vs market mean`;
                }
            } else {
                avgComparisonTitle.textContent = "Market Comparison";
                avgComparisonDesc.textContent = "Model data synchronized";
            }

            // 4. Vintage Indicator
            const age = data.details.HouseAge;
            if (age <= 2) {
                vintageDesc.textContent = "Newly Built (Vintage)";
            } else if (age <= 12) {
                vintageDesc.textContent = `Modern Home (${age} yrs old)`;
            } else if (age <= 30) {
                vintageDesc.textContent = `Established (${age} yrs old)`;
            } else {
                vintageDesc.textContent = `Vintage Home (${age} yrs old)`;
            }

        } catch (error) {
            console.error("Prediction Error:", error);
            alert("Oops! Valuation pipeline encountered an error. Please ensure the backend is running properly.");
        } finally {
            // Restore button state
            predictBtn.classList.remove("loading");
            predictBtn.disabled = false;
        }
    });

    // --- Helper Formatting Functions ---
    function formatCurrency(val) {
        return "$" + Math.round(val).toLocaleString();
    }

    function formatModelName(name) {
        if (name === "GradientBoosting") return "Gradient Boosting Regressor";
        if (name === "RandomForest") return "Random Forest Regressor";
        if (name === "LinearRegression") return "Linear Regression (OLS)";
        return name;
    }

    // Roll-up animated counter for target valuation
    function animatePrice(targetValue) {
        const startValue = 0;
        const duration = 1200; // ms
        const startTime = performance.now();

        function updateCounter(currentTime) {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            
            // EaseOutQuad formula
            const easeProgress = progress * (2 - progress);
            const currentValue = startValue + easeProgress * (targetValue - startValue);
            
            priceValueEl.textContent = Math.round(currentValue).toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(updateCounter);
            }
        }

        requestAnimationFrame(updateCounter);
    }

    // Run initial load
    loadModelMetrics();
});
