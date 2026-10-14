// LogiPulse AI - Client-side logic and Plotly charts (No emojis)

function initPredictPage() {
    const distSlider = document.getElementById('distance_km');
    const distVal = document.getElementById('dist-val');
    if (distSlider && distVal) {
        distSlider.addEventListener('input', () => distVal.textContent = distSlider.value);
    }

    const ratingSlider = document.getElementById('delivery_person_ratings');
    const ratingVal = document.getElementById('rating-val');
    if (ratingSlider && ratingVal) {
        ratingSlider.addEventListener('input', () => ratingVal.textContent = ratingSlider.value);
    }

    const ageSlider = document.getElementById('delivery_person_age');
    const ageVal = document.getElementById('age-val');
    if (ageSlider && ageVal) {
        ageSlider.addEventListener('input', () => ageVal.textContent = ageSlider.value);
    }

    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            executePrediction();
        });
        // Initial run
        executePrediction();
    }
}

function executePrediction() {
    const payload = {
        distance_km: parseFloat(document.getElementById('distance_km').value),
        road_traffic_density: document.getElementById('road_traffic_density').value,
        weather_conditions: document.getElementById('weather_conditions').value,
        type_of_vehicle: document.getElementById('type_of_vehicle').value,
        delivery_person_ratings: parseFloat(document.getElementById('delivery_person_ratings').value),
        delivery_person_age: parseFloat(document.getElementById('delivery_person_age').value),
        multiple_deliveries: parseFloat(document.getElementById('multiple_deliveries').value),
        vehicle_condition: parseFloat(document.getElementById('vehicle_condition').value),
        destination_node: document.getElementById('destination_node').value,
    };

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            const pred = data.prediction;
            document.getElementById('pred-minutes').textContent = pred.predicted_minutes;
            document.getElementById('pred-prob').textContent = pred.delay_probability_pct;

            const probBar = document.getElementById('prob-bar');
            probBar.style.width = pred.delay_probability_pct + '%';

            const badge = document.getElementById('risk-badge');
            badge.className = 'badge';
            if (pred.delay_risk_tier === 'High') {
                badge.classList.add('badge-danger');
                probBar.style.backgroundColor = '#f43f5e';
            } else if (pred.delay_risk_tier === 'Moderate') {
                badge.classList.add('badge-warning');
                probBar.style.backgroundColor = '#f59e0b';
            } else {
                badge.classList.add('badge-success');
                probBar.style.backgroundColor = '#10b981';
            }
            badge.textContent = pred.delay_risk_tier + ' Delay Risk';

            const setConsensusCell = (id, val) => {
                const el = document.getElementById(id);
                if (el) {
                    el.innerHTML = val === 1 
                        ? '<span class="badge badge-danger">Delayed</span>' 
                        : '<span class="badge badge-success">On-Time</span>';
                }
            };

            setConsensusCell('m-rf', pred.rf_prediction);
            setConsensusCell('m-dt-gini', pred.tree_gini_prediction);
            setConsensusCell('m-dt-entropy', pred.tree_entropy_prediction);
            setConsensusCell('m-lr', pred.logistic_prediction);

            // Search comparison
            const searchComps = data.search_comparison;
            if (searchComps && searchComps.length > 0) {
                const astar = searchComps[0];
                document.getElementById('route-path-text').textContent = astar.path_names.join(' -> ');

                const tableBody = document.getElementById('search-comparison-body');
                tableBody.innerHTML = '';
                searchComps.forEach(item => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${item.algorithm}</strong></td>
                        <td>${item.nodes_expanded} nodes</td>
                        <td>${item.total_distance_km} km</td>
                        <td><strong>${item.estimated_time_min} min</strong></td>
                    `;
                    tableBody.appendChild(tr);
                });
            }
        }
    })
    .catch(err => console.error('Prediction API error:', err));
}

function initRetrainPage() {
    const radioButtons = document.querySelectorAll('input[name="dataset_choice"]');
    const customGroup = document.getElementById('custom-file-group');

    radioButtons.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === 'custom') {
                customGroup.style.display = 'block';
            } else {
                customGroup.style.display = 'none';
            }
        });
    });

    const form = document.getElementById('retrain-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            const btn = document.getElementById('btn-retrain');
            const loading = document.getElementById('retrain-loading');
            btn.disabled = true;
            loading.style.display = 'block';

            const formData = new FormData(form);

            fetch('/api/retrain', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                loading.style.display = 'none';

                if (data.status === 'success') {
                    document.getElementById('delta-samples').textContent = data.samples_trained;
                    document.getElementById('delta-accuracy').textContent = data.updated_accuracy;

                    const deltaBadge = document.getElementById('delta-badge');
                    deltaBadge.className = 'badge badge-success';
                    deltaBadge.textContent = 'Retrained in ' + data.training_time_seconds + 's';

                    document.getElementById('meta-samples').textContent = data.samples_trained + ' records';
                    document.getElementById('meta-timestamp').textContent = data.timestamp;

                    const tableBody = document.getElementById('metrics-table-body');
                    tableBody.innerHTML = '';
                    data.updated_metrics.classification.forEach(m => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `
                            <td>${m.model_name}</td>
                            <td>${m.accuracy}%</td>
                            <td>${m.precision}%</td>
                            <td>${m.recall}%</td>
                            <td>${m.f1_score}%</td>
                            <td>${m.roc_auc}</td>
                        `;
                        tableBody.appendChild(tr);
                    });

                    document.getElementById('reg-mae').textContent = data.updated_metrics.regression.mae + ' min';
                    document.getElementById('reg-rmse').textContent = data.updated_metrics.regression.rmse + ' min';
                    document.getElementById('reg-r2').textContent = data.updated_metrics.regression.r2_score;
                }
            })
            .catch(err => {
                btn.disabled = false;
                loading.style.display = 'none';
                console.error('Retrain API error:', err);
            });
        });
    }
}

function initExplainCharts(metrics, clusters, featureImportances) {
    const darkThemeLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8', family: '-apple-system, sans-serif' },
        margin: { l: 45, r: 20, t: 30, b: 40 }
    };

    // 1. Confusion Matrix Heatmap
    if (metrics && metrics.classification && metrics.classification[0]) {
        const cm = metrics.classification[0].confusion_matrix;
        const cmData = [
            {
                z: [[cm.tn, cm.fp], [cm.fn, cm.tp]],
                x: ['Pred: On-Time (0)', 'Pred: Delayed (1)'],
                y: ['Actual: On-Time (0)', 'Actual: Delayed (1)'],
                type: 'heatmap',
                colorscale: [[0, '#0f172a'], [0.5, '#3b82f6'], [1, '#38bdf8']],
                showscale: false
            }
        ];
        const cmLayout = Object.assign({}, darkThemeLayout, {
            annotations: [
                { x: 'Pred: On-Time (0)', y: 'Actual: On-Time (0)', text: 'TN: ' + cm.tn, showarrow: false, font: { color: '#ffffff', size: 14 } },
                { x: 'Pred: Delayed (1)', y: 'Actual: On-Time (0)', text: 'FP: ' + cm.fp, showarrow: false, font: { color: '#ffffff', size: 14 } },
                { x: 'Pred: On-Time (0)', y: 'Actual: Delayed (1)', text: 'FN: ' + cm.fn, showarrow: false, font: { color: '#ffffff', size: 14 } },
                { x: 'Pred: Delayed (1)', y: 'Actual: Delayed (1)', text: 'TP: ' + cm.tp, showarrow: false, font: { color: '#ffffff', size: 14 } }
            ]
        });
        Plotly.newPlot('confusion-matrix-chart', cmData, cmLayout, { responsive: true, displayModeBar: false });

        // 2. ROC Curve
        const rocPoints = metrics.classification[0].roc_curve || [];
        const fpr = rocPoints.map(p => p.fpr);
        const tpr = rocPoints.map(p => p.tpr);
        const rocData = [
            { x: fpr, y: tpr, type: 'scatter', mode: 'lines', line: { color: '#38bdf8', width: 2.5 }, name: 'Model ROC' },
            { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines', line: { dash: 'dash', color: '#64748b' }, name: 'Random Chance' }
        ];
        const rocLayout = Object.assign({}, darkThemeLayout, {
            xaxis: { title: 'False Positive Rate', range: [0, 1] },
            yaxis: { title: 'True Positive Rate', range: [0, 1] },
            showlegend: false
        });
        Plotly.newPlot('roc-curve-chart', rocData, rocLayout, { responsive: true, displayModeBar: false });
    }

    // 3. K-Means Elbow Curve
    if (clusters && clusters.elbow_data) {
        const kVals = clusters.elbow_data.map(d => d.k);
        const inertias = clusters.elbow_data.map(d => d.inertia);
        const elbowData = [
            { x: kVals, y: inertias, type: 'scatter', mode: 'lines+markers', marker: { color: '#f59e0b', size: 8 }, line: { color: '#f59e0b', width: 2 } }
        ];
        const elbowLayout = Object.assign({}, darkThemeLayout, {
            xaxis: { title: 'Number of Clusters (k)', dtick: 1 },
            yaxis: { title: 'Within-Cluster Sum of Squares (WCSS)' }
        });
        Plotly.newPlot('elbow-chart', elbowData, elbowLayout, { responsive: true, displayModeBar: false });
    }

    // 4. PCA 2D Scatter
    if (clusters && clusters.pca_points) {
        const pts = clusters.pca_points;
        const colorPalette = ['#10b981', '#38bdf8', '#f43f5e'];
        const pcaData = [
            {
                x: pts.map(p => p.x),
                y: pts.map(p => p.y),
                text: pts.map(p => 'Transit: ' + p.time + ' min, Dist: ' + p.distance + ' km'),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 6,
                    color: pts.map(p => colorPalette[p.cluster % colorPalette.length]),
                    opacity: 0.8
                }
            }
        ];
        const pcaLayout = Object.assign({}, darkThemeLayout, {
            xaxis: { title: 'Principal Component 1' },
            yaxis: { title: 'Principal Component 2' }
        });
        Plotly.newPlot('pca-scatter-chart', pcaData, pcaLayout, { responsive: true, displayModeBar: false });
    }

    // 5. Feature Importance Horizontal Bar
    if (featureImportances && featureImportances.length > 0) {
        const topFeatures = featureImportances.slice(0, 8).reverse();
        const featData = [
            {
                x: topFeatures.map(f => f.importance),
                y: topFeatures.map(f => f.feature),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#6366f1' }
            }
        ];
        const featLayout = Object.assign({}, darkThemeLayout, {
            xaxis: { title: 'Gini Importance Score' },
            yaxis: { automargin: true }
        });
        Plotly.newPlot('feature-importance-chart', featData, featLayout, { responsive: true, displayModeBar: false });
    }
}
