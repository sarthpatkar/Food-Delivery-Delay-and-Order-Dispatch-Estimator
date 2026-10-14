const $ = id => document.getElementById(id);

function themeColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function plot(id, data, layout) {
    const muted = themeColor('--muted');
    const line = themeColor('--line');
    const axis = { gridcolor: line, zerolinecolor: line, linecolor: line };
    const base = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: muted, family: getComputedStyle(document.body).fontFamily, size: 12 },
        margin: { l: 50, r: 16, t: 10, b: 44 },
        showlegend: false,
    };
    const merged = Object.assign(base, layout);
    merged.hoverlabel = { font: { family: base.font.family } };
    merged.xaxis = Object.assign({}, axis, layout.xaxis);
    merged.yaxis = Object.assign({}, axis, layout.yaxis);
    Plotly.react(id, data, merged, { responsive: true, displayModeBar: false });
}

/* Predict page */

function initPredictPage() {
    const form = $('prediction-form');
    let timer = null;

    form.querySelectorAll('input[type=range]').forEach(slider => {
        slider.addEventListener('input', () => { $(slider.id + '-val').textContent = slider.value; });
    });
    form.addEventListener('input', e => {
        syncDropFields();
        if (e.target === $('start_node') || e.target === document.querySelector('.drop-field select')) syncTripDistance();
        clearTimeout(timer);
        timer = setTimeout(runPrediction, 200);
    });
    form.addEventListener('submit', e => e.preventDefault());
    syncDropFields();
    syncTripDistance();
    runPrediction();
}

function haversineKm(a, b) {
    const rad = d => d * Math.PI / 180;
    const h = Math.sin(rad(b.lat - a.lat) / 2) ** 2
        + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(rad(b.lon - a.lon) / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.sqrt(h));
}

// The model was trained on straight-line restaurant-to-customer distance, so feed it the same thing
function syncTripDistance() {
    const pickup = PLACES[$('start_node').value];
    const drop = PLACES[document.querySelector('.drop-field select').value];
    const slider = $('distance_km');
    slider.value = Math.max(0.5, haversineKm(pickup, drop)).toFixed(1);
    $('distance_km-val').textContent = slider.value;
}

// One drop-off for the main order plus one per stacked order
function syncDropFields() {
    const count = 1 + parseInt($('multiple_deliveries').value, 10);
    document.querySelectorAll('.drop-field').forEach((field, i) => { field.hidden = i >= count; });
}

function runPrediction() {
    const payload = {};
    new FormData($('prediction-form')).forEach((value, key) => {
        payload[key] = isNaN(value) || value === '' ? value : parseFloat(value);
    });
    payload.drops = [...document.querySelectorAll('.drop-field:not([hidden]) select')].map(s => s.value);

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
        .then(res => res.json())
        .then(data => {
            if (data.status !== 'success') return;
            showEstimate(data.prediction);
            showRoutes(data.search_comparison, data.map, data.stops);
        })
        .catch(err => console.error('predict failed', err));
}

function showEstimate(pred) {
    $('pred-minutes').textContent = pred.predicted_minutes;
    $('pred-prob').textContent = pred.delay_probability_pct;

    const tier = pred.delay_risk_tier.toLowerCase();
    const colors = { low: '--good', moderate: '--warn', high: '--bad' };
    $('risk-label').className = 'risk ' + tier;
    $('prob-tile').className = 'tile risk-' + tier;
    $('risk-label').textContent = pred.delay_risk_tier + ' risk';
    $('prob-bar').style.width = pred.delay_probability_pct + '%';
    $('prob-bar').style.background = themeColor(colors[tier]);

    const votes = {
        'm-rf': pred.rf_prediction,
        'm-dt-gini': pred.tree_gini_prediction,
        'm-dt-entropy': pred.tree_entropy_prediction,
        'm-lr': pred.logistic_prediction,
    };
    Object.entries(votes).forEach(([id, late]) => {
        $(id).textContent = late ? 'Late' : 'On time';
        $(id).className = late ? 'late' : 'ok';
    });
}

function stopClass(id, stops) {
    if (id === stops[0]) return 'pickup';
    return stops.includes(id) ? 'drop' : '';
}

function showRoutes(results, map, stops) {
    const [astar, greedy] = results;
    $('route-stops').innerHTML = astar.path
        .map((id, i) => `<li class="${stopClass(id, stops)}">${astar.path_names[i]}</li>`)
        .join('');

    const cheapest = Math.min(...results.map(r => r.path_cost));
    $('search-comparison-body').innerHTML = results.map(r => `
        <tr class="${r === astar ? 'best' : ''}">
            <td>${r.algorithm}</td>
            <td>${r.nodes_expanded}</td>
            <td>${r.total_distance_km} km</td>
            <td>${r.estimated_time_min} min</td>
            <td>${r.path_cost}${r.path_cost > cheapest ? '' : ' <span class="muted">(best)</span>'}</td>
        </tr>`).join('');

    drawRouteMap(map, astar, greedy, stops);
}

const BASEMAP = {
    light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
};
let routeMap = null;
let mapReady = false;
let pendingRoute = null;
const stopMarkers = {};

const line = coords => ({ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: coords } });

function initRouteMap(map) {
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    routeMap = new maplibregl.Map({
        container: 'route-map',
        style: dark ? BASEMAP.dark : BASEMAP.light,
        center: [72.865, 19.08],
        zoom: 11,
        attributionControl: { compact: true },
    });
    routeMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    routeMap.fitBounds([[72.805, 18.99], [72.92, 19.175]], { padding: 30, duration: 0 });

    map.nodes.forEach(n => {
        const el = document.createElement('div');
        el.className = 'stop-marker';
        el.innerHTML = '<span></span>' + n.name;
        stopMarkers[n.id] = el;
        new maplibregl.Marker({ element: el, anchor: 'left', offset: [-6, 0] }).setLngLat([n.lon, n.lat]).addTo(routeMap);
    });

    routeMap.on('load', () => {
        const empty = { type: 'FeatureCollection', features: [] };
        ['roads', 'greedy', 'astar'].forEach(id => routeMap.addSource(id, { type: 'geojson', data: empty }));
        routeMap.addLayer({
            id: 'roads', type: 'line', source: 'roads',
            layout: { 'line-cap': 'round' },
            paint: {
                'line-width': 3,
                'line-opacity': 0.7,
                'line-color': ['match', ['get', 'traffic'],
                    'Low', themeColor('--green'), 'High', themeColor('--saffron'), 'Jam', themeColor('--bad'),
                    themeColor('--muted')],
            },
        });
        routeMap.addLayer({
            id: 'greedy', type: 'line', source: 'greedy',
            paint: { 'line-color': themeColor('--violet'), 'line-width': 3, 'line-dasharray': [1, 1.5] },
        });
        routeMap.addLayer({
            id: 'astar', type: 'line', source: 'astar',
            layout: { 'line-cap': 'round', 'line-join': 'round' },
            paint: { 'line-color': themeColor('--teal'), 'line-width': 6 },
        });
        mapReady = true;
        if (pendingRoute) drawRouteMap(...pendingRoute);
    });
}

function drawRouteMap(map, astar, greedy, stops) {
    if (!routeMap) initRouteMap(map);
    if (!mapReady) {
        pendingRoute = [map, astar, greedy, stops];
        return;
    }
    routeMap.getSource('roads').setData({
        type: 'FeatureCollection',
        features: map.edges.map(e => Object.assign(line(e.coords), { properties: { traffic: e.traffic } })),
    });
    routeMap.getSource('greedy').setData(line(greedy.coords));
    routeMap.getSource('astar').setData(line(astar.coords));

    const onPath = new Set(astar.path);
    // Toggle only our classes; MapLibre keeps its positioning classes on the same element
    Object.entries(stopMarkers).forEach(([id, el]) => {
        const role = stopClass(id, stops);
        el.classList.toggle('on-path', onPath.has(id));
        el.classList.toggle('pickup', role === 'pickup');
        el.classList.toggle('drop', role === 'drop');
    });
}

/* Retrain page */

function initRetrainPage() {
    const form = $('retrain-form');
    form.querySelectorAll('input[name=dataset_choice]').forEach(radio => {
        radio.addEventListener('change', () => { $('custom-file-group').hidden = radio.value !== 'custom'; });
    });

    form.addEventListener('submit', e => {
        e.preventDefault();
        const btn = $('btn-retrain');
        btn.disabled = true;
        btn.textContent = 'Retraining...';
        $('retrain-error').hidden = true;

        fetch('/api/retrain', { method: 'POST', body: new FormData(form) })
            .then(res => res.json())
            .then(data => {
                if (data.status !== 'success') throw new Error(data.message || 'Retraining failed.');
                showRetrainResult(data);
            })
            .catch(err => {
                $('retrain-error').textContent = err.message;
                $('retrain-error').hidden = false;
            })
            .finally(() => {
                btn.disabled = false;
                btn.textContent = 'Retrain models';
            });
    });
}

function showRetrainResult(data) {
    const before = Object.fromEntries(data.previous_metrics.classification.map(m => [m.model_name, m.accuracy]));
    const delta = data.accuracy_delta;

    $('meta-samples').textContent = data.samples_trained;
    $('meta-timestamp').textContent = data.timestamp;
    $('prev-col').hidden = false;
    $('retrain-summary').innerHTML =
        `Added ${data.rows_added} rows in ${data.training_time_seconds}s. Random forest accuracy ` +
        `<span class="${delta >= 0 ? 'up' : 'down'}">${delta >= 0 ? '+' : ''}${delta} points</span>.`;

    $('metrics-table-body').innerHTML = data.updated_metrics.classification.map(m => `
        <tr>
            <td>${m.model_name}</td>
            <td>${m.accuracy}%</td>
            <td class="muted">${before[m.model_name] ?? '--'}%</td>
            <td>${m.precision}%</td>
            <td>${m.recall}%</td>
            <td>${m.f1_score}%</td>
            <td>${m.roc_auc}</td>
        </tr>`).join('');

    const reg = data.updated_metrics.regression;
    $('reg-mae').textContent = reg.mae + ' min';
    $('reg-rmse').textContent = reg.rmse + ' min';
    $('reg-r2').textContent = reg.r2_score;
}

/* How it works page */

function initExplainCharts(metrics, clusters, featureImportances) {
    const accent = themeColor('--accent');
    const ink = themeColor('--ink');
    const teal = themeColor('--teal');
    const violet = themeColor('--violet');
    const rf = metrics.classification[0];

    const cm = rf.confusion_matrix;
    const labels = ['On time', 'Late'];
    plot('confusion-matrix-chart', [{
        z: [[cm.tn, cm.fp], [cm.fn, cm.tp]],
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: [[0, themeColor('--paper')], [1, teal]],
        showscale: false,
        opacity: 0.75,
        hoverinfo: 'skip',
    }], {
        margin: { l: 80, r: 16, t: 10, b: 44 },
        xaxis: { title: 'Predicted', side: 'bottom' },
        yaxis: { title: 'Actual', autorange: 'reversed' },
        annotations: [
            ['On time', 'On time', 'TN', cm.tn], ['Late', 'On time', 'FP', cm.fp],
            ['On time', 'Late', 'FN', cm.fn], ['Late', 'Late', 'TP', cm.tp],
        ].map(([x, y, tag, n]) => ({
            x, y, text: `${tag}<br><b>${n}</b>`, showarrow: false, font: { color: ink, size: 14 },
        })),
    });

    plot('roc-curve-chart', [
        { x: [0, 1], y: [0, 1], mode: 'lines', line: { dash: 'dot', color: themeColor('--muted'), width: 1 }, hoverinfo: 'skip' },
        { x: rf.roc_curve.map(p => p.fpr), y: rf.roc_curve.map(p => p.tpr), mode: 'lines', line: { color: accent, width: 2.5 } },
    ], {
        xaxis: { title: 'False positive rate', range: [0, 1] },
        yaxis: { title: 'True positive rate', range: [0, 1.02] },
    });

    plot('elbow-chart', [{
        x: clusters.elbow_data.map(d => d.k),
        y: clusters.elbow_data.map(d => d.inertia),
        mode: 'lines+markers',
        line: { color: violet, width: 2 },
        marker: { size: clusters.elbow_data.map(d => (d.k === 3 ? 12 : 7)), color: clusters.elbow_data.map(d => (d.k === 3 ? accent : violet)) },
    }], {
        xaxis: { title: 'k', dtick: 1 },
        yaxis: { title: 'WCSS' },
    });

    const palette = [themeColor('--green'), themeColor('--saffron'), violet];
    plot('pca-scatter-chart', clusters.profiles.map((prof, rank) => {
        const pts = clusters.pca_points.filter(p => p.cluster === prof.cluster_id);
        return {
            name: prof.name,
            x: pts.map(p => p.x),
            y: pts.map(p => p.y),
            text: pts.map(p => `${p.time} min, ${p.distance.toFixed(1)} km`),
            mode: 'markers',
            hoverinfo: 'text',
            marker: { size: 5, color: palette[rank % palette.length], opacity: 0.75 },
        };
    }), {
        showlegend: true,
        legend: { orientation: 'h', y: 1.08, x: 0 },
        margin: { l: 50, r: 16, t: 30, b: 44 },
        xaxis: { title: 'PC 1' },
        yaxis: { title: 'PC 2' },
    });

    const top = featureImportances.slice(0, 8).reverse();
    plot('feature-importance-chart', [{
        x: top.map(f => f.importance),
        y: top.map(f => f.feature.replace(/_/g, ' ')),
        type: 'bar',
        orientation: 'h',
        marker: { color: top.map((f, i) => (i === top.length - 1 ? accent : teal)) },
    }], {
        margin: { l: 10, r: 16, t: 10, b: 44 },
        xaxis: { title: 'Importance' },
        yaxis: { automargin: true },
    });
}
