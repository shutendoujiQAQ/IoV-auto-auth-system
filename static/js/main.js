// Z3 Solver Visualization Interface Interaction Script

// Global variables for storing historical data
let historyData = {
    timestamps: [],
    scenes: [],
    confidences: [],
    safetyValues: [],
    speedValues: []
};

// History chart object
let historyChart;

// Initialize chart
function initChart() {
    const ctx = document.getElementById('history-chart').getContext('2d');
    historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Confidence',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Safety Threshold',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Speed Threshold',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Confidence'
                    },
                    min: 0,
                    max: 1
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Threshold'
                    },
                    min: 0,
                    max: 100,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Format JSON display
function formatJSON(json) {
    // If data is null or undefined, display empty object instead of "No data"
    if (json === null || json === undefined) {
        return '<span class="text-muted">{}</span>';
    }
    
    // If it's an empty object, display empty object format
    if (Object.keys(json).length === 0) {
        return '<span class="text-muted">{}</span>';
    }
    
    let html = '{';
    
    Object.entries(json).forEach(([key, value], index, array) => {
        html += `<br>&nbsp;&nbsp;<span class="json-key">${key}</span>: `;
        
        if (typeof value === 'object' && value !== null) {
            html += formatJSON(value);
        } else if (typeof value === 'number') {
            html += `<span class="json-number">${value}</span>`;
        } else {
            html += `<span class="json-value">${JSON.stringify(value)}</span>`;
        }
        
        if (index < array.length - 1) {
            html += ',';
        }
    });
    
    html += '<br>}';
    return html;
}

// Update input data display
function updateInputData(data) {
    // Update VLM data
    const vlmElement = document.getElementById('vlm-data');
    vlmElement.innerHTML = formatJSON(data.vlm);
    vlmElement.parentElement.parentElement.classList.add('updated');
    setTimeout(() => {
        vlmElement.parentElement.parentElement.classList.remove('updated');
    }, 2000);
    document.getElementById('vlm-time').textContent = new Date().toLocaleTimeString();
    
    // Update DL data
    const dlElement = document.getElementById('dl-data');
    dlElement.innerHTML = formatJSON(data.dl);
    dlElement.parentElement.parentElement.classList.add('updated');
    setTimeout(() => {
        dlElement.parentElement.parentElement.classList.remove('updated');
    }, 2000);
    document.getElementById('dl-time').textContent = new Date().toLocaleTimeString();
    
    // Update BUSDATA data
    const busElement = document.getElementById('bus-data');
    busElement.innerHTML = formatJSON(data.bus);
    busElement.parentElement.parentElement.classList.add('updated');
    setTimeout(() => {
        busElement.parentElement.parentElement.classList.remove('updated');
    }, 2000);
    document.getElementById('bus-time').textContent = new Date().toLocaleTimeString();
}

// Update result data display
function updateResultData(data) {
    if (!data || Object.keys(data).length === 0) {
        return;
    }
    
    // Update scene name
    document.getElementById('scene-name').textContent = data.scene || '-';
    
    // Update confidence
    const confidenceValue = data.confidence || 0;
    const confidencePercent = Math.round(confidenceValue * 100);
    document.getElementById('confidence-value').textContent = `${confidencePercent}%`;
    const confidenceBar = document.getElementById('confidence-bar');
    confidenceBar.style.width = `${confidencePercent}%`;
    confidenceBar.textContent = `${confidencePercent}%`;
    
    // Set color based on confidence
    if (confidencePercent >= 70) {
        confidenceBar.className = 'progress-bar bg-success';
    } else if (confidencePercent >= 40) {
        confidenceBar.className = 'progress-bar bg-warning';
    } else {
        confidenceBar.className = 'progress-bar bg-danger';
    }
    
    // Update threshold requirements
    document.getElementById('safety-min').textContent = data.safety_min || '-';
    document.getElementById('speed-min').textContent = data.speed_min || '-';
    
    // Update required types
    const needTypesElement = document.getElementById('need-types');
    needTypesElement.innerHTML = '';
    if (data.need_types && data.need_types.length > 0) {
        data.need_types.forEach(type => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-info type-badge';
            badge.textContent = type;
            needTypesElement.appendChild(badge);
        });
    } else {
        needTypesElement.innerHTML = '<span class="text-muted">-</span>';
    }
    
    // Update selected verification methods
    const methodsListElement = document.getElementById('methods-list');
    methodsListElement.innerHTML = '';
    if (data.selected_methods && data.selected_methods.length > 0) {
        data.selected_methods.forEach(method => {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center method-item';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = method.name;
            
            const badgesDiv = document.createElement('div');
            
            const typeBadge = document.createElement('span');
            typeBadge.className = 'badge bg-primary method-type';
            typeBadge.textContent = method.typ;
            
            const safeBadge = document.createElement('span');
            safeBadge.className = 'badge bg-success method-safe';
            safeBadge.textContent = `Safety: ${method.safe}`;
            
            const spdBadge = document.createElement('span');
            spdBadge.className = 'badge bg-info method-spd';
            spdBadge.textContent = `Speed: ${method.spd}`;
            
            badgesDiv.appendChild(typeBadge);
            badgesDiv.appendChild(safeBadge);
            badgesDiv.appendChild(spdBadge);
            
            listItem.appendChild(nameSpan);
            listItem.appendChild(badgesDiv);
            
            methodsListElement.appendChild(listItem);
        });
    } else {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item text-muted';
        listItem.textContent = 'No verification methods';
        methodsListElement.appendChild(listItem);
    }
    
    // Update result time
    document.getElementById('result-time').textContent = new Date().toLocaleTimeString();
    
    // Update historical data
    updateHistoryData(data);
}

// Update historical data and refresh chart
function updateHistoryData(data) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    // Limit the number of historical data points
    const maxDataPoints = 20;
    if (historyData.timestamps.length >= maxDataPoints) {
        historyData.timestamps.shift();
        historyData.scenes.shift();
        historyData.confidences.shift();
        historyData.safetyValues.shift();
        historyData.speedValues.shift();
    }
    
    // Add new data point
    historyData.timestamps.push(timeStr);
    historyData.scenes.push(data.scene);
    historyData.confidences.push(data.confidence || 0);
    historyData.safetyValues.push(data.safety_min || 0);
    historyData.speedValues.push(data.speed_min || 0);
    
    // Update chart
    historyChart.data.labels = historyData.timestamps;
    historyChart.data.datasets[0].data = historyData.confidences;
    historyChart.data.datasets[1].data = historyData.safetyValues.map(val => val / 100); // Scale to 0-1 range
    historyChart.data.datasets[2].data = historyData.speedValues.map(val => val / 100); // Scale to 0-1 range
    historyChart.update();
}

// Fetch data and update interface
function fetchDataAndUpdate() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            updateInputData(data);
            updateResultData(data.result);
        })
        .catch(error => {
            console.error('Failed to fetch data:', error);
            document.getElementById('status-badge').textContent = 'Connection Error';
            document.getElementById('status-badge').className = 'ms-auto badge bg-danger';
        });
}

// Initialize after page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart
    initChart();
    
    // First data fetch
    fetchDataAndUpdate();
    
    // Set interval for data fetching
    setInterval(fetchDataAndUpdate, 1000);
});