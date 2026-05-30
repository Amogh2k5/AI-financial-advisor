document.addEventListener('DOMContentLoaded', () => {
    const marketTicker = document.getElementById('market-ticker');
    const form = document.getElementById('advisor-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('loading-spinner');
    const activityFeed = document.getElementById('activity-feed');
    const recommendationContent = document.getElementById('recommendation-content');
    const statusBadge = document.getElementById('status-badge');

    // UI Elements for Drawers and Charts
    const activityDrawer = document.getElementById('activity-drawer');
    const toggleActivityBtn = document.getElementById('toggle-activity-btn');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    
    const toggleChartsBtn = document.getElementById('toggle-charts-btn');
    const chartsContainer = document.getElementById('charts-container');

    // Chart instances & Data storage
    let pieChartInstance = null;
    let lineChartInstance = null;
    let latestUserProfile = null;
    let latestFinalAnswer = "";
    
    const exportPdfBtn = document.getElementById('export-pdf-btn');

    // Toggle Drawer
    toggleActivityBtn.addEventListener('click', () => {
        activityDrawer.classList.toggle('open');
    });
    closeDrawerBtn.addEventListener('click', () => {
        activityDrawer.classList.remove('open');
    });

    // Toggle Charts
    toggleChartsBtn.addEventListener('click', () => {
        chartsContainer.classList.toggle('hidden');
        if (chartsContainer.classList.contains('hidden')) {
            toggleChartsBtn.innerHTML = 'Charts ↗';
            toggleChartsBtn.classList.remove('active');
        } else {
            toggleChartsBtn.innerHTML = 'Charts ↓';
            toggleChartsBtn.classList.add('active');
        }
    });

    // Export PDF
    exportPdfBtn.addEventListener('click', () => {
        generatePDF();
    });

    // Fetch live market data
    async function fetchMarketData() {
        try {
            const response = await fetch('/api/market');
            if (response.ok) {
                const res = await response.json();
                const data = res.data;
                const tickerText = Object.entries(data).map(([key, item]) => {
                    if (!item || item.error || typeof item.price !== 'number') {
                        return `<span style="color: var(--text-secondary); margin-right: 20px;">${item?.name || key}: N/A</span>`;
                    }
                    const change = item.percent_change || 0;
                    const symbol = change >= 0 ? '▲' : '▼';
                    const color = change >= 0 ? '#10b981' : '#ef4444';
                    return `<span style="color: ${color}; margin-right: 20px;">${item.name}: ${item.price.toFixed(2)} ${symbol} ${Math.abs(change).toFixed(2)}%</span>`;
                }).join('');
                marketTicker.innerHTML = `<marquee scrollamount="5">${tickerText}</marquee>`;
            } else {
                marketTicker.textContent = 'Market Data Unavailable';
            }
        } catch (err) {
            marketTicker.textContent = 'Market Data Error';
        }
    }

    // Initial fetch and interval
    fetchMarketData();
    setInterval(fetchMarketData, 60000); // Poll every minute

    function setStatus(status, text) {
        statusBadge.className = `badge ${status}`;
        statusBadge.textContent = text;
    }

    function addFeedItem(text, type = 'thinking') {
        const li = document.createElement('li');
        li.className = `feed-item ${type}`;
        li.textContent = text;
        activityFeed.appendChild(li);
        activityFeed.scrollTop = activityFeed.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Get values
        latestUserProfile = {
            name: document.getElementById('name').value,
            age: parseInt(document.getElementById('age').value),
            budget: parseFloat(document.getElementById('monthly_budget').value),
            risk_level: document.getElementById('risk_tolerance').value,
            goals: document.getElementById('goals').value,
            stocks: []
        };
        const payload = latestUserProfile;

        // UI Reset
        submitBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        spinner.classList.remove('hidden');
        activityFeed.innerHTML = '';
        recommendationContent.innerHTML = '';
        setStatus('active', 'Generating Strategy');
        
        // Hide charts on new run
        toggleChartsBtn.classList.add('hidden');
        chartsContainer.classList.add('hidden');
        toggleChartsBtn.innerHTML = 'Charts ↗';
        if (pieChartInstance) pieChartInstance.destroy();
        if (lineChartInstance) lineChartInstance.destroy();
        
        // Optionally auto-open drawer to show progress
        activityDrawer.classList.add('open');

        try {
            const response = await fetch('/api/analyze/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('API request failed');

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');

            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                
                for (let i = 0; i < lines.length - 1; i++) {
                    const line = lines[i].trim();
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            handleStreamEvent(data);
                        } catch (err) {
                            console.error('Error parsing SSE json:', err);
                        }
                    }
                }
                
                buffer = lines[lines.length - 1];
            }

            if (buffer.trim().startsWith('data: ')) {
                try {
                    const data = JSON.parse(buffer.trim().substring(6));
                    handleStreamEvent(data);
                } catch (err) {}
            }

            setStatus('done', 'Completed');

        } catch (error) {
            console.error(error);
            addFeedItem(`Error: ${error.message}`, 'error');
            setStatus('error', 'Failed');
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Generate Strategy";
            spinner.classList.add('hidden');
        }
    });

    function handleStreamEvent(data) {
        if (data.type === 'heartbeat') return;
        
        if (data.type === 'thinking') {
            addFeedItem(data.content, 'thinking');
        } else if (data.type === 'tool_call') {
            addFeedItem(data.content, 'tool');
        } else if (data.type === 'tool_result') {
            addFeedItem('Tool gathered data successfully.', 'result');
        } else if (data.type === 'final_answer') {
            addFeedItem('Writing final strategy...', 'thinking');
        } else if (data.type === 'error') {
            addFeedItem(`Error: ${data.content}`, 'error');
            setStatus('error', 'Error');
        } else if (data.type === 'done') {
            addFeedItem('Generation finished.', 'result');
            if (data.answer) {
                latestFinalAnswer = data.answer;
                recommendationContent.innerHTML = marked.parse(data.answer);
            }
            if (data.tool_results) {
                // IMPORTANT: Show the container BEFORE rendering so Chart.js can calculate dimensions
                chartsContainer.classList.remove('hidden');
                toggleChartsBtn.innerHTML = 'Charts ↘';
                toggleChartsBtn.classList.remove('hidden');
                
                renderCharts(data.tool_results);
                exportPdfBtn.classList.remove('hidden');
            }
            setStatus('done', 'Completed');
            // Suggest the user close the drawer
            setTimeout(() => { activityDrawer.classList.remove('open'); }, 3000);
        }
    }

    function renderCharts(results) {
        let showChartsBtn = false;

        // Pie Chart: Asset Allocation
        if (results.suggest_portfolio && results.suggest_portfolio.instruments) {
            const instruments = results.suggest_portfolio.instruments;
            const labels = instruments.map(i => i.name);
            const dataPts = instruments.map(i => i.weight_pct);
            
            if (pieChartInstance) {
                pieChartInstance.destroy();
            }
            
            pieChartInstance = new Chart(document.getElementById('pieChart'), {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: dataPts,
                        backgroundColor: [
                            '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#6366f1'
                        ],
                        borderWidth: 0,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#f3f4f6' } }
                    }
                }
            });
            showChartsBtn = true;
        }

        // Line Chart: Market Trend (Performance Comparison)
        if (results.get_market_trends && results.get_market_trends.trends) {
            const trends = results.get_market_trends;
            const dates = trends.dates;
            const colors = [
                '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', 
                '#06b6d4', '#f97316', '#6366f1', '#14b8a6', '#ef4444'
            ];

            const datasets = Object.keys(trends.trends).map((key, i) => {
                const labelName = (trends.names && trends.names[key]) ? trends.names[key] : key;
                return {
                    label: labelName,
                    data: trends.trends[key],
                    borderColor: colors[i % colors.length],
                    backgroundColor: colors[i % colors.length],
                    tension: 0.3,
                    fill: false,
                    pointRadius: 2,
                    borderWidth: 2
                }
            });

            if (lineChartInstance) {
                lineChartInstance.destroy();
            }

            lineChartInstance = new Chart(document.getElementById('lineChart'), {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    scales: {
                        x: { 
                            ticks: { color: '#9ca3af', font: { size: 10 } }, 
                            grid: { color: 'rgba(255,255,255,0.05)' } 
                        },
                        y: { 
                            ticks: { 
                                color: '#9ca3af',
                                callback: function(value) { return value + '%'; }
                            }, 
                            grid: { color: 'rgba(255,255,255,0.1)' } 
                        }
                    },
                    plugins: {
                        legend: { 
                            position: 'bottom',
                            labels: { color: '#f3f4f6', boxWidth: 12, padding: 15 } 
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) label += ': ';
                                    if (context.parsed.y !== null) label += context.parsed.y + '%';
                                    return label;
                                }
                            }
                        }
                    }
                }
            });
            showChartsBtn = true;
        }

        if (showChartsBtn) {
            toggleChartsBtn.classList.remove('hidden');
        }
    }

    // PDF GENERATION LOGIC
    async function generatePDF() {
        if (!latestUserProfile || !latestFinalAnswer) return;

        // 1. Create a light-themed container for the PDF
        const element = document.createElement('div');
        element.style.padding = '40px';
        element.style.color = '#1a1a1a';
        element.style.backgroundColor = '#ffffff';
        element.style.fontFamily = "'Inter', sans-serif";
        element.style.lineHeight = '1.6';

        // 2. Header / User Info
        const header = `
            <div style="border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 30px;">
                <h1 style="color: #3b82f6; margin-bottom: 10px;">Personal Wealth Strategy</h1>
                <p style="color: #666; font-size: 14px;">Generated by AI Wealth Advisor for ${latestUserProfile.name}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; font-size: 14px;">
                    <div><strong>Name:</strong> ${latestUserProfile.name}</div>
                    <div><strong>Age:</strong> ${latestUserProfile.age}</div>
                    <div><strong>Monthly Budget:</strong> INR ${latestUserProfile.budget.toLocaleString()}</div>
                    <div><strong>Risk Profile:</strong> ${latestUserProfile.risk_level.toUpperCase()}</div>
                    <div style="grid-column: span 2;"><strong>Financial Goals:</strong> ${latestUserProfile.goals}</div>
                </div>
            </div>
        `;

        // 3. Strategy Content
        const strategyHtml = `
            <div class="pdf-strategy" style="margin-bottom: 40px;">
                ${marked.parse(latestFinalAnswer)}
            </div>
        `;

        // 4. Charts as Images
        let chartsHtml = '<div style="page-break-before: always; padding-top: 20px;"><h2 style="color: #3b82f6; border-bottom: 1px solid #ddd; padding-bottom: 10px;">Data Visualizations</h2>';
        
        if (pieChartInstance) {
            const pieImg = pieChartInstance.toBase64Image();
            chartsHtml += `
                <div style="margin-top: 30px; text-align: center;">
                    <h3 style="margin-bottom: 15px;">Asset Allocation (Portfolio Mix)</h3>
                    <img src="${pieImg}" style="width: 80%; max-width: 500px; height: auto;" />
                </div>
            `;
        }

        if (lineChartInstance) {
            const lineImg = lineChartInstance.toBase64Image();
            chartsHtml += `
                <div style="margin-top: 50px; text-align: center;">
                    <h3 style="margin-bottom: 15px;">30-Day Growth Performance (%)</h3>
                    <img src="${lineImg}" style="width: 100%; height: auto;" />
                </div>
            `;
        }
        chartsHtml += '</div>';

        element.innerHTML = header + strategyHtml + chartsHtml;

        // 5. html2pdf Options
        const opt = {
            margin:       [10, 10, 10, 10],
            filename:     `Wealth_Strategy_${latestUserProfile.name.replace(/\s+/g, '_')}.pdf`,
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2, useCORS: true, letterRendering: true },
            jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };

        // 6. Generate!
        try {
            exportPdfBtn.textContent = 'Generating...';
            await html2pdf().from(element).set(opt).save();
            exportPdfBtn.textContent = 'Export PDF ↓';
        } catch (error) {
            console.error('PDF Export Error:', error);
            alert('Failed to generate PDF. Please try again.');
            exportPdfBtn.textContent = 'Export PDF ↓';
        }
    }
});
