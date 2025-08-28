/**
 * Dashboard JavaScript for Provider Network Optimization Application
 * Handles charts, metrics display, and dashboard interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    const DashboardApp = {
        charts: {},
        chartColors: {
            primary: getComputedStyle(document.documentElement).getPropertyValue('--bs-primary').trim() || '#0d6efd',
            success: getComputedStyle(document.documentElement).getPropertyValue('--bs-success').trim() || '#198754',
            danger: getComputedStyle(document.documentElement).getPropertyValue('--bs-danger').trim() || '#dc3545',
            warning: getComputedStyle(document.documentElement).getPropertyValue('--bs-warning').trim() || '#ffc107',
            info: getComputedStyle(document.documentElement).getPropertyValue('--bs-info').trim() || '#0dcaf0',
            secondary: getComputedStyle(document.documentElement).getPropertyValue('--bs-secondary').trim() || '#6c757d'
        },

        init: function() {
            this.loadChartData();
            this.initEventListeners();
            console.log('Dashboard initialized');
        },

        initEventListeners: function() {
            // Refresh data button
            const refreshBtn = document.getElementById('refreshData');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    this.loadChartData();
                });
            }

            // Chart resize handling
            window.addEventListener('resize', () => {
                Object.values(this.charts).forEach(chart => {
                    if (chart && typeof chart.resize === 'function') {
                        chart.resize();
                    }
                });
            });
        },

        loadChartData: function() {
            NetworkOptApp.utils.showLoading(document.querySelector('.chart-container'), 'Loading chart data...');
            
            fetch('/api/chart_data')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    this.createCharts(data);
                })
                .catch(error => {
                    console.error('Error loading chart data:', error);
                    NetworkOptApp.utils.showError(
                        document.querySelector('.chart-container'), 
                        'Failed to load chart data. Please try again.'
                    );
                });
        },

        createCharts: function(data) {
            if (!data) {
                console.error('No chart data provided');
                return;
            }

            // Create all charts
            this.createAccessChart(data.access_chart);
            this.createCostChart(data.cost_chart);
            this.createProviderUtilizationChart(data.provider_usage);
            this.createSourceTypeChart(data.source_type_analysis);

            // Update summary statistics
            this.updateSummaryStats(data);
        },

        createAccessChart: function(accessData) {
            const ctx = document.getElementById('accessChart');
            if (!ctx || !accessData) return;

            if (this.charts.accessChart) {
                this.charts.accessChart.destroy();
            }

            this.charts.accessChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Served Members', 'Unserved Members'],
                    datasets: [{
                        data: [accessData.served, accessData.unserved],
                        backgroundColor: [this.chartColors.success, this.chartColors.danger],
                        borderWidth: 2,
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-bg').trim()
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true,
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.parsed || 0;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return `${label}: ${NetworkOptApp.utils.formatNumber(value)} (${percentage}%)`;
                                }
                            }
                        }
                    },
                    cutout: '60%'
                }
            });

            // Add center text
            const centerText = document.createElement('div');
            centerText.style.position = 'absolute';
            centerText.style.top = '50%';
            centerText.style.left = '50%';
            centerText.style.transform = 'translate(-50%, -50%)';
            centerText.style.textAlign = 'center';
            centerText.style.pointerEvents = 'none';
            centerText.innerHTML = `
                <div style="font-size: 1.5rem; font-weight: bold; color: ${this.chartColors.success};">
                    ${NetworkOptApp.utils.formatPercentage(accessData.access_percentage, 1)}
                </div>
                <div style="font-size: 0.9rem; color: var(--bs-text-muted);">Access Rate</div>
            `;
            ctx.parentNode.style.position = 'relative';
            ctx.parentNode.appendChild(centerText);
        },

        createCostChart: function(costData) {
            const ctx = document.getElementById('costChart');
            if (!ctx || !costData) return;

            if (this.charts.costChart) {
                this.charts.costChart.destroy();
            }

            this.charts.costChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Original Cost', 'Optimized Cost', 'Savings'],
                    datasets: [{
                        label: 'Amount ($)',
                        data: [costData.original, costData.optimized, costData.savings],
                        backgroundColor: [
                            this.chartColors.secondary,
                            this.chartColors.primary,
                            costData.savings >= 0 ? this.chartColors.success : this.chartColors.danger
                        ],
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${NetworkOptApp.utils.formatCurrency(context.parsed.y)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            }
                        },
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return NetworkOptApp.utils.formatCurrency(value);
                                },
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            },
                            grid: {
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-border-color').trim()
                            }
                        }
                    }
                }
            });
        },

        createProviderUtilizationChart: function(providerData) {
            const ctx = document.getElementById('providerChart');
            if (!ctx || !providerData) return;

            if (this.charts.providerChart) {
                this.charts.providerChart.destroy();
            }

            this.charts.providerChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Used Providers', 'Unused Providers'],
                    datasets: [{
                        data: [providerData.used, providerData.unused],
                        backgroundColor: [this.chartColors.primary, this.chartColors.warning],
                        borderWidth: 2,
                        borderColor: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-bg').trim()
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true,
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.parsed || 0;
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return `${label}: ${NetworkOptApp.utils.formatNumber(value)} (${percentage}%)`;
                                }
                            }
                        }
                    },
                    cutout: '60%'
                }
            });

            // Add utilization percentage in center
            const utilizationRate = ((providerData.used / providerData.total) * 100).toFixed(1);
            const centerText = document.createElement('div');
            centerText.style.position = 'absolute';
            centerText.style.top = '50%';
            centerText.style.left = '50%';
            centerText.style.transform = 'translate(-50%, -50%)';
            centerText.style.textAlign = 'center';
            centerText.style.pointerEvents = 'none';
            centerText.innerHTML = `
                <div style="font-size: 1.5rem; font-weight: bold; color: ${this.chartColors.primary};">
                    ${utilizationRate}%
                </div>
                <div style="font-size: 0.9rem; color: var(--bs-text-muted);">Utilization</div>
            `;
            ctx.parentNode.style.position = 'relative';
            ctx.parentNode.appendChild(centerText);
        },

        createSourceTypeChart: function(sourceTypeData) {
            const ctx = document.getElementById('sourceTypeChart');
            if (!ctx || !sourceTypeData) return;

            if (this.charts.sourceTypeChart) {
                this.charts.sourceTypeChart.destroy();
            }

            const sourceTypes = Object.keys(sourceTypeData);
            const servedData = sourceTypes.map(type => sourceTypeData[type].served_members);
            const totalData = sourceTypes.map(type => sourceTypeData[type].total_members);
            const accessPercentages = sourceTypes.map(type => sourceTypeData[type].access_percentage);

            this.charts.sourceTypeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: sourceTypes,
                    datasets: [
                        {
                            label: 'Served Members',
                            data: servedData,
                            backgroundColor: this.chartColors.success,
                            borderRadius: 4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Total Members',
                            data: totalData,
                            backgroundColor: this.chartColors.secondary,
                            borderRadius: 4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Access %',
                            data: accessPercentages,
                            type: 'line',
                            borderColor: this.chartColors.primary,
                            backgroundColor: this.chartColors.primary,
                            borderWidth: 3,
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                usePointStyle: true,
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    if (context.datasetIndex === 2) {
                                        return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                                    }
                                    return `${context.dataset.label}: ${NetworkOptApp.utils.formatNumber(context.parsed.y)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return NetworkOptApp.utils.formatNumber(value);
                                },
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            },
                            grid: {
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-border-color').trim()
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                },
                                color: getComputedStyle(document.documentElement).getPropertyValue('--bs-body-color').trim()
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            });
        },

        updateSummaryStats: function(data) {
            // Update any dynamic summary statistics if needed
            console.log('Chart data loaded successfully', data);
        },

        // Export chart as image
        exportChart: function(chartName, filename) {
            const chart = this.charts[chartName];
            if (chart) {
                const url = chart.toBase64Image();
                const a = document.createElement('a');
                a.href = url;
                a.download = filename || `${chartName}.png`;
                a.click();
            }
        }
    };

    // Initialize dashboard
    DashboardApp.init();

    // Make dashboard functions globally available
    window.DashboardApp = DashboardApp;
});
