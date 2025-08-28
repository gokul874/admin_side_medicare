/**
 * Main JavaScript file for Provider Network Optimization Application
 * Contains common functionality used across all pages
 */

// Global application object
window.NetworkOptApp = {
    // Configuration
    config: {
        apiEndpoints: {
            uploadDataset: '/upload_dataset',
            optimizeNetwork: '/optimize_network',
            mapData: '/api/map_data',
            chartData: '/api/chart_data',
            downloadUnusedProviders: '/download_unused_providers'
        },
        fileTypes: {
            csv: ['text/csv', 'application/csv', '.csv']
        }
    },

    // Initialize application
    init: function() {
        this.initFeatherIcons();
        this.initTooltips();
        this.initCommonEventListeners();
        console.log('Network Optimization App initialized');
    },

    // Initialize Feather Icons
    initFeatherIcons: function() {
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    },

    // Initialize Bootstrap tooltips
    initTooltips: function() {
        if (typeof bootstrap !== 'undefined') {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    },

    // Initialize common event listeners
    initCommonEventListeners: function() {
        // Auto-dismiss alerts after 5 seconds
        const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(alert => {
            setTimeout(() => {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }, 5000);
        });

        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    },

    // Utility functions
    utils: {
        // Format numbers with commas
        formatNumber: function(num) {
            return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        },

        // Format currency
        formatCurrency: function(amount, currency = '$') {
            return currency + this.formatNumber(Math.round(amount * 100) / 100);
        },

        // Format percentage
        formatPercentage: function(value, decimals = 2) {
            return parseFloat(value).toFixed(decimals) + '%';
        },

        // Show loading state
        showLoading: function(element, message = 'Loading...') {
            if (element) {
                element.innerHTML = `
                    <div class="d-flex justify-content-center align-items-center p-3">
                        <div class="spinner-border spinner-border-sm me-2" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <span>${message}</span>
                    </div>
                `;
            }
        },

        // Show error state
        showError: function(element, message = 'An error occurred') {
            if (element) {
                element.innerHTML = `
                    <div class="alert alert-danger d-flex align-items-center" role="alert">
                        <i data-feather="alert-circle" class="me-2"></i>
                        <div>${message}</div>
                    </div>
                `;
                feather.replace();
            }
        },

        // Show empty state
        showEmpty: function(element, message = 'No data available') {
            if (element) {
                element.innerHTML = `
                    <div class="text-center text-muted p-4">
                        <i data-feather="inbox" size="48" class="mb-3"></i>
                        <p class="mb-0">${message}</p>
                    </div>
                `;
                feather.replace();
            }
        },

        // Create notification toast
        showToast: function(message, type = 'info', duration = 5000) {
            const toastContainer = this.getOrCreateToastContainer();
            const toastId = 'toast-' + Date.now();
            
            const toastHtml = `
                <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                    <div class="d-flex">
                        <div class="toast-body">
                            <i data-feather="${this.getToastIcon(type)}" class="me-2"></i>
                            ${message}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                    </div>
                </div>
            `;
            
            toastContainer.insertAdjacentHTML('beforeend', toastHtml);
            
            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, { delay: duration });
            
            feather.replace();
            toast.show();
            
            // Clean up after toast is hidden
            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        },

        // Get or create toast container
        getOrCreateToastContainer: function() {
            let container = document.querySelector('.toast-container');
            if (!container) {
                container = document.createElement('div');
                container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
                document.body.appendChild(container);
            }
            return container;
        },

        // Get appropriate icon for toast type
        getToastIcon: function(type) {
            const icons = {
                success: 'check-circle',
                danger: 'x-circle',
                warning: 'alert-triangle',
                info: 'info'
            };
            return icons[type] || 'info';
        },

        // Debounce function
        debounce: function(func, wait, immediate) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    timeout = null;
                    if (!immediate) func(...args);
                };
                const callNow = immediate && !timeout;
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
                if (callNow) func(...args);
            };
        },

        // Validate file type
        validateFileType: function(file, allowedTypes) {
            const fileName = file.name.toLowerCase();
            const fileType = file.type.toLowerCase();
            
            return allowedTypes.some(type => 
                fileName.endsWith(type) || fileType === type
            );
        },

        // Handle API errors
        handleApiError: function(error, defaultMessage = 'An unexpected error occurred') {
            console.error('API Error:', error);
            
            let errorMessage = defaultMessage;
            
            if (error.response) {
                // Server responded with error status
                errorMessage = error.response.data?.error || error.response.statusText || defaultMessage;
            } else if (error.request) {
                // Network error
                errorMessage = 'Network error. Please check your connection and try again.';
            } else if (error.message) {
                // Other error
                errorMessage = error.message;
            }
            
            this.showToast(errorMessage, 'danger', 8000);
            return errorMessage;
        }
    },

    // File handling utilities
    fileUtils: {
        // Read CSV file content
        readCSV: function(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = (e) => resolve(e.target.result);
                reader.onerror = (e) => reject(e);
                reader.readAsText(file);
            });
        },

        // Validate CSV structure
        validateCSVStructure: function(csvContent, requiredColumns) {
            try {
                const lines = csvContent.split('\n');
                if (lines.length < 2) {
                    throw new Error('CSV file must contain at least a header and one data row');
                }
                
                const headers = lines[0].split(',').map(h => h.trim());
                const missingColumns = requiredColumns.filter(col => !headers.includes(col));
                
                if (missingColumns.length > 0) {
                    throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
                }
                
                return { valid: true, headers, rowCount: lines.length - 1 };
            } catch (error) {
                return { valid: false, error: error.message };
            }
        },

        // Get file size in human readable format
        getFileSize: function(bytes) {
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            if (bytes === 0) return '0 Bytes';
            const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
            return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
        }
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    NetworkOptApp.init();
});

// Make utilities globally available
window.formatNumber = NetworkOptApp.utils.formatNumber.bind(NetworkOptApp.utils);
window.formatCurrency = NetworkOptApp.utils.formatCurrency.bind(NetworkOptApp.utils);
window.formatPercentage = NetworkOptApp.utils.formatPercentage.bind(NetworkOptApp.utils);
window.showToast = NetworkOptApp.utils.showToast.bind(NetworkOptApp.utils);
