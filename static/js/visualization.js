/**
 * Visualization JavaScript for Provider Network Optimization Application
 * Handles interactive map display using Leaflet.js
 */

document.addEventListener('DOMContentLoaded', function() {
    const VisualizationApp = {
        map: null,
        layers: {
            servedMembers: null,
            unservedMembers: null,
            usedProviders: null,
            unusedProviders: null
        },
        markers: {
            servedMembers: [],
            unservedMembers: [],
            usedProviders: [],
            unusedProviders: []
        },
        mapData: null,
        
        init: function() {
            this.showLoadingModal();
            this.initMap();
            this.loadMapData();
            this.initEventListeners();
            console.log('Visualization initialized');
        },

        showLoadingModal: function() {
            const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
            modal.show();
        },

        hideLoadingModal: function() {
            const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
            if (modal) {
                modal.hide();
            }
        },

        initMap: function() {
            // Initialize map
            this.map = L.map('map').setView([39.8283, -98.5795], 4); // Center of USA

            // Add tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 18
            }).addTo(this.map);

            // Create layer groups
            this.layers.servedMembers = L.layerGroup().addTo(this.map);
            this.layers.unservedMembers = L.layerGroup().addTo(this.map);
            this.layers.usedProviders = L.layerGroup().addTo(this.map);
            this.layers.unusedProviders = L.layerGroup();

            // Add legend
            this.addMapLegend();

            // Add scale
            L.control.scale({
                position: 'bottomleft',
                imperial: true,
                metric: true
            }).addTo(this.map);
        },

        addMapLegend: function() {
            const legend = L.control({ position: 'topright' });
            
            legend.onAdd = function() {
                const div = L.DomUtil.create('div', 'legend');
                div.innerHTML = `
                    <h6 style="margin-bottom: 10px; color: white; font-weight: bold;">Legend</h6>
                    <div class="legend-item">
                        <span class="legend-color" style="background-color: #28a745;"></span>
                        <span>Served Members</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background-color: #dc3545;"></span>
                        <span>Unserved Members</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background-color: #0d6efd; border-radius: 2px;"></span>
                        <span>Used Providers</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color" style="background-color: #ffc107; border-radius: 2px;"></span>
                        <span>Unused Providers</span>
                    </div>
                `;
                return div;
            };
            
            legend.addTo(this.map);
        },

        initEventListeners: function() {
            // Filter checkboxes
            document.getElementById('showServedMembers').addEventListener('change', (e) => {
                this.toggleLayer('servedMembers', e.target.checked);
            });

            document.getElementById('showUnservedMembers').addEventListener('change', (e) => {
                this.toggleLayer('unservedMembers', e.target.checked);
            });

            document.getElementById('showUsedProviders').addEventListener('change', (e) => {
                this.toggleLayer('usedProviders', e.target.checked);
            });

            document.getElementById('showUnusedProviders').addEventListener('change', (e) => {
                this.toggleLayer('unusedProviders', e.target.checked);
            });

            // Map control buttons
            document.getElementById('resetView').addEventListener('click', () => {
                this.resetMapView();
            });

            document.getElementById('fitBounds').addEventListener('click', () => {
                this.fitAllBounds();
            });
        },

        loadMapData: function() {
            fetch('/api/map_data')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    this.mapData = data;
                    this.createMapMarkers();
                    this.updateStats();
                    this.fitAllBounds();
                    this.hideLoadingModal();
                })
                .catch(error => {
                    console.error('Error loading map data:', error);
                    this.hideLoadingModal();
                    NetworkOptApp.utils.showToast('Failed to load map data. Please try again.', 'danger');
                });
        },

        createMapMarkers: function() {
            if (!this.mapData) return;

            // Clear existing markers
            this.clearAllMarkers();

            // Create served member markers
            this.mapData.served_members.forEach(member => {
                const marker = this.createMemberMarker(member, true);
                this.markers.servedMembers.push(marker);
                this.layers.servedMembers.addLayer(marker);
            });

            // Create unserved member markers
            this.mapData.unserved_members.forEach(member => {
                const marker = this.createMemberMarker(member, false);
                this.markers.unservedMembers.push(marker);
                this.layers.unservedMembers.addLayer(marker);
            });

            // Create provider markers
            this.mapData.providers.forEach(provider => {
                const marker = this.createProviderMarker(provider);
                if (provider.is_used) {
                    this.markers.usedProviders.push(marker);
                    this.layers.usedProviders.addLayer(marker);
                } else {
                    this.markers.unusedProviders.push(marker);
                    this.layers.unusedProviders.addLayer(marker);
                }
            });

            console.log('Map markers created:', {
                servedMembers: this.markers.servedMembers.length,
                unservedMembers: this.markers.unservedMembers.length,
                usedProviders: this.markers.usedProviders.length,
                unusedProviders: this.markers.unusedProviders.length
            });
        },

        createMemberMarker: function(member, isServed) {
            const color = isServed ? '#28a745' : '#dc3545';
            const icon = L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="
                    background-color: ${color};
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    border: 2px solid white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                "></div>`,
                iconSize: [12, 12],
                iconAnchor: [6, 6]
            });

            const marker = L.marker([member.lat, member.lng], { icon });

            // Create popup content
            let popupContent = `
                <div class="info-panel">
                    <h6><i data-feather="user" class="me-2"></i>Member ${member.id}</h6>
                    <p class="mb-2"><strong>Source Type:</strong> ${member.source_type}</p>
                    <p class="mb-2"><strong>Cost:</strong> ${NetworkOptApp.utils.formatCurrency(member.cost)}</p>
                    <p class="mb-2"><strong>Status:</strong> 
                        <span class="badge bg-${isServed ? 'success' : 'danger'}">
                            ${isServed ? 'Served' : 'Unserved'}
                        </span>
                    </p>
            `;

            if (isServed && member.provider_id) {
                popupContent += `
                    <hr>
                    <h6><i data-feather="map-pin" class="me-2"></i>Assigned Provider</h6>
                    <p class="mb-1"><strong>Provider ID:</strong> ${member.provider_id}</p>
                    <p class="mb-1"><strong>Distance:</strong> ${member.distance.toFixed(2)} km</p>
                    <p class="mb-0"><strong>Rating:</strong> ${member.provider_rating}/5</p>
                `;
            }

            popupContent += '</div>';
            marker.bindPopup(popupContent);

            return marker;
        },

        createProviderMarker: function(provider) {
            const color = provider.is_used ? '#0d6efd' : '#ffc107';
            const icon = L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="
                    background-color: ${color};
                    width: 8px;
                    height: 8px;
                    border-radius: 2px;
                    border: 2px solid white;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                "></div>`,
                iconSize: [8, 8],
                iconAnchor: [4, 4]
            });

            const marker = L.marker([provider.lat, provider.lng], { icon });

            // Create popup content
            const popupContent = `
                <div class="info-panel">
                    <h6><i data-feather="map-pin" class="me-2"></i>Provider ${provider.id}</h6>
                    <p class="mb-2"><strong>Name:</strong> ${provider.name}</p>
                    <p class="mb-2"><strong>Type:</strong> ${provider.type}</p>
                    <p class="mb-2"><strong>Rating:</strong> ${provider.rating}/5</p>
                    <p class="mb-2"><strong>Cost:</strong> ${NetworkOptApp.utils.formatCurrency(provider.cost)}</p>
                    <p class="mb-0"><strong>Status:</strong> 
                        <span class="badge bg-${provider.is_used ? 'primary' : 'warning'}">
                            ${provider.is_used ? 'In Use' : 'Not Used'}
                        </span>
                    </p>
                </div>
            `;
            marker.bindPopup(popupContent);

            return marker;
        },

        toggleLayer: function(layerName, show) {
            if (!this.layers[layerName]) return;

            if (show) {
                if (!this.map.hasLayer(this.layers[layerName])) {
                    this.map.addLayer(this.layers[layerName]);
                }
            } else {
                this.map.removeLayer(this.layers[layerName]);
            }
        },

        clearAllMarkers: function() {
            Object.values(this.layers).forEach(layer => {
                if (layer) layer.clearLayers();
            });
            
            Object.keys(this.markers).forEach(key => {
                this.markers[key] = [];
            });
        },

        updateStats: function() {
            if (!this.mapData || !this.mapData.stats) return;

            const stats = this.mapData.stats;
            
            document.getElementById('servedCount').textContent = NetworkOptApp.utils.formatNumber(stats.served_members);
            document.getElementById('unservedCount').textContent = NetworkOptApp.utils.formatNumber(stats.unserved_members);
            
            document.getElementById('usedProvidersCount').textContent = NetworkOptApp.utils.formatNumber(
                this.mapData.providers.filter(p => p.is_used).length
            );
            document.getElementById('unusedProvidersCount').textContent = NetworkOptApp.utils.formatNumber(
                this.mapData.providers.filter(p => !p.is_used).length
            );
        },

        resetMapView: function() {
            this.map.setView([39.8283, -98.5795], 4);
        },

        fitAllBounds: function() {
            if (!this.mapData) return;

            const allPoints = [
                ...this.mapData.served_members.map(m => [m.lat, m.lng]),
                ...this.mapData.unserved_members.map(m => [m.lat, m.lng]),
                ...this.mapData.providers.map(p => [p.lat, p.lng])
            ];

            if (allPoints.length > 0) {
                const bounds = L.latLngBounds(allPoints);
                this.map.fitBounds(bounds, { padding: [20, 20] });
            }
        },

        // Export map as image (requires additional plugin)
        exportMap: function() {
            NetworkOptApp.utils.showToast('Map export functionality would require additional plugins', 'info');
        }
    };

    // Initialize visualization
    VisualizationApp.init();

    // Make visualization functions globally available
    window.VisualizationApp = VisualizationApp;
});

// Update popup content after feather icons are loaded
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    }, 1000);
});
