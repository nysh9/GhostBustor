/**
 * GhostBustor Frontend Application
 * Interactive map interface for ghost net prediction zones
 */

// Configuration
const API_URL = 'http://localhost:8000';

// State
let map = null;
let zones = [];
let sightings = [];
let zoneLayers = [];
let sightingMarkers = [];
let selectedZone = null;

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
  initMap();
  await loadInitialData();
  setupEventListeners();
});

// Initialize Leaflet map
function initMap() {
  // Center on California coast (known ghost net hotspot)
  map = L.map('map').setView([36.0, -121.5], 8);
  
  // Add dark-themed tile layer
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);
  
  // Add ocean current visualization (simplified)
  addCurrentVisualization();
}

// Add simplified ocean current arrows
function addCurrentVisualization() {
  // This would integrate with real current data in production
  // For demo, we show simplified current patterns
  
  const currentPaths = [
    [[36.5, -123.0], [36.0, -122.0], [35.5, -121.0]],
    [[37.0, -124.0], [36.5, -123.0], [36.0, -122.0]],
    [[35.0, -121.0], [34.5, -120.0], [34.0, -119.0]],
  ];
  
  currentPaths.forEach(path => {
    const polyline = L.polyline(path, {
      color: '#0ea5e9',
      weight: 2,
      opacity: 0.3,
      dashArray: '5, 10'
    }).addTo(map);
  });
}

// Load initial data
async function loadInitialData() {
  showLoading(true);
  
  try {
    // Load stats
    const statsResponse = await fetch(`${API_URL}/stats`);
    const stats = await statsResponse.json();
    updateStats(stats);
    
    // Load sightings
    const sightingsResponse = await fetch(`${API_URL}/sightings`);
    sightings = await sightingsResponse.json();
    displaySightings(sightings);
    
    // Run initial prediction
    await runPrediction();
    
  } catch (error) {
    console.error('Error loading data:', error);
    // Use mock data if API is unavailable
    useMockData();
  } finally {
    showLoading(false);
  }
}

// Update stats display
function updateStats(stats) {
  document.getElementById('criticalZones').textContent = stats.critical_zones || 0;
  document.getElementById('predictedNets').textContent = stats.predicted_nets_total || 0;
  document.getElementById('totalSightings').textContent = stats.total_sightings || 0;
  document.getElementById('animalsSaved').textContent = stats.animals_affected || 0;
}

// Display historical sightings on map
function displaySightings(sightings) {
  // Clear existing markers
  sightingMarkers.forEach(marker => map.removeLayer(marker));
  sightingMarkers = [];
  
  sightings.forEach(sighting => {
    const { lat, lon } = sighting.location;
    
    // Create custom icon based on verification status
    const iconColor = sighting.verified ? '#22c55e' : '#94a3b8';
    const iconHtml = `
      <div style="
        width: 12px;
        height: 12px;
        background: ${iconColor};
        border: 2px solid white;
        border-radius: 50%;
        box-shadow: 0 0 10px ${iconColor};
      "></div>
    `;
    
    const icon = L.divIcon({
      html: iconHtml,
      className: 'sighting-marker',
      iconSize: [16, 16]
    });
    
    const marker = L.marker([lat, lon], { icon }).addTo(map);
    
    // Popup with sighting details
    const popupContent = `
      <div style="font-family: Inter, sans-serif; min-width: 200px;">
        <div style="font-weight: 600; margin-bottom: 8px;">Ghost Net Sighting</div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">
          Type: ${sighting.net_type}
        </div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">
          Size: ${sighting.estimated_size}
        </div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">
          Date: ${new Date(sighting.sighting_date).toLocaleDateString()}
        </div>
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">
          Reported by: ${sighting.reported_by}
        </div>
        ${sighting.animals_affected > 0 ? `
          <div style="font-size: 12px; color: #ef4444; margin-top: 8px;">
            ⚠️ ${sighting.animals_affected} animals affected
          </div>
        ` : ''}
        ${sighting.verified ? `
          <div style="font-size: 11px; color: #22c55e; margin-top: 8px;">
            ✓ Verified
          </div>
        ` : ''}
      </div>
    `;
    
    marker.bindPopup(popupContent);
    sightingMarkers.push(marker);
  });
}

// Run AI prediction
async function runPrediction() {
  showLoading(true);
  
  try {
    const response = await fetch(`${API_URL}/predict/region?min_lat=32&max_lat=40&min_lon=-125&max_lon=-117&days=7`);
    const data = await response.json();
    
    zones = data.zones || [];
    displayZones(zones);
    updateZoneList(zones);
    
    // Update zone count
    document.getElementById('zoneCount').textContent = `${zones.length} zones`;
    
  } catch (error) {
    console.error('Prediction error:', error);
    // Use mock zones if API fails
    useMockZones();
  } finally {
    showLoading(false);
  }
}

// Display prediction zones on map
function displayZones(zones) {
  // Clear existing zone layers
  zoneLayers.forEach(layer => map.removeLayer(layer));
  zoneLayers = [];
  
  zones.forEach(zone => {
    const { lat, lon } = zone.center;
    const radius = zone.radius_km * 1000; // Convert to meters
    
    // Color based on risk level
    let color, fillColor, fillOpacity;
    switch (zone.risk_level) {
      case 'critical':
        color = '#ef4444';
        fillColor = '#ef4444';
        fillOpacity = 0.3;
        break;
      case 'high':
        color = '#f59e0b';
        fillColor = '#f59e0b';
        fillOpacity = 0.25;
        break;
      case 'medium':
        color = '#0ea5e9';
        fillColor = '#0ea5e9';
        fillOpacity = 0.2;
        break;
      default:
        color = '#22c55e';
        fillColor = '#22c55e';
        fillOpacity = 0.15;
    }
    
    // Create circle for zone
    const circle = L.circle([lat, lon], {
      radius: radius,
      color: color,
      fillColor: fillColor,
      fillOpacity: fillOpacity,
      weight: 2
    }).addTo(map);
    
    // Add pulsing effect for critical zones
    if (zone.risk_level === 'critical') {
      addPulseEffect(lat, lon, radius, color);
    }
    
    // Click handler
    circle.on('click', () => selectZone(zone));
    
    // Popup
    const popupContent = `
      <div style="font-family: Inter, sans-serif;">
        <div style="font-weight: 600; margin-bottom: 8px;">${zone.id}</div>
        <div style="font-size: 12px; margin-bottom: 4px;">
          Risk: <span style="color: ${color}; font-weight: 600; text-transform: uppercase;">${zone.risk_level}</span>
        </div>
        <div style="font-size: 12px; margin-bottom: 4px;">
          Confidence: ${zone.confidence_score}%
        </div>
        <div style="font-size: 12px;">
          Predicted nets: ~${zone.predicted_net_count}
        </div>
      </div>
    `;
    circle.bindPopup(popupContent);
    
    zoneLayers.push(circle);
  });
}

// Add pulsing effect for critical zones
function addPulseEffect(lat, lon, radius, color) {
  // Create animated pulse using CSS
  const pulseIcon = L.divIcon({
    className: 'pulse-marker',
    iconSize: [radius / 500, radius / 500],
    html: `<div style="
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background: ${color};
      opacity: 0.4;
      animation: pulse 2s ease-in-out infinite;
    "></div>`
  });
  
  // Add pulse animation style if not already added
  if (!document.getElementById('pulse-style')) {
    const style = document.createElement('style');
    style.id = 'pulse-style';
    style.textContent = `
      @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.4; }
        50% { transform: scale(1.2); opacity: 0.2; }
      }
    `;
    document.head.appendChild(style);
  }
}

// Update sidebar zone list
function updateZoneList(zones) {
  const container = document.getElementById('zonesList');
  container.innerHTML = '';
  
  zones.forEach(zone => {
    const card = createZoneCard(zone);
    container.appendChild(card);
  });
}

// Create zone card element
function createZoneCard(zone) {
  const card = document.createElement('div');
  card.className = 'zone-card';
  card.dataset.zoneId = zone.id;
  card.dataset.risk = zone.risk_level;
  
  const confidenceClass = zone.confidence_score >= 70 ? 'high' : 
                          zone.confidence_score >= 50 ? 'medium' : 'low';
  
  card.innerHTML = `
    <div class="zone-header">
      <div class="zone-id">${zone.id}</div>
      <div class="zone-risk ${zone.risk_level}">${zone.risk_level}</div>
    </div>
    <div class="zone-confidence">
      <div class="confidence-bar">
        <div class="confidence-fill ${confidenceClass}" style="width: ${zone.confidence_score}%"></div>
      </div>
      <div class="confidence-value">${zone.confidence_score}%</div>
    </div>
    <div class="zone-details">
      ${zone.accumulation_reason}
    </div>
    <div class="zone-meta">
      <div class="zone-meta-item">
        <span>~${zone.predicted_net_count} nets</span>
      </div>
      <div class="zone-meta-item">
        <span>${zone.radius_km}km radius</span>
      </div>
    </div>
  `;
  
  card.addEventListener('click', () => selectZone(zone));
  
  return card;
}

// Select a zone
function selectZone(zone) {
  selectedZone = zone;
  
  // Update card selection
  document.querySelectorAll('.zone-card').forEach(c => c.classList.remove('active'));
  const card = document.querySelector(`.zone-card[data-zone-id="${zone.id}"]`);
  if (card) card.classList.add('active');
  
  // Pan map to zone
  map.panTo([zone.center.lat, zone.center.lon]);
  
  // Show detail panel
  showDetailPanel(zone);
}

// Show detail panel
function showDetailPanel(zone) {
  const panel = document.getElementById('detailPanel');
  
  document.getElementById('detailTitle').textContent = zone.id;
  document.getElementById('detailRisk').innerHTML = `
    <span style="color: ${getRiskColor(zone.risk_level)}; text-transform: uppercase; font-weight: 600;">
      ${zone.risk_level}
    </span>
  `;
  document.getElementById('detailConfidence').textContent = `${zone.confidence_score}%`;
  document.getElementById('detailNets').textContent = `~${zone.predicted_net_count} ghost nets`;
  document.getElementById('detailReason').textContent = zone.accumulation_reason;
  document.getElementById('detailAction').textContent = zone.recommended_action;
  
  panel.classList.add('active');
}

// Hide detail panel
function hideDetailPanel() {
  document.getElementById('detailPanel').classList.remove('active');
  selectedZone = null;
  
  // Remove card selection
  document.querySelectorAll('.zone-card').forEach(c => c.classList.remove('active'));
}

// Get color for risk level
function getRiskColor(risk) {
  switch (risk) {
    case 'critical': return '#ef4444';
    case 'high': return '#f59e0b';
    case 'medium': return '#0ea5e9';
    default: return '#22c55e';
  }
}

// Show/hide loading overlay
function showLoading(show) {
  document.getElementById('loadingOverlay').classList.toggle('active', show);
}

// Filter zones by risk level
function filterZones(riskLevel) {
  const cards = document.querySelectorAll('.zone-card');
  
  cards.forEach(card => {
    if (riskLevel === 'all' || card.dataset.risk === riskLevel) {
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  });
  
  // Also filter map layers
  zoneLayers.forEach((layer, index) => {
    const zone = zones[index];
    if (zone) {
      if (riskLevel === 'all' || zone.risk_level === riskLevel) {
        layer.setStyle({ opacity: 1, fillOpacity: layer.options.fillOpacity });
      } else {
        layer.setStyle({ opacity: 0.1, fillOpacity: 0.05 });
      }
    }
  });
}

// Setup event listeners
function setupEventListeners() {
  // Predict button
  document.getElementById('predictBtn').addEventListener('click', runPrediction);
  
  // Filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      // Update active state
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      
      // Apply filter
      filterZones(e.target.dataset.filter);
    });
  });
  
  // Detail panel close
  document.getElementById('detailClose').addEventListener('click', hideDetailPanel);
  
  // Plan mission button
  document.getElementById('planMissionBtn').addEventListener('click', () => {
    if (selectedZone) {
      alert(`Mission planning for ${selectedZone.id}\n\nCoordinates: ${selectedZone.center.lat.toFixed(4)}, ${selectedZone.center.lon.toFixed(4)}\nPredicted nets: ~${selectedZone.predicted_net_count}\n\nThis would open mission planning interface.`);
    }
  });
  
  // Share button
  document.getElementById('shareBtn').addEventListener('click', () => {
    if (selectedZone) {
      const shareText = `GhostBustor Alert: ${selectedZone.id} - ${selectedZone.risk_level.toUpperCase()} risk zone identified with ${selectedZone.confidence_score}% confidence. ~${selectedZone.predicted_net_count} ghost nets predicted.`;
      navigator.clipboard.writeText(shareText).then(() => {
        alert('Zone details copied to clipboard!');
      });
    }
  });
}

// Mock data fallback
function useMockData() {
  const mockStats = {
    critical_zones: 3,
    predicted_nets_total: 47,
    total_sightings: 8,
    animals_affected: 12
  };
  updateStats(mockStats);
  
  const mockSightings = [
    { location: { lat: 35.2, lon: -120.5 }, net_type: 'gillnet', estimated_size: 'large', sighting_date: '2024-08-15', reported_by: 'NOAA', verified: true, animals_affected: 3 },
    { location: { lat: 36.8, lon: -122.1 }, net_type: 'trawl_net', estimated_size: 'medium', sighting_date: '2024-09-03', reported_by: 'Fishing Vessel', verified: true, animals_affected: 1 },
    { location: { lat: 37.5, lon: -123.2 }, net_type: 'gillnet', estimated_size: 'large', sighting_date: '2024-10-05', reported_by: 'Coast Guard', verified: true, animals_affected: 5 },
  ];
  displaySightings(mockSightings);
  
  useMockZones();
}

// Mock zones fallback
function useMockZones() {
  const mockZones = [
    {
      id: 'zone_001',
      center: { lat: 36.6, lon: -121.9 },
      radius_km: 15,
      confidence_score: 87.5,
      risk_level: 'critical',
      predicted_net_count: 12,
      accumulation_reason: 'Zone identified due to historical sightings + current convergence',
      recommended_action: 'Immediate cleanup mission recommended. Deploy vessels within 48 hours.'
    },
    {
      id: 'zone_002',
      center: { lat: 34.2, lon: -119.8 },
      radius_km: 15,
      confidence_score: 72.3,
      risk_level: 'high',
      predicted_net_count: 8,
      accumulation_reason: 'Zone identified due to active fishing grounds + historical sightings',
      recommended_action: 'Schedule cleanup mission within 1 week. Monitor for changes.'
    },
    {
      id: 'zone_003',
      center: { lat: 37.9, lon: -123.0 },
      radius_km: 15,
      confidence_score: 65.8,
      risk_level: 'high',
      predicted_net_count: 6,
      accumulation_reason: 'Zone identified due to current convergence + active fishing grounds',
      recommended_action: 'Include in next patrol route. Aerial survey recommended.'
    },
    {
      id: 'zone_004',
      center: { lat: 35.8, lon: -121.3 },
      radius_km: 15,
      confidence_score: 58.2,
      risk_level: 'medium',
      predicted_net_count: 4,
      accumulation_reason: 'Zone identified due to historical sightings',
      recommended_action: 'Low priority. Monitor via satellite.'
    },
    {
      id: 'zone_005',
      center: { lat: 36.3, lon: -122.8 },
      radius_km: 15,
      confidence_score: 52.1,
      risk_level: 'medium',
      predicted_net_count: 3,
      accumulation_reason: 'Zone identified due to current convergence',
      recommended_action: 'Low priority. Monitor via satellite.'
    }
  ];
  
  zones = mockZones;
  displayZones(zones);
  updateZoneList(zones);
  document.getElementById('zoneCount').textContent = `${zones.length} zones`;
}
