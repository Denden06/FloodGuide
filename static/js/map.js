// ===============================
// FLOODGUIDE MAP SCRIPT (Leaflet)
// ===============================

// Initialize map centered at Mandaue Bridge
var map = L.map("map").setView([10.3305, 123.9560], 14);

// Load OpenStreetMap tiles
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap contributors"
}).addTo(map);

// Marker storage
let markers = {};

// ===============================
// ICON STYLE BASED ON RISK COLOR
// ===============================
function createRiskIcon(color) {
    return L.divIcon({
        className: "",
        html: `
            <div style="
                background:${color};
                width:18px;
                height:18px;
                border-radius:50%;
                border:3px solid white;
                box-shadow: 0 0 6px black;
            "></div>
        `,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
    });
}

// ===============================
// LOAD DATA FROM FLASK API
// ===============================
function loadFloodMarkers() {

    fetch("/api/map-data")
        .then(response => response.json())
        .then(data => {

            data.locations.forEach(loc => {

                let icon = createRiskIcon(loc.risk_color);

                // If marker already exists, update it
                if (markers[loc.id]) {
                    markers[loc.id].setLatLng([loc.lat, loc.lng]);
                    markers[loc.id].setIcon(icon);
                    markers[loc.id].setPopupContent(loc.popup_html);
                }

                // Otherwise create marker
                else {
                    let marker = L.marker([loc.lat, loc.lng], { icon: icon })
                        .addTo(map)
                        .bindPopup(loc.popup_html);

                    markers[loc.id] = marker;
                }

            });

        })
        .catch(err => console.error("❌ Marker Load Error:", err));
}

// Load once immediately
loadFloodMarkers();

// Auto refresh every 60 seconds
setInterval(loadFloodMarkers, 60000);
// ===============================
// ICON STYLE BASED ON RISK COLOR WITH PULSE
// ===============================
function createRiskIcon(color) {
    const isHigh = color === "red"; // Only High risk pulses
    return L.divIcon({
        className: "",
        html: `
            <div style="
                background:${color};
                width:18px;
                height:18px;
                border-radius:50%;
                border:3px solid white;
                box-shadow: 0 0 6px black;
                ${isHigh ? 'animation: pulse 1.2s infinite;' : ''}
            "></div>
        `,
        iconSize: [18, 18],
        iconAnchor: [9, 9]
    });
}

// Add CSS for pulse animation
const style = document.createElement('style');
style.innerHTML = `
@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.6); opacity: 0.6; }
    100% { transform: scale(1); opacity: 1; }
}
`;
document.head.appendChild(style);