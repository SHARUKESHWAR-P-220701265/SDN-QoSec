/* ═══════════════════════════════════════════════════════════════════════════
   SDQN Dashboard — app.js
   Polls /api/state and renders: D3 topology, Chart.js metrics, tables
   ═══════════════════════════════════════════════════════════════════════════ */

const API = "";
const POLL_MS = 350;

// ── State ─────────────────────────────────────────────────────────────────
let prevTick = -1;
let attackedLinks = new Set();  // "u-v" strings currently under attack

// ── Chart.js Setup ────────────────────────────────────────────────────────
Chart.defaults.color = "#6b7280";
Chart.defaults.borderColor = "rgba(30,37,54,.6)";
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 10;
Chart.defaults.animation = false;

function makeChart(id, label, color, yLabel, suggestedMax) {
    const ctx = document.getElementById(id).getContext("2d");
    return new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label,
                data: [],
                borderColor: color,
                backgroundColor: color + "18",
                borderWidth: 1.5,
                pointRadius: 0,
                fill: true,
                tension: 0.3,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { boxWidth: 8, padding: 6, font: { size: 9 } } },
            },
            scales: {
                x: { display: true, ticks: { maxTicksLimit: 8, font: { size: 8 } }, grid: { display: false } },
                y: {
                    display: true,
                    title: { display: true, text: yLabel, font: { size: 9 } },
                    suggestedMin: 0,
                    suggestedMax,
                    ticks: { font: { size: 8 }, maxTicksLimit: 5 },
                },
            },
        },
    });
}

const chartBlocking = makeChart("chart-blocking", "Service Blocking Rate", "#e04f5f", "Rate", 0.5);
const chartKcurr = makeChart("chart-kcurr", "Avg Key Buffer (bits)", "#00d2c6", "Bits", 1200);
const chartQber = makeChart("chart-qber", "Max QBER", "#f0b849", "QBER", 0.2);

function updateChart(chart, labels, data) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update("none");
}

// ── D3 Topology Setup ─────────────────────────────────────────────────────
const svg = d3.select("#topo-svg");
let gLinks, gNodes, gEdgeLabels, simulation;
let topoInited = false;
let d3Nodes = [], d3Links = [];

function initTopo(nodes, links) {
    const container = document.getElementById("topo-container");
    const W = container.clientWidth;
    const H = container.clientHeight;

    svg.attr("viewBox", `0 0 ${W} ${H}`);

    // Build node/link data
    d3Nodes = nodes.map(id => ({ id }));
    const nodeMap = {};
    d3Nodes.forEach(n => nodeMap[n.id] = n);

    d3Links = links.map(l => ({
        source: nodeMap[l.u],
        target: nodeMap[l.v],
        u: l.u, v: l.v,
        K_curr: l.K_curr,
        QBER: l.QBER,
        pruned: l.pruned,
    }));

    // Force simulation
    simulation = d3.forceSimulation(d3Nodes)
        .force("link", d3.forceLink(d3Links).id(d => d.id).distance(90).strength(0.7))
        .force("charge", d3.forceManyBody().strength(-320))
        .force("center", d3.forceCenter(W / 2, H / 2))
        .force("collision", d3.forceCollide(30))
        .alphaDecay(0.02);

    // Link lines
    gLinks = svg.append("g").attr("class", "links")
        .selectAll("line")
        .data(d3Links)
        .enter().append("line")
        .attr("class", "link")
        .attr("stroke-width", 3);

    // Edge labels (K_curr)
    gEdgeLabels = svg.append("g").attr("class", "edge-labels")
        .selectAll("text")
        .data(d3Links)
        .enter().append("text")
        .attr("class", "edge-label");

    // Node groups
    const nodeG = svg.append("g").attr("class", "nodes")
        .selectAll("g")
        .data(d3Nodes)
        .enter().append("g")
        .call(d3.drag()
            .on("start", dragStart)
            .on("drag", dragging)
            .on("end", dragEnd));

    nodeG.append("circle")
        .attr("class", "node-circle")
        .attr("r", 18)
        .attr("fill", "#1a1f2e");

    nodeG.append("text")
        .attr("class", "node-label")
        .text(d => d.id);

    gNodes = nodeG;

    simulation.on("tick", () => {
        gLinks
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        gEdgeLabels
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2 - 6)
            .text(d => d.pruned ? "✕" : Math.round(d.K_curr));

        gNodes.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    topoInited = true;
}

function linkColor(K_curr, QBER, pruned, attacked) {
    if (attacked) return "#e04f5f";
    if (pruned) return "#3a3f4b";
    if (QBER >= 0.10) return "#e04f5f";
    if (K_curr < 200) return "#e04f5f";
    if (K_curr < 500) return "#f0b849";
    return "#00d2c6";
}

function updateTopo(links) {
    if (!topoInited) return;
    const linkMap = {};
    links.forEach(l => {
        const key = l.u < l.v ? `${l.u}-${l.v}` : `${l.v}-${l.u}`;
        linkMap[key] = l;
    });

    d3Links.forEach(dl => {
        const key = dl.u < dl.v ? `${dl.u}-${dl.v}` : `${dl.v}-${dl.u}`;
        const data = linkMap[key];
        if (data) {
            dl.K_curr = data.K_curr;
            dl.QBER = data.QBER;
            dl.pruned = data.pruned;
        }
    });

    gLinks
        .attr("stroke", d => {
            const key = d.u < d.v ? `${d.u}-${d.v}` : `${d.v}-${d.u}`;
            return linkColor(d.K_curr, d.QBER, d.pruned, attackedLinks.has(key));
        })
        .attr("stroke-width", d => {
            const key = d.u < d.v ? `${d.u}-${d.v}` : `${d.v}-${d.u}`;
            return attackedLinks.has(key) ? 5 : (d.pruned ? 1.5 : 3);
        })
        .attr("stroke-opacity", d => d.pruned ? 0.3 : 0.85)
        .classed("link-attacked", d => {
            const key = d.u < d.v ? `${d.u}-${d.v}` : `${d.v}-${d.u}`;
            return attackedLinks.has(key);
        });
}

// Drag handlers
function dragStart(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}
function dragging(event, d) { d.fx = event.x; d.fy = event.y; }
function dragEnd(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}

// ── Table Renderers ───────────────────────────────────────────────────────

function renderLinkTable(links) {
    const tbody = document.getElementById("link-table-body");
    tbody.innerHTML = links.map(l => {
        const cls = l.pruned ? "row-pruned" : (l.QBER >= 0.10 ? "row-attacked" : "");
        const cost = l.qosec_cost === "inf" ? "∞ (pruned)" : Number(l.qosec_cost).toFixed(4);
        return `<tr class="${cls}">
            <td>${l.u} ↔ ${l.v}</td>
            <td>${l.distance.toFixed(1)}</td>
            <td>${l.K_curr}</td>
            <td>${l.QBER.toFixed(4)}</td>
            <td>${cost}</td>
        </tr>`;
    }).join("");
}

function renderTrafficTable(traffic) {
    const tbody = document.getElementById("traffic-table-body");
    // Show latest first
    const rows = traffic.slice().reverse().slice(0, 30);
    tbody.innerHTML = rows.map(r => {
        const cls = r.success ? "row-ok" : "row-fail";
        const path = r.path ? r.path.join(" → ") : "—";
        const status = r.success ? "✓ OK" : `✗ ${r.failure_reason.substring(0, 30)}`;
        return `<tr class="${cls}">
            <td>${r.tick}</td>
            <td>${r.src} → ${r.dst}</td>
            <td>${path}</td>
            <td>${r.key_bits}</td>
            <td>${status}</td>
        </tr>`;
    }).join("");
}

function renderEveTable(attacks) {
    const tbody = document.getElementById("eve-table-body");
    const rows = attacks.slice().reverse();
    tbody.innerHTML = rows.map(a => {
        const [u, v] = a.link;
        return `<tr class="row-attacked">
            <td>${a.tick}</td>
            <td>${u} ↔ ${v}</td>
            <td>${a.qber_injected.toFixed(3)}</td>
            <td>${a.k_curr_at_attack.toFixed(0)}</td>
        </tr>`;
    }).join("");
    document.getElementById("eve-count").textContent = `${attacks.length} attack${attacks.length !== 1 ? "s" : ""}`;
}

// ── Header Updaters ───────────────────────────────────────────────────────
function updateHeader(state) {
    document.getElementById("tick-display").textContent = state.tick;
    document.getElementById("tick-max").textContent = `/ ${state.max_ticks}`;
    document.getElementById("blocking-display").textContent = (state.blocking_rate * 100).toFixed(2) + "%";
    document.getElementById("requests-display").textContent = state.total_requests;

    const dot = document.getElementById("status-dot");
    dot.className = "status-dot";
    if (state.finished) { dot.classList.add("finished"); dot.title = "Finished"; }
    else if (state.running) { dot.classList.add("running"); dot.title = "Running"; }
    else { dot.classList.add("paused"); dot.title = "Paused"; }

    // Params
    const p = state.params;
    document.getElementById("params-display").innerHTML =
        `<span class="param">α=${p.alpha}</span>
         <span class="param">β=${p.beta}</span>
         <span class="param">γ=${p.gamma}</span>
         <span class="param">η=${p.eta}</span>`;
}

// ── Build attacked-links set from attack_log ──────────────────────────────
function refreshAttackedLinks(attackLog, currentTick) {
    attackedLinks.clear();
    // Mark links attacked in the last 50 ticks as "under attack"
    attackLog.forEach(a => {
        if (currentTick - a.tick < 50) {
            const [u, v] = a.link;
            const key = u < v ? `${u}-${v}` : `${v}-${u}`;
            attackedLinks.add(key);
        }
    });
}

// ── Main Poll Loop ────────────────────────────────────────────────────────
async function poll() {
    try {
        const resp = await fetch(`${API}/api/state`);
        if (!resp.ok) return;
        const s = await resp.json();

        // Init topology on first data
        if (!topoInited && s.nodes && s.links) {
            initTopo(s.nodes, s.links);
        }

        // Only update DOM when tick changes
        if (s.tick !== prevTick) {
            prevTick = s.tick;

            refreshAttackedLinks(s.attack_log, s.tick);
            updateTopo(s.links);
            renderLinkTable(s.links);
            renderTrafficTable(s.traffic);
            renderEveTable(s.attack_log);
            updateHeader(s);

            // Charts
            const ts = s.timeseries;
            updateChart(chartBlocking, ts.ticks, ts.blocking);
            updateChart(chartKcurr, ts.ticks, ts.avg_kcurr);
            updateChart(chartQber, ts.ticks, ts.max_qber);
        }
    } catch (e) {
        // Silently retry
    }
}

setInterval(poll, POLL_MS);

// ── Control Buttons ───────────────────────────────────────────────────────
function sendControl(action) {
    fetch(`${API}/api/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
    });
}

document.getElementById("btn-start").addEventListener("click", () => sendControl("start"));
document.getElementById("btn-pause").addEventListener("click", () => sendControl("pause"));
document.getElementById("btn-step").addEventListener("click", () => sendControl("step"));
document.getElementById("btn-reset").addEventListener("click", () => {
    sendControl("reset");
    prevTick = -1;
});

// Speed slider
const speedSlider = document.getElementById("speed-slider");
const speedLabel = document.getElementById("speed-label");
speedSlider.addEventListener("input", () => {
    const ms = parseInt(speedSlider.value);
    speedLabel.textContent = ms + " ms";
    fetch(`${API}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed_ms: ms }),
    });
});

// ── Initial poll ──────────────────────────────────────────────────────────
poll();
