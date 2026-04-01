function pct(x) {
  return `${(100 * parseFloat(x)).toFixed(2)}%`;
}

function num(x, d = 2) {
  const n = parseFloat(x);
  if (!isFinite(n)) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: d,
    maximumFractionDigits: d
  });
}

// Book-favorable decimal odds
function cleanDecimalOdds(x) {
  let n = parseFloat(x);
  if (!isFinite(n)) return "—";

  n = Math.min(n, 51);

  if (n > 100) {
    n = Math.floor(n / 5) * 5;
  } else {
    n = Math.floor(n);
  }

  return n.toString();
}

function americanOdds(prob) {
  const p = parseFloat(prob);
  if (p <= 0 || p >= 1) return "—";

  let american;
  if (p > 0.5) {
    american = -100 * p / (1 - p);
  } else {
    american = 100 * (1 - p) / p;
  }

  if (american > 0) american = Math.min(american, 5000);
  if (american < 0) american = Math.max(american, -5000);

  if (american > 0) {
    american = Math.floor(american / 5) * 5;
    return `+${american}`;
  } else {
    american = Math.floor(Math.abs(american) / 5) * 5;
    return `-${american}`;
  }
}

function buildTable(el, columns, rows) {
  const table = document.getElementById(el);
  let html = "<thead><tr>";
  for (const c of columns) html += `<th>${c.label}</th>`;
  html += "</tr></thead><tbody>";

  for (const row of rows) {
    html += "<tr>";
    for (const c of columns) {
      const val = typeof c.render === "function" ? c.render(row) : row[c.key];
      html += `<td class="${c.num ? "num" : ""}">${val}</td>`;
    }
    html += "</tr>";
  }

  html += "</tbody>";
  table.innerHTML = html;
}

function buildMobileCards(el, rows, probKey, oddsKey, probLabel, bookProbKey) {
  const container = document.getElementById(el);
  container.innerHTML = rows.map(row => `
    <div class="mobile-card">
      <div class="mobile-card-top">
        <div class="mobile-name">${row.player}</div>
        <div class="prob-pill">${pct(row[probKey])}</div>
      </div>
      <div class="mobile-lines">
        <div class="label">${probLabel}</div>
        <div>${pct(row[probKey])}</div>
        <div class="label">Decimal</div>
        <div>${cleanDecimalOdds(row[oddsKey])}</div>
        <div class="label">American</div>
        <div>${americanOdds(row[bookProbKey])}</div>
      </div>
    </div>
  `).join("");
}

async function loadCSV(path) {
  const res = await fetch(path);
  const text = await res.text();
  const parsed = Papa.parse(text, { header: true, dynamicTyping: true });
  return parsed.data.filter(r => Object.values(r).some(v => v !== null && v !== ""));
}

async function init() {
  const market = await loadCSV("market_results.csv");
  const standings = await loadCSV("official_current_standings.csv");

  market.sort((a, b) => b.win_prob - a.win_prob);
  standings.sort((a, b) => b.current_total - a.current_total);

  const topBoard = document.getElementById("top-board");
  topBoard.innerHTML = market.slice(0, 4).map((row, i) => `
    <div class="top-card">
      <div class="rank">Rank ${i + 1}</div>
      <div class="name">${row.player}</div>
      <div class="prob">${pct(row.win_prob)} to win</div>
      <div class="odds">Decimal ${cleanDecimalOdds(row.win_odds_book)} · American ${americanOdds(row.win_prob_book)}</div>
    </div>
  `).join("");

  buildTable("winner-table", [
    { label: "Player", key: "player" },
    { label: "Win %", render: r => `<span class="prob-pill">${pct(r.win_prob)}</span>` },
    { label: "Decimal", render: r => cleanDecimalOdds(r.win_odds_book), num: true },
    { label: "American", render: r => americanOdds(r.win_prob_book), num: true },
    { label: "Expected Final", render: r => num(r.exp_final), num: true },
  ], market);

  const podiumRows = [...market].sort((a, b) => b.top3_prob - a.top3_prob);
  buildTable("podium-table", [
    { label: "Player", key: "player" },
    { label: "Top 3 %", render: r => `<span class="prob-pill">${pct(r.top3_prob)}</span>` },
    { label: "Decimal", render: r => cleanDecimalOdds(r.top3_odds_book), num: true },
    { label: "American", render: r => americanOdds(r.top3_prob_book), num: true },
  ], podiumRows);

  const eatersRows = [...market].sort((a, b) => b.top8_prob - a.top8_prob);
  buildTable("eaters-table", [
    { label: "Player", key: "player" },
    { label: "Top 8 %", render: r => `<span class="prob-pill">${pct(r.top8_prob)}</span>` },
    { label: "Decimal", render: r => cleanDecimalOdds(r.top8_odds_book), num: true },
    { label: "American", render: r => americanOdds(r.top8_prob_book), num: true },
  ], eatersRows);

  buildMobileCards("podium-cards", podiumRows, "top3_prob", "top3_odds_book", "Top 3", "top3_prob_book");
  buildMobileCards("eaters-cards", eatersRows, "top8_prob", "top8_odds_book", "Top 8", "top8_prob_book");

  buildTable("standings-table", [
    { label: "Player", key: "player" },
    { label: "Current Total", render: r => num(r.current_total), num: true },
  ], standings);

  buildTable("standings-table-mobile", [
    { label: "Player", key: "player" },
    { label: "Current Total", render: r => num(r.current_total), num: true },
  ], standings);

  const expectedFinishRows = [...market].sort((a, b) => b.estimated_mean - a.estimated_mean);

  buildTable("model-table", [
    { label: "Player", key: "player" },
    { label: "Expected Mean", render: r => num(r.estimated_mean), num: true },
    { label: "Volatility", render: r => num(r.resid_sd), num: true },
    { label: "P10", render: r => num(r.p10_final), num: true },
    { label: "P50", render: r => num(r.p50_final), num: true },
    { label: "P90", render: r => num(r.p90_final), num: true },
  ], expectedFinishRows);
}

init();