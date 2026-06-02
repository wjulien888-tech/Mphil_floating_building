const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const INPUT = path.join(ROOT, "output", "dispatch_rbc_15min_raw_battery215_2023.csv");
const SUMMARY = path.join(ROOT, "output", "dispatch_rbc_15min_raw_battery215_summary.csv");
const PLOTS = path.join(ROOT, "plots");

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      const value = values[index];
      row[header] = value === undefined ? "" : value;
    });
    return row;
  });
}

function num(row, key) {
  return Number(row[key]);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

function polyline(points, color, width = 1.5) {
  return `<polyline points="${points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
}

function axisLabels(width, height, margin, title, yLabel) {
  return `
  <rect width="${width}" height="${height}" fill="#ffffff"/>
  <text x="${width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#1f2933">${title}</text>
  <text x="22" y="${height / 2}" transform="rotate(-90 22 ${height / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="#52606d">${yLabel}</text>
  <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#9aa5b1"/>
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#9aa5b1"/>
  `;
}

function writeSvg(filename, body, width = 1100, height = 420) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
  fs.writeFileSync(path.join(PLOTS, filename), svg);
}

function plotSoc(rows) {
  const width = 1100;
  const height = 420;
  const margin = { top: 52, right: 30, bottom: 48, left: 72 };
  const x = scaleLinear(0, rows.length - 1, margin.left, width - margin.right);
  const y = scaleLinear(0, 215, height - margin.bottom, margin.top);
  const points = rows.map((row, i) => [x(i), y(num(row, "battery_soc_kWh"))]);
  const monthTicks = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334].map((day, i) => {
    const xx = x(day * 96);
    return `<line x1="${xx}" y1="${height - margin.bottom}" x2="${xx}" y2="${height - margin.bottom + 5}" stroke="#9aa5b1"/><text x="${xx}" y="${height - 22}" text-anchor="middle" font-family="Arial" font-size="10" fill="#52606d">${i + 1}</text>`;
  }).join("");
  const grid = [0, 50, 100, 150, 200].map((v) => {
    const yy = y(v);
    return `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="#e4e7eb"/><text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#52606d">${v}</text>`;
  }).join("");
  writeSvg(
    "dispatch_rbc_15min_raw_battery215_soc.svg",
    axisLabels(width, height, margin, "Battery SOC - RBC 15-min raw disturbed PV (215 kWh)", "SOC (kWh)") + grid + polyline(points, "#2f6f73", 1.2) + monthTicks,
    width,
    height
  );
}

function plotWorstDay(rows) {
  const byDate = new Map();
  rows.forEach((row) => {
    const d = row.datetime.slice(0, 10);
    byDate.set(d, (byDate.get(d) || 0) + num(row, "unmet_load_kW") * 0.25);
  });
  const [worstDate] = [...byDate.entries()].sort((a, b) => b[1] - a[1])[0];
  const day = rows.filter((row) => row.datetime.startsWith(worstDate));

  const width = 1100;
  const height = 460;
  const margin = { top: 58, right: 32, bottom: 58, left: 72 };
  const values = day.flatMap((row) => [num(row, "load_kW"), num(row, "pv_kW"), num(row, "battery_charge_kW"), -num(row, "battery_discharge_kW")]);
  const ymin = Math.min(...values, -55);
  const ymax = Math.max(...values, 55);
  const x = scaleLinear(0, 24, margin.left, width - margin.right);
  const y = scaleLinear(ymin, ymax, height - margin.bottom, margin.top);
  const loadPts = day.map((row, i) => [x(i / 4), y(num(row, "load_kW"))]);
  const pvPts = day.map((row, i) => [x(i / 4), y(num(row, "pv_kW"))]);
  const zero = y(0);
  const bars = day.map((row, i) => {
    const xx = x(i / 4);
    const chargeY = y(num(row, "battery_charge_kW"));
    const dischargeY = y(-num(row, "battery_discharge_kW"));
    return `<rect x="${xx - 2}" y="${Math.min(chargeY, zero)}" width="4" height="${Math.abs(zero - chargeY)}" fill="#5aa469" opacity="0.55"/><rect x="${xx - 2}" y="${Math.min(dischargeY, zero)}" width="4" height="${Math.abs(zero - dischargeY)}" fill="#c44e52" opacity="0.55"/>`;
  }).join("");
  const ticks = [0, 4, 8, 12, 16, 20, 24].map((v) => `<line x1="${x(v)}" y1="${height - margin.bottom}" x2="${x(v)}" y2="${height - margin.bottom + 5}" stroke="#9aa5b1"/><text x="${x(v)}" y="${height - 27}" text-anchor="middle" font-family="Arial" font-size="10" fill="#52606d">${v}</text>`).join("");
  const legend = `<text x="${margin.left}" y="42" font-family="Arial" font-size="12" fill="#334e68">Load</text><text x="${margin.left + 58}" y="42" font-family="Arial" font-size="12" fill="#d99000">PV</text><text x="${margin.left + 98}" y="42" font-family="Arial" font-size="12" fill="#5aa469">Charge</text><text x="${margin.left + 165}" y="42" font-family="Arial" font-size="12" fill="#c44e52">Discharge</text>`;
  writeSvg(
    "dispatch_rbc_15min_raw_battery215_worst_day.svg",
    axisLabels(width, height, margin, `Worst unmet-load day dispatch (${worstDate})`, "kW") + `<line x1="${margin.left}" y1="${zero}" x2="${width - margin.right}" y2="${zero}" stroke="#e4e7eb"/>` + bars + polyline(loadPts, "#334e68", 2) + polyline(pvPts, "#d99000", 2) + ticks + legend,
    width,
    height
  );
}

function plotMonthly(rows) {
  const monthly = new Map();
  rows.forEach((row) => {
    const month = Number(row.datetime.slice(5, 7));
    if (!monthly.has(month)) monthly.set(month, { load: 0, pv: 0, curtail: 0, unmet: 0 });
    const m = monthly.get(month);
    m.load += num(row, "load_kW") * 0.25;
    m.pv += num(row, "pv_kW") * 0.25;
    m.curtail += num(row, "curtailment_kW") * 0.25;
    m.unmet += num(row, "unmet_load_kW") * 0.25;
  });
  const data = [...monthly.entries()].map(([month, value]) => ({ month, ...value }));
  const width = 1100;
  const height = 470;
  const margin = { top: 58, right: 34, bottom: 58, left: 78 };
  const ymax = Math.max(...data.flatMap((d) => [d.load, d.pv, d.curtail, d.unmet])) * 1.12;
  const x = scaleLinear(0, 12, margin.left, width - margin.right);
  const y = scaleLinear(0, ymax, height - margin.bottom, margin.top);
  const barW = 18;
  const bars = data.map((d, i) => {
    const cx = x(i + 0.5);
    return [
      ["load", "#334e68", -barW],
      ["pv", "#d99000", 0],
      ["curtail", "#8e6c8a", barW],
    ].map(([key, color, offset]) => {
      const v = d[key];
      return `<rect x="${cx + Number(offset) - barW / 2}" y="${y(v)}" width="${barW}" height="${height - margin.bottom - y(v)}" fill="${color}"/>`;
    }).join("") + `<circle cx="${cx}" cy="${y(d.unmet)}" r="4" fill="#c44e52"/>`;
  }).join("");
  const ticks = data.map((d, i) => `<text x="${x(i + 0.5)}" y="${height - 28}" text-anchor="middle" font-family="Arial" font-size="10" fill="#52606d">${d.month}</text>`).join("");
  const legend = `<text x="${margin.left}" y="42" font-family="Arial" font-size="12" fill="#334e68">Load</text><text x="${margin.left + 55}" y="42" font-family="Arial" font-size="12" fill="#d99000">PV</text><text x="${margin.left + 88}" y="42" font-family="Arial" font-size="12" fill="#8e6c8a">Curtailment</text><text x="${margin.left + 180}" y="42" font-family="Arial" font-size="12" fill="#c44e52">Unmet load</text>`;
  writeSvg(
    "dispatch_rbc_15min_raw_battery215_monthly.svg",
    axisLabels(width, height, margin, "Monthly energy summary - RBC 15-min raw disturbed PV", "kWh") + bars + ticks + legend,
    width,
    height
  );
}

function main() {
  ensureDir(PLOTS);
  const rows = parseCsv(fs.readFileSync(INPUT, "utf8"));
  plotSoc(rows);
  plotWorstDay(rows);
  plotMonthly(rows);
  console.log("Wrote SVG plots to plots/");
}

main();
