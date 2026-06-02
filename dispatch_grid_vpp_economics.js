const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const OUT_DIR = path.join(ROOT, "output");
const PLOTS_DIR = path.join(ROOT, "plots");

const RBC_CSV = path.join(OUT_DIR, "dispatch_rbc_15min_raw_battery215_2023.csv");
const DETAIL_CSV = path.join(OUT_DIR, "dispatch_grid_vpp_economics_2023.csv");
const SUMMARY_CSV = path.join(OUT_DIR, "dispatch_grid_vpp_economics_summary.csv");

const DT_HOURS = 0.25;
const EXPORT_PRICE_CNY_PER_KWH = 0.36;

// Guangzhou / Pearl River Delta five-city general industrial/commercial
// single-rate 1-10(20) kV proxy from Guangdong Grid agent-purchase TOU table.
const IMPORT_TARIFFS = {
  valley: 0.29526875,
  flat: 0.73196875,
  peak: 1.22496875,
  sharp_peak: 1.52436875,
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] === undefined ? "" : values[index];
    });
    return row;
  });
}

function formatCsvValue(value) {
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(filename, rows, headers) {
  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => formatCsvValue(row[header] ?? "")).join(",")),
  ];
  fs.writeFileSync(filename, `${lines.join("\n")}\n`);
}

function num(row, key) {
  return Number(row[key]) || 0;
}

function dateKey(datetime) {
  return datetime.slice(0, 10);
}

function monthKey(datetime) {
  return Number(datetime.slice(5, 7));
}

function hourOfDay(datetime) {
  return Number(datetime.slice(11, 13)) + Number(datetime.slice(14, 16)) / 60;
}

function buildHighTemperatureDays(rows) {
  const maxByDate = new Map();
  rows.forEach((row) => {
    const d = dateKey(row.datetime);
    const temp = num(row, "temperature_C");
    maxByDate.set(d, Math.max(maxByDate.get(d) ?? -Infinity, temp));
  });
  return new Set([...maxByDate.entries()].filter(([, temp]) => temp >= 35).map(([d]) => d));
}

function tariffPeriod(datetime, highTemperatureDays) {
  const month = monthKey(datetime);
  const hour = hourOfDay(datetime);
  const isSharpMonth = month >= 7 && month <= 9;
  const isHighTemperatureDay = highTemperatureDays.has(dateKey(datetime));
  const isSharpHour = (hour >= 11 && hour < 12) || (hour >= 15 && hour < 17);

  if ((isSharpMonth || isHighTemperatureDay) && isSharpHour) return "sharp_peak";
  if (hour >= 0 && hour < 8) return "valley";
  if ((hour >= 10 && hour < 12) || (hour >= 14 && hour < 19)) return "peak";
  return "flat";
}

function addEconomics(rows) {
  const highTemperatureDays = buildHighTemperatureDays(rows);
  let cumulativeCost = 0;
  let cumulativeRevenue = 0;

  return rows.map((row) => {
    const period = tariffPeriod(row.datetime, highTemperatureDays);
    const importTariff = IMPORT_TARIFFS[period];
    const gridImportKw = num(row, "unmet_load_kW");
    const gridExportKw = num(row, "curtailment_kW");
    const gridImportKwh = gridImportKw * DT_HOURS;
    const gridExportKwh = gridExportKw * DT_HOURS;
    const importCost = gridImportKwh * importTariff;
    const exportRevenue = gridExportKwh * EXPORT_PRICE_CNY_PER_KWH;
    cumulativeCost += importCost;
    cumulativeRevenue += exportRevenue;

    return {
      ...row,
      tariff_period: period,
      import_tariff_CNY_per_kWh: importTariff,
      export_price_CNY_per_kWh: EXPORT_PRICE_CNY_PER_KWH,
      grid_import_kW: gridImportKw,
      grid_export_kW: gridExportKw,
      grid_import_kWh: gridImportKwh,
      grid_export_kWh: gridExportKwh,
      import_cost_CNY: importCost,
      export_revenue_CNY: exportRevenue,
      net_grid_cost_CNY: importCost - exportRevenue,
      cumulative_import_cost_CNY: cumulativeCost,
      cumulative_export_revenue_CNY: cumulativeRevenue,
      cumulative_net_grid_cost_CNY: cumulativeCost - cumulativeRevenue,
    };
  });
}

function summarize(rows) {
  const byPeriod = {};
  Object.keys(IMPORT_TARIFFS).forEach((period) => {
    byPeriod[period] = { importKwh: 0, importCost: 0 };
  });

  const summary = rows.reduce(
    (acc, row) => {
      const period = row.tariff_period;
      acc.grid_import_kWh += row.grid_import_kWh;
      acc.grid_export_kWh += row.grid_export_kWh;
      acc.import_cost_CNY += row.import_cost_CNY;
      acc.export_revenue_CNY += row.export_revenue_CNY;
      acc.net_grid_cost_CNY += row.net_grid_cost_CNY;
      acc.max_grid_import_kW = Math.max(acc.max_grid_import_kW, row.grid_import_kW);
      acc.max_grid_export_kW = Math.max(acc.max_grid_export_kW, row.grid_export_kW);
      byPeriod[period].importKwh += row.grid_import_kWh;
      byPeriod[period].importCost += row.import_cost_CNY;
      return acc;
    },
    {
      scenario: "grid_import_plus_vpp_export",
      export_price_CNY_per_kWh: EXPORT_PRICE_CNY_PER_KWH,
      grid_import_kWh: 0,
      grid_export_kWh: 0,
      import_cost_CNY: 0,
      export_revenue_CNY: 0,
      net_grid_cost_CNY: 0,
      max_grid_import_kW: 0,
      max_grid_export_kW: 0,
    }
  );

  summary.average_import_price_CNY_per_kWh =
    summary.grid_import_kWh ? summary.import_cost_CNY / summary.grid_import_kWh : 0;
  summary.average_export_price_CNY_per_kWh = EXPORT_PRICE_CNY_PER_KWH;
  Object.entries(byPeriod).forEach(([period, values]) => {
    summary[`${period}_import_kWh`] = values.importKwh;
    summary[`${period}_import_cost_CNY`] = values.importCost;
  });
  return summary;
}

function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
}

function polyline(points, color, width = 1.5) {
  return `<polyline points="${points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
}

function frame(width, height, margin, title, yLabel) {
  return `
  <rect width="${width}" height="${height}" fill="#ffffff"/>
  <text x="${width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" fill="#1f2933">${title}</text>
  <text x="22" y="${height / 2}" transform="rotate(-90 22 ${height / 2})" text-anchor="middle" font-family="Arial" font-size="12" fill="#52606d">${yLabel}</text>
  <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#9aa5b1"/>
  <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#9aa5b1"/>
  `;
}

function writeSvg(filename, body, width = 1200, height = 480) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${body}</svg>`;
  fs.writeFileSync(path.join(PLOTS_DIR, filename), svg);
}

function monthAggregates(rows) {
  const monthly = new Map();
  rows.forEach((row) => {
    const month = monthKey(row.datetime);
    if (!monthly.has(month)) {
      monthly.set(month, { month, importKwh: 0, exportKwh: 0, importCost: 0, exportRevenue: 0, netCost: 0 });
    }
    const item = monthly.get(month);
    item.importKwh += row.grid_import_kWh;
    item.exportKwh += row.grid_export_kWh;
    item.importCost += row.import_cost_CNY;
    item.exportRevenue += row.export_revenue_CNY;
    item.netCost += row.net_grid_cost_CNY;
  });
  return [...monthly.values()].sort((a, b) => a.month - b.month);
}

function plotMonthlyEconomics(rows) {
  const data = monthAggregates(rows);
  const width = 1200;
  const height = 500;
  const margin = { top: 58, right: 44, bottom: 58, left: 84 };
  const maxEnergy = Math.max(...data.flatMap((d) => [d.importKwh, d.exportKwh])) * 1.15;
  const minCost = Math.min(0, ...data.map((d) => d.netCost));
  const maxCost = Math.max(...data.flatMap((d) => [d.importCost, d.exportRevenue, d.netCost])) * 1.15;
  const x = scaleLinear(0, 12, margin.left, width - margin.right);
  const yEnergy = scaleLinear(0, maxEnergy, height - margin.bottom, margin.top);
  const yCost = scaleLinear(minCost, maxCost, height - margin.bottom, margin.top);
  const barW = 20;

  const bars = data.map((d, i) => {
    const cx = x(i + 0.5);
    return `
      <rect x="${cx - barW - 2}" y="${yEnergy(d.importKwh)}" width="${barW}" height="${height - margin.bottom - yEnergy(d.importKwh)}" fill="#334e68" opacity="0.85"/>
      <rect x="${cx + 2}" y="${yEnergy(d.exportKwh)}" width="${barW}" height="${height - margin.bottom - yEnergy(d.exportKwh)}" fill="#d99000" opacity="0.78"/>
    `;
  }).join("");
  const netLine = polyline(data.map((d, i) => [x(i + 0.5), yCost(d.netCost)]), "#c44e52", 2);
  const ticks = data.map((d, i) => `<text x="${x(i + 0.5)}" y="${height - 28}" text-anchor="middle" font-family="Arial" font-size="10" fill="#52606d">${d.month}</text>`).join("");
  const grid = [0, maxEnergy / 2, maxEnergy].map((v) => `<line x1="${margin.left}" y1="${yEnergy(v)}" x2="${width - margin.right}" y2="${yEnergy(v)}" stroke="#e4e7eb"/><text x="${margin.left - 8}" y="${yEnergy(v) + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#52606d">${Math.round(v)}</text>`).join("");
  const legend = `<text x="${margin.left}" y="44" font-family="Arial" font-size="12" fill="#334e68">Grid import kWh</text><text x="${margin.left + 120}" y="44" font-family="Arial" font-size="12" fill="#d99000">VPP export kWh</text><text x="${margin.left + 245}" y="44" font-family="Arial" font-size="12" fill="#c44e52">Net cost CNY</text>`;

  writeSvg(
    "grid_vpp_monthly_import_export_cost.svg",
    frame(width, height, margin, "Monthly grid import, VPP export and net electricity cost", "Energy (kWh)") + grid + bars + netLine + ticks + legend,
    width,
    height
  );
}

function plotDailyImportAndTariff(rows) {
  const byDate = new Map();
  rows.forEach((row) => {
    const d = dateKey(row.datetime);
    if (!byDate.has(d)) byDate.set(d, { date: d, importKwh: 0, cost: 0 });
    const item = byDate.get(d);
    item.importKwh += row.grid_import_kWh;
    item.cost += row.import_cost_CNY;
  });
  const daily = [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
  daily.forEach((d) => {
    d.avgTariff = d.importKwh ? d.cost / d.importKwh : 0;
  });

  const width = 1200;
  const height = 460;
  const margin = { top: 58, right: 44, bottom: 56, left: 84 };
  const maxImport = Math.max(...daily.map((d) => d.importKwh)) * 1.15;
  const maxTariff = Math.max(...Object.values(IMPORT_TARIFFS)) * 1.05;
  const x = scaleLinear(0, daily.length, margin.left, width - margin.right);
  const yImport = scaleLinear(0, maxImport, height - margin.bottom, margin.top);
  const yTariff = scaleLinear(0, maxTariff, height - margin.bottom, margin.top);
  const bars = daily.map((d, i) => `<rect x="${x(i)}" y="${yImport(d.importKwh)}" width="${Math.max(1, x(i + 1) - x(i) - 0.25)}" height="${height - margin.bottom - yImport(d.importKwh)}" fill="#334e68" opacity="0.72"/>`).join("");
  const tariffLine = polyline(daily.map((d, i) => [x(i + 0.5), yTariff(d.avgTariff)]), "#c44e52", 1.6);
  const monthStarts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  const labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const ticks = monthStarts.map((day, i) => `<line x1="${x(day)}" y1="${height - margin.bottom}" x2="${x(day)}" y2="${height - margin.bottom + 5}" stroke="#9aa5b1"/><text x="${x(day)}" y="${height - 24}" text-anchor="middle" font-family="Arial" font-size="10" fill="#52606d">${labels[i]}</text>`).join("");
  const grid = [0, maxImport / 2, maxImport].map((v) => `<line x1="${margin.left}" y1="${yImport(v)}" x2="${width - margin.right}" y2="${yImport(v)}" stroke="#e4e7eb"/><text x="${margin.left - 8}" y="${yImport(v) + 4}" text-anchor="end" font-family="Arial" font-size="10" fill="#52606d">${Math.round(v)}</text>`).join("");
  const legend = `<text x="${margin.left}" y="44" font-family="Arial" font-size="12" fill="#334e68">Daily grid import</text><text x="${margin.left + 130}" y="44" font-family="Arial" font-size="12" fill="#c44e52">Average import tariff</text>`;

  writeSvg(
    "grid_vpp_daily_import_and_tariff.svg",
    frame(width, height, margin, "Daily grid import and effective TOU purchase tariff", "Grid import (kWh/day)") + grid + bars + tariffLine + ticks + legend,
    width,
    height
  );
}

function main() {
  if (!fs.existsSync(RBC_CSV)) {
    throw new Error(`Missing ${RBC_CSV}; run dispatch_rbc_15min.py first.`);
  }
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(PLOTS_DIR, { recursive: true });

  const inputRows = parseCsv(fs.readFileSync(RBC_CSV, "utf8"));
  const rows = addEconomics(inputRows);
  const summary = summarize(rows);

  writeCsv(DETAIL_CSV, rows, Object.keys(rows[0]));
  writeCsv(SUMMARY_CSV, [summary], Object.keys(summary));
  plotMonthlyEconomics(rows);
  plotDailyImportAndTariff(rows);

  console.log(`Grid import: ${summary.grid_import_kWh.toFixed(1)} kWh`);
  console.log(`VPP export: ${summary.grid_export_kWh.toFixed(1)} kWh`);
  console.log(`Import cost: ${summary.import_cost_CNY.toFixed(2)} CNY`);
  console.log(`Export revenue: ${summary.export_revenue_CNY.toFixed(2)} CNY`);
  console.log(`Net grid cost: ${summary.net_grid_cost_CNY.toFixed(2)} CNY`);
  console.log(`Saved: ${path.relative(ROOT, DETAIL_CSV)}`);
  console.log(`Saved: ${path.relative(ROOT, SUMMARY_CSV)}`);
}

main();
