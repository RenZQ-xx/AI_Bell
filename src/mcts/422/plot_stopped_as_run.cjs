/* Offline scientific chart using the bundled sharp runtime, not search code. */
const fs = require('node:fs');
const path = require('node:path');
const [input, output, sharpPath] = process.argv.slice(2);
if (!input || !output || !sharpPath) throw new Error('usage: input.json output.png sharp-module');
const sharp = require(sharpPath);
const summary = JSON.parse(fs.readFileSync(input, 'utf8'));
const rows = summary.windows_10000;
const width = 1100, height = 1040, left = 100, right = 45, chartWidth = width-left-right;
const panelHeight = 215, tops = [140, 445, 750];
const x = i => left+(i+0.5)*chartWidth/rows.length;
const escape = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const content = [`<rect width="100%" height="100%" fill="white"/>`];
function text(x, y, value, size=14, anchor='start', extra='') {
  content.push(`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" ${extra}>${escape(value)}</text>`);
}
text(left, 38, 'Bell 4-2-2 v31 stopped at 90,000 iterations', 25);
text(left, 69, `${summary.canonical_classes.toLocaleString('en-US')} recorded classes | same-run 10,000-iteration windows`, 16);
function panel(index, label, maximum) {
  const top = tops[index], bottom = top+panelHeight;
  const y = value => bottom-value/maximum*panelHeight;
  text(left, top-24, label, 18);
  for (let i=0; i<=5; i++) {
    const value = maximum*i/5;
    content.push(`<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="#e4e7eb"/>`);
    text(left-13, y(value)+5, maximum <= 4 ? value.toFixed(1) : Math.round(value), 13, 'end');
  }
  content.push(`<line x1="${left}" x2="${width-right}" y1="${bottom}" y2="${bottom}" stroke="#555"/>`);
  rows.forEach((row, i) => text(x(i), bottom+24, `${row.end_iteration/1000}k`, 13, 'middle'));
  return y;
}
const y0 = panel(0, 'New canonical classes per window', 2500);
const barWidth = Math.min(62, chartWidth/rows.length*0.7);
rows.forEach((row, i) => {
  const bottom = tops[0]+panelHeight;
  content.push(`<rect x="${x(i)-barWidth/2}" y="${y0(row.catalog_unmatched)}" width="${barWidth}" height="${bottom-y0(row.catalog_unmatched)}" fill="#3978a8"/>`);
  content.push(`<rect x="${x(i)-barWidth/2}" y="${y0(row.new_classes)}" width="${barWidth}" height="${y0(row.catalog_unmatched)-y0(row.new_classes)}" fill="#3d935a"/>`);
  text(x(i), y0(row.new_classes)-10, row.new_classes, 13, 'middle');
});
content.push('<rect x="650" y="88" width="14" height="14" fill="#3978a8"/><rect x="650" y="111" width="14" height="14" fill="#3d935a"/>');
text(674, 100, 'Catalog unmatched', 13);
text(674, 123, 'Catalog matched', 13);
function line(index, title, maximum, values, color, format) {
  const y = panel(index, title, maximum);
  content.push(`<polyline points="${values.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="3"/>`);
  values.forEach((value, i) => {
    content.push(`<circle cx="${x(i)}" cy="${y(value)}" r="4" fill="${color}"/>`);
    text(x(i), y(value)-13, format(value), 13, 'middle');
  });
}
line(1, 'Seconds per iteration (lower is faster)', 3, rows.map(r=>r.seconds_per_iteration), '#c54b46', v=>v.toFixed(2));
line(2, 'New canonical classes per search hour', 2000, rows.map(r=>r.classes_per_hour), '#3978a8', v=>Math.round(v));
text(width/2, 1025, 'Iteration window end', 15, 'middle');
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><g font-family="Arial, sans-serif" fill="#20252a">${content.join('')}</g></svg>`;
const svgPath = path.join(path.dirname(output), path.basename(output, path.extname(output))+'.svg');
fs.writeFileSync(svgPath, svg);
sharp(Buffer.from(svg)).png().toFile(output).then(info=>process.stdout.write(JSON.stringify(info)+'\n'));
