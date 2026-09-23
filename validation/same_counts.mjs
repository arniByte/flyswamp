// Are two spike-count files (run_lif.mjs schema) identical, neuron by neuron and trial by trial?
// Used to show that an engine change leaves every spike of a validated run in place.
// usage: node validation/same_counts.mjs a.json b.json
import fs from 'node:fs';

const [a, b] = process.argv.slice(2).map((f) => JSON.parse(fs.readFileSync(f, 'utf8')).counts);
const ids = new Set([...Object.keys(a), ...Object.keys(b)]);
let differ = 0;
for (const id of ids) if (JSON.stringify(a[id] ?? null) !== JSON.stringify(b[id] ?? null)) differ++;
console.log(`${ids.size} neurons, ${differ} differ → ${differ ? 'DIFFERENT' : 'identical'}`);
process.exit(differ ? 1 : 0);
