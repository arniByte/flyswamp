// Repackage web/dist as a claude.ai Artifact: page content without the html/head/body skeleton
// (the host adds its own), CSS inlined, JS bundle and data published as sibling files.
// usage: node scripts/artifact.mjs   (after `npm run build` in web/)
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const dist = path.join(root, 'web', 'dist');
const out = path.join(root, 'web', 'dist-artifact');

let html = fs.readFileSync(path.join(dist, 'index.html'), 'utf8');
html = html.replace(/<link rel="stylesheet"[^>]*href="\.\/(assets\/[^"]+\.css)"[^>]*>/, (_, css) =>
  `<style>\n${fs.readFileSync(path.join(dist, css), 'utf8')}\n</style>`);
html = html
  .replace(/<!doctype html>/i, '')
  .replace(/<\/?html[^>]*>/g, '')
  .replace(/<\/?head>/g, '')
  .replace(/<\/?body>/g, '')
  .replace(/<meta charset[^>]*>\s*/, '')
  .replace(/<meta name="viewport"[^>]*>\s*/, '')
  .trim();
const title = html.match(/<title>.*?<\/title>/)[0];
html = `${title}\n${html.replace(title, '')}`;
// Artifacts serve no binary types: ship .bin/.glb as base64 text and tell data.js to decode them.
html = html.replace('<script type="module"', '<script>window.FLYSWAMP_B64 = true;</script>\n  <script type="module"');

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'index.html'), html + '\n');
const files = {};
for (const dir of ['assets', 'data', 'data/cns']) {
  if (!fs.existsSync(path.join(dist, dir))) continue;
  for (const f of fs.readdirSync(path.join(dist, dir))) {
    if (f.endsWith('.css') || fs.statSync(path.join(dist, dir, f)).isDirectory()) continue;
    fs.mkdirSync(path.join(out, dir), { recursive: true });
    const binary = /\.(bin|glb)(\.\d+)?$/.test(f); // cns.bin.0, cns.bin.1, ...: chunks of the whole-CNS graph
    const name = binary ? `${f}.b64.txt` : f;
    if (binary) fs.writeFileSync(path.join(out, dir, name), fs.readFileSync(path.join(dist, dir, f)).toString('base64'));
    else fs.copyFileSync(path.join(dist, dir, f), path.join(out, dir, name));
    files[`${dir}/${name}`] = path.relative(root, path.join(out, dir, name));
  }
}
console.log(JSON.stringify(files, null, 1));
