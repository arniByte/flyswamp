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

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'index.html'), html + '\n');
const files = {};
for (const dir of ['assets', 'data']) {
  for (const f of fs.readdirSync(path.join(dist, dir))) {
    if (f.endsWith('.css')) continue;
    fs.mkdirSync(path.join(out, dir), { recursive: true });
    fs.copyFileSync(path.join(dist, dir, f), path.join(out, dir, f));
    files[`${dir}/${f}`] = path.relative(root, path.join(out, dir, f));
  }
}
console.log(JSON.stringify(files, null, 1));
