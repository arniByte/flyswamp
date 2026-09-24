// DOM overlay: specimen label, memory bars, brain read-out, learning chart, event log, toolbar.
import { GROUP_STYLE } from '../render/brain.js';
import { W_INNATE, W_MB } from '../sim/world.js';

const STATE_TEXT = {
  fly: ['летит, нюхает', '#8a9c92'],
  land: ['садится', '#ffb347'],
  feed: ['ест · сахар → PAM', '#ffc933'],
  trapped: ['в росянке · PPL1', '#ff4f8b'],
  escape: ['побег · DNp01', '#eaf6ff'],
};
const $ = (id) => document.getElementById(id);
const plural = (n, one, few, many) => {
  const d = n % 10, h = n % 100;
  return d === 1 && h !== 11 ? one : d >= 2 && d <= 4 && (h < 12 || h > 14) ? few : many;
};

export class Hud {
  constructor() {
    this.el = {
      state: $('state-text'), dot: $('state-dot'), clock: $('clock'), energy: $('energy-bar'), counts: $('spec-counts'), cns: $('cns-status'),
      kc: $('kc-pct'), kcBar: $('kc-bar'), app: $('mbon-app'), appBar: $('app-bar'), av: $('mbon-av'), avBar: $('av-bar'),
      pam: $('pam'), pamBar: $('pam-bar'), ppl: $('ppl'), pplBar: $('ppl-bar'), drive: $('drive'),
      eaten: $('c-eaten'), trapped: $('c-trapped'), strikes: $('c-strikes'), log: $('log'), chart: $('chart'),
      legend: $('brain-legend'), tooltip: $('tooltip'), pip: $('pip-label'),
    };
    this.memRows = [...document.querySelectorAll('.mem-row')].map((r) => ({
      k: r.dataset.k, bar: r.querySelector('.bar i'), val: r.querySelector('.val'), split: r.querySelector('.split'),
    }));
    this.lastLog = -1;
    this.history = [];
    this._t = 0;
  }

  loading(text) {
    $('loading-text').textContent = text;
  }

  ready(circuit) {
    const n = circuit.neurons.bodyId.length;
    const syn = Object.values(circuit.edges).reduce((s, e) => { for (let i = 2; i < e.length; i += 3) s += e[i]; return s; }, 0);
    this.el.counts.textContent = `в симуляции: ${n.toLocaleString('ru-RU')} нейронов · ${(syn / 1e6).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} млн синапсов`;
    $('loading').classList.add('done');
  }

  cnsStatus(text) {
    if (text === this._cnsText) return;
    this._cnsText = text;
    this.el.cns.hidden = !text;
    this.el.cns.textContent = text ?? '';
    // the label grew or shrank: keep the left panel just below it
    const bottom = $('specimen').getBoundingClientRect().bottom;
    document.documentElement.style.setProperty('--left-top', `${Math.max(176, Math.ceil(bottom) + 8)}px`);
  }

  // the whole CNS is running: the counts line now describes it, the rate-model panel is marked as a prototype
  cnsReady(header) {
    const n = header.n;
    this.el.counts.textContent = `в симуляции: ${n.toLocaleString('ru-RU')} ${plural(n, 'нейрон', 'нейрона', 'нейронов')} · ${(header.nnz / 1e6).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} млн связей`;
    this.el.counts.title = `MaleCNS v1.0, связи от ${header.graph.endsWith('min2') ? 2 : 1} синапсов; LIF Shiu et al. 2024, w_syn ${header.engine.wSyn} mV (гейт P2)`;
    document.body.classList.add('cns-live');
  }

  cnsUpdate(cns, speed) {
    const slow = cns.factor < 0.8 * speed ? ` · отстаёт: ×${cns.factor.toFixed(1)} из ×${speed}` : '';
    const taste = cns.tasting ? ' · вкус → GRN' : '';
    this.cnsStatus(cns.status === 'работает'
      ? `спайки: ${Math.round(cns.spikesPerSec).toLocaleString('ru-RU')}/с · MN9 хоботок ${cns.mn9Hz.toFixed(0)} Hz${taste}${slow}`
      : `спайковый CNS: ${cns.status}`);
  }

  bind(handlers) {
    document.querySelectorAll('#cam-seg button').forEach((b) => b.addEventListener('click', () => handlers.camera(b.dataset.cam)));
    document.querySelectorAll('[data-place]').forEach((b) => b.addEventListener('click', () => handlers.place(b.dataset.place)));
    $('btn-sugar').addEventListener('click', handlers.sugar);
    $('btn-shock').addEventListener('click', handlers.shock);
    $('btn-speed').addEventListener('click', () => { $('btn-speed').textContent = `×${handlers.speed()}`; });
    $('btn-arms').addEventListener('click', () => $('btn-arms').setAttribute('aria-pressed', String(handlers.arms())));
    $('btn-plumes').addEventListener('click', () => $('btn-plumes').setAttribute('aria-pressed', String(handlers.plumes())));
    $('btn-wind').addEventListener('click', handlers.wind);
    $('btn-reset').addEventListener('click', handlers.reset);
    const about = $('about'), aboutBtn = $('btn-about');
    const toggleAbout = (open) => { about.hidden = !open; aboutBtn.setAttribute('aria-expanded', String(open)); };
    aboutBtn.addEventListener('click', () => toggleAbout(about.hidden));
    $('btn-about-close').addEventListener('click', () => toggleAbout(false));
    window.addEventListener('keydown', (e) => {
      if (e.target.closest?.('input, textarea')) return;
      const cams = { 1: 'follow', 2: 'xray', 3: 'brain', 4: 'map' };
      if (cams[e.key]) handlers.camera(cams[e.key]);
      if (e.key === 'Escape') { handlers.place(null); toggleAbout(false); }
    });
  }

  setCamera(mode) {
    document.querySelectorAll('#cam-seg button').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.cam === mode)));
    document.body.className = document.body.className.replace(/mode-\w+/g, '').trim() + ` mode-${mode}`;
    this.el.legend.hidden = !(mode === 'xray' || mode === 'brain');
  }

  setPlacing(kind) {
    document.querySelectorAll('[data-place]').forEach((b) => b.classList.toggle('active', b.dataset.place === kind));
    document.body.classList.toggle('placing', !!kind);
  }

  buildLegend(groups, onToggle) {
    this.el.legend.innerHTML = '';
    for (const g of groups) {
      const s = GROUP_STYLE[g.name];
      const b = document.createElement('button');
      b.setAttribute('aria-pressed', 'true');
      b.innerHTML = `<span class="swatch" style="background:#${s.color.toString(16).padStart(6, '0')}"></span>${s.label} <span class="mono dim">${g.count}</span>`;
      b.addEventListener('click', () => {
        const on = b.getAttribute('aria-pressed') !== 'true';
        b.setAttribute('aria-pressed', String(on));
        onToggle(g.name, on);
      });
      this.el.legend.appendChild(b);
    }
  }

  tooltip(info, x, y) {
    const t = this.el.tooltip;
    if (!info) { t.hidden = true; return; }
    t.hidden = false;
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
    t.innerHTML = info;
  }

  placePip(rect) {
    const p = this.el.pip;
    p.hidden = !rect;
    if (rect) { p.style.left = `${rect.x + 8}px`; p.style.top = `${rect.y + 6}px`; }
  }

  // called ~10 times per second
  update(world, readout) {
    const f = world.fly;
    const [text, color] = STATE_TEXT[f.state] ?? STATE_TEXT.fly;
    this.el.state.textContent = text;
    this.el.dot.style.background = color;
    this.el.dot.style.color = color;
    const m = Math.floor(world.t / 60), s = Math.floor(world.t % 60);
    this.el.clock.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    this.el.energy.style.width = `${(f.energy * 100).toFixed(0)}%`;

    for (const r of this.memRows) {
      const v = world.memory[r.k];
      const d = Math.max(-1, Math.min(1, v.drive));
      r.bar.style.left = d >= 0 ? '50%' : `${50 + 50 * d}%`;
      r.bar.style.width = `${Math.abs(d) * 50}%`;
      r.bar.style.background = d >= 0 ? 'var(--app)' : 'var(--av)';
      r.val.textContent = signed(v.drive);
      r.split.textContent = `врождённое ${signed(W_INNATE * v.innate)} · выучено ${signed(W_MB * v.mb)}`;
    }

    const pct = (x) => `${(x * 100).toFixed(1)}%`;
    this.el.kc.textContent = pct(readout.kcFrac);
    this.el.kcBar.style.width = `${Math.min(100, readout.kcFrac * 500)}%`;
    this.el.app.textContent = readout.approach.toFixed(2);
    this.el.appBar.style.width = `${Math.min(100, readout.approach * 60)}%`;
    this.el.av.textContent = readout.avoid.toFixed(2);
    this.el.avBar.style.width = `${Math.min(100, readout.avoid * 60)}%`;
    this.el.pam.textContent = readout.pam.toFixed(2);
    this.el.pamBar.style.width = `${Math.min(100, readout.pam * 100)}%`;
    this.el.ppl.textContent = readout.ppl.toFixed(2);
    this.el.pplBar.style.width = `${Math.min(100, readout.ppl * 100)}%`;
    const dr = world.sense.drive;
    this.el.drive.textContent = world.sense.intensity < 0.03 ? 'нет запаха — ищет' : dr > 0.1 ? `тянет к запаху ${signed(dr)}` : dr < -0.1 ? `уходит от запаха ${signed(dr)}` : `сомневается ${signed(dr)}`;

    this.el.eaten.textContent = world.stats.eaten;
    this.el.trapped.textContent = world.stats.trapped;
    this.el.strikes.textContent = `${world.stats.strikes}`;

    const ev = world.events;
    const last = ev.length ? ev[ev.length - 1] : null;
    if (last && last !== this.lastEvent) {
      this.lastEvent = last;
      this.el.log.innerHTML = ev.slice(-7).reverse().map((e) =>
        `<li class="${e.type}"><time>${fmt(e.t)}</time><span>${e.text}</span></li>`).join('');
    }

    // learning chart samples every 2 s of simulated time
    if (!this.history.length || world.t - this.history[this.history.length - 1].t >= 2) {
      this.history.push({ t: world.t, fruit: world.memory.fruit.drive, sundew: world.memory.sundew.drive, frog: world.memory.frog.drive, mimic: world.memory.mimicry });
      if (this.history.length > 900) this.history.shift();
      this.drawChart(world.armsRace);
    }
  }

  resetHistory() {
    this.history = [];
  }

  drawChart(arms) {
    const c = this.el.chart, g = c.getContext('2d');
    const W = c.width, H = c.height, padL = 34, padB = 22, padT = 8, padR = 8;
    g.clearRect(0, 0, W, H);
    const h = this.history;
    const t0 = h[0]?.t ?? 0, t1 = Math.max(t0 + 60, h[h.length - 1]?.t ?? 60);
    const x = (t) => padL + ((t - t0) / (t1 - t0)) * (W - padL - padR);
    const y = (v) => padT + (1 - (Math.max(-1.5, Math.min(1.5, v)) + 1.5) / 3) * (H - padT - padB);
    g.font = '18px "IBM Plex Mono", monospace';
    g.fillStyle = '#5a6961';
    g.strokeStyle = 'rgba(160,196,176,0.12)';
    g.lineWidth = 1;
    for (const v of [-1, 0, 1]) {
      g.beginPath(); g.moveTo(padL, y(v)); g.lineTo(W - padR, y(v)); g.stroke();
      g.fillText(v > 0 ? '+1' : String(v), 0, y(v) + 6);
    }
    g.fillText('0', padL - 4, H - 2);
    const mins = (t1 - t0) / 60;
    g.textAlign = 'right';
    g.fillText(`${mins.toFixed(mins < 10 ? 1 : 0)} мин`, W - padR, H - 2);
    g.textAlign = 'left';
    const line = (key, color, dash) => {
      g.strokeStyle = color; g.lineWidth = 3; g.setLineDash(dash ?? []);
      g.beginPath();
      h.forEach((p, i) => (i ? g.lineTo(x(p.t), y(p[key])) : g.moveTo(x(p.t), y(p[key]))));
      g.stroke();
      const e = h[h.length - 1];
      if (e) { g.fillStyle = color; g.beginPath(); g.arc(x(e.t), y(e[key]), 5, 0, Math.PI * 2); g.fill(); }
    };
    if (arms) line('mimic', '#8a9c92', [6, 5]);
    line('frog', '#7bd88f');
    line('sundew', '#ff4f8b');
    line('fruit', '#ffb347');
    g.setLineDash([]);
  }
}

function signed(v) {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
}
function fmt(t) {
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
}
