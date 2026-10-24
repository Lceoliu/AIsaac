// Feedback: the game's sounds (Web Audio) and Isaac's thumbs up (the player's "Happy" animation, a
// strip from web/roomart.py). Sounds are fetched and decoded ahead of time and only play after the
// player has clicked something; the mute switch is remembered.
const MUTE = 'hr-mute';
let ctx = null;
const buffers = new Map();     // name -> Promise<AudioBuffer | null>
let muted = false;
try { muted = localStorage.getItem(MUTE) === '1'; } catch { /* storage unavailable */ }

export function isMuted() { return muted; }
export function setMuted(v) {
  muted = !!v;
  try { localStorage.setItem(MUTE, muted ? '1' : '0'); } catch { /* storage unavailable */ }
}

function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  return ctx;
}

// fetch and decode the sounds named in art.json's ui.sounds
export function loadSounds(ui) {
  const ac = audio();
  if (!ac || !ui || !ui.sounds) return;
  for (const [name, file] of Object.entries(ui.sounds)) {
    if (buffers.has(name)) continue;
    buffers.set(name, fetch('art/' + file).then((r) => r.arrayBuffer())
      .then((data) => new Promise((resolve, reject) => ac.decodeAudioData(data, resolve, reject)))
      .catch(() => null));
  }
}

export async function playSound(name, volume = 0.8) {
  if (muted || !buffers.has(name)) return;
  const ac = audio();
  if (ac.state === 'suspended') await ac.resume().catch(() => {});
  const buffer = await buffers.get(name);
  if (!buffer || muted) return;
  const src = ac.createBufferSource();
  const gain = ac.createGain();
  gain.gain.value = volume;
  src.buffer = buffer;
  src.connect(gain).connect(ac.destination);
  src.start();
}

// Isaac, one frame of the strip, as an element scaled by `scale` (pixels stay square)
export function isaacSprite(ui, scale = 2, frame = null) {
  const h = ui && ui.happy;
  const el = document.createElement('span');
  el.className = 'isaac';
  if (!h) return el;
  el.style.width = `${h.w * scale}px`;
  el.style.height = `${h.h * scale}px`;
  el.style.backgroundImage = `url(art/${h.file})`;
  el.style.backgroundSize = `${h.w * scale * h.delays.length}px ${h.h * scale}px`;
  setFrame(el, h, scale, frame === null ? h.thumb : frame);
  return el;
}
function setFrame(el, h, scale, i) {
  el.style.backgroundPosition = `${-i * h.w * scale}px 0`;
}

// play the thumbs up once on an element made by isaacSprite
export function animateIsaac(el, ui, scale = 2) {
  const h = ui && ui.happy;
  if (!h || matchMedia('(prefers-reduced-motion: reduce)').matches) return Promise.resolve();
  return new Promise((resolve) => {
    let i = 0;
    const step = () => {
      setFrame(el, h, scale, i);
      const wait = (h.delays[i] * 1000) / h.fps;
      i += 1;
      if (i < h.delays.length) setTimeout(step, wait);
      else setTimeout(() => { setFrame(el, h, scale, h.thumb); resolve(); }, wait);
    };
    step();
  });
}

// a thumbs up and "+100" popping up at (x, y) inside `host` (a positioned element), then fading
// a floor with no miss: the Perfection trinket (满分考卷, the "A+" paper) stamped on the map
export function stampPerfect(host, ui, text) {
  if (!ui || !ui.perfect) return;
  const box = document.createElement('div');
  box.className = 'stamp';
  box.appendChild(perfectIcon(ui, 5));
  const label = document.createElement('b');
  label.textContent = text;
  box.appendChild(label);
  host.appendChild(box);
  setTimeout(() => box.remove(), 2600);
}
export function perfectIcon(ui, scale) {
  const img = document.createElement('img');
  img.src = 'art/' + ui.perfect.file;
  img.alt = '满分考卷';
  img.width = ui.perfect.w * scale;
  img.height = ui.perfect.h * scale;
  img.className = 'pixel perfect';
  return img;
}

export function celebrate(host, ui, x, y, text, good = true) {
  const box = document.createElement('div');
  box.className = good ? 'cheer' : 'cheer miss';
  box.style.left = `${x}px`;
  box.style.top = `${y}px`;
  if (good) {
    const isaac = isaacSprite(ui, 3, 0);
    box.appendChild(isaac);
    animateIsaac(isaac, ui, 3);
  }
  const label = document.createElement('b');
  label.textContent = text;
  box.appendChild(label);
  host.appendChild(box);
  setTimeout(() => box.remove(), good ? 1700 : 1100);
}
