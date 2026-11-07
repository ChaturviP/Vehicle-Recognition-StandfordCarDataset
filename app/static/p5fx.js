// Motion runtime for the Streamlit app: page effects, scroll reveals and the 3D hero.
// Loaded as a module from ui.hero(); every effect is progressive enhancement.
try { CSS.registerProperty({ name: '--p5p', syntax: '<number>', inherits: false, initialValue: '0' }); } catch {}
const W = window, D = document;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fine = matchMedia('(pointer: fine)').matches;
const scroller = () => D.querySelector('[data-testid="stMain"]') || D.scrollingElement;
let gsap = null;
const gsapReady = import('https://cdn.jsdelivr.net/npm/gsap@3.12.5/index.js').then(m => (gsap = m.gsap || m.default))
  .catch(() => null);

/* ---------- page-wide effects, installed once ---------- */
if (!W.__p5fx) {
  W.__p5fx = true;
  const bar = D.createElement('div'); bar.className = 'p5-progress'; D.body.appendChild(bar);

  // Scroll reveals for app content that React mounts and remounts on every rerun.
  const REVEAL = ['[data-testid="stFileUploader"]', '.st-key-examples [data-testid="stColumn"]',
    '.st-key-result [data-testid="stColumn"]', '.p5-sec', '.p5-secp', '.p5-branches tbody tr', '.p5-stages li',
    '.p5-card', '.p5-flow li', '.st-key-plots [data-testid="stImage"]', '.st-key-arch [data-testid="stImage"]',
    '.p5-empty'].join(',');
  const seen = new WeakSet(), counted = new WeakSet();
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.dataset.rv = '1'; io.unobserve(e.target);
    countUp(e.target);
  }), { rootMargin: '0px 0px -6% 0px' });
  const scan = () => {
    D.querySelectorAll(REVEAL).forEach(el => {
      if (seen.has(el)) return; seen.add(el);
      if (reduce) return;
      const sib = el.parentElement ? [...el.parentElement.children].indexOf(el) : 0;
      el.style.setProperty('--rd', `${Math.min(sib, 6) * 0.07}s`);
      el.dataset.rv = '0'; io.observe(el);
    });
    D.querySelectorAll('[data-to]').forEach(el => { if (!el.closest('[data-rv="0"]')) countUp(el); });
    D.querySelectorAll('.p5-stage:not([data-gl])').forEach(st => initHero(st));
  };
  let queued = false;
  new MutationObserver(() => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; scan(); }); } })
    .observe(D.body, { childList: true, subtree: true });
  scan();

  // Number count-ups: the final value is in the HTML; JS only animates towards it.
  function countUp(root) {
    const els = root.matches?.('[data-to]') ? [root] : [...(root.querySelectorAll?.('[data-to]') || [])];
    els.forEach(el => {
      if (counted.has(el) || reduce) return; counted.add(el);
      const to = parseFloat(el.dataset.to), dec = +(el.dataset.dec || 0), suf = el.dataset.suf || '';
      const fmt = v => v.toLocaleString('en', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suf;
      const o = { v: 0 }; el.textContent = fmt(0);
      gsapReady.then(g => {
        if (!g) { el.textContent = fmt(to); return; }
        g.to(o, { v: to, duration: 1.2, ease: 'power3.out', onUpdate: () => (el.textContent = fmt(o.v)) });
      });
    });
  }
  W.__p5count = countUp;

  // Scroll: progress bar and hero parallax (CSS variable read by the hero and the 3D scene).
  let ticking = false;
  D.addEventListener('scroll', () => {
    if (ticking) return; ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      const s = scroller(); if (!s) return;
      const max = s.scrollHeight - s.clientHeight;
      bar.style.transform = `scaleX(${max > 0 ? s.scrollTop / max : 0})`;
      const hero = D.getElementById('p5-hero');
      if (hero && !reduce) {
        const p = Math.min(Math.max(s.scrollTop / (hero.offsetHeight || 1), 0), 1);
        W.__p5heroP = p;
        const copy = hero.querySelector('.p5-hero-copy');
        if (copy) { copy.style.transform = `translate3d(0, ${-p * 70}px, 0)`; copy.style.opacity = String(1 - p * 0.85); }
      }
    });
  }, { capture: true, passive: true });

  // Pointer-following highlight on buttons and cards, and magnetic hero buttons.
  if (fine) {
    D.addEventListener('pointermove', e => {
      const t = e.target.closest?.('.stButton > button, .p5-btn, .st-key-examples [data-testid="stColumn"]');
      if (!t) return;
      const r = t.getBoundingClientRect();
      t.style.setProperty('--mx', `${e.clientX - r.left}px`); t.style.setProperty('--my', `${e.clientY - r.top}px`);
      if (t.classList.contains('p5-btn') && !reduce) {
        const dx = (e.clientX - r.left - r.width / 2) * 0.22, dy = (e.clientY - r.top - r.height / 2) * 0.3;
        gsap ? gsap.to(t, { x: dx, y: dy, duration: .4, ease: 'power3.out' }) : (t.style.translate = `${dx}px ${dy}px`);
      }
    }, { passive: true });
    D.addEventListener('pointerout', e => {
      const t = e.target.closest?.('.p5-btn');
      if (t && !t.contains(e.relatedTarget)) gsap ? gsap.to(t, { x: 0, y: 0, duration: .7, ease: 'elastic.out(1, .4)' }) : (t.style.translate = '');
    });
  }

  // Hero buttons: scroll to the uploader, or open a tab.
  D.addEventListener('click', e => {
    const a = e.target.closest?.('[data-act]'); if (!a) return;
    e.preventDefault();
    const act = a.dataset.act;
    if (act.startsWith('tab:')) {
      const tab = [...D.querySelectorAll('[data-testid="stTab"]')].find(b => b.textContent.trim() === act.slice(4));
      tab?.click();
    } else {
      const tab = D.querySelector('[data-testid="stTab"]'); if (tab && tab.getAttribute('aria-selected') !== 'true') tab.click();
    }
    const target = D.querySelector('[data-testid="stTabs"]');
    target?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  });
}

/* ---------- 3D hero: the example photo being scanned, with the real detection box ---------- */
// A module runs once per page, but Streamlit can remount the hero, so stages are picked up by
// the mutation observer above as they appear.
async function initHero(stage) {
  if (!stage || stage.dataset.gl) return;
  stage.dataset.gl = '1';
  let THREE;
  try { THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js'); } catch { return; }
  const canvas = D.createElement('canvas');
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' }); }
  catch { return; }
  renderer.setPixelRatio(Math.min(W.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const tex = await new THREE.TextureLoader().loadAsync(stage.dataset.img).catch(() => null);
  if (!tex) { renderer.dispose(); return; }
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  stage.prepend(canvas);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(30, 1, .1, 60);
  const aspect = tex.image.width / tex.image.height, PH = 1.9, PW = PH * aspect;
  const box = JSON.parse(stage.dataset.box);
  const U = {
    uTex: { value: tex }, uTime: { value: 0 }, uScan: { value: reduce ? 1.1 : -0.06 }, uSweep: { value: -1 },
    uTexel: { value: new THREE.Vector2(1 / tex.image.width, 1 / tex.image.height) },
    uAccent: { value: new THREE.Color('#5B8CFF') }, uOk: { value: new THREE.Color('#3DDC84') },
    uBox: { value: new THREE.Vector4(0.004, 0.004, 0.996, 0.996) }, uBoxA: { value: reduce ? 1 : 0 },
    uAspect: { value: aspect }, uFade: { value: reduce ? 1 : 0 },
  };
  if (reduce) U.uBox.value.set(box[0], box[1], box[2], box[3]);
  const vert = `
    uniform float uAspect; varying vec2 vUv;
    void main() {
      vUv = uv;
      vec3 p = position;
      float x = p.x / (${(PW / 2).toFixed(3)});
      p.z -= x * x * 0.22;                       // gently curved, like a visor
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }`;
  const frag = (mirror) => `
    uniform sampler2D uTex; uniform float uTime, uScan, uSweep, uBoxA, uAspect, uFade;
    uniform vec2 uTexel; uniform vec3 uAccent, uOk; uniform vec4 uBox; varying vec2 vUv;
    float lum(vec3 c) { return dot(c, vec3(.299, .587, .114)); }
    void main() {
      vec2 uv = vUv;
      vec3 c = texture2D(uTex, uv).rgb;
      vec2 o = uTexel * 1.6;
      float gx = lum(texture2D(uTex, uv + vec2(o.x, 0.)).rgb) - lum(texture2D(uTex, uv - vec2(o.x, 0.)).rgb);
      float gy = lum(texture2D(uTex, uv + vec2(0., o.y)).rgb) - lum(texture2D(uTex, uv - vec2(0., o.y)).rgb);
      float edge = smoothstep(.04, .22, length(vec2(gx, gy)));
      vec2 g = abs(fract(uv * vec2(28. * uAspect, 28.)) - .5);
      float grid = step(.47, max(g.x, g.y)) * .07;
      vec3 blueprint = vec3(.012, .018, .035) + uAccent * (edge * 1.25 + grid);
      float pending = smoothstep(uScan - .015, uScan + .015, uv.x);
      vec3 col = mix(c, blueprint, pending);
      float line = exp(-pow((uv.x - uScan) * 70., 2.)) + .7 * exp(-pow((uv.x - uSweep) * 110., 2.));
      col += uAccent * line * 1.3 + vec3(.7, .8, 1.) * line * .35;
      // detection brackets (drawn in image space so they follow the curvature)
      vec2 px = vec2(uTexel.x, uTexel.y) * 3.2;
      vec2 a = uBox.xy, b = uBox.zw; vec2 q = vec2(uv.x, 1. - uv.y);
      float arm = min(b.x - a.x, b.y - a.y) * .2;
      float inX = step(a.x - px.x, q.x) * step(q.x, b.x + px.x), inY = step(a.y - px.y, q.y) * step(q.y, b.y + px.y);
      float onV = (step(abs(q.x - a.x), px.x) + step(abs(q.x - b.x), px.x)) * inY;
      float onH = (step(abs(q.y - a.y), px.y) + step(abs(q.y - b.y), px.y)) * inX;
      float nearX = step(q.x, a.x + arm) + step(b.x - arm, q.x), nearY = step(q.y, a.y + arm) + step(b.y - arm, q.y);
      float br = clamp(onV * min(nearY, 1.) + onH * min(nearX, 1.), 0., 1.);
      float frame = clamp(onV + onH, 0., 1.) * .25;
      col = mix(col, uOk, max(br, frame) * uBoxA);
      vec2 e = uv * (1. - uv);
      float alpha = smoothstep(0., .012, e.x) * smoothstep(0., .02, e.y) * uFade;
      ${mirror ? 'alpha *= (1. - smoothstep(0., .42, uv.y)) * .2; col *= .7;' : ''}
      gl_FragColor = vec4(col, alpha);
      #include <colorspace_fragment>
    }`;
  const geo = new THREE.PlaneGeometry(PW, PH, 48, 1);
  const group = new THREE.Group(); scene.add(group);
  const photo = new THREE.Mesh(geo, new THREE.ShaderMaterial({ uniforms: U, vertexShader: vert, fragmentShader: frag(false), transparent: true }));
  const mirror = new THREE.Mesh(geo, new THREE.ShaderMaterial({ uniforms: U, vertexShader: vert, fragmentShader: frag(true), transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  mirror.scale.y = -1; mirror.position.y = -PH - 0.03;
  group.add(photo, mirror);

  // Dust in the headlight beam.
  const N = 340, pos = new Float32Array(N * 3), spd = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    pos[i * 3] = (Math.random() - .5) * PW * 1.7; pos[i * 3 + 1] = (Math.random() - .5) * PH * 2.2;
    pos[i * 3 + 2] = -2.6 + Math.random() * 3.4; spd[i] = .04 + Math.random() * .1;
  }
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const dot = D.createElement('canvas'); dot.width = dot.height = 32;
  const dc = dot.getContext('2d'); const gr = dc.createRadialGradient(16, 16, 0, 16, 16, 16);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); dc.fillStyle = gr; dc.fillRect(0, 0, 32, 32);
  const pts = new THREE.Points(pg, new THREE.PointsMaterial({ size: .045, map: new THREE.CanvasTexture(dot), color: 0x9bb8ff,
    transparent: true, opacity: .55, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(pts);

  const tag = stage.querySelector('.p5-tag3d');
  const v3 = new THREE.Vector3();
  function placeTag() {
    const b = U.uBox.value, x = (b.x - .5) * PW, y = (.5 - b.y) * PH, nx = x / (PW / 2);
    v3.set(x, y, -nx * nx * .22).applyMatrix4(photo.matrixWorld).project(cam);
    const w = stage.clientWidth, h = stage.clientHeight;
    tag.style.transform = `translate(${(v3.x * .5 + .5) * w}px, ${(-v3.y * .5 + .5) * h - 8}px) translateY(-100%)`;
  }

  function fit() {
    const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
    const t = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    cam.position.set(0, 0, Math.max((PW * 1.04) / (2 * t * cam.aspect), (PH * 1.42) / (2 * t)));
    group.position.y = PH * .19;
  }
  fit(); new ResizeObserver(() => { fit(); if (!running) frame(performance.now()); }).observe(stage);

  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
  if (fine && !reduce) stage.addEventListener('pointermove', e => {
    const r = stage.getBoundingClientRect();
    mouse.tx = (e.clientX - r.left) / r.width - .5; mouse.ty = (e.clientY - r.top) / r.height - .5;
  });
  stage.addEventListener('pointerleave', () => { mouse.tx = 0; mouse.ty = 0; });

  const ease = t => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);
  const lerp = (a, b, t) => a + (b - a) * t;
  const t0 = performance.now();
  let running = false, visible = true, raf = 0, last = t0;
  function frame(now) {
    if (!stage.isConnected) { cancelAnimationFrame(raf); renderer.dispose(); geo.dispose(); tex.dispose(); return; }
    const t = (now - t0) / 1000, dt = Math.min((now - last) / 1000, .05); last = now;
    if (!reduce) {
      U.uTime.value = t;
      U.uFade.value = ease(t / .9);
      U.uScan.value = lerp(-0.06, 1.08, ease((t - .35) / 2.0));
      const k = ease((t - 2.0) / .9);
      U.uBox.value.set(lerp(.004, box[0], k), lerp(.004, box[1], k), lerp(.996, box[2], k), lerp(.996, box[3], k));
      U.uBoxA.value = Math.min(Math.max((t - 1.9) / .4, 0), 1);
      const cyc = (t - 4) % 7; U.uSweep.value = t > 4 && cyc < 2.2 ? lerp(-.05, 1.05, cyc / 2.2) : -1;
      mouse.x += (mouse.tx - mouse.x) * .06; mouse.y += (mouse.ty - mouse.y) * .06;
      const sp = W.__p5heroP || 0;
      const intro = 1 - ease(t / 1.6);
      group.rotation.y = mouse.x * .32 + Math.sin(t * .35) * .05 - intro * .55;
      group.rotation.x = mouse.y * .16 + sp * .5;
      group.position.z = -intro * 1.2 - sp * .8;
      const p = pg.attributes.position.array;
      for (let i = 0; i < N; i++) { p[i * 3 + 1] += spd[i] * dt; if (p[i * 3 + 1] > PH * 1.1) p[i * 3 + 1] = -PH * 1.1; }
      pg.attributes.position.needsUpdate = true;
      pts.rotation.y = mouse.x * .12;
    }
    renderer.render(scene, cam);
    placeTag();
    if (tag && (reduce || t > 2.9)) tag.classList.add('on');
    if (running) raf = requestAnimationFrame(frame);
  }
  const start = () => { if (!running && !reduce && visible && !D.hidden) { running = true; last = performance.now(); raf = requestAnimationFrame(frame); } };
  const stop = () => { running = false; cancelAnimationFrame(raf); };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? start() : stop(); }).observe(stage);
  D.addEventListener('visibilitychange', () => (D.hidden ? stop() : start()));
  stage.classList.add('gl-on');
  frame(performance.now());
  start();
}
