/* Accent lab (experimental, fork only): how close a take's American English accent sits to an
   anchor. The server embeds the take into a gender-invariant accent space (accent/space.py);
   this page reads the take's region profile against the anchor's, and places both among the
   reference speakers. Readout and calibration are explained in accent/build_index.py. */
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const SVG = 'http://www.w3.org/2000/svg';
const BASELINE = 'koenami-accent-baseline';
const RATE = 16000;

type Region = { id: string; label: string; n: number; centroid: number[] };
type Speaker = { id: string; region: string; state: string; city: string; age: number | null; z: number[] };
type Reference = {
	embedder: string;
	passage: string;
	source: string;
	regions: Region[];
	explained: number[];
	heldout_accuracy: number;
	temperature: number;
	pair_auc: { overlap: number; distance: number };
	calibration: { overlap_same: number[]; overlap_diff: number[] };
	speakers: Speaker[];
};
type Take = { z: number[]; seconds: number; at: number };
type Anchor = { kind: string; label: string; z: number[]; region?: string; speaker?: string };

let ref: Reference;
const takes: Take[] = [];

/* ---------------------------------------------------------------- accent maths */

function profile(z: number[]): number[] {
	// Softened by the temperature fitted on held-out speakers (accent/build_index.py).
	const d2 = ref.regions.map((r) => r.centroid.reduce((s, c, k) => s + (z[k] - c) ** 2, 0) / ref.temperature);
	const min = Math.min(...d2);
	const p = d2.map((d) => Math.exp(-0.5 * (d - min)));
	const sum = p.reduce((a, b) => a + b, 0);
	return p.map((v) => v / sum);
}

const overlap = (p: number[], q: number[]) => p.reduce((s, v, k) => s + Math.min(v, q[k]), 0);

function quantile(sorted: number[], q: number) {
	const i = (sorted.length - 1) * q,
		lo = Math.floor(i);
	return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo);
}

/* Share of a distribution below v; `table` is either raw values or 101 stored quantiles. */
function percentile(table: number[], v: number, stored = false) {
	if (stored) {
		const i = table.findIndex((x) => x > v);
		return i < 0 ? 100 : i;
	}
	return (100 * table.filter((x) => x < v).length) / Math.max(1, table.length);
}

const regionLabel = (id: string) => ref.regions.find((r) => r.id === id)?.label ?? 'Mixed / transitional';
const shortLabel = (id: string) => regionLabel(id).split(' (')[0];
const pct = (v: number) => `${Math.round(v * 100)} %`;

/* ---------------------------------------------------------------- audio */

async function toSamples(blob: Blob): Promise<Float32Array> {
	const ctx = new AudioContext();
	const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
	await ctx.close();
	const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * RATE), RATE);
	const src = offline.createBufferSource();
	src.buffer = decoded;
	src.connect(offline.destination);
	src.start();
	const x = (await offline.startRendering()).getChannelData(0);
	for (let i = 0; i < x.length; i++) x[i] = Math.max(-1, Math.min(1, x[i]));
	return x;
}

async function embed(x: Float32Array): Promise<Take> {
	const res = await fetch('/api/accent', { method: 'POST', body: x.buffer as ArrayBuffer });
	if (!res.ok) throw new Error(await res.text());
	const r = await res.json();
	return { z: r.z, seconds: r.seconds, at: Date.now() };
}

/* One recorder at a time; `onDone` receives the take's samples. */
let recorder: MediaRecorder | null = null;
async function toggleRecording(button: HTMLButtonElement, idle: string, onDone: (b: Blob) => void) {
	if (recorder) {
		recorder.stop();
		return;
	}
	const stream = await navigator.mediaDevices.getUserMedia({
		audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
	});
	const chunks: Blob[] = [];
	recorder = new MediaRecorder(stream);
	recorder.ondataavailable = (e) => chunks.push(e.data);
	recorder.onstop = () => {
		stream.getTracks().forEach((t) => t.stop());
		recorder = null;
		button.classList.remove('recording');
		button.textContent = idle;
		onDone(new Blob(chunks, { type: chunks[0]?.type }));
	};
	recorder.start();
	button.classList.add('recording');
	button.textContent = 'Stop';
}

/* ---------------------------------------------------------------- anchor */

function baseline(): Take | null {
	try {
		return JSON.parse(localStorage.getItem(BASELINE) as string);
	} catch {
		return null;
	}
}

function anchor(): Anchor | null {
	const kind = (document.querySelector('input[name=anchor]:checked') as HTMLInputElement).value;
	if (kind === 'own') {
		const b = baseline();
		return b && { kind, label: 'your everyday accent', z: b.z };
	}
	if (kind === 'region') {
		const r = ref.regions.find((r) => r.id === $<HTMLSelectElement>('region-select').value)!;
		return { kind, label: `the ${shortLabel(r.id)} average`, z: r.centroid, region: r.id };
	}
	const s = ref.speakers.find((s) => s.id === $<HTMLSelectElement>('speaker-select').value)!;
	return { kind, label: `the speaker from ${place(s)}`, z: s.z, region: s.region, speaker: s.id };
}

const title = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());
const place = (s: Speaker) => `${title(s.city)}, ${title(s.state)}`;

/* ---------------------------------------------------------------- drawing helpers */

function el<K extends keyof SVGElementTagNameMap>(parent: Element, tag: K, attrs: Record<string, string | number>) {
	const e = document.createElementNS(SVG, tag);
	for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
	parent.appendChild(e);
	return e;
}

function clear(svg: SVGSVGElement) {
	[...svg.children].forEach((c) => c.tagName !== 'title' && c.remove());
}

/* Drawing width from the chart's container: clientWidth on an <svg> is unreliable. */
const widthOf = (svg: Element) => Math.round(svg.parentElement!.getBoundingClientRect().width) || 700;

const diamond = (x: number, y: number, r: number) => `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`;

function legend(id: string, items: [string, string][]) {
	$(id).innerHTML = items
		.map(
			([shape, text]) =>
				`<span><svg viewBox="0 0 12 12">${shape}</svg>${text}</span>`
		)
		.join('');
}
const LEGEND_TAKE = '<circle cx="6" cy="6" r="5" fill="var(--take)"/>';
const LEGEND_ANCHOR = '<path d="M6,0.5L11.5,6L6,11.5L0.5,6Z" fill="var(--anchor)"/>';
const LEGEND_HL = '<circle cx="6" cy="6" r="3.5" fill="var(--region-hl)"/>';
const LEGEND_REF = '<circle cx="6" cy="6" r="3.5" fill="var(--ref-dot)"/>';

/* ---------------------------------------------------------------- readout */

function renderCloseness(take: Take, a: Anchor) {
	const pt = profile(take.z),
		pa = profile(a.z),
		v = overlap(pt, pa);
	$('overlap-value').textContent = v.toFixed(2);
	let same: number[], other: number[], sameName: string, sentence: string;
	if (a.region && a.region !== 'mixed') {
		// Real speakers against this very anchor: those from its region, and the rest.
		const rows = ref.speakers.filter((s) => s.region !== 'mixed' && s.id !== a.speaker);
		const ov = rows.map((s) => ({ s, o: overlap(profile(s.z), pa) }));
		same = ov.filter((r) => r.s.region === a.region).map((r) => r.o).sort((x, y) => x - y);
		other = ov.filter((r) => r.s.region !== a.region).map((r) => r.o).sort((x, y) => x - y);
		sameName = `${shortLabel(a.region)} speakers`;
		const p = Math.round(percentile(same, v));
		sentence =
			`Your take overlaps ${a.label} more than ${p} % of reference ${sameName} do. ` +
			`They reach a median of ${quantile(same, 0.5).toFixed(2)}; speakers from other regions, ` +
			`${quantile(other, 0.5).toFixed(2)}.`;
	} else {
		same = ref.calibration.overlap_same;
		other = ref.calibration.overlap_diff;
		sameName = 'same-region pairs';
		const p = percentile(same, v, true);
		sentence =
			`Your take overlaps ${a.label} more than ${p} % of pairs of different speakers from the same region do ` +
			`(median ${quantile(same, 0.5).toFixed(2)}; pairs from different regions, ${quantile(other, 0.5).toFixed(2)}). ` +
			`Two takes by one person usually overlap more than two people do, so a value near the pair median means ` +
			`the take reads like a different speaker from your region, not like you.`;
	}
	$('overlap-sentence').textContent = sentence;
	renderStrip(v, [
		[sameName, same],
		[a.region && a.region !== 'mixed' ? 'speakers from other regions' : 'different-region pairs', other]
	]);
}

function renderStrip(v: number, rows: [string, number[]][]) {
	const svg = $<SVGSVGElement & HTMLElement>('strip');
	clear(svg);
	const w = widthOf(svg),
		left = 170,
		right = 20,
		rowH = 26;
	const h = rows.length * rowH + 34;
	svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
	svg.setAttribute('height', String(h));
	const x = (o: number) => left + o * (w - left - right);
	rows.forEach(([name, values], i) => {
		const y = 8 + i * rowH;
		el(svg, 'text', { x: left - 10, y: y + 13, 'text-anchor': 'end' }).textContent = `${name} (middle half)`;
		el(svg, 'rect', {
			x: x(quantile(values, 0.25)),
			y,
			width: Math.max(2, x(quantile(values, 0.75)) - x(quantile(values, 0.25))),
			height: 18,
			rx: 4,
			class: i ? 'band-other' : 'band-same'
		});
		el(svg, 'line', { x1: x(quantile(values, 0.5)), x2: x(quantile(values, 0.5)), y1: y, y2: y + 18, class: 'axis', 'stroke-width': 2 });
	});
	const axisY = rows.length * rowH + 12;
	el(svg, 'line', { x1: x(0), x2: x(1), y1: axisY, y2: axisY, class: 'axis' });
	for (const t of [0, 0.25, 0.5, 0.75, 1])
		el(svg, 'text', { x: x(t), y: axisY + 14, 'text-anchor': 'middle' }).textContent = t.toFixed(2);
	el(svg, 'line', { x1: x(v), x2: x(v), y1: 2, y2: axisY, stroke: 'var(--take)', 'stroke-width': 2 });
	el(svg, 'circle', { cx: x(v), cy: 4, r: 5, class: 'take' });
}

function renderMap(a: Anchor | null) {
	const svg = $<SVGSVGElement & HTMLElement>('map');
	clear(svg);
	const w = widthOf(svg),
		h = Math.round(w * 0.62),
		pad = 34;
	svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
	svg.setAttribute('height', String(h));
	const pts = [...ref.speakers.map((s) => s.z), ...takes.map((t) => t.z), ...(a ? [a.z] : [])];
	const xs = pts.map((z) => z[0]),
		ys = pts.map((z) => z[1]);
	const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
	const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (w - 2 * pad);
	const sy = (v: number) => h - pad - ((v - y0) / (y1 - y0 || 1)) * (h - 2 * pad);

	// Axis meaning: the regions whose averages sit at either end of each axis.
	const ends = (k: number) => {
		const sorted = [...ref.regions].sort((p, q) => p.centroid[k] - q.centroid[k]);
		return [shortLabel(sorted[0].id), shortLabel(sorted[sorted.length - 1].id)];
	};
	const [xl, xr] = ends(0),
		[yb, yt] = ends(1);
	el(svg, 'line', { x1: pad, x2: w - pad, y1: h - pad + 8, y2: h - pad + 8, class: 'axis' });
	el(svg, 'text', { x: pad, y: h - 6 }).textContent = `← more ${xl}-like`;
	el(svg, 'text', { x: w - pad, y: h - 6, 'text-anchor': 'end' }).textContent = `more ${xr}-like →`;
	el(svg, 'text', { x: 4, y: pad - 14 }).textContent = `↑ more ${yt}-like`;
	el(svg, 'text', { x: 4, y: h - pad - 2 }).textContent = `↓ more ${yb}-like`;

	const dots: { x: number; y: number; s: Speaker; node: SVGCircleElement }[] = [];
	for (const s of ref.speakers) {
		const node = el(svg, 'circle', {
			cx: sx(s.z[0]),
			cy: sy(s.z[1]),
			r: 3.5,
			class: `ref${a?.region && s.region === a.region ? ' hl' : ''}`
		});
		dots.push({ x: sx(s.z[0]), y: sy(s.z[1]), s, node });
	}
	for (const r of ref.regions) {
		const cx = sx(r.centroid[0]),
			cy = sy(r.centroid[1]);
		el(svg, 'path', { d: `M${cx - 5},${cy}h10M${cx},${cy - 5}v10`, class: 'centroid' });
		el(svg, 'text', { x: cx + 7, y: cy - 6, class: 'label-strong' }).textContent = shortLabel(r.id);
	}
	if (a) {
		const last = takes[takes.length - 1];
		if (last)
			el(svg, 'line', { x1: sx(last.z[0]), y1: sy(last.z[1]), x2: sx(a.z[0]), y2: sy(a.z[1]), class: 'link' });
		el(svg, 'path', { d: diamond(sx(a.z[0]), sy(a.z[1]), 9), class: 'anchor' });
	}
	takes.forEach((t, i) =>
		el(svg, 'circle', {
			cx: sx(t.z[0]),
			cy: sy(t.z[1]),
			r: i === takes.length - 1 ? 7 : 5,
			class: `take${i === takes.length - 1 ? '' : ' past'}`
		})
	);

	// Hover: the nearest reference speaker within 12 px (a larger target than the 3.5 px dot).
	const tip = $('tooltip');
	let current: (typeof dots)[number] | null = null;
	svg.onmousemove = (e) => {
		const box = svg.getBoundingClientRect(),
			mx = ((e.clientX - box.left) / box.width) * w,
			my = ((e.clientY - box.top) / box.height) * h;
		let best = null,
			bd = 12 ** 2;
		for (const d of dots) {
			const dd = (d.x - mx) ** 2 + (d.y - my) ** 2;
			if (dd < bd) [best, bd] = [d, dd];
		}
		current?.node.classList.remove('hover');
		current = best;
		if (!best) {
			tip.hidden = true;
			svg.style.cursor = '';
			return;
		}
		best.node.classList.add('hover');
		svg.style.cursor = 'pointer';
		tip.hidden = false;
		tip.innerHTML = `<strong>${place(best.s)}</strong><br>${regionLabel(best.s.region)}${best.s.age ? ` · age ${best.s.age}` : ''}<br><small>Click to listen</small>`;
		tip.style.left = `${(best.x / w) * box.width + 12}px`;
		tip.style.top = `${(best.y / h) * box.height - 10}px`;
	};
	svg.onmouseleave = () => {
		tip.hidden = true;
		current?.node.classList.remove('hover');
	};
	svg.onclick = () => current && play(current.s.id);

	const share = Math.round(100 * (ref.explained[0] + ref.explained[1]));
	$('map-note').textContent =
		`Each grey dot is one reference speaker, placed by a model fitted without them. The two axes carry ${share} % ` +
		`of what separates the regions; the closeness score above uses all of it. + marks each region's average.`;
	legend('map-legend', [
		[LEGEND_TAKE, takes.length > 1 ? 'Your takes (latest largest)' : 'Your take'],
		[LEGEND_ANCHOR, 'Anchor'],
		...(a?.region && a.region !== 'mixed' ? [[LEGEND_HL, `${shortLabel(a.region)} speakers`] as [string, string]] : []),
		[LEGEND_REF, 'Other reference speakers']
	]);
}

function renderBars(take: Take, a: Anchor) {
	const svg = $<SVGSVGElement & HTMLElement>('bars');
	clear(svg);
	const pt = profile(take.z),
		pa = profile(a.z);
	const w = widthOf(svg),
		left = 190,
		right = 50,
		rowH = 30,
		barH = 10;
	const h = ref.regions.length * rowH + 10;
	svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
	svg.setAttribute('height', String(h));
	const max = Math.max(...pt, ...pa, 0.25);
	const x = (v: number) => (v / max) * (w - left - right);
	const top = pt.indexOf(Math.max(...pt));
	ref.regions.forEach((r, i) => {
		const y = 6 + i * rowH;
		el(svg, 'text', { x: left - 10, y: y + barH + 3, 'text-anchor': 'end' }).textContent = shortLabel(r.id);
		const bt = el(svg, 'rect', { x: left, y, width: Math.max(1, x(pt[i])), height: barH, rx: 3, class: 'bar-take' });
		const ba = el(svg, 'rect', { x: left, y: y + barH + 2, width: Math.max(1, x(pa[i])), height: barH, rx: 3, class: 'bar-anchor' });
		el(bt, 'title', {}).textContent = `${regionLabel(r.id)}: take ${pct(pt[i])}`;
		el(ba, 'title', {}).textContent = `${regionLabel(r.id)}: anchor ${pct(pa[i])}`;
		if (i === top)
			el(svg, 'text', { x: left + x(pt[i]) + 6, y: y + barH - 1, class: 'label-strong' }).textContent = pct(pt[i]);
	});
	legend('bars-legend', [
		[LEGEND_TAKE, 'Your take'],
		[LEGEND_ANCHOR, `Anchor (${a.label})`]
	]);
	$('profile-table').innerHTML =
		'<tr><th>Region</th><th>Speakers</th><th>Take</th><th>Anchor</th></tr>' +
		ref.regions
			.map((r, i) => `<tr><td>${regionLabel(r.id)}</td><td>${r.n}</td><td>${pct(pt[i])}</td><td>${pct(pa[i])}</td></tr>`)
			.join('');
}

function render() {
	const a = anchor();
	const take = takes[takes.length - 1];
	renderMap(a);
	if (!take) return;
	$('result').hidden = false;
	if (!a) {
		$('overlap-value').textContent = '–';
		$('overlap-sentence').textContent = 'Record your everyday accent first, or pick another anchor.';
		return;
	}
	renderCloseness(take, a);
	renderMap(a);
	renderBars(take, a);
}

/* ---------------------------------------------------------------- page */

let audio: HTMLAudioElement | null = null; // created on first use: the page is prerendered without a DOM
function play(id: string) {
	audio ??= new Audio();
	audio.src = `/api/accent/audio/${id}`;
	void audio.play();
}

async function process(blob: Blob, status: HTMLElement) {
	status.textContent = 'Measuring…';
	const x = await toSamples(blob);
	return embed(x);
}

export async function mountAccent() {
	function setTheme(value: string) {
		try {
			localStorage.setItem('voice-theme', value);
		} catch {}
		document.documentElement.dataset.theme = value;
		$('theme-button').querySelector('use')!.setAttribute('href', value === 'dark' ? '#i-sun' : '#i-moon');
	}
	$('theme-button').onclick = () => {
		setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
		render();
	};
	$('theme-button')
		.querySelector('use')!
		.setAttribute('href', document.documentElement.dataset.theme === 'dark' ? '#i-sun' : '#i-moon');

	const res = await fetch('/api/accent/reference');
	if (!res.ok) {
		$('caveat').textContent = 'The accent index is not available on this server (run accent/build_index.py).';
		return;
	}
	ref = await res.json();
	const labelled = ref.speakers.filter((s) => s.region !== 'mixed').length;
	$('caveat').textContent =
		`Experimental. ${ref.speakers.length} US-born speakers from the Speech Accent Archive, ${labelled} of them grouped into ` +
		`${ref.regions.length} regions by birthplace. On speakers it never saw, it names the right region ` +
		`${Math.round(ref.heldout_accuracy * 100)} % of the time (chance ${Math.round(100 / ref.regions.length)} %), and its ` +
		`overlap score only modestly separates same-region from different-region speakers (AUC ${ref.pair_auc.overlap.toFixed(2)}, ` +
		`where 0.5 is chance). Read it as a rough tendency, never a verdict on anyone's accent. Gender-predictive directions are ` +
		`removed so that changing how your voice reads for gender should not move you here; that was checked on the reference ` +
		`speakers, not yet on people in training. Embedder: ${ref.embedder}.`;
	$('passage').textContent = ref.passage;
	$('sources').innerHTML =
		`Reference recordings: ${ref.source} (<a href="https://accent.gmu.edu/">accent.gmu.edu</a>, ` +
		`<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>). Regions loosely follow the Atlas of ` +
		`North American English; a birthplace is only a proxy for where someone's accent formed. The mapping is editable in ` +
		`accent/regions.json.`;

	$<HTMLSelectElement>('region-select').innerHTML = ref.regions
		.map((r) => `<option value="${r.id}">${r.label} (${r.n} speakers)</option>`)
		.join('');
	$<HTMLSelectElement>('speaker-select').innerHTML = [...ref.speakers]
		.sort((p, q) => shortLabel(p.region).localeCompare(shortLabel(q.region)) || p.state.localeCompare(q.state))
		.map((s) => `<option value="${s.id}">${shortLabel(s.region)} · ${place(s)}${s.age ? `, ${s.age}` : ''}</option>`)
		.join('');

	const panes = [...document.querySelectorAll<HTMLElement>('.anchor-pane')];
	for (const input of document.querySelectorAll<HTMLInputElement>('input[name=anchor]'))
		input.onchange = () => {
			panes.forEach((p) => (p.hidden = p.dataset.kind !== input.value));
			render();
		};
	$('region-select').onchange = render;
	$('speaker-select').onchange = render;
	$('speaker-play').onclick = () => play($<HTMLSelectElement>('speaker-select').value);

	const showBaseline = () => {
		const b = baseline();
		$('baseline-status').textContent = b
			? `Baseline saved ${new Date(b.at).toLocaleString()} (${b.seconds} s).`
			: 'No baseline yet. Read the paragraph in the voice you use day to day.';
		$('baseline-clear').hidden = !b;
	};
	showBaseline();
	$('baseline-clear').onclick = () => {
		localStorage.removeItem(BASELINE);
		showBaseline();
		render();
	};
	const baselineButton = $<HTMLButtonElement>('baseline-record');
	baselineButton.onclick = () =>
		toggleRecording(baselineButton, 'Record my everyday accent', async (blob) => {
			try {
				const t = await process(blob, $('baseline-status'));
				localStorage.setItem(BASELINE, JSON.stringify(t));
				showBaseline();
				render();
			} catch (e) {
				$('baseline-status').textContent = String((e as Error).message);
			}
		});

	const takeButton = $<HTMLButtonElement>('take-record');
	const takeIdle = takeButton.innerHTML;
	const addTake = async (blob: Blob) => {
		try {
			takes.push(await process(blob, $('take-status')));
			if (takes.length > 6) takes.shift();
			$('take-status').textContent = `Take ${takes.length}: ${takes[takes.length - 1].seconds} s measured.`;
			render();
		} catch (e) {
			$('take-status').textContent = String((e as Error).message);
		}
	};
	takeButton.onclick = () =>
		toggleRecording(takeButton, 'Record a take', (b) => {
			takeButton.innerHTML = takeIdle;
			void addTake(b);
		});
	$<HTMLInputElement>('take-upload').onchange = (e) => {
		const f = (e.target as HTMLInputElement).files?.[0];
		if (f) void addTake(f);
	};
	document.addEventListener('keydown', (e) => {
		const t = e.target as HTMLElement;
		if (e.key.toLowerCase() === 'r' && !['INPUT', 'SELECT', 'TEXTAREA'].includes(t.tagName) && !e.metaKey && !e.ctrlKey)
			takeButton.click();
	});
	addEventListener('resize', render);
	render();
}
