"use strict";

const $ = (id) => document.getElementById(id);
const NUM_FIELDS = ["age", "gestational_age", "num_pregnancies", "parity", "prenatal_checkups", "prev_cesarean_count"];
const SELECT_FIELDS = ["mode_of_delivery", "education_level"];
const CHECK_FIELDS = ["multiple_births", "hypertensive_disorders", "gestational_diabetes", "cardiac_disease", "liver_disease"];
const NAMES = {
  age: "Maternal age", gestational_age: "Gestational age", num_pregnancies: "Number of pregnancies",
  parity: "Previous births", prenatal_checkups: "Prenatal checkups", prev_cesarean_count: "Previous cesarean sections",
};
const LABEL_OVERRIDE = { mode_of_delivery: "Mode of delivery", macrosomia: "Macrosomia (4 kg or more)" };
const PLAIN_NUMBERS = ["age", "gestational_age", "parity", "num_pregnancies", "prenatal_checkups", "prev_cesarean_count", "education_level"];

const PRESETS = {
  typical: { age: 28, gestational_age: 39, num_pregnancies: 2, parity: 1, prenatal_checkups: 8, prev_cesarean_count: 0,
    mode_of_delivery: 1, education_level: 1, multiple_births: 0, hypertensive_disorders: 0, gestational_diabetes: 0,
    cardiac_disease: 0, liver_disease: 0 },
  higher: { age: 31, gestational_age: 37, num_pregnancies: 3, parity: 1, prenatal_checkups: 12, prev_cesarean_count: 0,
    mode_of_delivery: 1, education_level: 2, multiple_births: 1, hypertensive_disorders: 1, gestational_diabetes: 0,
    cardiac_disease: 0, liver_disease: 0 },
};

/* ---------- tiny DOM helper: builds nodes without innerHTML ---------- */
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") Object.assign(node.style, v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c);
  return node;
}
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

/* ---------- form behaviour ---------- */
function applyPreset(name) {
  const p = PRESETS[name];
  NUM_FIELDS.concat(SELECT_FIELDS).forEach((f) => { $(f).value = String(p[f]); });
  CHECK_FIELDS.forEach((f) => { $(f).checked = !!p[f]; });
  document.querySelector('input[name="macrosomia"][value="unknown"]').checked = true;
  $("us_mode").value = "none";
  updateUltrasoundBoxes();
  clearError();
}

function updateUltrasoundBoxes() {
  const mode = $("us_mode").value;
  $("us-image").hidden = mode !== "image";
  $("us-hc").hidden = mode !== "hc";
  $("us-note").hidden = mode === "none";
}

function showError(messages) {
  const box = $("form-error");
  box.replaceChildren(messages.length === 1 ? document.createTextNode(messages[0]) : h("ul", {}, messages.map((m) => h("li", { text: m }))));
  box.hidden = false;
}
function clearError() {
  $("form-error").hidden = true;
  document.querySelectorAll("[aria-invalid]").forEach((el) => el.removeAttribute("aria-invalid"));
}

function validate() {
  const errors = [];
  const bad = (id, msg) => { $(id).setAttribute("aria-invalid", "true"); errors.push(msg); };
  for (const f of NUM_FIELDS) {
    const el = $(f), raw = el.value.trim(), v = Number(raw);
    const lo = Number(el.min), hi = Number(el.max);
    if (raw === "" || Number.isNaN(v)) bad(f, `${NAMES[f]} is required.`);
    else if (v < lo || v > hi) bad(f, `${NAMES[f]} must be between ${lo} and ${hi}.`);
  }
  const num = Number($("num_pregnancies").value), par = Number($("parity").value), cs = Number($("prev_cesarean_count").value);
  if (!$("parity").hasAttribute("aria-invalid") && !$("num_pregnancies").hasAttribute("aria-invalid")) {
    if (par >= num) bad("parity", "Previous births must be fewer than the number of pregnancies (which includes this one).");
    else if (cs > par) bad("prev_cesarean_count", "Previous cesarean sections cannot exceed previous births.");
  }
  const mode = $("us_mode").value;
  if (mode === "image") {
    if (!$("image").files.length) bad("image", "Choose an ultrasound image, or switch the ultrasound option to \"Not used\".");
    const ps = Number($("pixel_size").value);
    if (!(ps > 0)) bad("pixel_size", "Enter the pixel size (mm per pixel) of the ultrasound image.");
  } else if (mode === "hc") {
    const hc = Number($("hc_mm").value);
    if (!($("hc_mm").value !== "" && hc > 20 && hc < 450)) bad("hc_mm", "Head circumference must be between 20 and 450 mm.");
  }
  return errors;
}

function collectPatient() {
  const p = {};
  NUM_FIELDS.forEach((f) => { p[f] = Number($(f).value); });
  SELECT_FIELDS.forEach((f) => { p[f] = Number($(f).value); });
  CHECK_FIELDS.forEach((f) => { p[f] = $(f).checked ? 1 : 0; });
  const m = document.querySelector('input[name="macrosomia"]:checked').value;
  if (m !== "unknown") p.macrosomia = Number(m);
  return p;
}

function describeApiError(detail) {
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) return detail.map((d) => (d.field ? `${d.field}: ${d.problem}` : d.msg || JSON.stringify(d)));
  return ["Something went wrong. Please check the inputs and try again."];
}

async function onSubmit(event) {
  event.preventDefault();
  clearError();
  const errors = validate();
  if (errors.length) { showError(errors); return; }

  const fd = new FormData();
  fd.append("patient", JSON.stringify(collectPatient()));
  fd.append("rho", $("rho").value);
  const mode = $("us_mode").value;
  if (mode === "image") {
    fd.append("image", $("image").files[0]);
    fd.append("pixel_size", $("pixel_size").value);
  } else if (mode === "hc") {
    fd.append("hc_mm", $("hc_mm").value);
  }

  const button = $("submit");
  button.disabled = true;
  button.textContent = mode === "image" ? "Measuring the head and estimating…" : "Estimating…";
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
    if (!res.ok) { showError(describeApiError(data && data.detail)); return; }
    renderResult(data);
  } catch (err) {
    showError(["Cannot reach the server. Is it running? (uvicorn backend.main:app)"]);
  } finally {
    button.disabled = false;
    button.textContent = "Estimate risk";
  }
}

/* ---------- rendering ---------- */
function riskCard(r) {
  const band = r.band, p = r.probability, info = r.band_info || {};
  const overall = info.overall_rate_in_holdout;
  const nodes = [
    h("p", { class: "sub", text: "Estimated risk of postpartum hemorrhage" }),
    h("div", { class: `pct band-${band}`, text: pct(p) }),
    h("span", { class: `chip band-${band}`, text: `${band} risk` }),
    h("p", { class: "sub", text: `For comparison: a patient with average characteristics has an estimated ${pct(r.average_patient_risk)}` + (overall ? `, and the overall PPH rate in the data was ${pct(overall)}.` : ".") }),
  ];
  if (info.cutoffs) {
    const c1 = info.cutoffs.low_to_moderate, c2 = info.cutoffs.moderate_to_high;
    const axisMax = Math.max(c2 * 1.8, p * 1.15);
    const seg = (w, color) => h("div", { class: "seg", style: { width: `${w * 100}%`, background: color } });
    nodes.push(
      h("div", { class: "scale", "aria-hidden": "true" },
        seg(c1 / axisMax, "var(--low)"), seg((c2 - c1) / axisMax, "var(--mod)"), seg(1 - c2 / axisMax, "var(--high)"),
        h("div", { class: "marker", style: { left: `${Math.min(p / axisMax, 1) * 100}%` } })),
      h("div", { class: "scale-labels" }, h("span", { text: "0%" }), h("span", { text: pct(axisMax, 0) })),
    );
    const obs = info.observed_rate_in_holdout || {};
    const row = (name, range) => h("tr", { class: name === band ? "here" : "" },
      h("td", { text: name }), h("td", { text: range }), h("td", { text: obs[name] !== undefined ? pct(obs[name]) : "–" }));
    nodes.push(h("table", { class: "band-table" },
      h("thead", {}, h("tr", {}, h("th", { text: "Band" }), h("th", { text: "Estimated risk" }), h("th", { text: "PPH actually seen" }))),
      h("tbody", {}, row("Low", `below ${pct(c1)}`), row("Moderate", `${pct(c1)} to ${pct(c2)}`), row("High", `${pct(c2)} or more`))));
    nodes.push(h("p", { class: "hint", text: "Bands are relative: Low is the lower-risk half of patients in the training data, Moderate the next 40%, High the top 10%. “PPH actually seen” is the observed rate in each band among held-out deliveries the model never saw during development." }));
  }
  return h("div", { class: "card risk" }, nodes);
}

function valueText(c, r) {
  const v = c.value;
  if (c.feature === "macrosomia") {
    const src = r.macrosomia_input.source;
    if (src === "ultrasound image" || src === "entered head circumference") return `${(v * 100).toFixed(0)}% est.`;
    if (v === 1) return "yes";
    if (v === 0) return "no";
    return `${(v * 100).toFixed(0)}% typical`;
  }
  if (c.feature === "mode_of_delivery") return v === 1 ? "vaginal" : "cesarean";
  if (PLAIN_NUMBERS.includes(c.feature)) return String(v);
  return v ? "yes" : "no";
}

function driversCard(r) {
  const top = r.contributions.slice(0, 8);
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.log_odds)), 0.0001);
  const rows = top.map((c) => {
    const w = (Math.abs(c.log_odds) / maxAbs) * 48, up = c.log_odds >= 0;
    return h("div", { class: "driver" },
      h("div", { class: "name" }, LABEL_OVERRIDE[c.feature] || c.label, " ", h("small", { text: valueText(c, r) })),
      h("div", { class: "track" }, h("div", { class: `bar ${up ? "up" : "down"}`, style: { left: `${up ? 50 : 50 - w}%`, width: `${w}%` } })),
      h("div", { class: "or", text: `×${c.odds_ratio_vs_average.toFixed(2)}` }));
  });
  return h("div", { class: "card" },
    h("h2", { text: "What drives this estimate" }),
    h("p", { class: "hint", text: "Compared with an average patient in the training data. The × number multiplies the odds of PPH. It shows association, not cause." }),
    h("div", { class: "drivers" }, rows),
    h("div", { class: "legend" }, h("span", {}, h("i", { style: { background: "var(--up)" } }), "raises risk"), h("span", {}, h("i", { style: { background: "var(--down)" } }), "lowers risk")));
}

function macrosomiaCard(r) {
  const used = r.macrosomia_input, im = r.imaging;
  const usedValue = used.value === 0 || used.value === 1 ? (used.value ? "yes" : "no") : `${(used.value * 100).toFixed(1)}% chance`;
  const nodes = [h("h2", { text: "Large-baby (macrosomia) input" })];
  if (im) {
    if (r.overlay_image) nodes.push(h("img", { src: `data:image/jpeg;base64,${r.overlay_image}`, alt: "Ultrasound with the fitted head outline" }));
    const fact = (value, label) => h("div", { class: "fact" }, h("b", { text: value }), h("span", { text: label }));
    nodes.push(h("div", { class: "facts" },
      fact(`${im.hc_mm.toFixed(0)} mm`, im.source === "ultrasound image" ? "Head circumference, measured by the model" : "Head circumference, entered"),
      fact(`${im.expected_hc_mm.toFixed(0)} mm`, `Typical at ${im.ga_used.toFixed(0)} weeks`),
      fact(`${im.z_score >= 0 ? "+" : ""}${im.z_score.toFixed(2)}`, `Head-size z-score (percentile ${im.percentile.toFixed(0)})`),
      fact(pct(im.p_macrosomia), "Estimated chance of a baby of 4 kg or more")));
    nodes.push(h("p", { class: "hint", text: `Based on the INTERGROWTH-21st head-size standard and the base rate in the training data, assuming a head-size / birth-weight correlation of ρ = ${im.rho}. Head size alone is a weak proxy for fetal weight.` }));
  }
  nodes.push(h("p", { class: "sub", text: `Used in the model: ${usedValue} (${used.source}).` }));
  return h("div", { class: "card imaging" }, nodes);
}

function renderResult(r) {
  const body = $("result-body");
  const parts = [riskCard(r), driversCard(r), macrosomiaCard(r)];
  if (r.warnings && r.warnings.length) {
    parts.push(h("div", { class: "alert warn", role: "status" }, h("ul", {}, r.warnings.map((w) => h("li", { text: w })))));
  }
  parts.push(h("p", { class: "disclaimer", text: r.disclaimer }));
  body.replaceChildren(...parts);
  body.hidden = false;
  $("empty-state").hidden = true;
  if (window.innerWidth <= 880) body.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- wiring ---------- */
document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => applyPreset(b.dataset.preset)));
$("us_mode").addEventListener("change", updateUltrasoundBoxes);
$("rho").addEventListener("input", () => { $("rho-out").textContent = Number($("rho").value).toFixed(1); });
$("patient-form").addEventListener("submit", onSubmit);
updateUltrasoundBoxes();