// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
  var WB = window.__WB_API || {};
  var STATE = WB.STATE || {};
  var DATA = WB.DATA || {};
  const PHASE_T_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  const PHASE_S_GRID = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576
  ];

  (function mountPhaseDom() {
    const railSlot = document.getElementById('wb-extra-rail-slot');
    if (railSlot) {
      railSlot.innerHTML =
        '<div class="wb-sec-title">4. Trace Calibration Dropzone</div>' +
        '<div class="wb-knob-group" id="wb-cal-dropzone" ' +
        'ondragover="event.preventDefault()" ' +
        'ondrop="wbOnDropCalibration(event)">' +
        '<div style="font-size:10px;color:var(--muted);">' +
        'Drop <code>advisor.measurements.v1</code> or Chrome trace ' +
        '(<code>.json</code> / <code>.json.gz</code>) to upgrade ' +
        '<code>ASSUMED</code> priors to <code>MEASURED</code>.' +
        '</div>' +
        '<input id="cal-file-input" type="file" ' +
        'accept=".json,.gz,application/json,application/gzip" ' +
        'onchange="wbOnPickCalibration(event)" ' +
        'style="font-size:10px;width:100%;"/>' +
        '<div id="wb-cal-status" style="font-size:10px;color:var(--muted);">' +
        'Status: uncalibrated hardware priors</div></div>';
    }
    const stageSlot = document.getElementById('wb-extra-stage-slot');
    if (stageSlot) {
      stageSlot.innerHTML =
        '<div id="wb-stage-phase" ' +
        'style="display:none;flex-direction:column;gap:14px;">' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Continuous 2D Winning-Scope Phase Map ' +
        '(Active Tokens &times; Context)</h2>' +
        '<span class="kpi-sub">Click any (T, Context) cell to inspect ' +
        'in Arena</span></div><div id="wb-phase-map-svg"></div></div>' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Advisory Token-Bucket Schedule JSON (Offline Export)</h2>' +
        '<button class="wb-mini-btn" id="btn-copy-bucket-schedule" ' +
        'type="button" onclick="wbCopyBucketSchedule()">' +
        'Copy Bucket Schedule JSON</button></div>' +
        '<div class="kpi-sub" style="margin-bottom:6px;">' +
        'Schema <code>advisor.bucket_schedule.v1</code> ' +
        '(advisory offline schedule; unwired to runtime runner dispatch).' +
        '</div><pre class="wb-formula" id="wb-bucket-schedule-pre" ' +
        'style="max-height:200px;overflow-y:auto;margin:0;"></pre>' +
        '</div></div>';
    }
  })();

  window.wbFocusedTopicCard = function(wlA, evalA, evalB, lblB, topic, sc) {
    if (!topic || !sc) return '';
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    if (topic.kind === 'vmem') {
      const spK = (evalA.vmemParts.spineBytes / 1024).toFixed(1);
      const vmap = {
        dense: ['Dense Weight Ring', sc.dense_stage_mib,
          `slots=${sc.selected_nbd} * 2 MiB (adaptive ring)`],
        exp: ['Routed Expert Ring', sc.expert_stage_mib,
          `slots=${sc.selected_nbe} * 1 MiB (adaptive ring)`],
        scr: ['Static Op Scratch', sc.op_scratch_mib,
          'sequential attention / linear / FFN tile scratch'],
        pad: ['Arena Alignment Reserve', sc.pad_mib,
          '2.0 MiB on-chip arena alignment reserve'],
        spn: ['Dynamic [T, H] Residual Spine', sc.spine_dyn_mib,
          `T=${evalA.activeTokens} tok * ${spK} KiB/tok`],
        tok: ['Dynamic Per-Token Op Scratch', sc.op_dyn_mib,
          `alias=${wlA.vmemAlias ? 'enabled (max)' : 'disabled (sum)'} (#ctl-alias)`]
      };
      const info = vmap[topic.sub] || ['VMEM Region', sc.vmem_static_mib, sc.scope];
      const pct = ((info[1] / Math.max(1, evalA.vmemBudgetMib)) * 100).toFixed(1);
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Selected VMEM: ${esc(info[0])}` +
        `</strong><span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">${info[1].toFixed(2)} MiB (${pct}% of ` +
        `${evalA.vmemBudgetMib.toFixed(1)} MiB budget)\n${esc(info[2])}</div></div>`;
    }
    if (topic.kind === 'waterfall') {
      const perOp = evalA.scopes.find(s => s.scope === 'per_op');
      const scB = evalB ? evalB.winner : (perOp || evalA.runnerUp);
      const cmpLbl = evalB ? `${lblB} (${scB.scope})` : `Baseline (${scB.scope})`;
      const idx = parseInt(topic.sub || '0', 10) || 0;
      const envA = Math.max(sc.compute_us, sc.hbm_us);
      const envB = Math.max(scB.compute_us, scB.hbm_us);
      const fA = sc.scope === 'whole_step' || sc.scope === 'layer_group';
      const fB = scB.scope === 'whole_step' || scB.scope === 'layer_group';
      const cA = fA ? Math.max(0, sc.comm_us - envA) : sc.comm_us;
      const cB = fB ? Math.max(0, scB.comm_us - envB) : scB.comm_us;
      const rows = [
        ['Compute/HBM Roofline Core', envA, envB, 'max(t_comp, t_hbm) overlap floor'],
        ['Exposed Collective Sync & Bounce', cA, cB,
          'Exposed ICI/DCN sync beyond compute/HBM overlap (#ctl-coll-hop)'],
        ['Cold Pipeline Fill/Drain', sc.fill_drain_us, scB.fill_drain_us,
          'Cold HBM slot fill + synchronous stage barriers'],
        ['Multi-Pass HBM Weight Re-stream', sc.repass_spill_us, scB.repass_spill_us,
          `n_splits=${sc.n_splits} vs ${scB.n_splits} (knob: #ctl-vmem)`],
        ['Dispatch / Boundary Bubble',
          sc.ops_per_step * (evalA.launchGapPriorUs ?? evalA.launchGapLoUs),
          scB.ops_per_step * (evalA.launchGapPriorUs ?? evalA.launchGapLoUs),
          `ops=${sc.ops_per_step} vs ${scB.ops_per_step} * gap (#ctl-launch-gap)`]
      ];
      const r = rows[idx] || rows[0];
      const d = r[1] - r[2], sgn = d >= 0 ? '+' : '';
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Waterfall: ${esc(r[0])}</strong>` +
        `<span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">[A] ${sc.scope}: ${r[1].toFixed(2)} us\n` +
        `[B] ${esc(cmpLbl)}: ${r[2].toFixed(2)} us\n` +
        `Delta ([A] - [B]): ${sgn}${d.toFixed(2)} us\n${esc(r[3])}</div></div>`;
    }
    return '';
  };

  function resolveEpDpAndRowFloor(wlBase) {
    const ENG = window.__MK_ADVISOR_ENGINE;
    const tp = Math.max(1, wlBase.tp || 1), dcp = Math.max(1, wlBase.dcp || 1);
    const ep = Math.max(1, wlBase.ep || 1);
    const effTp = tp === 1 ? Math.max(1, dcp) : tp;
    const epDp = Math.max(1, Math.floor(ep / ENG.gcd(ep, effTp)));
    const rFloor = ENG.rowFloor(ENG.PROF.expert_dtype);
    return { epDp, rFloor };
  }

  function buildLiveBucketSchedule(wlBase) {
    const intervals = [];
    let cur = null;
    const { epDp, rFloor } = resolveEpDpAndRowFloor(wlBase);
    const dp = Math.max(1, wlBase.dp || 1);
    for (const tVal of PHASE_T_GRID) {
      const wl = Object.assign({}, wlBase, {
        phase: 'decode', batch: tVal * dp, seqTokens: 1,
        moeMode: (tVal * epDp >= rFloor) ? 'grouped_on_chip' : 'masked_on_chip'
      });
      const res = window.__WB_API.evaluateWorkload(wl);
      const win = res.winner;
      if (cur && cur.recommended_scope === win.scope &&
          cur.recommended_moe_mode === res.moeModeRec) {
        cur.max_active_tokens = tVal;
        cur.step_bound_us = win.step_bound_us;
      } else {
        cur = {
          phase: 'decode',
          layout: document.getElementById('lbl-layout').textContent,
          min_active_tokens: tVal,
          max_active_tokens: tVal,
          effective_context: wlBase.context,
          recommended_scope: win.scope,
          recommended_moe_mode: res.moeModeRec,
          max_vmem_tokens: win.t_vmem_max,
          num_vmem_splits: win.n_splits,
          step_bound_us: win.step_bound_us
        };
        intervals.push(cur);
      }
    }
    return {
      schema: 'advisor.bucket_schedule.v1',
      consumer_status: 'advisory_offline_v1',
      model: DATA.model, chip: DATA.chip,
      default_layout: document.getElementById('lbl-layout').textContent,
      calibrated: STATE.calibrated,
      buckets: intervals
    };
  }

  window.wbCalNotes = function(sc) {
    const esc = (window.__WB_API && window.__WB_API.escH) || String;
    const pTag = t => `<span class="prov-tag prov-tag-${t}">${t}</span>`;
    const line = STATE.calibrated
      ? `${pTag('MEASURED')} Source: ${esc(STATE.calSource)}`
      : `${pTag('ASSUMED')} Settle: calibrate --trace &lt;p&gt;`;
    const ratio = STATE.calStepTimeUs
      ? (sc.step_bound_us / Math.max(0.1, STATE.calStepTimeUs) * 100).toFixed(1) : '';
    const anc = STATE.calStepTimeUs !== null
      ? '<div class="wb-deriv-box"><strong>Measured Step:</strong> ' +
        `<span class="tabular">${STATE.calStepTimeUs.toFixed(1)} us</span> ` +
        `(${ratio}%)</div>`
      : '';
    return { line, anc };
  };

  window.wbSelectPhaseCell = function(tVal, sVal) {
    const wlCur = (WB.currentWorkloadA && WB.currentWorkloadA()) || {};
    const dp = Math.max(1, wlCur.dp || 1);
    const bVal = tVal * dp, bEl = document.getElementById('ctl-batch');
    STATE.customDirty = true;
    document.getElementById('ctl-phase').value = 'decode';
    if (bEl) {
      bEl.max = String(Math.max(1024, bVal));
      bEl.value = String(bVal);
    }
    document.getElementById('ctl-seq-tokens').value = '1';
    document.getElementById('ctl-context').value = String(sVal);
    window.wbSetStageTab('arena');
    window.wbRenderAll();
  };

  function fmtPhaseCtx(sVal) {
    return sVal >= 1048576 ? (sVal / 1048576) + 'M' : (sVal / 1024) + 'K';
  }

  window.wbRenderPhaseExtension = function(wlA, evalA) {
    const mapEl = document.getElementById('wb-phase-map-svg');
    const preEl = document.getElementById('wb-bucket-schedule-pre');
    if (!mapEl || !preEl) return;
    const colors = window.__WB_API.SCOPE_COLORS || {};
    const W = 760, padL = 72, padT = 28, cellW = 64, cellH = 26;
    const actT = evalA ? evalA.activeTokens : wlA.batch;
    const { epDp, rFloor } = resolveEpDpAndRowFloor(wlA);
    const dp = Math.max(1, wlA.dp || 1);
    let cells = '';
    PHASE_S_GRID.forEach((sVal, rIdx) => {
      const y = padT + (PHASE_S_GRID.length - 1 - rIdx) * cellH;
      const sLbl = fmtPhaseCtx(sVal);
      cells += `<text x="${padL - 8}" y="${y + 17}" text-anchor="end" ` +
        `fill="var(--muted)" font-size="11" class="tabular">${sLbl}</text>`;
      PHASE_T_GRID.forEach((tVal, cIdx) => {
        const x = padL + cIdx * cellW;
        const wl = Object.assign({}, wlA, {
          phase: 'decode', batch: tVal * dp, seqTokens: 1, context: sVal,
          moeMode: (tVal * epDp >= rFloor) ? 'grouped_on_chip' : 'masked_on_chip'
        });
        const ev = window.__WB_API.evaluateWorkload(wl);
        const isOom = Boolean(
          ev.hbmPoolOverflow || !ev.hbmFits ||
          (ev.invalidReasons && ev.invalidReasons.includes('hbm_pool_overflow'))
        );
        const col = isOom
          ? 'url(#wb-oom-hatch)'
          : (colors[ev.winner.scope] || '#64748B');
        const shortSc = isOom ? 'OOM'
          : (ev.winner.scope === 'whole_step' ? 'WS'
          : (ev.winner.scope === 'layer_group' ? 'LG'
          : (ev.winner.scope === 'single_layer' ? '1L' : 'OP')));
        const hypMark = (!isOom && ev.winner.support === 'hypothetical') ? '?' : '';
        const tieMark = (!isOom && ev.marginPct < 5.0) ? '*' : '';
        const isAct = tVal === actT && sVal === wlA.context;
        const stk = isAct ? ' stroke="#FFFFFF" stroke-width="2.2"' : '';
        const esc = (window.__WB_API && window.__WB_API.escH) || String;
        const hbmG = (ev.hbmUsedBytes / 1e9).toFixed(1);
        const wSc = ev.winner.scope;
        const wSup = ev.winner.support || 'baseline';
        const wFl = ev.winner.step_bound_us;
        const tip = esc(
          isOom
            ? `T=${tVal}, S=${sLbl} -> OOM > 80% HBM (${hbmG} GB)`
            : `T=${tVal}, S=${sLbl} -> ${wSc} [${wSup}] (${ev.moeModeRec}), floor=${wFl} us`
        );
        cells += `<g class="wb-clickable" ` +
          `onclick="wbSelectPhaseCell(${tVal},${sVal})">` +
          `<rect x="${x + 1}" y="${y + 1}" width="${cellW - 2}" ` +
          `height="${cellH - 2}" rx="3" fill="${col}" fill-opacity="0.88"${stk}>` +
          `<title>${tip}</title></rect>` +
          `<text x="${x + cellW / 2}" y="${y + 17}" text-anchor="middle" ` +
          `fill="#FFFFFF" font-size="10" font-weight="700">` +
          `${shortSc}${hypMark}${tieMark}</text></g>`;
      });
    });
    const yTick = padT + PHASE_S_GRID.length * cellH + 16;
    const xTicks = PHASE_T_GRID.map((tVal, cIdx) =>
      `<text x="${padL + cIdx * cellW + cellW / 2}" y="${yTick}" ` +
      `text-anchor="middle" fill="var(--muted)" font-size="11" ` +
      `class="tabular">${tVal}</text>`
    ).join('');
    const svgH = padT + PHASE_S_GRID.length * cellH + 38;
    const defs = '<defs><pattern id="wb-oom-hatch" width="8" height="8" ' +
      'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">' +
      '<rect width="8" height="8" fill="#7F1D1D"/>' +
      '<line x1="0" y1="0" x2="0" y2="8" stroke="#EF4444" stroke-width="3"/>' +
      '</pattern></defs>';
    mapEl.innerHTML = `<svg viewBox="0 0 ${W} ${svgH}" class="chart-svg" ` +
      `role="img" aria-label="2D Winning-Scope Phase Map">${defs}` +
      `<text x="${padL}" y="16" fill="var(--muted)" font-size="11">` +
      `Y: Context Length | X: Active Tokens (T) | * = &lt;5% margin | ` +
      `OOM = OOM &gt; 80% HBM</text>${cells}${xTicks}</svg>`;
    preEl.textContent = JSON.stringify(buildLiveBucketSchedule(wlA), null, 2);
  };

  function applyCalibrationObject(obj, fileName) {
    if (!obj || obj.schema !== 'advisor.measurements.v1') {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Rejected: expected advisor.measurements.v1';
      return;
    }
    if (obj.launch_gap_us && typeof obj.launch_gap_us.p50 === 'number') {
      const p50 = obj.launch_gap_us.p50;
      document.getElementById('ctl-launch-gap').value = String(p50);
      STATE.launchGapHiUs = typeof obj.launch_gap_us.p90 === 'number'
        ? Math.max(p50, obj.launch_gap_us.p90) : p50;
    }
    if (Array.isArray(obj.collectives) && obj.collectives.length > 0) {
      let minHop = null;
      for (const c of obj.collectives) {
        if (c && typeof c.latency_us === 'number') {
          const grp = Math.max(2, parseInt(c.group_size || 2, 10));
          const hopUs = c.latency_us / Math.max(1, Math.round(Math.log2(grp)));
          if (minHop === null || hopUs < minHop) minHop = hopUs;
        }
      }
      if (minHop !== null) {
        const cVal = Number(minHop.toFixed(3));
        document.getElementById('ctl-coll-hop').value = String(cVal);
        STATE.collHopHiUs = cVal;
      }
    }
    if (obj.ring_plan && typeof obj.ring_plan === 'object') {
      const rp = obj.ring_plan;
      if (rp.dense_slots) {
        document.getElementById('ctl-dma').value = String(rp.dense_slots);
      }
      if (rp.dense_slot_bytes) STATE.denseSlotBytes = rp.dense_slot_bytes;
      if (rp.expert_slot_bytes) STATE.expertSlotBytes = rp.expert_slot_bytes;
    }
    const stRaw = obj.step_time_us;
    STATE.calStepTimeUs = (stRaw && typeof stRaw.p50 === 'number') ? stRaw.p50
      : (typeof stRaw === 'number' ? stRaw : null);
    STATE.calibrated = true;
    STATE.hwDirty = true;
    STATE.calSource = (obj.source && obj.source.file)
      ? obj.source.file : fileName;
    const st = document.getElementById('wb-cal-status');
    if (st) st.textContent = 'Calibrated (MEASURED): ' + STATE.calSource;
    window.wbRenderAll();
  }
  function quantileSorted(arr, p) {
    if (!arr.length) return 0.0;
    const pos = p * (arr.length - 1), lo = Math.floor(pos);
    const hi = Math.min(arr.length - 1, lo + 1), frac = pos - lo;
    return arr[lo] * (1.0 - frac) + arr[hi] * frac;
  }
  function traceEventsToMeas(raw, fileName) {
    const evs = Array.isArray(raw) ? raw
      : (raw && Array.isArray(raw.traceEvents) ? raw.traceEvents : null);
    if (!evs) return raw;
    const valid = evs.filter(e => e && (!e.ph || e.ph === 'X') &&
      typeof e.ts === 'number' && typeof e.dur === 'number' && e.dur > 0);
    const isStep = e => /step/i.test(String(e.name || '')) ||
      /step/i.test(String(e.cat || ''));
    const isWrap = e => /^(jit[(]|pjit[(]|xla_computation[(]|module_main|__main__)/i
      .test(String(e.name || '').trim()) ||
      /xla modules/i.test(String(e.cat || ''));
    const steps = valid.filter(isStep).sort((a, b) => a.ts - b.ts);
    const spans = valid.filter(e => !isStep(e) && !isWrap(e))
      .sort((a, b) => a.ts - b.ts);
    const numLayers = (window.__MK_ADVISOR_ENGINE &&
      window.__MK_ADVISOR_ENGINE.PROF &&
      window.__MK_ADVISOR_ENGINE.PROF.num_layers) || 60;
    if (steps.length > 0) {
      const minRest = steps.length > 1
        ? Math.min(...steps.slice(1).map(s => s.dur)) : 0.0;
      const dropWarmup = steps.length >= 3 ||
        (steps.length === 2 && steps[0].dur > 2.0 * minRest);
      const steady = dropWarmup ? steps.slice(1) : steps;
      const perBnd = [], stepIdles = [], stepDurs = [];
      for (const st of steady) {
        const s0 = st.ts, s1 = st.ts + st.dur;
        const clipped = [];
        for (const sp of spans) {
          const a = Math.max(s0, sp.ts), b = Math.min(s1, sp.ts + sp.dur);
          if (b > a) clipped.push([a, b]);
        }
        clipped.sort((u, v) => u[0] - v[0]);
        let busy = 0.0;
        if (clipped.length > 0) {
          let curA = clipped[0][0], curB = clipped[0][1];
          for (let i = 1; i < clipped.length; i++) {
            if (clipped[i][0] <= curB) curB = Math.max(curB, clipped[i][1]);
            else { busy += (curB - curA); curA = clipped[i][0]; curB = clipped[i][1]; }
          }
          busy += (curB - curA);
        }
        const idle = Math.max(0.0, st.dur - busy);
        stepIdles.push(idle);
        stepDurs.push(st.dur);
        perBnd.push(idle / (2.0 * numLayers));
      }
      perBnd.sort((a, b) => a - b);
      stepIdles.sort((a, b) => a - b);
      stepDurs.sort((a, b) => a - b);
      return {
        schema: 'advisor.measurements.v1',
        source: { file: fileName },
        step_time_us: {
          p50: Number(quantileSorted(stepDurs, 0.5).toFixed(2)),
          p90: Number(quantileSorted(stepDurs, 0.9).toFixed(2)),
          n: steady.length
        },
        launch_gap_us: {
          p50: Number(quantileSorted(perBnd, 0.5).toFixed(4)),
          p90: Number(quantileSorted(perBnd, 0.9).toFixed(4)),
          n: steady.length,
          step_idle_gap_us: Number(quantileSorted(stepIdles, 0.5).toFixed(2)),
          num_layers_norm: numLayers,
          unit: 'us_per_sublayer_boundary'
        }
      };
    }
    const gaps = [];
    for (let i = 1; i < spans.length; i++) {
      const g = spans[i].ts - (spans[i - 1].ts + spans[i - 1].dur);
      if (g > 0 && g <= 500) gaps.push(g);
    }
    gaps.sort((a, b) => a - b);
    return {
      schema: 'advisor.measurements.v1',
      source: { file: fileName },
      launch_gap_us: {
        p50: Number(quantileSorted(gaps, 0.5).toFixed(4)),
        p90: Number(quantileSorted(gaps, 0.9).toFixed(4)),
        n: gaps.length
      }
    };
  }
  window.__WB_TRACE_TO_MEAS = traceEventsToMeas;
  async function readCalibrationFile(file) {
    if (!file) return;
    try {
      let text = '';
      if (file.name.endsWith('.gz') && typeof DecompressionStream !== 'undefined') {
        const ds = file.stream().pipeThrough(new DecompressionStream('gzip'));
        text = await new Response(ds).text();
      } else {
        text = await new Promise((res, rej) => {
          const reader = new FileReader();
          reader.onload = ev => res(String((ev.target && ev.target.result) || '{}'));
          reader.onerror = () => rej(new Error('read error'));
          reader.readAsText(file);
        });
      }
      const parsed = traceEventsToMeas(JSON.parse(text), file.name);
      applyCalibrationObject(parsed, file.name);
    } catch (_err) {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Invalid JSON or gzip trace file';
    }
  }
  window.wbOnPickCalibration = function(ev) {
    const files = ev.target && ev.target.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbOnDropCalibration = function(ev) {
    ev.preventDefault();
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbCopyBucketSchedule = function() {
    const preEl = document.getElementById('wb-bucket-schedule-pre');
    const btn = document.getElementById('btn-copy-bucket-schedule');
    if (preEl && navigator.clipboard) {
      navigator.clipboard.writeText(preEl.textContent || '');
    }
    if (btn) {
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy Bucket Schedule JSON'; }, 1200);
    }
  };
