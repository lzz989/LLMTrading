function qs(sel) {
  return document.querySelector(sel);
}

function qsa(sel) {
  return Array.from(document.querySelectorAll(sel));
}

function setHealth(ok, text) {
  const dot = qs("#healthDot");
  const t = qs("#healthText");
  dot.classList.remove("ok", "bad");
  dot.classList.add(ok ? "ok" : "bad");
  t.textContent = text;
}

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${txt}`);
  }
  return res.json();
}

async function fetchJsonSafe(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`fetch失败：${res.status} ${res.statusText}: ${txt}`);
  }
  const txt = (await res.text()) || "";
  try {
    return JSON.parse(txt);
  } catch {
    // 兼容旧输出：Python json.dumps 默认会吐 NaN/Infinity（非标准 JSON），浏览器 JSON.parse 直接炸。
    const fixed = txt
      .replace(/\bNaN\b/g, "null")
      .replace(/\bInfinity\b/g, "null")
      .replace(/\b-Infinity\b/g, "null");
    return JSON.parse(fixed);
  }
}

function toNumber(x, def = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : def;
}

function parseProgress(logTail) {
  if (!logTail) return null;
  const re = /\[(\d+)\s*\/\s*(\d+)\]/g;
  let m = null;
  let last = null;
  // eslint-disable-next-line no-cond-assign
  while ((m = re.exec(logTail)) !== null) {
    const done = Number(m[1]);
    const total = Number(m[2]);
    if (Number.isFinite(done) && Number.isFinite(total) && total > 0 && done >= 0) {
      last = { done, total };
    }
  }
  if (!last) return null;
  const pct = Math.max(0, Math.min(100, Math.round((last.done / last.total) * 100)));
  return { ...last, pct };
}

function setTextAll(selector, text) {
  qsa(selector).forEach((el) => {
    el.textContent = text;
  });
}

function setLogAll(text) {
  const t = text || "";
  if (qs("#jobLog")) qs("#jobLog").textContent = t;
  qsa("[data-job='log']").forEach((el) => {
    el.textContent = t;
  });
}

function setOutDirAll(outDir) {
  const href = outDir && outDir !== "-" ? outDir : "#";
  const text = outDir || "-";
  const a = qs("#jobOutDir");
  if (a) {
    a.textContent = text;
    a.href = href;
  }
  qsa("[data-job='outdir']").forEach((el) => {
    el.textContent = text;
    el.href = href;
  });
}

function setProgressAll(progress) {
  let text = "-";
  let pct = 0;
  if (progress && Number.isFinite(progress.pct)) {
    pct = progress.pct;
    text = `${progress.done}/${progress.total} (${pct}%)`;
  }

  if (qs("#jobProgressText")) qs("#jobProgressText").textContent = text;
  if (qs("#jobProgressBar")) qs("#jobProgressBar").style.width = `${pct}%`;
  qsa("[data-job='progressText']").forEach((el) => {
    el.textContent = text;
  });
  qsa("[data-job='progressBar']").forEach((el) => {
    el.style.width = `${pct}%`;
  });
}

function setJobInfo(job) {
  const jobId = job?.job_id || "-";
  const status = job?.status || "-";

  if (qs("#jobId")) qs("#jobId").textContent = jobId;
  setTextAll("[data-job='id']", jobId);

  if (qs("#jobStatus")) qs("#jobStatus").textContent = status;
  setTextAll("[data-job='status']", status);

  setOutDirAll(job?.out_dir || "-");
  setLogAll(job?.log_tail || "");

  let progress = null;
  if (status === "succeeded") {
    progress = { done: 1, total: 1, pct: 100 };
  } else if (status === "failed") {
    progress = parseProgress(job?.log_tail || "");
  } else if (status === "running") {
    progress = parseProgress(job?.log_tail || "");
  }
  setProgressAll(progress);
}

function renderImages(job) {
  const grid = qs("#resultGrid");
  const empty = qs("#resultEmpty");
  grid.innerHTML = "";

  const pngs = job?.artifacts?.png || [];
  if (!pngs.length) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  const base = job.out_dir;
  for (const rel of pngs) {
    const url = `${base}/${rel}`;
    const cap = rel.replace(/^([^/]+\/)?/, "");
    const box = document.createElement("div");
    box.className = "img-card";
    box.innerHTML = `
      <div class="cap">
        <span>${cap}</span>
        <a href="${url}" target="_blank" rel="noreferrer">打开</a>
      </div>
      <img src="${url}" alt="${cap}" loading="lazy" />
    `;
    grid.appendChild(box);
  }
}

function renderJson(result) {
  const pre = qs("#resultJson");
  if (!result?.json) {
    pre.textContent = "";
    return;
  }
  pre.textContent = pretty(result.json);
}

function findJsonBySuffix(result, suffix) {
  const js = result?.json || {};
  for (const [k, v] of Object.entries(js)) {
    if (k === suffix || k.endsWith(`/${suffix}`)) return v;
  }
  return null;
}

function renderInstitution(result) {
  const pre = qs("#institutionText");
  const empty = qs("#institutionEmpty");
  if (!pre) return;

  pre.textContent = "";
  if (empty) {
    empty.textContent = "如果你选择 method=all 或 institution，任务结束后会在这里显示“机构痕迹”的量化摘要。";
    empty.style.display = "block";
  }

  const inst = findJsonBySuffix(result, "institution.json");
  if (!inst || typeof inst !== "object") return;

  const sum = inst.summary || {};
  const state = String(sum.state || "unknown");
  const stateMap = {
    accumulation: "疑似吸筹",
    distribution: "疑似派发",
    neutral: "中性/看不出",
    unknown: "未知",
  };

  const lines = [];
  lines.push(`状态: ${stateMap[state] || state}`);
  if (sum.score !== undefined && sum.score !== null) lines.push(`得分: ${sum.score}`);
  if (sum.confidence !== undefined && sum.confidence !== null) {
    const conf = toNumber(sum.confidence, NaN);
    if (Number.isFinite(conf)) lines.push(`置信: ${(conf * 100).toFixed(0)}%`);
  }
  if (sum.last_date) lines.push(`截止: ${sum.last_date}`);

  const evidence = sum.evidence || [];
  if (Array.isArray(evidence) && evidence.length) {
    lines.push("");
    lines.push("证据:");
    for (const e of evidence.slice(0, 10)) {
      lines.push(`- ${String(e)}`);
    }
  }

  const ff = inst.fund_flow;
  if (ff && typeof ff === "object") {
    const main5 = ff.main_net_5d;
    const main20 = ff.main_net_20d;
    const pct5 = ff.main_pct_avg_5d;
    lines.push("");
    lines.push("资金流(摘要):");
    if (main5 !== undefined) lines.push(`- 主力净流入(5D): ${fmtAmount(main5)}`);
    if (main20 !== undefined) lines.push(`- 主力净流入(20D): ${fmtAmount(main20)}`);
    if (pct5 !== undefined && pct5 !== null) lines.push(`- 主力净占比均值(5D): ${toNumber(pct5, 0).toFixed(2)}%`);
    if (ff.last_date) lines.push(`- 资金流截止: ${ff.last_date}`);
  }

  pre.textContent = lines.join("\n");
  if (empty) empty.style.display = pre.textContent.trim() ? "none" : "block";
}

async function renderNarrative(job) {
  const pre = qs("#narrativeText");
  const empty = qs("#narrativeEmpty");
  if (!pre) return;

  pre.textContent = "";
  if (empty) {
    empty.textContent = "如果勾选了“综合解读”，任务结束后会在这里显示自然语言解读。";
    empty.style.display = "block";
  }
  if (!job?.out_dir) return;

  const base = job.out_dir;
  const tryFetch = async (url) => {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.text()) || "";
  };

  try {
    const summary = await tryFetch(`${base}/summary.md`);
    if (summary !== null) {
      pre.textContent = summary;
      if (empty) empty.style.display = summary.trim() ? "none" : "block";
      return;
    }
  } catch {
    // ignore
  }

  try {
    const err = await tryFetch(`${base}/summary_error.txt`);
    if (err !== null) {
      pre.textContent = `解读生成失败：\n${err}`;
      if (empty) empty.style.display = "none";
      return;
    }
  } catch {
    // ignore
  }

  if (empty) {
    empty.textContent = "本次任务未生成解读（大概率是你没勾选“综合解读”）。";
    empty.style.display = "block";
  }
}

let currentJobId = null;
let pollTimer = null;

async function pollJob(jobId) {
  currentJobId = jobId;
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setJobInfo(job);
      if (job.status === "succeeded") {
        renderImages(job);
        const result = await api(`/api/jobs/${jobId}/result`);
        renderJson(result);
        renderInstitution(result);
        await renderNarrative(job);
        if (job.type === "scan_etf") {
          await renderScanTables(jobId);
        } else if (job.type === "scan_stock") {
          await renderStockTables(jobId);
        }
        clearInterval(pollTimer);
        pollTimer = null;
      } else if (job.status === "failed") {
        renderImages(job);
        await renderNarrative(job);
        clearInterval(pollTimer);
        pollTimer = null;
      }
    } catch (e) {
      setHealth(false, `接口异常：${e.message}`);
    }
  };
  await tick();
  pollTimer = setInterval(tick, 1200);
}

async function startAnalyze(payload) {
  const form = qs("#analyzeForm");
  if (form) {
    if (payload.narrate === undefined && form.narrate) payload.narrate = Boolean(form.narrate.checked);
    if (payload.narrate_provider === undefined && form.narrate_provider) payload.narrate_provider = form.narrate_provider.value || "openai";
    if (payload.narrate_schools === undefined && form.narrate_schools) {
      payload.narrate_schools = form.narrate_schools.value.trim() || "chan,wyckoff,ichimoku,turtle,momentum";
    }
  }
  if (qs("#institutionText")) qs("#institutionText").textContent = "";
  if (qs("#institutionEmpty")) {
    qs("#institutionEmpty").textContent = "如果你选择 method=all 或 institution，任务结束后会在这里显示“机构痕迹”的量化摘要。";
    qs("#institutionEmpty").style.display = "block";
  }
  const res = await api("/api/jobs/analyze", { method: "POST", body: JSON.stringify(payload) });
  await pollJob(res.job_id);
}

async function startScan(payload) {
  const res = await api("/api/jobs/scan-etf", { method: "POST", body: JSON.stringify(payload) });
  await pollJob(res.job_id);
  await renderScanTables(res.job_id);
}

async function startScanStock(payload) {
  const res = await api("/api/jobs/scan-stock", { method: "POST", body: JSON.stringify(payload) });
  await pollJob(res.job_id);
  await renderStockTables(res.job_id);
}

function tabSwitch(name) {
  qsa(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  qsa(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
}

async function renderScanTables(jobId) {
  const job = await api(`/api/jobs/${jobId}`);
  if (job.status !== "succeeded") return;
  const out = job.out_dir;
  const trendUrl = `${out}/top_trend.json`;
  const swingUrl = `${out}/top_swing.json`;
  const bbbUrl = `${out}/top_bbb.json`;

  const trend = await fetchJsonSafe(trendUrl);
  const swing = await fetchJsonSafe(swingUrl);
  let bbb = null;
  try {
    bbb = await fetchJsonSafe(bbbUrl);
  } catch {
    bbb = null;
  }

  const trendItems = trend.items || [];
  const swingItems = swing.items || [];
  const bbbItems = bbb?.items || [];
  const bbbCfg = bbb?.bbb || {};

  const empty = qs("#scanEmpty");
  if (empty) {
    if (!bbbItems.length) {
      empty.textContent = "当前没有 BBB 能买候选（高级榜可能有，但别上头，宁愿空仓也别套山上）。";
      empty.style.display = "block";
    } else {
      empty.style.display = "none";
    }
  }
  renderBBBEtfMeta(bbb, bbbItems);
  if (qs("#bbbEtfTable")) qs("#bbbEtfTable").innerHTML = buildBBBEtfTable(bbbItems, bbbCfg);
  if (qs("#bbbEtfEmpty")) qs("#bbbEtfEmpty").style.display = bbbItems.length ? "none" : "block";
  if (qs("#trendTable")) qs("#trendTable").innerHTML = buildEtfTable(trendItems, "trend");
  if (qs("#swingTable")) qs("#swingTable").innerHTML = buildEtfTable(swingItems, "swing");

  // bind buttons
  qsa("[data-action='analyze-etf']").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const symbol = btn.getAttribute("data-symbol");
      const freq = qs("#scanForm [name='freq']").value || "weekly";
      tabSwitch("analyze");
      qs("#analyzeForm [name='asset']").value = "etf";
      qs("#analyzeForm [name='symbol']").value = symbol;
      qs("#analyzeForm [name='freq']").value = freq;
      qs("#analyzeForm [name='method']").value = "all";
      qs("#analyzeForm [name='window']").value = "500";
      await startAnalyze({
        asset: "etf",
        symbol,
        freq,
        method: "all",
        window: 500,
      });
    });
  });
}

async function renderStockTables(jobId) {
  const job = await api(`/api/jobs/${jobId}`);
  if (job.status !== "succeeded") return;
  const out = job.out_dir;
  const trendUrl = `${out}/top_trend.json`;
  const swingUrl = `${out}/top_swing.json`;
  const dipUrl = `${out}/top_dip.json`;

  const trend = await fetchJsonSafe(trendUrl);
  const swing = await fetchJsonSafe(swingUrl);
  let dip = null;
  try {
    dip = await fetchJsonSafe(dipUrl);
  } catch {
    dip = null;
  }
  const rankH = trend.rank_horizon_weeks || 8;

  const trendItems = trend.items || [];
  const swingItems = swing.items || [];
  const dipItems = dip?.items || [];

  const empty = qs("#stockEmpty");
  if (empty) {
    if (!trendItems.length && !swingItems.length && !dipItems.length) {
      empty.textContent = "当前没有候选（可能没有触发信号，或源站数据抽风）。";
      empty.style.display = "block";
    } else {
      empty.style.display = "none";
    }
  }
  qs("#stockTrendTable").innerHTML = buildStockTable(trendItems, "trend", rankH);
  qs("#stockSwingTable").innerHTML = buildStockTable(swingItems, "swing", rankH);
  if (qs("#stockDipTable")) qs("#stockDipTable").innerHTML = buildStockTable(dipItems, "dip", rankH);

  qsa("[data-action='analyze-stock']").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const symbol = btn.getAttribute("data-symbol");
      const freq = qs("#scanStockForm [name='freq']").value || "weekly";
      tabSwitch("analyze");
      qs("#analyzeForm [name='asset']").value = "stock";
      qs("#analyzeForm [name='symbol']").value = symbol;
      qs("#analyzeForm [name='freq']").value = freq;
      qs("#analyzeForm [name='method']").value = "all";
      qs("#analyzeForm [name='window']").value = "500";
      await startAnalyze({
        asset: "stock",
        symbol,
        freq,
        method: "all",
        window: 500,
      });
    });
  });
}

function buildEtfTable(items, kind) {
  const head = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>标的</th>
          <th>close</th>
          <th>score</th>
          <th>支撑/压力</th>
          <th>状态</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
  `;
  const rows = (items || []).slice(0, 30).map((it, idx) => {
    const lv = it.levels || {};
    const sc = it.scores || {};
    const mom = it.momentum || {};
    const ich = it.ichimoku || {};
    const close = it.close ?? "";
    const score = kind === "trend" ? sc.trend : sc.swing;
    const sup = lv.support_20w ?? "-";
    const res = lv.resistance_20w ?? "-";
    const cutoff = it.last_daily_date || it.last_date || "-";
    const state = `${ich.position || "-"} / ${mom.macd_state || "-"} / ADX:${toNumber(mom.adx, 0).toFixed(1)} / 截止:${cutoff}`;
    return `
      <tr>
        <td>${idx + 1}</td>
        <td><div><code>${it.symbol}</code></div><div class="muted">${it.name || ""}</div></td>
        <td><code>${close}</code></td>
        <td><span class="pill">${score ?? "-"}</span></td>
        <td><div class="muted">S:${sup}</div><div class="muted">R:${res}</div></td>
        <td class="muted">${state}</td>
        <td><button class="btn" data-action="analyze-etf" data-symbol="${it.symbol}">一键分析</button></td>
      </tr>
    `;
  });
  const tail = `</tbody></table>`;
  return head + rows.join("") + tail;
}

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || x === "") return "-";
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
}

function fmtAmount(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1e8) return `${(n / 1e8).toFixed(2)}亿`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e4).toFixed(0)}万`;
  return `${n.toFixed(0)}`;
}

function bbbDistToMA50(it) {
  const close = toNumber(it?.close, NaN);
  const ma50 = toNumber(it?.levels?.ma50, NaN);
  if (!Number.isFinite(close) || !Number.isFinite(ma50) || ma50 <= 0) return "-";
  const pct = ((close - ma50) / ma50) * 100.0;
  return `${pct.toFixed(1)}%`;
}

function bbbRoomToUpper(it) {
  const close = toNumber(it?.close, NaN);
  const upper = toNumber(it?.levels?.resistance_20w, NaN);
  if (!Number.isFinite(close) || !Number.isFinite(upper) || upper <= 0) return "-";
  const pct = ((upper - close) / upper) * 100.0;
  return `${pct.toFixed(1)}%`;
}

function bbbReasonText(it, cfg) {
  const ev = it?.bbb;
  if (ev?.why) return String(ev.why);

  const distMax = toNumber(cfg?.dist_ma50_max, 0.12);
  const aboveMax = toNumber(cfg?.max_above_20w, 0.05);
  const minWeeks = toNumber(cfg?.min_weekly_bars_total, 60);

  const wk = toNumber(it?.bars?.weekly_total, 0);
  const close = toNumber(it?.close, NaN);
  const ma50 = toNumber(it?.levels?.ma50, NaN);
  const upper = toNumber(it?.levels?.resistance_20w, NaN);

  const mom = it?.momentum || {};
  const daily = it?.daily || {};

  const okWeeks = wk >= minWeeks;
  const okPos = Number.isFinite(close) && Number.isFinite(ma50) && ma50 > 0 && Math.abs(close - ma50) / ma50 <= distMax;
  const okNotChase = Number.isFinite(close) && Number.isFinite(upper) && upper > 0 ? close <= upper * (1.0 + aboveMax) : false;
  const okWMACD = mom?.macd_state === "bullish" && toNumber(mom?.macd, NaN) > 0;
  const okDMACD = daily?.macd_state === "bullish";

  const fails = [];
  if (!okWeeks) fails.push(`周K不足(${wk}/${minWeeks})`);
  if (!okPos) fails.push("位置偏离MA50");
  if (!okNotChase) fails.push("追高风险");
  if (!okWMACD) fails.push("周MACD不够强");
  if (!okDMACD) fails.push("日MACD没转多");

  if (!fails.length) {
    return `通过：周MACD多且>0 / 日MACD多 / 位置靠MA50(${bbbDistToMA50(it)}) / 未追高(离20W上轨${bbbRoomToUpper(it)})`;
  }
  return `差：${fails.slice(0, 3).join(" / ")}${fails.length > 3 ? " …" : ""}`;
}

function renderBBBEtfMeta(bbbJson, bbbItems) {
  const cfg = bbbJson?.bbb || {};
  const ruleEl = qs("#bbbEtfRule");
  if (ruleEl) {
    const dist = toNumber(cfg.dist_ma50_max, 0.12) * 100;
    const above = toNumber(cfg.max_above_20w, 0.05) * 100;
    const minWeeks = toNumber(cfg.min_weekly_bars_total, 60);
    let t = `规则：周MACD多且>0；日MACD多；距MA50≤${dist.toFixed(0)}%；不追高（不高于20W上轨${above.toFixed(0)}%）；周K≥${minWeeks}`;
    const minAvg20 = cfg.min_daily_amount_avg20;
    if (minAvg20) t += `；20日均成交额≥${fmtAmount(minAvg20)}`;
    ruleEl.textContent = t;
  }

  const statsEl = qs("#bbbEtfFailStats");
  if (statsEl) {
    const failStats = cfg.fail_stats || {};
    const entries = Object.entries(failStats)
      .filter(([, v]) => Number(v) > 0)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 6);

    if (bbbItems?.length || !entries.length) {
      statsEl.textContent = "";
      statsEl.style.display = "none";
    } else {
      statsEl.textContent = `主要卡点：${entries.map(([k, v]) => `${k} ${v}`).join("；")}`;
      statsEl.style.display = "block";
    }
  }
}

function buildBBBEtfTable(items, cfg) {
  const head = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>标的</th>
          <th>周K数</th>
          <th>close</th>
          <th>距MA50</th>
          <th>离20W上轨</th>
          <th>周MACD</th>
          <th>日MACD</th>
          <th>理由</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
  `;
  const rows = (items || []).slice(0, 50).map((it, idx) => {
    const mom = it.momentum || {};
    const daily = it.daily || {};
    const wk = it.bars?.weekly_total ?? "-";
    const close = it.close ?? "";
    const w = `${mom.macd_state || "-"} (${toNumber(mom.macd, 0).toFixed(3)})`;
    const d = `${daily.macd_state || "-"} (${toNumber(daily.macd, 0).toFixed(3)})`;
    const cutoff = it.last_daily_date || it.last_date || "-";
    const why = bbbReasonText(it, cfg);
    return `
      <tr>
        <td>${idx + 1}</td>
        <td><div><code>${it.symbol}</code></div><div class="muted">${it.name || ""}</div><div class="muted">截止 ${cutoff}</div></td>
        <td class="muted">${wk}</td>
        <td><code>${close}</code></td>
        <td><span class="pill">${bbbDistToMA50(it)}</span></td>
        <td class="muted">${bbbRoomToUpper(it)}</td>
        <td class="muted">${w}</td>
        <td class="muted">${d}</td>
        <td class="muted">${why}</td>
        <td><button class="btn" data-action="analyze-etf" data-symbol="${it.symbol}">一键分析</button></td>
      </tr>
    `;
  });
  const tail = `</tbody></table>`;
  return head + rows.join("") + tail;
}

function buildStockTable(items, kind, rankHorizonWeeks) {
  const head = `
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>标的</th>
          <th>信号</th>
          <th>close</th>
          <th>周成交额</th>
          <th>score</th>
          <th>${rankHorizonWeeks}W 胜率</th>
          <th>${rankHorizonWeeks}W 均值</th>
          <th>${rankHorizonWeeks}W 磨损(MAE)</th>
          <th>样本数</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
  `;
  const rows = (items || []).slice(0, 30).map((it, idx) => {
    const sc = it.scores || {};
    const sig = it.signals || {};
    const fw = (it.forward || {})[kind] || {};
    const key = `${rankHorizonWeeks}w`;
    const st = fw[key] || {};
    const close = it.close ?? "";
    const amount = it.amount ?? "";
    const score = sc[kind] ?? sc.trend ?? sc.swing;
    const win = fmtPct(st.win_rate, 1);
    const avg = fmtPct(st.avg_return, 2);
    const mae = fmtPct(st.avg_mae, 2);
    const trades = st.trades ?? "-";
    const flag = sig[kind] ? "触发" : "观察";
    return `
      <tr>
        <td>${idx + 1}</td>
        <td><div><code>${it.symbol}</code></div><div class="muted">${it.name || ""}</div></td>
        <td><span class="pill">${flag}</span></td>
        <td><code>${close}</code></td>
        <td class="muted">${fmtAmount(amount)}</td>
        <td><span class="pill">${score ?? "-"}</span></td>
        <td><span class="pill">${win}</span></td>
        <td class="muted">${avg}</td>
        <td class="muted">${mae}</td>
        <td class="muted">${trades}</td>
        <td><button class="btn" data-action="analyze-stock" data-symbol="${it.symbol}">一键分析</button></td>
      </tr>
    `;
  });
  const tail = `</tbody></table>`;
  return head + rows.join("") + tail;
}

async function init() {
  try {
    const h = await api("/api/health");
    setHealth(true, `OK · py${h.python}`);
  } catch (e) {
    setHealth(false, `不可用：${e.message}`);
  }

  qsa(".tab").forEach((b) => {
    b.addEventListener("click", () => tabSwitch(b.dataset.tab));
  });

  qs("#scanDemo").addEventListener("click", async () => {
    qs("#scanForm [name='freq']").value = "weekly";
    qs("#scanForm [name='limit']").value = "200";
    qs("#scanForm [name='top_k']").value = "30";
    qs("#scanForm [name='min_amount']").value = "50000000";
    if (qs("#scanForm [name='min_amount_avg20']")) qs("#scanForm [name='min_amount_avg20']").value = "50000000";
    if (qs("#scanForm [name='min_weeks']")) qs("#scanForm [name='min_weeks']").value = "60";
    if (qs("#scanForm [name='workers']")) qs("#scanForm [name='workers']").value = "8";
    await startScan({
      freq: "weekly",
      window: 400,
      min_weeks: 60,
      limit: 200,
      top_k: 30,
      min_amount: 50000000,
      min_amount_avg20: 50000000,
      workers: 8,
    });
  });

  qs("#scanStockDemo").addEventListener("click", async () => {
    qs("#scanStockForm [name='freq']").value = "weekly";
    qs("#scanStockForm [name='limit']").value = "300";
    qs("#scanStockForm [name='top_k']").value = "50";
    qs("#scanStockForm [name='min_amount']").value = "50000000";
    qs("#scanStockForm [name='workers']").value = "8";
    qs("#scanStockForm [name='daily_filter']").value = "macd";
    await startScanStock({
      freq: "weekly",
      window: 500,
      start_date: "20100101",
      end_date: null,
      adjust: null,
      daily_filter: "macd",
      horizons: "4,8,12",
      rank_horizon: 8,
      min_weeks: 120,
      min_trades: 12,
      min_amount: 50000000,
      limit: 300,
      top_k: 50,
      workers: 8,
      buy_cost: 0.001,
      sell_cost: 0.002,
      include_st: false,
      exclude_bj: false,
      cache_ttl_hours: 24.0,
    });
  });

  qs("#analyzeForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const payload = {
      asset: form.asset.value,
      symbol: form.symbol.value.trim(),
      freq: form.freq.value,
      method: form.method.value,
      window: toNumber(form.window.value, 500),
      start_date: form.start_date.value.trim() || null,
      end_date: form.end_date.value.trim() || null,
      adjust: form.adjust.value.trim() || null,
      narrate: Boolean(form.narrate?.checked),
      narrate_provider: form.narrate_provider?.value || "openai",
      narrate_schools: form.narrate_schools?.value?.trim() || "chan,wyckoff,ichimoku,turtle,momentum",
    };
    await startAnalyze(payload);
  });

  qs("#scanForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const payload = {
      freq: form.freq.value,
      window: toNumber(form.window.value, 400),
      min_weeks: toNumber(form.min_weeks?.value, 60),
      min_amount: toNumber(form.min_amount.value, 0),
      min_amount_avg20: toNumber(form.min_amount_avg20?.value, 0),
      limit: toNumber(form.limit.value, 0),
      top_k: toNumber(form.top_k.value, 30),
      workers: toNumber(form.workers?.value, 8),
    };
    await startScan(payload);
  });

  qs("#scanStockForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const nearHighPct = toNumber(form.tt_near_high_pct?.value, 25);
    const aboveLowPct = toNumber(form.tt_above_low_pct?.value, 30);
    const slopeWeeks = toNumber(form.tt_slope_weeks?.value, 4);
    const payload = {
      freq: form.freq.value,
      window: 500,
      start_date: form.start_date.value.trim() || "20100101",
      end_date: form.end_date.value.trim() || null,
      adjust: null,
      daily_filter: form.daily_filter.value,
      base_filters: form.base_filters.value || "trend_template",
      tt_near_high: nearHighPct / 100.0,
      tt_above_low: aboveLowPct / 100.0,
      tt_slope_weeks: slopeWeeks,
      horizons: "4,8,12",
      rank_horizon: 8,
      min_weeks: 120,
      min_trades: 12,
      min_price: toNumber(form.min_price.value, 0),
      max_price: toNumber(form.max_price.value, 0),
      min_amount: toNumber(form.min_amount.value, 0),
      limit: toNumber(form.limit.value, 0),
      top_k: toNumber(form.top_k.value, 50),
      workers: toNumber(form.workers.value, 8),
      buy_cost: 0.001,
      sell_cost: 0.002,
      include_st: Boolean(form.include_st.checked),
      exclude_bj: Boolean(form.exclude_bj.checked),
      cache_ttl_hours: 24.0,
    };
    await startScanStock(payload);
  });
}

init();
