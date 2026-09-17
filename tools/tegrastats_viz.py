"""Turn tegrastats output into time-series charts you can open in a browser.

Record while something runs, then plot:

    sudo stdbuf -oL tegrastats --interval 500 \\
        | python tools/tegrastats_viz.py record tegra.log

    python tools/tegrastats_viz.py plot tegra.log --stages mem_r2.json --out r2.html

`record` stamps every line with the wall-clock time it arrived, so the charts
line up to the millisecond with the stage timestamps written by
`tools/device_memory_breakdown.py --json`. `stdbuf -oL` keeps tegrastats from
buffering its output when piped, which would bunch the timestamps together.

`plot` also reads a log written by `tegrastats --logfile` directly. It then falls
back to the date tegrastats prints on each line, or to the sampling interval.

The page is a single self-contained HTML file with no network access needed.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
CPU = re.compile(r"\bCPU \[([^\]]*)\]")
CORE = re.compile(r"(\d+)%@")
GPU = re.compile(r"\bGR3D_FREQ (\d+)%")
TEMP = re.compile(r"\b([A-Za-z][\w]*)@(-?\d+(?:\.\d+)?)C\b")
POWER = re.compile(r"\b([A-Z][A-Z0-9_]+) (\d+)mW/(\d+)mW")
DEVICE_DATE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
STAMP = re.compile(r"^(\d+\.\d+)\t(.*)$")

PREFERRED_TEMPS = ("cpu", "gpu", "tj")
PREFERRED_RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")
MAX_SERIES = 3  # the first three palette slots stay distinguishable in every pairing


def record(path: Path) -> int:
    if sys.stdin.isatty():
        print("pipe tegrastats into this command; see --help", file=sys.stderr)
        return 2
    count = 0
    with path.open("a", encoding="utf-8") as out:
        try:
            for line in sys.stdin:
                line = line.rstrip("\n")
                if not line:
                    continue
                out.write(f"{time.time():.3f}\t{line}\n")
                out.flush()
                count += 1
                ram = RAM.search(line)
                gpu = GPU.search(line)
                print(
                    f"\r{count:6d} samples  "
                    f"RAM {ram.group(1) if ram else '?':>5} MB  "
                    f"GPU {gpu.group(1) if gpu else '?':>3}%",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            pass
    print(f"\nwrote {count} samples to {path}")
    return 0


def parse_log(path: Path, interval: float) -> list[dict]:
    samples = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stamp = STAMP.match(raw)
        line = stamp.group(2) if stamp else raw
        ram = RAM.search(line)
        if not ram:
            continue
        sample = {
            "line": index,
            "ram": int(ram.group(1)),
            "ram_total": int(ram.group(2)),
        }
        if stamp:
            sample["time"] = float(stamp.group(1))
        elif date := DEVICE_DATE.match(line):
            sample["device_second"] = datetime.strptime(
                date.group(1), "%m-%d-%Y %H:%M:%S"
            ).timestamp()
        if swap := SWAP.search(line):
            sample["swap"] = int(swap.group(1))
        if cpu := CPU.search(line):
            cores = [int(value) for value in CORE.findall(cpu.group(1))]
            if cores:
                sample["cpu_mean"] = sum(cores) / len(cores)
                sample["cpu_max"] = max(cores)
        if gpu := GPU.search(line):
            sample["gpu"] = int(gpu.group(1))
        sample["temps"] = {
            name: float(value)
            for name, value in TEMP.findall(line)
            if -40.0 < float(value) < 150.0  # idle sensors report sentinel values
        }
        sample["power"] = {name: int(now) for name, now, _avg in POWER.findall(line)}
        samples.append(sample)
    assign_times(samples, interval)
    return samples


def assign_times(samples: list[dict], interval: float) -> None:
    """Give every sample an absolute time, using the best clock the log has."""
    if samples and all("time" in s for s in samples):
        return
    if samples and all("device_second" in s for s in samples):
        # tegrastats prints whole seconds; spread samples evenly inside each one.
        by_second: dict[float, list[dict]] = {}
        for sample in samples:
            by_second.setdefault(sample["device_second"], []).append(sample)
        for second, group in by_second.items():
            for position, sample in enumerate(group):
                sample["time"] = second + position / len(group)
        return
    for position, sample in enumerate(samples):
        sample["time"] = position * interval


def pick(names, preferred):
    ordered = [name for name in preferred if name in names]
    ordered += [name for name in sorted(names) if name not in ordered]
    return ordered[:MAX_SERIES]


def build_payload(samples: list[dict], stages_path: Path | None, title: str) -> dict:
    start = samples[0]["time"]
    t = [round(s["time"] - start, 3) for s in samples]

    def column(key):
        return [s.get(key) for s in samples]

    charts = [
        {
            "id": "ram",
            "title": "Memory in use",
            "unit": "MB",
            "area": True,
            "series": [{"name": "RAM used", "values": column("ram")}],
            "limit": {"label": "Total", "value": samples[0]["ram_total"]},
        }
    ]
    if any(s.get("swap") for s in samples):
        charts.append(
            {
                "id": "swap",
                "title": "Swap in use",
                "unit": "MB",
                "area": True,
                "series": [{"name": "Swap used", "values": column("swap")}],
            }
        )
    if any("gpu" in s for s in samples):
        charts.append(
            {
                "id": "gpu",
                "title": "GPU load",
                "unit": "%",
                "ymax": 100,
                "area": True,
                "series": [{"name": "GPU load", "values": column("gpu")}],
            }
        )
    if any("cpu_mean" in s for s in samples):
        charts.append(
            {
                "id": "cpu",
                "title": "CPU load",
                "unit": "%",
                "ymax": 100,
                "series": [
                    {"name": "Average of all cores", "values": column("cpu_mean")},
                    {"name": "Busiest core", "values": column("cpu_max")},
                ],
            }
        )
    temp_names = pick({n for s in samples for n in s["temps"]}, PREFERRED_TEMPS)
    if temp_names:
        charts.append(
            {
                "id": "temp",
                "title": "Temperature",
                "unit": "°C",
                "series": [
                    {"name": name, "values": [s["temps"].get(name) for s in samples]}
                    for name in temp_names
                ],
            }
        )
    rail_names = pick({n for s in samples for n in s["power"]}, PREFERRED_RAILS)
    if rail_names:
        charts.append(
            {
                "id": "power",
                "title": "Power draw",
                "unit": "mW",
                "series": [
                    {"name": name, "values": [s["power"].get(name) for s in samples]}
                    for name in rail_names
                ],
            }
        )

    stages = []
    if stages_path:
        report = json.loads(stages_path.read_text(encoding="utf-8"))
        for row in report.get("stages", []):
            if "time" in row:
                offset = round(row["time"] - start, 3)
                stages.append({"t": offset, "name": row["stage"]})

    ram = [v for v in column("ram") if v is not None]
    tiles = [
        {"label": "Peak memory in use", "value": f"{max(ram):,} MB"},
        {"label": "Growth over the recording", "value": f"{max(ram) - ram[0]:+,} MB"},
        {"label": "Duration", "value": f"{t[-1]:,.1f} s"},
    ]
    gpu = [v for v in column("gpu") if v is not None]
    if gpu:
        tiles.append({"label": "Peak GPU load", "value": f"{max(gpu)} %"})
    if "VDD_IN" in rail_names:
        watts = max(s["power"].get("VDD_IN", 0) for s in samples) / 1000
        tiles.append({"label": "Peak input power", "value": f"{watts:.2f} W"})

    return {"title": title, "t": t, "charts": charts, "stages": stages, "tiles": tiles}


def plot(args: argparse.Namespace) -> int:
    samples = parse_log(args.log, args.interval_ms / 1000)
    if len(samples) < 2:
        print(f"fewer than two tegrastats samples in {args.log}", file=sys.stderr)
        return 2
    title = args.title or args.log.stem
    payload = build_payload(samples, args.stages, title)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)
    out = args.out or args.log.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(samples)} samples, {len(payload['stages'])} stages)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="stamp tegrastats lines from stdin")
    rec.add_argument("log", type=Path)
    draw = commands.add_parser("plot", help="write an HTML page of charts")
    draw.add_argument("log", type=Path)
    draw.add_argument("--stages", type=Path, help="JSON from device_memory_breakdown")
    draw.add_argument("--out", type=Path)
    draw.add_argument("--title")
    draw.add_argument(
        "--interval-ms",
        type=float,
        default=500,
        help="sampling interval, used only when the log carries no timestamps",
    )
    args = parser.parse_args()
    return record(args.log) if args.command == "record" else plot(args)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · tegrastats</title>
<style>
.viz-root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page: #0d0d0d;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  min-height: 100vh;
  background: var(--page);
  color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding-block: 32px 48px;
  padding-inline: 16px;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 24px; }
.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.tile, .card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
}
.tile { padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.card { padding: 16px 16px 8px; margin-bottom: 12px; position: relative; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0; }
.card .unit { color: var(--text-muted); font-weight: 400; }
.legend {
  display: flex; flex-wrap: wrap; gap: 4px 16px;
  color: var(--text-secondary); font-size: 13px; margin: 6px 0 0;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
svg { display: block; width: 100%; overflow: visible; cursor: crosshair;
  touch-action: pan-y; user-select: none; }
.reset {
  font: inherit; font-size: 13px; color: var(--text-primary);
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 6px; padding: 2px 10px; margin-left: 4px; cursor: pointer;
}
.reset:hover { border-color: var(--axis); }
svg:focus { outline: none; }
.card:focus-within { border-color: var(--axis); }
.tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.stage-label { fill: var(--text-secondary); font-size: 11px; }
.tooltip {
  position: fixed; pointer-events: none; z-index: 10;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 13px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12); min-width: 170px;
}
.tooltip .head { color: var(--text-secondary); margin-bottom: 4px; }
.tooltip .row { display: flex; align-items: center; gap: 8px; }
.tooltip .row strong { font-weight: 600; font-variant-numeric: tabular-nums; }
.tooltip .row span:last-child { color: var(--text-secondary); }
details { margin-top: 20px; }
summary { cursor: pointer; color: var(--text-secondary); }
.table-wrap { overflow-x: auto; margin-top: 8px; }
table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td {
  text-align: right; padding: 4px 10px;
  border-bottom: 1px solid var(--grid); white-space: nowrap;
}
th { color: var(--text-secondary); font-weight: 600; }
td.text, th.text { text-align: left; }
</style>
</head>
<body>
<div class="viz-root">
  <div class="wrap">
    <h1 id="title"></h1>
    <p class="sub" id="subtitle"></p>
    <div class="tiles" id="tiles"></div>
    <div id="charts"></div>
    <details>
      <summary>Data table</summary>
      <div class="table-wrap"><table id="table"></table></div>
    </details>
  </div>
  <div class="tooltip" id="tooltip" hidden></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("data").textContent);
  const SVG = "http://www.w3.org/2000/svg";
  const HEIGHT = 170, M = { top: 14, right: 20, bottom: 26, left: 52 };
  const STAGE_BAND = 22;
  const t = data.t, tEnd = t[t.length - 1] || 1;
  const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
  let view = [0, tEnd];
  let active = null;
  let clipCount = 0;

  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const text = (tag, content, cls) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    node.textContent = content;
    return node;
  };

  // A snapshot is taken when a stage finishes, so each stage spans from the
  // previous snapshot to its own.
  const spans = [];
  for (let i = 1; i < data.stages.length; i++) {
    spans.push({ start: data.stages[i - 1].t, end: data.stages[i].t,
      name: data.stages[i].name });
  }
  const stageAt = (seconds) => {
    if (!data.stages.length) return null;
    if (seconds < data.stages[0].t) return "before the run";
    for (const span of spans) if (seconds <= span.end) return span.name;
    return "after the run";
  };

  document.getElementById("title").textContent = data.title;
  const subtitle = document.getElementById("subtitle");
  const resetButton = document.createElement("button");
  resetButton.type = "button";
  resetButton.className = "reset";
  resetButton.textContent = "Reset zoom";
  resetButton.hidden = true;
  const describe = () => {
    subtitle.replaceChildren(document.createTextNode(
      `tegrastats · ${t.length} samples over ${nf.format(tEnd)} s` +
      (data.stages.length ? ` · ${spans.length} stages marked` : "") +
      " · drag across a chart to zoom, double-click to reset "), resetButton);
  };
  describe();

  const tiles = document.getElementById("tiles");
  for (const tile of data.tiles) {
    const box = text("div", "", "tile");
    box.append(text("div", tile.label, "label"), text("div", tile.value, "value"));
    tiles.append(box);
  }

  const niceStep = (span, target) => {
    const raw = span / Math.max(1, target);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  };
  const nearestIndex = (seconds) => {
    let lo = 0, hi = t.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (t[mid] < seconds) lo = mid; else hi = mid;
    }
    return Math.abs(t[lo] - seconds) <= Math.abs(t[hi] - seconds) ? lo : hi;
  };

  const tooltip = document.getElementById("tooltip");
  const charts = [];
  const container = document.getElementById("charts");

  data.charts.forEach((chart, chartIndex) => {
    const card = document.createElement("section");
    card.className = "card";
    const heading = text("h2", chart.title + " ");
    heading.append(text("span", `(${chart.unit})`, "unit"));
    card.append(heading);
    if (chart.series.length > 1) {
      const legend = text("div", "", "legend");
      chart.series.forEach((series, i) => {
        const item = document.createElement("span");
        const key = document.createElement("i");
        key.className = "key";
        key.style.background = `var(--series-${i + 1})`;
        item.append(key, document.createTextNode(series.name));
        legend.append(item);
      });
      card.append(legend);
    }
    const svg = el("svg", {
      role: "img",
      tabindex: "0",
      "aria-label": `${chart.title} over time. Arrow keys step through samples.`,
    }, card);
    container.append(card);
    charts.push({ chart, svg, first: chartIndex === 0 });
  });

  function render() {
    const [a, b] = view;
    for (const entry of charts) {
      const { chart, svg, first } = entry;
      svg.replaceChildren();
      const width = svg.getBoundingClientRect().width || 800;
      const top = M.top + (first && spans.length ? STAGE_BAND : 0);
      const height = HEIGHT + top + M.bottom;
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("height", height);
      const plotW = width - M.left - M.right;

      const values = chart.series.flatMap((s) => s.values.filter((v) => v != null));
      let yMax = chart.ymax ?? Math.max(...values, chart.limit ? chart.limit.value : 0);
      let yMin = 0;
      if (chart.id === "temp" || chart.id === "power") {
        const lo = Math.min(...values), hi = Math.max(...values);
        const pad = Math.max((hi - lo) * 0.15, chart.id === "temp" ? 1 : 50);
        yMin = Math.max(0, lo - pad);
        yMax = hi + pad;
      }
      const yStep = niceStep(yMax - yMin || 1, 4);
      yMin = Math.floor(yMin / yStep) * yStep;
      yMax = Math.max(Math.ceil(yMax / yStep) * yStep, yMin + yStep);
      const x = (s) => M.left + ((s - a) / (b - a)) * plotW;
      const y = (v) => top + HEIGHT - ((v - yMin) / (yMax - yMin)) * HEIGHT;
      Object.assign(entry, { x, y, top, plotW });

      const clipId = `clip-${clipCount++}`;
      const defs = el("defs", {}, svg);
      el("rect", { x: M.left, y: top - 6, width: plotW, height: HEIGHT + 12 },
        el("clipPath", { id: clipId }, defs));

      for (let v = yMin; v <= yMax + 1e-9; v += yStep) {
        el("line", { x1: M.left, x2: M.left + plotW, y1: y(v), y2: y(v),
          stroke: v === yMin ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, svg);
        el("text", { x: M.left - 8, y: y(v) + 4, "text-anchor": "end", class: "tick" }, svg)
          .textContent = nf.format(v);
      }
      const xStep = niceStep(b - a, Math.max(2, Math.floor(plotW / 90)));
      for (let s = Math.ceil(a / xStep) * xStep; s <= b + 1e-9; s += xStep) {
        el("text", { x: x(s), y: top + HEIGHT + 18, "text-anchor": "middle", class: "tick" },
          svg).textContent = `${nf.format(s)} s`;
      }

      if (chart.limit) {
        const ly = y(chart.limit.value);
        el("line", { x1: M.left, x2: M.left + plotW, y1: ly, y2: ly,
          stroke: "var(--axis)", "stroke-width": 1 }, svg);
        el("text", { x: M.left + plotW, y: ly - 6, "text-anchor": "end", class: "tick" }, svg)
          .textContent = `${chart.limit.label} ${chart.limit.value.toLocaleString()} ${chart.unit}`;
      }

      let lastLine = -Infinity;
      for (const stage of data.stages) {
        if (stage.t < a || stage.t > b) continue;
        const sx = x(stage.t);
        if (sx - lastLine < 3) continue;
        el("line", { x1: sx, x2: sx, y1: top, y2: top + HEIGHT,
          stroke: "var(--axis)", "stroke-width": 1 }, svg);
        lastLine = sx;
      }
      if (first) {
        for (const span of spans) {
          const left = Math.max(x(span.start), M.left);
          const right = Math.min(x(span.end), M.left + plotW);
          if (right - left < span.name.length * 6.2 + 10) continue;
          el("text", { x: left + 5, y: M.top + 10, class: "stage-label" }, svg)
            .textContent = span.name;
        }
      }

      const marks = el("g", { "clip-path": `url(#${clipId})` }, svg);
      chart.series.forEach((series, i) => {
        const color = `var(--series-${i + 1})`;
        let d = "", area = "", open = false, startX = 0, prevX = 0;
        const close = () => {
          if (open && chart.area) area += `L${prevX},${y(yMin)}L${startX},${y(yMin)}Z`;
          open = false;
        };
        series.values.forEach((v, k) => {
          if (v == null) return close();
          const px = x(t[k]), py = y(v);
          if (!open) {
            d += `M${px},${py}`;
            if (chart.area) area += `M${px},${y(yMin)}L${px},${py}`;
            startX = px; open = true;
          } else {
            d += `L${px},${py}`;
            if (chart.area) area += `L${px},${py}`;
          }
          prevX = px;
        });
        close();
        if (chart.area && chart.series.length === 1) {
          el("path", { d: area, fill: color, opacity: 0.1 }, marks);
        }
        el("path", { d, fill: "none", stroke: color, "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round" }, marks);
      });

      entry.brush = el("rect", { y: top, height: HEIGHT, fill: "var(--text-muted)",
        opacity: 0.14, visibility: "hidden" }, svg);
      entry.cross = el("line", { y1: top, y2: top + HEIGHT, stroke: "var(--text-muted)",
        "stroke-width": 1, visibility: "hidden" }, svg);
      entry.dots = chart.series.map((_, i) => el("circle", { r: 4,
        fill: `var(--series-${i + 1})`, stroke: "var(--surface-1)", "stroke-width": 2,
        visibility: "hidden" }, svg));
    }
    resetButton.hidden = view[0] === 0 && view[1] === tEnd;
    if (active !== null) show(active.index, active.entry, false);
  }

  function show(index, owner, moveTooltip = true, clientX = 0, clientY = 0) {
    active = { index, entry: owner };
    for (const entry of charts) {
      const cx = entry.x(t[index]);
      entry.cross.setAttribute("x1", cx);
      entry.cross.setAttribute("x2", cx);
      entry.cross.setAttribute("visibility", "visible");
      entry.chart.series.forEach((series, i) => {
        const v = series.values[index];
        const dot = entry.dots[i];
        if (v == null) return dot.setAttribute("visibility", "hidden");
        dot.setAttribute("cx", cx);
        dot.setAttribute("cy", entry.y(v));
        dot.setAttribute("visibility", "visible");
      });
    }
    tooltip.replaceChildren();
    const stage = stageAt(t[index]);
    tooltip.append(text("div", `${nf.format(t[index])} s` + (stage ? ` · ${stage}` : ""),
      "head"));
    owner.chart.series.forEach((series, i) => {
      const v = series.values[index];
      const row = text("div", "", "row");
      const key = document.createElement("i");
      key.className = "key";
      key.style.background = `var(--series-${i + 1})`;
      row.append(key,
        text("strong", v == null ? "–" : `${nf.format(v)} ${owner.chart.unit}`),
        text("span", series.name));
      tooltip.append(row);
    });
    tooltip.hidden = false;
    const box = tooltip.getBoundingClientRect();
    if (!moveTooltip) {
      const r = owner.svg.getBoundingClientRect();
      clientX = r.left + owner.x(t[index]);
      clientY = r.top + owner.top;
    }
    let left = clientX + 16, topPx = clientY + 16;
    if (left + box.width > window.innerWidth - 8) left = clientX - box.width - 16;
    if (topPx + box.height > window.innerHeight - 8) topPx = clientY - box.height - 16;
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, topPx)}px`;
  }

  function hide() {
    active = null;
    tooltip.hidden = true;
    for (const entry of charts) {
      entry.cross.setAttribute("visibility", "hidden");
      entry.dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
    }
  }

  const secondsAt = (entry, clientX) => {
    const r = entry.svg.getBoundingClientRect();
    const frac = (clientX - r.left - M.left) / entry.plotW;
    return view[0] + Math.min(1, Math.max(0, frac)) * (view[1] - view[0]);
  };
  let drag = null;

  for (const entry of charts) {
    const { svg } = entry;
    svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      drag = { entry, from: secondsAt(entry, event.clientX), x0: event.clientX };
      try { svg.setPointerCapture(event.pointerId); } catch (_) { /* synthetic events */ }
    });
    svg.addEventListener("pointermove", (event) => {
      const r = svg.getBoundingClientRect();
      const px = event.clientX - r.left;
      if (drag && drag.entry === entry) {
        const x1 = Math.max(M.left, Math.min(drag.x0 - r.left, M.left + entry.plotW));
        const x2 = Math.max(M.left, Math.min(px, M.left + entry.plotW));
        for (const other of charts) {
          other.brush.setAttribute("x", Math.min(x1, x2));
          other.brush.setAttribute("width", Math.abs(x2 - x1));
          other.brush.setAttribute("visibility", "visible");
        }
      }
      if (px < M.left - 12 || px > M.left + entry.plotW + 12) return hide();
      show(nearestIndex(secondsAt(entry, event.clientX)), entry, true,
        event.clientX, event.clientY);
    });
    svg.addEventListener("pointerup", (event) => {
      if (!drag || drag.entry !== entry) return;
      const to = secondsAt(entry, event.clientX);
      const wide = Math.abs(event.clientX - drag.x0) > 8;
      const [lo, hi] = [Math.min(drag.from, to), Math.max(drag.from, to)];
      drag = null;
      for (const other of charts) other.brush.setAttribute("visibility", "hidden");
      if (wide && hi - lo > 0) { view = [lo, hi]; render(); }
    });
    svg.addEventListener("dblclick", () => { view = [0, tEnd]; render(); });
    svg.addEventListener("pointerleave", () => { if (!drag) hide(); });
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let index = active && active.entry === entry ? active.index : nearestIndex(view[0]);
      if (event.key === "ArrowLeft") index = Math.max(0, index - 1);
      if (event.key === "ArrowRight") index = Math.min(t.length - 1, index + 1);
      if (event.key === "Home") index = nearestIndex(view[0]);
      if (event.key === "End") index = nearestIndex(view[1]);
      show(index, entry, false);
    });
  }
  resetButton.addEventListener("click", () => { view = [0, tEnd]; render(); });

  const table = document.getElementById("table");
  const head = document.createElement("tr");
  head.append(text("th", "Time (s)"), text("th", "Stage", "text"));
  for (const chart of data.charts) {
    for (const series of chart.series) {
      head.append(text("th", `${series.name} (${chart.unit})`));
    }
  }
  table.append(head);
  t.forEach((seconds, k) => {
    const row = document.createElement("tr");
    row.append(text("td", nf.format(seconds)), text("td", stageAt(seconds) || "", "text"));
    for (const chart of data.charts) {
      for (const series of chart.series) {
        const v = series.values[k];
        row.append(text("td", v == null ? "" : nf.format(v)));
      }
    }
    table.append(row);
  });

  let lastWidth = 0;
  new ResizeObserver((entries) => {
    const width = Math.round(entries[0].contentRect.width);
    if (width !== lastWidth) { lastWidth = width; requestAnimationFrame(render); }
  }).observe(container);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
