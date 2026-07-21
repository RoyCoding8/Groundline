type Point = { tick: number; value: number };

const coordinates = (points: Point[], width: number, height: number) => {
  const maxTick = Math.max(...points.map((point) => point.tick), 1);
  return points.map((point) => ({
    x: 34 + ((width - 48) * (point.tick - 1)) / Math.max(maxTick - 1, 1),
    y: 10 + (height - 34) * (1 - point.value),
  }));
};

export function SignalChart({ truth, belief }: { truth: Point[]; belief: Point[] }) {
  const width = 760;
  const height = 236;
  const truthCoords = coordinates(truth, width, height);
  const beliefCoords = coordinates(belief, width, height);

  const path = (points: Array<{ x: number; y: number }>) =>
    points
      .map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`)
      .join(" ");

  return (
    <svg
      className="signal-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="World truth and executive belief over time"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((value) => {
        const y = 10 + (height - 34) * (1 - value);
        return (
          <g key={value}>
            <line
              x1="34"
              y1={y}
              x2={width - 14}
              y2={y}
              className="grid-line"
            />
            <text x="0" y={y + 4}>
              {Math.round(value * 100)}
            </text>
          </g>
        );
      })}
      <path d={path(truthCoords)} className="truth-line" />
      <path d={path(beliefCoords)} className="belief-line" />
      {truthCoords.map((point, index) => (
        <circle
          key={`t${index}`}
          cx={point.x}
          cy={point.y}
          r="3"
          className="truth-dot"
        />
      ))}
      {beliefCoords.map((point, index) => (
        <circle
          key={`b${index}`}
          cx={point.x}
          cy={point.y}
          r="3"
          className="belief-dot"
        />
      ))}
    </svg>
  );
}

export function DistortionLadder({
  levels,
}: {
  levels: Array<{ depth: number; value: number }>;
}) {
  const width = 410;
  const height = 190;
  const max = Math.max(
    ...levels.map((level) => Math.abs(level.value)),
    0.01,
  );
  const ordered = [...levels].sort((a, b) => b.depth - a.depth);
  const points = ordered.map((level, index) => ({
    x: 118 + (240 * Math.max(0, level.value)) / max,
    y: 32 + index * (126 / Math.max(ordered.length - 1, 1)),
    ...level,
  }));
  const labels = ["CONTRIBUTORS", "MANAGERS", "EXECUTIVE"];

  return (
    <svg
      className="ladder-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Optimism bias by hierarchy level"
    >
      <line
        x1="118"
        y1="18"
        x2="118"
        y2="174"
        className="zero-line"
      />
      <polyline
        points={points.map((point) => `${point.x},${point.y}`).join(" ")}
        className="ladder-line"
      />
      {points.map((point, index) => (
        <g key={point.depth}>
          <text x="4" y={point.y + 4}>
            {labels[index] ?? `DEPTH ${point.depth}`}
          </text>
          <circle cx={point.x} cy={point.y} r="7" />
          <text
            x={point.x + 14}
            y={point.y + 4}
            className="value-label"
          >
            {point.value >= 0 ? "+" : ""}
            {(point.value * 100).toFixed(1)}
          </text>
        </g>
      ))}
    </svg>
  );
}
