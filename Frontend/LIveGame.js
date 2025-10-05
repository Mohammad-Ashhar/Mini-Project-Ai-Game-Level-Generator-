import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Tilemap Adventure — single-file mini game
 * - Arrow keys / WASD to move
 * - Collect all ✨ gems
 * - Reach the 🌀 portal to win
 * - You can't walk on water 🌊 or lava 🔥; mountains ⛰ block movement
 */

const WALKABLE = new Set(["grass", "sand", "snow", "dirt", "ice", "tree", "stone", "rock", "desert"]);
const BLOCKED  = new Set(["water", "lava", "mountain"]);

const TILE_COLORS = {
  grass: "#4ade80",
  water: "#3b82f6",
  stone: "#6b7280",
  dirt: "#92400e",
  sand: "#f59e0b",
  tree: "#16a34a",
  rock: "#374151",
  mountain: "#78716c",
  desert: "#eab308",
  snow: "#f8fafc",
  lava: "#dc2626",
  ice: "#06b6d4",
};

function makeTilemap({ genre = "fantasy", size = "20x20", seed = "" }) {
  // deterministic-ish noise if seed provided
  let rng = mulberry32(seed ? hashStr(seed) : Math.floor(Math.random() * 1e9));
  const [w, h] = size.split("x").map(Number);
  const m = { width: w, height: h, genre, tiles: [] };

  for (let y = 0; y < h; y++) {
    const row = [];
    for (let x = 0; x < w; x++) {
      const n = (Math.sin((x + rng()) * 0.25) + Math.cos((y + rng()) * 0.23)) * 0.5;
      const d = Math.hypot(x - w / 2, y - h / 2);
      let type = "grass";

      if (genre === "fantasy") {
        if (n > 0.35) type = "tree";
        else if (n < -0.35) type = "water";
        else if (d > Math.min(w, h) * 0.36) type = "mountain";
        else type = "grass";
      } else if (genre === "desert") {
        if (n > 0.4) type = "rock";
        else if (n < -0.25) type = "dirt";
        else type = "sand";
      } else if (genre === "arctic") {
        if (n > 0.25) type = "ice";
        else if (d > Math.min(w, h) * 0.3) type = "mountain";
        else type = "snow";
      } else if (genre === "volcanic") {
        if (n > 0.3) type = "lava";
        else if (d > Math.min(w, h) * 0.35) type = "mountain";
        else type = "stone";
      } else {
        // tropical
        if (n > 0.35) type = "tree";
        else if (n < -0.35) type = "water";
        else type = "grass";
      }

      row.push({ type });
    }
    m.tiles.push(row);
  }
  return m;
}

function hashStr(s) {
  let h = 1779033703 ^ s.length;
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(h ^ s.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  h = Math.imul(h ^ (h >>> 16), 2246822507);
  h = Math.imul(h ^ (h >>> 13), 3266489909);
  return (h ^ (h >>> 16)) >>> 0;
}
function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export default function App() {
  // --- UI state
  const [genre, setGenre] = useState("fantasy");
  const [size, setSize] = useState("20x20");
  const [seed, setSeed] = useState("");
  const [runningSeed, setRunningSeed] = useState("");

  // --- game state
  const [map, setMap] = useState(null);
  const [player, setPlayer] = useState({ x: 1, y: 1, hp: 3, gems: 0 });
  const [portal, setPortal] = useState({ x: 0, y: 0 });
  const [gems, setGems] = useState([]); // [{x,y,active}]
  const [enemies, setEnemies] = useState([]); // [{x,y}]
  const [status, setStatus] = useState("idle"); // idle | playing | won | lost

  const canvasRef = useRef(null);
  const keysRef = useRef({});

  // generate / restart
  const generate = () => {
    const m = makeTilemap({ genre, size, seed });
    // place player at first walkable from top-left
    const [w, h] = size.split("x").map(Number);
    const p = findFirstWalkable(m);
    // portal at bottom-right walkable
    const q = findLastWalkable(m);

    const rng = mulberry32(seed ? hashStr(seed) : Math.floor(Math.random() * 1e9));

    // scatter gems (5% tiles, capped)
    const targetGemCount = Math.max(5, Math.floor((w * h) * 0.03));
    const g = [];
    while (g.length < targetGemCount) {
      const x = Math.floor(rng() * w);
      const y = Math.floor(rng() * h);
      if (WALKABLE.has(m.tiles[y][x].type) && !(x === p.x && y === p.y) && !(x === q.x && y === q.y)) {
        if (!g.some(o => o.x === x && o.y === y)) g.push({ x, y, active: true });
      }
    }

    // a few enemies (2–5) that roam on walkable tiles
    const enemyCount = Math.max(2, Math.min(5, Math.floor((w + h) / 20)));
    const e = [];
    while (e.length < enemyCount) {
      const x = Math.floor(rng() * w);
      const y = Math.floor(rng() * h);
      if (WALKABLE.has(m.tiles[y][x].type) && !(x === p.x && y === p.y)) {
        e.push({ x, y });
      }
    }

    setMap(m);
    setPlayer({ x: p.x, y: p.y, hp: 3, gems: 0 });
    setPortal(q);
    setGems(g);
    setEnemies(e);
    setStatus("playing");
    setRunningSeed(seed || String(Math.floor(Math.random() * 1e9)));
  };

  // keyboard
  useEffect(() => {
    const onDown = (e) => {
      keysRef.current[e.key.toLowerCase()] = true;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "r") {
        e.preventDefault();
        setSeed(String(Math.floor(Math.random() * 1e6)));
      }
    };
    const onUp = (e) => (keysRef.current[e.key.toLowerCase()] = false);
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, []);

  // game loop
  useEffect(() => {
    if (!map || status !== "playing") return;
    let last = 0;
    let acc = 0;
    const step = 1 / 8; // 8 moves per second

    function loop(ts) {
      if (!last) last = ts;
      const dt = (ts - last) / 1000;
      last = ts;
      acc += dt;
      while (acc >= step) {
        tick(); // fixed update
        acc -= step;
      }
      draw();
      req = requestAnimationFrame(loop);
    }
    let req = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(req);
    // eslint-disable-next-line
  }, [map, status, player, enemies, gems]);

  const tick = () => {
    if (!map) return;

    // 1) player movement (grid step)
    const dir = inputToDir(keysRef.current);
    if (dir.dx || dir.dy) {
      const nx = clamp(player.x + dir.dx, 0, map.width - 1);
      const ny = clamp(player.y + dir.dy, 0, map.height - 1);
      if (!BLOCKED.has(map.tiles[ny][nx].type)) {
        setPlayer((p) => ({ ...p, x: nx, y: ny }));
      }
    }

    // 2) collect gems
    setGems((old) => {
      let collected = 0;
      const next = old.map((g) => {
        if (g.active && g.x === player.x && g.y === player.y) {
          collected++;
          return { ...g, active: false };
        }
        return g;
      });
      if (collected) setPlayer((p) => ({ ...p, gems: p.gems + collected }));
      return next;
    });

    // 3) enemies wander towards player if close
    setEnemies((es) =>
      es.map((e) => {
        const dist = Math.abs(e.x - player.x) + Math.abs(e.y - player.y);
        let dx = 0,
          dy = 0;
        if (dist <= 6) {
          dx = Math.sign(player.x - e.x);
          dy = Math.sign(player.y - e.y);
          // move one axis at a time
          if (Math.abs(player.x - e.x) > Math.abs(player.y - e.y)) dy = 0;
          else dx = 0;
        } else {
          // random wander
          const r = Math.floor(Math.random() * 4);
          dx = [1, -1, 0, 0][r];
          dy = [0, 0, 1, -1][r];
        }
        const nx = clamp(e.x + dx, 0, map.width - 1);
        const ny = clamp(e.y + dy, 0, map.height - 1);
        if (!BLOCKED.has(map.tiles[ny][nx].type)) return { x: nx, y: ny };
        return e;
      }),
    );

    // 4) enemy collision damages player
    if (enemies.some((e) => e.x === player.x && e.y === player.y)) {
      setPlayer((p) => ({ ...p, hp: Math.max(0, p.hp - 1) }));
    }

    // 5) hazards
    const t = map.tiles[player.y][player.x].type;
    if (t === "lava" || t === "water") {
      // falling in costs all HP
      setPlayer((p) => ({ ...p, hp: 0 }));
    }

    // 6) win/lose
    const allCollected = gems.every((g) => !g.active);
    if (player.hp <= 0) setStatus("lost");
    if (allCollected && player.x === portal.x && player.y === portal.y) setStatus("won");
  };

  const tileSize = useMemo(() => {
    if (!map) return 24;
    const max = 520; // canvas size cap
    return Math.floor(Math.min(max / map.width, max / map.height));
  }, [map]);

  const draw = () => {
    if (!map) return;
    const ctx = canvasRef.current.getContext("2d");
    const W = map.width * tileSize;
    const H = map.height * tileSize;

    canvasRef.current.width = W;
    canvasRef.current.height = H;

    // tiles
    for (let y = 0; y < map.height; y++) {
      for (let x = 0; x < map.width; x++) {
        const type = map.tiles[y][x].type;
        ctx.fillStyle = TILE_COLORS[type] || "#d1d5db";
        ctx.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
        ctx.strokeStyle = "rgba(255,255,255,0.25)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x * tileSize, y * tileSize, tileSize, tileSize);
      }
    }

    // portal
    drawEmoji(ctx, "🌀", portal.x, portal.y, tileSize);

    // gems
    for (const g of gems) if (g.active) drawEmoji(ctx, "✨", g.x, g.y, tileSize);

    // enemies
    for (const e of enemies) drawCircle(ctx, e.x, e.y, tileSize, "#111827");

    // player
    drawCircle(ctx, player.x, player.y, tileSize, "#10b981");

    // vignette for win/lose
    if (status === "won" || status === "lost") {
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fillRect(0, 0, W, H);
    }
  };

  // ------- UI -------
  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-6xl mx-auto space-y-4">
        <header className="text-center">
          <h1 className="text-3xl font-bold">🗺️ Tilemap Adventure</h1>
          <p className="text-slate-300">Built from your 1-week tilemap generator</p>
        </header>

        {/* Controls */}
        <div className="grid md:grid-cols-5 gap-3 items-end">
          <div className="space-y-1 md:col-span-1">
            <label className="text-sm text-slate-300">Genre</label>
            <select className="w-full bg-slate-800 rounded px-3 py-2" value={genre} onChange={(e) => setGenre(e.target.value)}>
              <option value="fantasy">🧙 Fantasy</option>
              <option value="desert">🏜️ Desert</option>
              <option value="arctic">❄️ Arctic</option>
              <option value="volcanic">🌋 Volcanic</option>
              <option value="tropical">🌴 Tropical</option>
            </select>
          </div>
          <div className="space-y-1 md:col-span-1">
            <label className="text-sm text-slate-300">Map Size</label>
            <select className="w-full bg-slate-800 rounded px-3 py-2" value={size} onChange={(e) => setSize(e.target.value)}>
              <option value="10x10">10×10</option>
              <option value="20x20">20×20</option>
              <option value="30x30">30×30</option>
              <option value="40x40">40×40</option>
            </select>
          </div>
          <div className="space-y-1 md:col-span-1">
            <label className="text-sm text-slate-300">Seed (optional)</label>
            <input className="w-full bg-slate-800 rounded px-3 py-2" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="e.g. 12345" />
          </div>
          <button onClick={generate} className="bg-purple-600 hover:bg-purple-700 rounded px-4 py-2 font-semibold md:col-span-1">
            ⚡ Generate / Restart
          </button>
          <div className="text-sm text-slate-400 md:col-span-1">
            Controls: <b>WASD / Arrows</b>. Ctrl+R to randomize seed.
          </div>
        </div>

        {/* Game area + HUD */}
        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div className="bg-slate-800 rounded-xl p-3 border border-slate-700">
              <canvas ref={canvasRef} className="w-full h-auto block rounded" />
            </div>
          </div>

          <aside className="bg-slate-800 rounded-xl p-4 border border-slate-700 space-y-3">
            <div className="text-lg font-semibold">HUD</div>
            <div className="flex gap-2 items-center">
              <span>❤️</span>
              <span className={player.hp > 0 ? "text-white" : "text-red-400"}>{player.hp}</span>
            </div>
            <div className="flex gap-2 items-center">
              <span>✨</span>
              <span>
                {player.gems}/{gems.filter((g) => g.active || true).length}
              </span>
            </div>
            <div>Seed in play: <code className="text-slate-300">{runningSeed || "(none)"}</code></div>

            {status === "won" && <div className="text-green-400 font-semibold">🎉 You collected all gems and reached the portal!</div>}
            {status === "lost" && <div className="text-red-400 font-semibold">💀 You died! Press “Generate / Restart”.</div>}
            {!map && <div className="text-slate-400">Click “Generate / Restart” to begin.</div>}
          </aside>
        </div>

        {/* Legend */}
        {map && (
          <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
            <div className="text-sm text-slate-300 mb-2">Legend</div>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 text-sm">
              {Object.entries(TILE_COLORS).map(([name, color]) => (
                <div key={name} className="flex items-center gap-2">
                  <span style={{ background: color }} className="inline-block w-4 h-4 rounded border border-white/20" />
                  <span className="capitalize">{name}</span>
                </div>
              ))}
              <div className="flex items-center gap-2"><span>🌀</span>Portal</div>
              <div className="flex items-center gap-2"><span>✨</span>Gem</div>
              <div className="flex items-center gap-2"><span>●</span>Enemy</div>
              <div className="flex items-center gap-2"><span style={{ color: "#10b981" }}>●</span>Player</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- helpers for drawing/input ---------- */

function drawCircle(ctx, gx, gy, s, color) {
  const cx = gx * s + s / 2;
  const cy = gy * s + s / 2;
  ctx.beginPath();
  ctx.arc(cx, cy, Math.max(4, s * 0.35), 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawEmoji(ctx, emoji, gx, gy, s) {
  ctx.font = `${Math.floor(s * 0.8)}px serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(emoji, gx * s + s / 2, gy * s + s / 2 + 1);
}

function inputToDir(keys) {
  const up = keys["arrowup"] || keys["w"];
  const down = keys["arrowdown"] || keys["s"];
  const left = keys["arrowleft"] || keys["a"];
  const right = keys["arrowright"] || keys["d"];
  const dx = (right ? 1 : 0) - (left ? 1 : 0);
  const dy = (down ? 1 : 0) - (up ? 1 : 0);
  // prevent diagonals (grid feel)
  if (dx && dy) return Math.random() < 0.5 ? { dx, dy: 0 } : { dx: 0, dy };
  return { dx, dy };
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function findFirstWalkable(m) {
  for (let y = 0; y < m.height; y++)
    for (let x = 0; x < m.width; x++)
      if (!BLOCKED.has(m.tiles[y][x].type)) return { x, y };
  return { x: 0, y: 0 };
}
function findLastWalkable(m) {
  for (let y = m.height - 1; y >= 0; y--)
    for (let x = m.width - 1; x >= 0; x--)
      if (!BLOCKED.has(m.tiles[y][x].type)) return { x, y };
  return { x: m.width - 1, y: m.height - 1 };
}
