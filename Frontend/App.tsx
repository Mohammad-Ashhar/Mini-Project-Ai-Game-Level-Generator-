import React, { useState } from "react";

type TilemapResponse = {
  grid: number[][]; // backend returns 2D grid of numbers
};

const TILE_SIZE = 32;

const App: React.FC = () => {
  const [genre, setGenre] = useState("dungeon");
  const [difficulty, setDifficulty] = useState("easy");
  const [size, setSize] = useState(10);
  const [seed, setSeed] = useState("1234");
  const [tilemap, setTilemap] = useState<number[][] | null>(null);

  const handleGenerate = async () => {
    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ genre, difficulty, size, seed }),
    });
    const data: TilemapResponse = await response.json();
    setTilemap(data.grid);
  };

  const handleExport = () => {
    if (!tilemap) return;
    const blob = new Blob([JSON.stringify(tilemap, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tilemap.json";
    a.click();
  };

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Tilemap Generator</h1>

      {/* Input Form */}
      <div className="space-x-2">
        <input
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
          placeholder="Genre"
          className="border p-1"
        />
        <input
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value)}
          placeholder="Difficulty"
          className="border p-1"
        />
        <input
          type="number"
          value={size}
          onChange={(e) => setSize(Number(e.target.value))}
          placeholder="Size"
          className="border p-1 w-20"
        />
        <input
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          placeholder="Seed"
          className="border p-1"
        />
        <button
          onClick={handleGenerate}
          className="bg-blue-500 text-white px-3 py-1 rounded"
        >
          Generate
        </button>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-3 py-1 rounded ml-2"
          title="Export JSON (Ctrl+E)"
        >
          Export JSON
        </button>
      </div>

      {/* Canvas Render */}
      {tilemap && (
        <canvas
          id="tilemapCanvas"
          width={size * TILE_SIZE}
          height={size * TILE_SIZE}
          ref={(canvas) => {
            if (canvas && tilemap) {
              const ctx = canvas.getContext("2d");
              if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                tilemap.forEach((row, y) =>
                  row.forEach((tile, x) => {
                    ctx.fillStyle =
                      tile === 0
                        ? "lightblue"
                        : tile === 1
                        ? "green"
                        : tile === 2
                        ? "brown"
                        : "gray";
                    ctx.fillRect(
                      x * TILE_SIZE,
                      y * TILE_SIZE,
                      TILE_SIZE,
                      TILE_SIZE
                    );
                  })
                );
              }
            }
          }}
        />
      )}
    </div>
  );
};

export default App;
