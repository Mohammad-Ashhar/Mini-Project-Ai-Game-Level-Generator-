import React, { useState, useRef, useEffect } from 'react';

const TilemapGeneratorUI = () => {
  const [formData, setFormData] = useState({
    genre: 'fantasy',
    difficulty: 'medium',
    size: '20x20',
    seed: ''
  });

  const [tilemap, setTilemap] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const canvasRef = useRef(null);

  // Tile type to color mapping for visualization
  const tileColors = {
    grass: '#4ade80',
    water: '#3b82f6', 
    stone: '#6b7280',
    dirt: '#92400e',
    sand: '#f59e0b',
    tree: '#16a34a',
    rock: '#374151',
    mountain: '#78716c',
    desert: '#eab308',
    snow: '#f8fafc',
    lava: '#dc2626',
    ice: '#06b6d4'
  };

  // Generate random seed
  const generateRandomSeed = () => {
    const seed = Math.floor(Math.random() * 1000000).toString();
    setFormData(prev => ({ ...prev, seed }));
  };

  // Handle form input changes
  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  // Mock API call (since we don't have a real backend)
  const mockApiCall = async (data) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const [width, height] = data.size.split('x').map(Number);
    const tileTypes = Object.keys(tileColors);
    const mockTilemap = {
      width,
      height,
      seed: data.seed || Math.floor(Math.random() * 1000000),
      genre: data.genre,
      difficulty: data.difficulty,
      tiles: []
    };

    for (let y = 0; y < height; y++) {
      const row = [];
      for (let x = 0; x < width; x++) {
        let tileType;
        const noise = (Math.sin(x * 0.1) + Math.cos(y * 0.1)) * 0.5;
        const distance = Math.sqrt((x - width/2) ** 2 + (y - height/2) ** 2);
        if (data.genre === 'fantasy') {
          if (noise > 0.3) tileType = 'tree';
          else if (noise < -0.3) tileType = 'water';
          else if (distance > Math.min(width, height) * 0.3) tileType = 'mountain';
          else tileType = 'grass';
        } else if (data.genre === 'desert') {
          if (noise > 0.4) tileType = 'rock';
          else if (noise < -0.2) tileType = 'dirt';
          else tileType = 'sand';
        } else if (data.genre === 'arctic') {
          if (noise > 0.2) tileType = 'ice';
          else if (distance > Math.min(width, height) * 0.25) tileType = 'mountain';
          else tileType = 'snow';
        } else {
          tileType = tileTypes[Math.floor(Math.random() * tileTypes.length)];
        }
        row.push({ type: tileType, x, y });
      }
      mockTilemap.tiles.push(row);
    }
    return mockTilemap;
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    setError('');
    try {
      const data = await mockApiCall(formData);
      setTilemap(data);
    } catch (err) {
      setError('Failed to generate tilemap. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (!tilemap || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const tileSize = Math.min(400 / tilemap.width, 400 / tilemap.height);
    canvas.width = tilemap.width * tileSize;
    canvas.height = tilemap.height * tileSize;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < tilemap.height; y++) {
      for (let x = 0; x < tilemap.width; x++) {
        const tile = tilemap.tiles[y][x];
        ctx.fillStyle = tileColors[tile.type] || '#d1d5db';
        ctx.fillRect(x * tileSize, y * tileSize, tileSize, tileSize);
        ctx.strokeStyle = '#ffffff40';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x * tileSize, y * tileSize, tileSize, tileSize);
      }
    }
  }, [tilemap]);

  const exportTilemap = () => {
    if (!tilemap) return;
    const dataStr = JSON.stringify(tilemap, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `tilemap-${tilemap.seed}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-4">🗺️ Tilemap Generator</h1>
      <div className="space-y-4">
        <select value={formData.genre} onChange={(e) => handleInputChange('genre', e.target.value)}>
          <option value="fantasy">🧙 Fantasy</option>
          <option value="desert">🏜️ Desert</option>
          <option value="arctic">❄️ Arctic</option>
          <option value="tropical">🌴 Tropical</option>
          <option value="volcanic">🌋 Volcanic</option>
        </select>

        <select value={formData.difficulty} onChange={(e) => handleInputChange('difficulty', e.target.value)}>
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
          <option value="extreme">Extreme</option>
        </select>

        <select value={formData.size} onChange={(e) => handleInputChange('size', e.target.value)}>
          <option value="10x10">10x10</option>
          <option value="20x20">20x20</option>
          <option value="30x30">30x30</option>
          <option value="50x50">50x50</option>
        </select>

        <input
          type="text"
          value={formData.seed}
          onChange={(e) => handleInputChange('seed', e.target.value)}
          placeholder="Seed"
        />

        <button onClick={generateRandomSeed}>🎲 Random Seed</button>
        <button onClick={handleGenerate}>{isLoading ? '⏳ Generating...' : '⚡ Generate'}</button>
        {tilemap && <button onClick={exportTilemap}>💾 Export JSON</button>}
      </div>

      <div className="mt-6">
        {error && <div className="text-red-500">{error}</div>}
        <canvas ref={canvasRef} className="border mt-4" />
      </div>
    </div>
  );
};

export default TilemapGeneratorUI;
