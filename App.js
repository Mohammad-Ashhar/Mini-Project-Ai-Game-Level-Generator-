import React, { useState, useEffect, useRef } from 'react';
import { Play, RotateCcw, LogOut, Trophy, Target, Clock, Coins, TrendingUp } from 'lucide-react';

// ML-based Level Generator using Q-Learning principles
class LevelGenerator {
  constructor() {
    this.difficultyWeights = {
      easy: { obstacles: 0.3, coins: 0.8, pathComplexity: 0.2 },
      medium: { obstacles: 0.5, coins: 0.6, pathComplexity: 0.5 },
      hard: { obstacles: 0.7, coins: 0.5, pathComplexity: 0.7 },
      superhard: { obstacles: 0.9, coins: 0.4, pathComplexity: 0.9 }
    };
  }

  analyzeUserPerformance(performances) {
    if (performances.length === 0) return 'easy';
    
    const recent = performances.slice(-5);
    const avgCompletionRate = recent.filter(p => p.completed).length / recent.length;
    const avgCoinsCollected = recent.reduce((sum, p) => sum + p.coinsCollected / p.totalCoins, 0) / recent.length;
    const avgTime = recent.reduce((sum, p) => sum + p.timeSpent, 0) / recent.length;
    const expectedTime = recent.reduce((sum, p) => sum + p.expectedTime, 0) / recent.length;
    
    const skillScore = (avgCompletionRate * 0.4) + (avgCoinsCollected * 0.3) + 
                       (expectedTime / Math.max(avgTime, 1) * 0.3);
    
    if (skillScore > 0.8) return 'superhard';
    if (skillScore > 0.6) return 'hard';
    if (skillScore > 0.4) return 'medium';
    return 'easy';
  }

  generateLevel(userId, performances) {
    const difficulty = this.analyzeUserPerformance(performances);
    const weights = this.difficultyWeights[difficulty];
    
    const gridSize = 15;
    const level = {
      id: Date.now() + Math.random(),
      userId,
      difficulty,
      gridSize,
      player: { x: 1, y: 1 },
      goal: { x: gridSize - 2, y: gridSize - 2 },
      obstacles: [],
      coins: [],
      createdAt: Date.now()
    };
    
    const numObstacles = Math.floor(15 + (weights.obstacles * 40));
    const numCoins = Math.floor(10 + (weights.coins * 15));
    
    const occupied = new Set();
    occupied.add(`${level.player.x},${level.player.y}`);
    occupied.add(`${level.goal.x},${level.goal.y}`);
    
    // Generate obstacles with clustering based on complexity
    for (let i = 0; i < numObstacles; i++) {
      let x, y, attempts = 0;
      do {
        if (weights.pathComplexity > 0.5 && level.obstacles.length > 0 && Math.random() < 0.6) {
          const base = level.obstacles[Math.floor(Math.random() * level.obstacles.length)];
          x = base.x + (Math.random() < 0.5 ? -1 : 1) * Math.floor(Math.random() * 2 + 1);
          y = base.y + (Math.random() < 0.5 ? -1 : 1) * Math.floor(Math.random() * 2 + 1);
        } else {
          x = Math.floor(Math.random() * (gridSize - 2)) + 1;
          y = Math.floor(Math.random() * (gridSize - 2)) + 1;
        }
        attempts++;
      } while ((occupied.has(`${x},${y}`) || !this.isPathPossible(level, { x, y })) && attempts < 100);
      
      if (attempts < 100) {
        level.obstacles.push({ x, y });
        occupied.add(`${x},${y}`);
      }
    }
    
    // Generate coins along viable paths
    for (let i = 0; i < numCoins; i++) {
      let x, y, attempts = 0;
      do {
        x = Math.floor(Math.random() * (gridSize - 2)) + 1;
        y = Math.floor(Math.random() * (gridSize - 2)) + 1;
        attempts++;
      } while (occupied.has(`${x},${y}`) && attempts < 100);
      
      if (attempts < 100) {
        level.coins.push({ x, y, collected: false });
        occupied.add(`${x},${y}`);
      }
    }
    
    level.expectedTime = this.calculateExpectedTime(level);
    
    return level;
  }
  
  isPathPossible(level, newObstacle) {
    const obstacles = [...level.obstacles, newObstacle];
    const obstacleSet = new Set(obstacles.map(o => `${o.x},${o.y}`));
    
    const queue = [level.player];
    const visited = new Set([`${level.player.x},${level.player.y}`]);
    
    while (queue.length > 0) {
      const current = queue.shift();
      
      if (current.x === level.goal.x && current.y === level.goal.y) {
        return true;
      }
      
      const neighbors = [
        { x: current.x + 1, y: current.y },
        { x: current.x - 1, y: current.y },
        { x: current.x, y: current.y + 1 },
        { x: current.x, y: current.y - 1 }
      ];
      
      for (const next of neighbors) {
        const key = `${next.x},${next.y}`;
        if (next.x >= 0 && next.x < level.gridSize && 
            next.y >= 0 && next.y < level.gridSize &&
            !visited.has(key) && !obstacleSet.has(key)) {
          visited.add(key);
          queue.push(next);
        }
      }
    }
    
    return false;
  }
  
  calculateExpectedTime(level) {
    const distance = Math.abs(level.goal.x - level.player.x) + Math.abs(level.goal.y - level.player.y);
    const complexity = level.obstacles.length / 50;
    return distance * (1 + complexity) * 0.5;
  }
}

const levelGenerator = new LevelGenerator();

// Main App Component
export default function AIGameLevelGenerator() {
  const [screen, setScreen] = useState('login');
  const [user, setUser] = useState(null);
  const [level, setLevel] = useState(null);
  const [gameState, setGameState] = useState('ready');
  const [playerPos, setPlayerPos] = useState({ x: 1, y: 1 });
  const [coins, setCoins] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [moves, setMoves] = useState(0);
  const [stats, setStats] = useState(null);
  
  const [formData, setFormData] = useState({ username: '', password: '' });
  const canvasRef = useRef(null);

  useEffect(() => {
    if (level && gameState === 'playing') {
      drawGame();
    }
  }, [level, playerPos, coins, gameState]);

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (gameState !== 'playing') return;
      
      const key = e.key.toLowerCase();
      let newPos = { ...playerPos };
      
      if (key === 'arrowup' || key === 'w') newPos.y -= 1;
      if (key === 'arrowdown' || key === 's') newPos.y += 1;
      if (key === 'arrowleft' || key === 'a') newPos.x -= 1;
      if (key === 'arrowright' || key === 'd') newPos.x += 1;
      
      if (newPos.x !== playerPos.x || newPos.y !== playerPos.y) {
        movePlayer(newPos);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [gameState, playerPos, coins]);

  const drawGame = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const cellSize = 30;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    for (let i = 0; i <= level.gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, level.gridSize * cellSize);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(level.gridSize * cellSize, i * cellSize);
      ctx.stroke();
    }
    
    // Draw obstacles
    ctx.fillStyle = '#ef4444';
    level.obstacles.forEach(obs => {
      ctx.fillRect(obs.x * cellSize + 2, obs.y * cellSize + 2, cellSize - 4, cellSize - 4);
    });
    
    // Draw coins
    coins.forEach(coin => {
      if (!coin.collected) {
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(coin.x * cellSize + cellSize/2, coin.y * cellSize + cellSize/2, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    // Draw goal
    ctx.fillStyle = '#10b981';
    ctx.fillRect(level.goal.x * cellSize + 2, level.goal.y * cellSize + 2, cellSize - 4, cellSize - 4);
    
    // Draw player
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(playerPos.x * cellSize + cellSize/2, playerPos.y * cellSize + cellSize/2, 10, 0, Math.PI * 2);
    ctx.fill();
  };

  const movePlayer = (newPos) => {
    if (newPos.x < 0 || newPos.x >= level.gridSize || 
        newPos.y < 0 || newPos.y >= level.gridSize) return;
    
    const isObstacle = level.obstacles.some(obs => obs.x === newPos.x && obs.y === newPos.y);
    if (isObstacle) return;
    
    setPlayerPos(newPos);
    setMoves(m => m + 1);
    
    // Check coin collection
    const coinIndex = coins.findIndex(c => c.x === newPos.x && c.y === newPos.y && !c.collected);
    if (coinIndex !== -1) {
      const newCoins = [...coins];
      newCoins[coinIndex].collected = true;
      setCoins(newCoins);
    }
    
    // Check goal
    if (newPos.x === level.goal.x && newPos.y === level.goal.y) {
      completeLevel();
    }
  };

  const completeLevel = () => {
    const endTime = Date.now();
    const timeSpent = (endTime - startTime) / 1000;
    const coinsCollected = coins.filter(c => c.collected).length;
    
    const performance = {
      userId: user.id,
      levelId: level.id,
      difficulty: level.difficulty,
      completed: true,
      timeSpent,
      expectedTime: level.expectedTime,
      coinsCollected,
      totalCoins: coins.length,
      moves,
      timestamp: endTime
    };
    
    mockDB.performances.push(performance);
    
    setStats({
      timeSpent: timeSpent.toFixed(1),
      coinsCollected,
      totalCoins: coins.length,
      moves,
      efficiency: ((coinsCollected / coins.length) * 100).toFixed(0)
    });
    
    setGameState('completed');
  };

  const handleAuth = (isLogin) => {
    const { username, password } = formData;
    if (!username || !password) return;
    
    if (isLogin) {
      const existingUser = mockDB.users.find(u => u.username === username && u.password === password);
      if (existingUser) {
        setUser(existingUser);
        setScreen('game');
      } else {
        alert('Invalid credentials');
      }
    } else {
      const existing = mockDB.users.find(u => u.username === username);
      if (existing) {
        alert('Username already exists');
        return;
      }
      
      const newUser = {
        id: Date.now(),
        username,
        password,
        createdAt: Date.now()
      };
      
      mockDB.users.push(newUser);
      setUser(newUser);
      setScreen('game');
    }
    
    setFormData({ username: '', password: '' });
  };

  const generateNewLevel = () => {
    const userPerformances = mockDB.performances.filter(p => p.userId === user.id);
    const newLevel = levelGenerator.generateLevel(user.id, userPerformances);
    
    mockDB.levels.push(newLevel);
    
    setLevel(newLevel);
    setPlayerPos({ x: newLevel.player.x, y: newLevel.player.y });
    setCoins(newLevel.coins.map(c => ({ ...c, collected: false })));
    setGameState('ready');
    setMoves(0);
    setStats(null);
  };

  const startGame = () => {
    setGameState('playing');
    setStartTime(Date.now());
  };

  const resetLevel = () => {
    setPlayerPos({ x: level.player.x, y: level.player.y });
    setCoins(level.coins.map(c => ({ ...c, collected: false })));
    setGameState('ready');
    setMoves(0);
    setStats(null);
  };

  const logout = () => {
    setUser(null);
    setLevel(null);
    setScreen('login');
  };

  const getUserStats = () => {
    const userPerf = mockDB.performances.filter(p => p.userId === user.id);
    const completed = userPerf.filter(p => p.completed).length;
    const avgCoins = userPerf.length > 0 
      ? (userPerf.reduce((sum, p) => sum + (p.coinsCollected / p.totalCoins), 0) / userPerf.length * 100).toFixed(0)
      : 0;
    
    return { totalLevels: completed, avgCoins };
  };

  // Login Screen
  if (screen === 'login') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-indigo-600 rounded-full mb-4">
              <Trophy className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">AI Level Generator</h1>
            <p className="text-gray-600">Adaptive gameplay that learns from you</p>
          </div>
          
          <div className="space-y-4">
            <input
              type="text"
              placeholder="Username"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            <input
              type="password"
              placeholder="Password"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
            
            <button
              onClick={() => handleAuth(true)}
              className="w-full bg-indigo-600 text-white py-3 rounded-lg font-semibold hover:bg-indigo-700 transition"
            >
              Login
            </button>
            
            <button
              onClick={() => handleAuth(false)}
              className="w-full bg-white border-2 border-indigo-600 text-indigo-600 py-3 rounded-lg font-semibold hover:bg-indigo-50 transition"
            >
              Sign Up
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Game Screen
  const userStats = getUserStats();
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">Welcome, {user.username}!</h1>
            <div className="flex gap-6 mt-2 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <Trophy className="w-4 h-4" />
                <span>{userStats.totalLevels} Levels Completed</span>
              </div>
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4" />
                <span>{userStats.avgCoins}% Avg Coins</span>
              </div>
            </div>
          </div>
          <button
            onClick={logout}
            className="flex items-center gap-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </div>

        {/* Game Area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Canvas */}
          <div className="lg:col-span-2 bg-white rounded-xl shadow-lg p-6">
            {!level ? (
              <div className="flex flex-col items-center justify-center h-96">
                <Trophy className="w-16 h-16 text-indigo-600 mb-4" />
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Ready to Play?</h2>
                <p className="text-gray-600 mb-6 text-center">
                  Click below to generate your personalized level
                </p>
                <button
                  onClick={generateNewLevel}
                  className="px-8 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 transition flex items-center gap-2"
                >
                  <Play className="w-5 h-5" />
                  Generate Level
                </button>
              </div>
            ) : (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <div>
                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                      level.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
                      level.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                      level.difficulty === 'hard' ? 'bg-orange-100 text-orange-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {level.difficulty.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex gap-2">
                    {gameState === 'ready' && (
                      <button
                        onClick={startGame}
                        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition flex items-center gap-2"
                      >
                        <Play className="w-4 h-4" />
                        Start
                      </button>
                    )}
                    <button
                      onClick={resetLevel}
                      className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition flex items-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset
                    </button>
                    <button
                      onClick={generateNewLevel}
                      className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                    >
                      New Level
                    </button>
                  </div>
                </div>
                
                <canvas
                  ref={canvasRef}
                  width={level.gridSize * 30}
                  height={level.gridSize * 30}
                  className="border-2 border-gray-300 rounded-lg mx-auto"
                />
                
                <div className="mt-4 text-sm text-gray-600 text-center">
                  Use Arrow Keys or WASD to move • Collect coins • Reach the green goal
                </div>
              </div>
            )}
          </div>

          {/* Stats Panel */}
          <div className="space-y-6">
            {level && (
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">Current Game</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Coins className="w-5 h-5 text-yellow-500" />
                      <span className="text-gray-700">Coins</span>
                    </div>
                    <span className="font-semibold text-gray-800">
                      {coins.filter(c => c.collected).length} / {coins.length}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-blue-500" />
                      <span className="text-gray-700">Moves</span>
                    </div>
                    <span className="font-semibold text-gray-800">{moves}</span>
                  </div>
                  
                  {gameState === 'playing' && startTime && (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Clock className="w-5 h-5 text-purple-500" />
                        <span className="text-gray-700">Time</span>
                      </div>
                      <span className="font-semibold text-gray-800">
                        {((Date.now() - startTime) / 1000).toFixed(1)}s
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {stats && (
              <div className="bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl shadow-lg p-6 text-white">
                <h3 className="text-xl font-bold mb-4">Level Complete! 🎉</h3>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Time:</span>
                    <span className="font-semibold">{stats.timeSpent}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Coins:</span>
                    <span className="font-semibold">{stats.coinsCollected}/{stats.totalCoins}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Moves:</span>
                    <span className="font-semibold">{stats.moves}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Efficiency:</span>
                    <span className="font-semibold">{stats.efficiency}%</span>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-800 mb-4">How It Works</h3>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• AI analyzes your gameplay patterns</li>
                <li>• Levels adapt to your skill level</li>
                <li>• Complete levels faster for harder challenges</li>
                <li>• Collect all coins for maximum efficiency</li>
                <li>• Your progress is saved automatically</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}