import { renderPressureField, renderFlowField } from './rendering.js';
import { spawnEnemy, exportEnemies } from './enemies.js';

export const enemyKernels = {}; // Central storage for enemy kernels

export async function fetchEnemyKernels() {
    try {
        const response = await fetch('http://3.226.12.50:8000/enemy-kernels'); // Backend endpoint to serve the JSON
        const kernels = await response.json();
        Object.assign(enemyKernels, kernels); // Store the fetched kernels
        console.log("Fetched enemy kernels:", enemyKernels);
    } catch (err) {
        console.error("Failed to fetch enemy kernels:", err);
    }
}

export const worldSetupConfig = {}; // Central storage for world setup configuration

export async function fetchWorldSetup() {
    try {
        const response = await fetch('http://3.226.12.50:8000/world-setup'); // Backend endpoint
        const setup = await response.json();
        Object.assign(worldSetupConfig, setup); // Store the fetched world setup
        console.log("Fetched world setup:", worldSetupConfig);
    } catch (err) {
        console.error("Failed to fetch world setup:", err);
    }
}

export async function sendFrameUpdate(player, enemies, scene, shadowGenerator) {
  const { enemyStates, kinds } = exportEnemies(enemies);

  const playerState = [
    player.position.x,
    player.position.y,
    player.position.z,
    player.velocity.x,
    player.velocity.y,
    player.velocity.z,
    player.acceleration.x,
    player.acceleration.y,
    player.acceleration.z,
  ];

  const payload = {
    playerState: playerState,
    enemies: {
      states: enemyStates,
      kinds: kinds,
    },
  };

  try {
    const res = await fetch('http://3.226.12.50:8000/frame-update', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (data.pressure_field) {
      renderPressureField(data.pressure_field);
    }

    if (data.flow_field) {
      renderFlowField(data.flow_field);
    }

    if (data.difficulty !== undefined) {
        const difficultyLabel = document.getElementById("difficultyLabel");
        difficultyLabel.textContent = `Difficulty: ${data.difficulty}`;
    }

    data.spawn.forEach(spawn => {
      if (!enemies[spawn.id]) {
        spawnEnemy(spawn.id, spawn.position, spawn.kind, scene, enemies, shadowGenerator);
      }
    });
  } catch (err) {
    console.error("Frame update failed:", err);
  }
}
