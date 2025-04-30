import { createPlayer, updatePlayer } from './player.js';
import { updateEnemies } from './enemies.js';
import { shootBullet, updateBullets } from './bullets.js';
import { setupInput } from './input.js';
import { startGame, gameOver, updateHealthBar, gameState } from './gameState.js';
import { sendFrameUpdate, fetchEnemyKernels, fetchWorldSetup } from './network.js';
import { setupScene } from './sceneSetup.js';

window.addEventListener('DOMContentLoaded', async () => {
  const renderCanvas = document.getElementById('renderCanvas');
  const pressureCanvas = document.getElementById("overlay");

  const startButton = document.getElementById("startButton");
  const restartButton = document.getElementById("restartButton");

  if (!startButton || !restartButton) {
    console.error("Start or Restart button not found in the DOM!");
  } else {
    startButton.addEventListener("click", () => startGame(player, enemies, bullets));
    restartButton.addEventListener("click", () => startGame(player, enemies, bullets));
  }

  const engine = new BABYLON.Engine(renderCanvas, true, { antialias: true });

  await fetchWorldSetup(); // Fetch world setup from the server
  const { scene, camera, shadowGenerator } = setupScene(engine);

  // Initialize game objects
  const player = createPlayer(scene);
  const enemies = {};
  const bullets = [];

  await fetchEnemyKernels(); // Fetch enemy kernels from the server

  setInterval(() => {
    if (gameState === "running") {
      sendFrameUpdate(player, enemies, scene, shadowGenerator);
    }
  }, 500);

  // Input handling
  setupInput(renderCanvas, player, () => shootBullet(scene, player, bullets), scene);

  // Game loop
  scene.onBeforeRenderObservable.add(() => {
    if (gameState === "running") {
      camera.target.copyFrom(player.mesh.position);
      updatePlayer(player);
      updateEnemies(enemies, player);
      updateBullets(bullets, enemies);
      updateHealthBar(player);

      if (player.health <= 0) {
        gameOver();
      }
    }
  });

  engine.runRenderLoop(() => scene.render());
});