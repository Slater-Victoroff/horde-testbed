export let gameState = "start";

export function startGame(player, enemies, bullets) {
    console.log("Starting game...");
    gameState = "running";

    // Reset player health
    player.health = 100;

    // Clear enemies and bullets
    Object.keys(enemies).forEach((id) => {
        enemies[id].mesh.dispose(); // Remove enemy mesh from the scene
        delete enemies[id];
    });
    bullets.forEach((bullet) => bullet.mesh.dispose()); // Remove bullet meshes
    bullets.length = 0; // Clear the bullets array

    // Hide UI screens
    document.getElementById("startScreen").style.display = "none";
    document.getElementById("gameOverScreen").style.display = "none";

    // Reset health bar
    updateHealthBar(player);
}
  
export function gameOver() {
    gameState = "gameOver";
    document.getElementById("gameOverScreen").style.display = "flex";
}

export function updateHealthBar(player) {
    const healthBar = document.getElementById("healthBar");
    healthBar.style.width = `${player.health * 2}px`; // Scale health to 200px max
}
