export function shootBullet(scene, player, bullets) {
  const bullet = BABYLON.MeshBuilder.CreateSphere("bullet", { diameter: 0.2 }, scene);
  bullet.position.copyFrom(player.mesh.position);
  bullet.material = new BABYLON.StandardMaterial("bulletMaterial", scene);
  bullet.material.diffuseColor = new BABYLON.Color3(1, 1, 0);

  const direction = new BABYLON.Vector3(
    Math.sin(player.mesh.rotation.y),
    0,
    Math.cos(player.mesh.rotation.y)
  ).normalize();

  const speed = 0.5;
  bullets.push({ mesh: bullet, direction, speed });

  setTimeout(() => bullet.dispose(), 3000); // Dispose after 3 seconds
}

export function updateBullets(bullets, enemies) {
  bullets.forEach((bullet, bulletIndex) => {
    bullet.mesh.position.addInPlace(bullet.direction.scale(bullet.speed));

    Object.keys(enemies).forEach((enemyId) => {
        const enemy = enemies[enemyId];
        const toEnemy = enemy.mesh.position.subtract(bullet.mesh.position);
        if (toEnemy.lengthSquared() < 0.5) { // Collision detected
          enemy.mesh.dispose(); // Remove enemy
          delete enemies[enemyId]; // Remove enemy from list
          bullet.mesh.dispose(); // Remove bullet
          bullets.splice(bulletIndex, 1); // Remove bullet from list
        }
      });
  });
}
