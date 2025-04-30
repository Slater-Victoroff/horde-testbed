export function renderPressureField(pressure) {
    const pressureCanvas = document.getElementById("scalarMap");
    const pressureCtx = pressureCanvas.getContext("2d");

    pressureCtx.clearRect(0, 0, pressureCanvas.width, pressureCanvas.height);
    const size = pressure.length;
    const cellW = pressureCanvas.width / size;
    const cellH = pressureCanvas.height / size;

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const value = pressure[j][i];
            const alpha = Math.min(value * 5, 1.0);  // scale for visibility
            pressureCtx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
            pressureCtx.fillRect(i * cellW, (size - 1 - j) * cellH, cellW, cellH); // Flip the y-axis
        }
    }
}

export function renderFlowField(flowField) {
    const canvas = document.getElementById("vectorMap");
    const ctx = canvas.getContext("2d");

    const size = flowField.length;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cellW = canvas.width / size;
    const cellH = canvas.height / size;

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const [vx, vz] = flowField[j][i];
            const x = i * cellW + cellW / 2;
            const y = (size - 1 - j) * cellH + cellH / 2; // Flip the y-axis

            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + vx * 10, y - vz * 10); // Scale vector for visibility
            ctx.strokeStyle = "rgba(0, 255, 0, 0.9)";
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
}
