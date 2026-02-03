'use client';

import { useEffect, useRef } from 'react';

export default function FluidBackground() {
    const canvasRef = useRef(null);
    const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });
    const blobsRef = useRef([]);
    const animationRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let width = window.innerWidth;
        let height = window.innerHeight;

        const resize = () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
            initBlobs();
        };

        // Very subtle, large blobs
        const initBlobs = () => {
            blobsRef.current = [];
            const blobCount = 5;
            const colors = [
                { r: 240, g: 235, b: 228 },  // Warm cream
                { r: 235, g: 232, b: 245 },  // Very light lavender
                { r: 232, g: 242, b: 245 },  // Very light blue
                { r: 245, g: 238, b: 230 },  // Warm beige
                { r: 238, g: 235, b: 230 },  // Neutral warm
            ];

            for (let i = 0; i < blobCount; i++) {
                const color = colors[i % colors.length];
                blobsRef.current.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: 0,
                    vy: 0,
                    baseX: Math.random() * width,
                    baseY: Math.random() * height,
                    radius: 300 + Math.random() * 400,
                    color: color,
                    opacity: 0.4 + Math.random() * 0.2,
                    phase: Math.random() * Math.PI * 2,
                    speed: 0.0002 + Math.random() * 0.0003,
                });
            }
        };

        const handleMouseMove = (e) => {
            mouseRef.current.targetX = e.clientX;
            mouseRef.current.targetY = e.clientY;
        };

        const animate = () => {
            // Smooth mouse following
            mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.04;
            mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.04;

            // Cream background
            ctx.fillStyle = '#FAF9F7';
            ctx.fillRect(0, 0, width, height);

            const mouse = mouseRef.current;
            const time = Date.now();

            // Update and draw blobs
            blobsRef.current.forEach((blob) => {
                // Very gentle floating motion
                blob.baseX += Math.sin(time * blob.speed + blob.phase) * 0.15;
                blob.baseY += Math.cos(time * blob.speed * 0.8 + blob.phase) * 0.15;

                // Keep base within bounds
                if (blob.baseX < -blob.radius) blob.baseX = width + blob.radius;
                if (blob.baseX > width + blob.radius) blob.baseX = -blob.radius;
                if (blob.baseY < -blob.radius) blob.baseY = height + blob.radius;
                if (blob.baseY > height + blob.radius) blob.baseY = -blob.radius;

                // Subtle mouse interaction
                const dx = mouse.x - blob.x;
                const dy = mouse.y - blob.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const maxDist = 400;

                if (dist < maxDist && dist > 0) {
                    const force = ((maxDist - dist) / maxDist) * 1.5;
                    blob.vx -= (dx / dist) * force;
                    blob.vy -= (dy / dist) * force;
                }

                // Apply velocity with damping
                blob.vx *= 0.96;
                blob.vy *= 0.96;

                // Return to base position
                blob.vx += (blob.baseX - blob.x) * 0.004;
                blob.vy += (blob.baseY - blob.y) * 0.004;

                blob.x += blob.vx;
                blob.y += blob.vy;

                // Draw blob with very soft gradient
                const blobGradient = ctx.createRadialGradient(
                    blob.x, blob.y, 0,
                    blob.x, blob.y, blob.radius
                );
                blobGradient.addColorStop(0, `rgba(${blob.color.r}, ${blob.color.g}, ${blob.color.b}, ${blob.opacity})`);
                blobGradient.addColorStop(0.6, `rgba(${blob.color.r}, ${blob.color.g}, ${blob.color.b}, ${blob.opacity * 0.3})`);
                blobGradient.addColorStop(1, `rgba(${blob.color.r}, ${blob.color.g}, ${blob.color.b}, 0)`);

                ctx.beginPath();
                ctx.arc(blob.x, blob.y, blob.radius, 0, Math.PI * 2);
                ctx.fillStyle = blobGradient;
                ctx.fill();
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        resize();
        window.addEventListener('resize', resize);
        window.addEventListener('mousemove', handleMouseMove);
        animate();

        return () => {
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', handleMouseMove);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: -1
            }}
        />
    );
}
