'use client';

import Link from 'next/link';
import { useState, useEffect, useRef } from 'react';
import styles from './Hero.module.css';
import {
    IconChart,
    IconRobot,
    IconTrend,
    IconSearch,
    IconLightning,
    IconShield,
    IconDatabase,
    IconSparkles,
    IconTarget,
    IconRefresh
} from '@/components/icons';

// Feature icons that float/scatter - using custom SVG icons
const featureIcons = [
    { Icon: IconChart, label: 'Analytics' },
    { Icon: IconRobot, label: 'AI' },
    { Icon: IconTrend, label: 'Trends' },
    { Icon: IconSearch, label: 'Search' },
    { Icon: IconLightning, label: 'Fast' },
    { Icon: IconShield, label: 'Secure' },
    { Icon: IconDatabase, label: 'Data' },
    { Icon: IconSparkles, label: 'Insights' },
    { Icon: IconTarget, label: 'Accurate' },
    { Icon: IconRefresh, label: 'Sync' },
];

function TypewriterHero({ texts, speed = 40 }) {
    const [currentTextIndex, setCurrentTextIndex] = useState(0);
    const [displayText, setDisplayText] = useState('');
    const [isDeleting, setIsDeleting] = useState(false);

    useEffect(() => {
        const currentFullText = texts[currentTextIndex];

        const timeout = setTimeout(() => {
            if (!isDeleting) {
                if (displayText.length < currentFullText.length) {
                    setDisplayText(currentFullText.slice(0, displayText.length + 1));
                } else {
                    // Pause at end
                    setTimeout(() => setIsDeleting(true), 2000);
                }
            } else {
                if (displayText.length > 0) {
                    setDisplayText(displayText.slice(0, -1));
                } else {
                    setIsDeleting(false);
                    setCurrentTextIndex((prev) => (prev + 1) % texts.length);
                }
            }
        }, isDeleting ? speed / 2 : speed);

        return () => clearTimeout(timeout);
    }, [displayText, isDeleting, currentTextIndex, texts, speed]);

    return (
        <span className={styles.typewriterText}>
            {displayText}
            <span className={styles.cursor}>|</span>
        </span>
    );
}

export default function Hero() {
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const heroRef = useRef(null);

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (heroRef.current) {
                const rect = heroRef.current.getBoundingClientRect();
                setMousePos({
                    x: (e.clientX - rect.left - rect.width / 2) / 30,
                    y: (e.clientY - rect.top - rect.height / 2) / 30,
                });
            }
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);

    const heroTexts = [
        'Your data has stories to tell.',
        'Insights emerge from chaos.',
        'Patterns reveal themselves.',
    ];

    return (
        <section ref={heroRef} className={styles.hero}>
            <div className={styles.content}>
                <h1 className={styles.title}>
                    <TypewriterHero texts={heroTexts} speed={50} />
                    <br />
                    <span className={styles.subtitle}>
                        We help you <span className={styles.italic}>listen.</span>
                    </span>
                </h1>

                <p className={styles.description}>
                    NeuralAnalyst is your AI-powered data analyst. Ask questions in plain English,
                    get insights in seconds. No coding required.
                </p>

                <div className={styles.actions}>
                    <Link href="/register" className={styles.primaryBtn}>
                        Start analyzing
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                    <Link href="/docs" className={styles.secondaryBtn}>
                        Documentation
                    </Link>
                </div>
            </div>

            <div className={styles.scrollIndicator}>
                <span>Scroll to explore</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 5v14M5 12l7 7 7-7" />
                </svg>
            </div>
        </section>
    );
}
