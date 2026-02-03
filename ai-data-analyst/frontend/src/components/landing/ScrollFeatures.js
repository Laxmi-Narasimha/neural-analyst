'use client';

import { useEffect, useRef, useState } from 'react';
import styles from './ScrollFeatures.module.css';

const categories = [
    {
        id: 'intelligent-analysis',
        title: 'Intelligent Analysis',
        description: 'Our AI understands your data at a fundamental level, identifying patterns, anomalies, and insights that would take humans weeks to discover.',
        features: [
            'Natural language queries',
            'Automatic pattern detection',
            'Anomaly identification',
            'Trend forecasting'
        ],
        visual: 'analysis'
    },
    {
        id: 'visualization',
        title: 'Visual Storytelling',
        description: 'Transform raw numbers into compelling narratives. Every chart, every graph, carefully crafted to communicate your data\'s story.',
        features: [
            'Auto-generated charts',
            'Interactive dashboards',
            'Custom visualizations',
            'Export-ready reports'
        ],
        visual: 'visualization'
    },
    {
        id: 'ml-models',
        title: 'Machine Learning',
        description: 'Build predictive models without writing a single line of code. Our platform handles the complexity so you can focus on decisions.',
        features: [
            'AutoML pipeline',
            'Model comparison',
            'Feature engineering',
            'Performance tracking'
        ],
        visual: 'ml'
    },
    {
        id: 'security',
        title: 'Enterprise Security',
        description: 'Your data never leaves your control. Bring your own API keys, maintain full ownership, and meet the strictest compliance requirements.',
        features: [
            'BYOK architecture',
            'SOC 2 certified',
            'Data encryption',
            'Access controls'
        ],
        visual: 'security'
    }
];

export default function ScrollFeatures() {
    const [activeIndex, setActiveIndex] = useState(0);
    const [scrollProgress, setScrollProgress] = useState(0);
    const containerRef = useRef(null);
    const sectionRefs = useRef([]);

    useEffect(() => {
        const handleScroll = () => {
            if (!containerRef.current) return;

            const container = containerRef.current;
            const rect = container.getBoundingClientRect();
            const viewportHeight = window.innerHeight;

            // Calculate how far we've scrolled through the container
            const containerTop = rect.top;
            const containerHeight = rect.height;
            const scrolled = -containerTop;
            const totalScrollable = containerHeight - viewportHeight;

            if (scrolled >= 0 && scrolled <= totalScrollable) {
                const progress = scrolled / totalScrollable;
                setScrollProgress(progress);

                // Determine active section
                const sectionProgress = progress * categories.length;
                const newIndex = Math.min(Math.floor(sectionProgress), categories.length - 1);
                setActiveIndex(Math.max(0, newIndex));
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <section ref={containerRef} className={styles.container}>
            <div className={styles.sticky}>
                {/* Left Panel - Category Info */}
                <div className={styles.leftPanel}>
                    <div className={styles.categoryIndicator}>
                        <span className={styles.categoryNumber}>
                            {String(activeIndex + 1).padStart(2, '0')}
                        </span>
                        <span className={styles.categoryTotal}>
                            / {String(categories.length).padStart(2, '0')}
                        </span>
                    </div>

                    <div className={styles.categoryContent}>
                        {categories.map((category, index) => (
                            <div
                                key={category.id}
                                className={`${styles.categoryItem} ${index === activeIndex ? styles.active : ''}`}
                            >
                                <h2 className={styles.categoryTitle}>{category.title}</h2>
                                <p className={styles.categoryDescription}>{category.description}</p>

                                <ul className={styles.featureList}>
                                    {category.features.map((feature, i) => (
                                        <li key={i} className={styles.featureItem}>
                                            <span className={styles.featureDot} />
                                            {feature}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>

                    {/* Progress Dots */}
                    <div className={styles.progressDots}>
                        {categories.map((_, index) => (
                            <button
                                key={index}
                                className={`${styles.progressDot} ${index === activeIndex ? styles.activeDot : ''}`}
                                onClick={() => {
                                    const container = containerRef.current;
                                    if (container) {
                                        const targetScroll = container.offsetTop + (index / categories.length) * (container.offsetHeight - window.innerHeight);
                                        window.scrollTo({ top: targetScroll, behavior: 'smooth' });
                                    }
                                }}
                            />
                        ))}
                    </div>
                </div>

                {/* Right Panel - Visual */}
                <div className={styles.rightPanel}>
                    <div className={styles.visualContainer}>
                        {categories.map((category, index) => (
                            <div
                                key={category.id}
                                className={`${styles.visual} ${index === activeIndex ? styles.activeVisual : ''}`}
                            >
                                <div className={styles.visualWindow}>
                                    <div className={styles.windowHeader}>
                                        <div className={styles.windowDots}>
                                            <span className={styles.dot} style={{ background: '#FF5F57' }} />
                                            <span className={styles.dot} style={{ background: '#FEBC2E' }} />
                                            <span className={styles.dot} style={{ background: '#28C840' }} />
                                        </div>
                                        <span className={styles.windowTitle}>{category.title}</span>
                                    </div>
                                    <div className={styles.windowContent}>
                                        <VisualContent type={category.visual} />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}

function VisualContent({ type }) {
    switch (type) {
        case 'analysis':
            return (
                <div className={styles.codeBlock}>
                    <div className={styles.codeLine}>
                        <span className={styles.codeComment}>// Natural language query</span>
                    </div>
                    <div className={styles.codeLine}>
                        <span className={styles.codeKeyword}>analyze</span>
                        <span className={styles.codeString}>"Show me revenue trends by region"</span>
                    </div>
                    <div className={styles.codeLine}>&nbsp;</div>
                    <div className={styles.codeLine}>
                        <span className={styles.codeComment}>// AI Response</span>
                    </div>
                    <div className={styles.codeLine}>
                        <span className={styles.codeProperty}>insights</span>: [
                    </div>
                    <div className={styles.codeLine}>
                        &nbsp;&nbsp;<span className={styles.codeString}>"APAC grew 47% YoY"</span>,
                    </div>
                    <div className={styles.codeLine}>
                        &nbsp;&nbsp;<span className={styles.codeString}>"Europe shows seasonal pattern"</span>,
                    </div>
                    <div className={styles.codeLine}>
                        &nbsp;&nbsp;<span className={styles.codeString}>"Anomaly detected Q3"</span>
                    </div>
                    <div className={styles.codeLine}>]</div>
                </div>
            );
        case 'visualization':
            return (
                <div className={styles.chartPreview}>
                    <div className={styles.chartBars}>
                        <div className={styles.chartBar} style={{ height: '60%' }} />
                        <div className={styles.chartBar} style={{ height: '80%' }} />
                        <div className={styles.chartBar} style={{ height: '45%' }} />
                        <div className={styles.chartBar} style={{ height: '90%' }} />
                        <div className={styles.chartBar} style={{ height: '70%' }} />
                        <div className={styles.chartBar} style={{ height: '85%' }} />
                    </div>
                    <div className={styles.chartLine} />
                    <div className={styles.chartLabels}>
                        <span>Jan</span><span>Feb</span><span>Mar</span><span>Apr</span><span>May</span><span>Jun</span>
                    </div>
                </div>
            );
        case 'ml':
            return (
                <div className={styles.mlPreview}>
                    <div className={styles.mlStep}>
                        <span className={styles.mlIcon}>◆</span>
                        <span>Data Preparation</span>
                        <span className={styles.mlCheck}>✓</span>
                    </div>
                    <div className={styles.mlStep}>
                        <span className={styles.mlIcon}>◆</span>
                        <span>Feature Engineering</span>
                        <span className={styles.mlCheck}>✓</span>
                    </div>
                    <div className={styles.mlStep}>
                        <span className={styles.mlIcon}>◆</span>
                        <span>Model Training</span>
                        <span className={styles.mlProgress}>87%</span>
                    </div>
                    <div className={styles.mlStep} style={{ opacity: 0.4 }}>
                        <span className={styles.mlIcon}>◇</span>
                        <span>Validation</span>
                    </div>
                    <div className={styles.mlStep} style={{ opacity: 0.4 }}>
                        <span className={styles.mlIcon}>◇</span>
                        <span>Deployment</span>
                    </div>
                </div>
            );
        case 'security':
            return (
                <div className={styles.securityPreview}>
                    <div className={styles.securityRow}>
                        <span className={styles.securityLabel}>Encryption</span>
                        <span className={styles.securityValue}>AES-256</span>
                    </div>
                    <div className={styles.securityRow}>
                        <span className={styles.securityLabel}>API Keys</span>
                        <span className={styles.securityValue}>Your Own</span>
                    </div>
                    <div className={styles.securityRow}>
                        <span className={styles.securityLabel}>Data Storage</span>
                        <span className={styles.securityValue}>Client-side</span>
                    </div>
                    <div className={styles.securityRow}>
                        <span className={styles.securityLabel}>Compliance</span>
                        <span className={styles.securityValue}>SOC 2, GDPR</span>
                    </div>
                    <div className={styles.securityShield}>
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                            <path d="M9 12l2 2 4-4" />
                        </svg>
                    </div>
                </div>
            );
        default:
            return null;
    }
}
