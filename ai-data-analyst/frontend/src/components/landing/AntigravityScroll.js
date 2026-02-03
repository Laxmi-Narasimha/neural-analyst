'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './AntigravityScroll.module.css';

// Core capabilities - from basic to advanced
const capabilities = [
    {
        id: 'data-ingestion',
        category: 'Data Foundation',
        headline: 'Upload any data format. We handle the complexity.',
        description: 'CSV, Excel, JSON, Parquet, SQL databases — our intelligent parser understands structure, infers types, and cleanses automatically.',
        features: [
            { icon: 'upload', label: 'Multi-format import' },
            { icon: 'columns', label: 'Schema detection' },
            { icon: 'clean', label: 'Auto-cleansing' },
            { icon: 'types', label: 'Type inference' },
            { icon: 'preview', label: 'Live preview' },
            { icon: 'sync', label: 'Real-time sync' },
        ],
        visual: 'ingestion',
        level: 'Foundation',
    },
    {
        id: 'natural-language',
        category: 'Conversational AI',
        headline: 'Ask questions in plain English. Get answers instantly.',
        description: 'Our AI understands context, remembers your previous queries, and provides answers with full transparency into its reasoning.',
        features: [
            { icon: 'chat', label: 'Natural queries' },
            { icon: 'context', label: 'Context memory' },
            { icon: 'explain', label: 'Explainable AI' },
            { icon: 'suggest', label: 'Smart suggestions' },
            { icon: 'history', label: 'Query history' },
            { icon: 'refine', label: 'Iterative refinement' },
        ],
        visual: 'nlp',
        level: 'Core',
    },
    {
        id: 'analysis-engine',
        category: 'Analysis Engine',
        headline: 'Patterns emerge. Anomalies surface. Insights crystallize.',
        description: 'Statistical analysis, trend detection, correlation discovery, and anomaly identification — all automated, all explained.',
        features: [
            { icon: 'stats', label: 'Statistical analysis' },
            { icon: 'trend', label: 'Trend detection' },
            { icon: 'correlation', label: 'Correlations' },
            { icon: 'anomaly', label: 'Anomaly detection' },
            { icon: 'segment', label: 'Segmentation' },
            { icon: 'forecast', label: 'Forecasting' },
        ],
        visual: 'analysis',
        level: 'Advanced',
    },
    {
        id: 'visualization',
        category: 'Visual Intelligence',
        headline: 'Data becomes visual. Stories become clear.',
        description: 'Auto-generated charts, interactive dashboards, and custom visualizations — each designed to communicate your data\'s narrative.',
        features: [
            { icon: 'chart', label: 'Smart charts' },
            { icon: 'dashboard', label: 'Dashboards' },
            { icon: 'interact', label: 'Interactivity' },
            { icon: 'export', label: 'Export-ready' },
            { icon: 'theme', label: 'Custom themes' },
            { icon: 'embed', label: 'Embeddable' },
        ],
        visual: 'visualization',
        level: 'Advanced',
    },
    {
        id: 'ml-models',
        category: 'Machine Learning',
        headline: 'Build predictive models. No code required.',
        description: 'AutoML pipeline with model comparison, feature engineering, hyperparameter tuning, and one-click deployment.',
        features: [
            { icon: 'automl', label: 'AutoML' },
            { icon: 'compare', label: 'Model comparison' },
            { icon: 'feature', label: 'Feature engineering' },
            { icon: 'tune', label: 'Hyperparameter tuning' },
            { icon: 'validate', label: 'Cross-validation' },
            { icon: 'deploy', label: 'One-click deploy' },
        ],
        visual: 'ml',
        level: 'Expert',
    },
    {
        id: 'enterprise',
        category: 'Enterprise Security',
        headline: 'Your data. Your keys. Your complete control.',
        description: 'BYOK architecture, SOC 2 certified, end-to-end encryption, role-based access, and audit logging for complete governance.',
        features: [
            { icon: 'key', label: 'BYOK' },
            { icon: 'encrypt', label: 'Encryption' },
            { icon: 'soc2', label: 'SOC 2' },
            { icon: 'rbac', label: 'Access control' },
            { icon: 'audit', label: 'Audit logs' },
            { icon: 'compliance', label: 'Compliance' },
        ],
        visual: 'security',
        level: 'Enterprise',
    },
];

// Icons SVGs
const iconPaths = {
    upload: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12',
    columns: 'M12 3v18M3 9h18M3 15h18',
    clean: 'M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5zM16 8L2 22M17.5 15H9',
    types: 'M4 7V4h16v3M9 20h6M12 4v16',
    preview: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
    sync: 'M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15',
    chat: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
    context: 'M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zM12 6a1 1 0 0 0-1 1v5a1 1 0 0 0 .29.71l3 3a1 1 0 0 0 1.42-1.42L13 11.59V7a1 1 0 0 0-1-1z',
    explain: 'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 8v4M12 16h.01',
    suggest: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
    history: 'M12 8v4l3 3m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0z',
    refine: 'M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z',
    stats: 'M18 20V10M12 20V4M6 20v-6',
    trend: 'M23 6l-9.5 9.5-5-5L1 18',
    correlation: 'M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM2 12h20',
    anomaly: 'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01',
    segment: 'M21.21 15.89A10 10 0 1 1 8 2.83M22 12A10 10 0 0 0 12 2v10z',
    forecast: 'M22 12h-4l-3 9L9 3l-3 9H2',
    chart: 'M3 3v18h18M18 17V9M13 17V5M8 17v-3',
    dashboard: 'M3 3h7v7H3zM14 3h7v4h-7zM14 10h7v7h-7zM3 14h7v7H3z',
    interact: 'M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122',
    export: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3',
    theme: 'M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z',
    embed: 'M16 18l6-6-6-6M8 6l-6 6 6 6',
    automl: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
    compare: 'M9 9V4.5M9 9H4.5M9 9L3.5 3.5M15 9V4.5M15 9h4.5M15 9l5.5-5.5M9 15v4.5M9 15H4.5M9 15l-5.5 5.5M15 15h4.5M15 15v4.5m0-4.5l5.5 5.5',
    feature: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
    tune: 'M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.49 8.49l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.49-8.49l2.83-2.83',
    validate: 'M22 11.08V12a10 10 0 1 1-5.93-9.14M22 4L12 14.01l-3-3',
    deploy: 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z',
    key: 'M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4',
    encrypt: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z',
    soc2: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0 1 12 2.944a11.955 11.955 0 0 1-8.618 3.04A12.02 12.02 0 0 0 3 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z',
    rbac: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75M9 7a4 4 0 1 0 0 8 4 4 0 0 0 0-8z',
    audit: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M16 13H8M16 17H8M10 9H8',
    compliance: 'M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2M9 5a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2M9 5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2m-6 9l2 2 4-4',
};

function FeatureIcon({ icon, label, delay, isVisible }) {
    const path = iconPaths[icon] || iconPaths.chart;

    return (
        <div
            className={`${styles.featureIcon} ${isVisible ? styles.iconVisible : ''}`}
            style={{ transitionDelay: `${delay}ms` }}
        >
            <div className={styles.iconCircle}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d={path} />
                </svg>
            </div>
            <span className={styles.iconLabel}>{label}</span>
        </div>
    );
}

function TypewriterText({ text, isActive, speed = 30 }) {
    const [displayedText, setDisplayedText] = useState('');
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (!isActive) {
            setDisplayedText('');
            setIsComplete(false);
            return;
        }

        let index = 0;
        setDisplayedText('');
        setIsComplete(false);

        const interval = setInterval(() => {
            if (index < text.length) {
                setDisplayedText(text.slice(0, index + 1));
                index++;
            } else {
                setIsComplete(true);
                clearInterval(interval);
            }
        }, speed);

        return () => clearInterval(interval);
    }, [text, isActive, speed]);

    return (
        <span className={styles.typewriter}>
            {displayedText}
            {!isComplete && <span className={styles.cursor}>|</span>}
        </span>
    );
}

export default function AntigravityScroll() {
    const [activeIndex, setActiveIndex] = useState(0);
    const [scrollProgress, setScrollProgress] = useState(0);
    const [iconsVisible, setIconsVisible] = useState(false);
    const containerRef = useRef(null);
    const prevIndexRef = useRef(0);

    useEffect(() => {
        const handleScroll = () => {
            if (!containerRef.current) return;

            const container = containerRef.current;
            const rect = container.getBoundingClientRect();
            const viewportHeight = window.innerHeight;

            const containerTop = rect.top;
            const containerHeight = rect.height;
            const scrolled = -containerTop;
            const totalScrollable = containerHeight - viewportHeight;

            if (scrolled >= 0 && scrolled <= totalScrollable) {
                const progress = scrolled / totalScrollable;
                setScrollProgress(progress);

                const sectionProgress = progress * capabilities.length;
                const newIndex = Math.min(Math.floor(sectionProgress), capabilities.length - 1);

                if (newIndex !== prevIndexRef.current) {
                    setIconsVisible(false);
                    setTimeout(() => setIconsVisible(true), 400);
                    prevIndexRef.current = newIndex;
                }

                setActiveIndex(Math.max(0, newIndex));
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();
        setTimeout(() => setIconsVisible(true), 500);

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const currentCapability = capabilities[activeIndex];

    return (
        <section ref={containerRef} className={styles.container}>
            <div className={styles.sticky}>
                {/* Left Panel - Content */}
                <div className={styles.leftPanel}>
                    <div className={styles.levelBadge}>
                        <span className={styles.levelDot} />
                        {currentCapability.level}
                    </div>

                    <div className={styles.headlineWrapper}>
                        <h2 className={styles.headline}>
                            <TypewriterText
                                text={currentCapability.headline}
                                isActive={true}
                                speed={25}
                            />
                        </h2>
                    </div>

                    <p className={styles.description}>
                        {currentCapability.description}
                    </p>

                    {/* Floating Feature Icons */}
                    <div className={styles.featureIcons}>
                        {currentCapability.features.map((feature, index) => (
                            <FeatureIcon
                                key={`${currentCapability.id}-${feature.icon}`}
                                icon={feature.icon}
                                label={feature.label}
                                delay={index * 80}
                                isVisible={iconsVisible}
                            />
                        ))}
                    </div>

                    {/* Progress Indicator */}
                    <div className={styles.progressBar}>
                        <div className={styles.progressTrack}>
                            {capabilities.map((cap, index) => (
                                <button
                                    key={cap.id}
                                    className={`${styles.progressStep} ${index === activeIndex ? styles.activeStep : ''} ${index < activeIndex ? styles.completedStep : ''}`}
                                    onClick={() => {
                                        const container = containerRef.current;
                                        if (container) {
                                            const targetScroll = container.offsetTop + (index / capabilities.length) * (container.offsetHeight - window.innerHeight);
                                            window.scrollTo({ top: targetScroll, behavior: 'smooth' });
                                        }
                                    }}
                                >
                                    <span className={styles.stepNumber}>{String(index + 1).padStart(2, '0')}</span>
                                    <span className={styles.stepLabel}>{cap.category}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Panel - Visual */}
                <div className={styles.rightPanel}>
                    <div className={styles.visualContainer}>
                        {capabilities.map((cap, index) => (
                            <div
                                key={cap.id}
                                className={`${styles.visual} ${index === activeIndex ? styles.activeVisual : ''}`}
                            >
                                <AdvancedVisual type={cap.visual} isActive={index === activeIndex} />
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </section>
    );
}

function AdvancedVisual({ type, isActive }) {
    switch (type) {
        case 'ingestion':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>Data Import</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.ingestionFlow}>
                            <div className={styles.fileCards}>
                                <div className={`${styles.fileCard} ${isActive ? styles.fileActive : ''}`}>
                                    <span className={styles.fileIcon}>📊</span>
                                    <span className={styles.fileName}>sales_2024.csv</span>
                                    <span className={styles.fileSize}>2.4 MB</span>
                                </div>
                                <div className={`${styles.fileCard} ${isActive ? styles.fileActive : ''}`} style={{ animationDelay: '0.1s' }}>
                                    <span className={styles.fileIcon}>📈</span>
                                    <span className={styles.fileName}>metrics.xlsx</span>
                                    <span className={styles.fileSize}>1.8 MB</span>
                                </div>
                                <div className={`${styles.fileCard} ${isActive ? styles.fileActive : ''}`} style={{ animationDelay: '0.2s' }}>
                                    <span className={styles.fileIcon}>🗄️</span>
                                    <span className={styles.fileName}>customers.json</span>
                                    <span className={styles.fileSize}>4.1 MB</span>
                                </div>
                            </div>
                            <div className={styles.arrow}>→</div>
                            <div className={styles.processBox}>
                                <div className={styles.processIcon}>⚙️</div>
                                <div className={styles.processText}>Processing</div>
                                <div className={styles.processBar}><span /></div>
                            </div>
                            <div className={styles.arrow}>→</div>
                            <div className={styles.resultBox}>
                                <span>✓</span> Ready
                            </div>
                        </div>
                    </div>
                </div>
            );

        case 'nlp':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>AI Assistant</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.chatInterface}>
                            <div className={styles.chatMessage + ' ' + styles.userMessage}>
                                <p>Show me revenue trends by region for Q4</p>
                            </div>
                            <div className={styles.chatMessage + ' ' + styles.aiMessage}>
                                <div className={styles.aiAvatar}>AI</div>
                                <div className={styles.aiContent}>
                                    <p>Analyzing Q4 regional revenue data...</p>
                                    <div className={styles.aiInsight}>
                                        <span className={styles.insightIcon}>📊</span>
                                        <div>
                                            <strong>Key Finding:</strong> APAC region shows 47% YoY growth, outperforming other regions by 2.3x
                                        </div>
                                    </div>
                                    <div className={styles.aiTags}>
                                        <span>Trend Analysis</span>
                                        <span>Regional</span>
                                        <span>Q4 2024</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );

        case 'analysis':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>Analysis Engine</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.analysisGrid}>
                            <div className={styles.analysisCard}>
                                <div className={styles.analysisHeader}>
                                    <span className={styles.analysisIcon}>📈</span>
                                    <span>Trend Detection</span>
                                </div>
                                <div className={styles.miniChart}>
                                    <svg viewBox="0 0 100 40">
                                        <path d="M0 35 Q25 30 40 25 T70 15 T100 5" fill="none" stroke="#10b981" strokeWidth="2" />
                                    </svg>
                                </div>
                                <span className={styles.analysisValue}>+34% growth</span>
                            </div>
                            <div className={styles.analysisCard}>
                                <div className={styles.analysisHeader}>
                                    <span className={styles.analysisIcon}>⚠️</span>
                                    <span>Anomalies</span>
                                </div>
                                <div className={styles.anomalyDots}>
                                    <span className={styles.normalDot} />
                                    <span className={styles.normalDot} />
                                    <span className={styles.anomalyDot} />
                                    <span className={styles.normalDot} />
                                </div>
                                <span className={styles.analysisValue}>1 detected</span>
                            </div>
                            <div className={styles.analysisCard}>
                                <div className={styles.analysisHeader}>
                                    <span className={styles.analysisIcon}>🔗</span>
                                    <span>Correlations</span>
                                </div>
                                <div className={styles.correlationBars}>
                                    <div style={{ width: '90%' }} />
                                    <div style={{ width: '75%' }} />
                                    <div style={{ width: '60%' }} />
                                </div>
                                <span className={styles.analysisValue}>3 strong</span>
                            </div>
                        </div>
                    </div>
                </div>
            );

        case 'visualization':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>Dashboard</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.dashboardGrid}>
                            <div className={styles.dashCard}>
                                <div className={styles.dashBars}>
                                    <div style={{ height: '60%' }} />
                                    <div style={{ height: '80%' }} />
                                    <div style={{ height: '45%' }} />
                                    <div style={{ height: '90%' }} />
                                    <div style={{ height: '70%' }} />
                                </div>
                            </div>
                            <div className={styles.dashCard}>
                                <div className={styles.dashPie}>
                                    <div className={styles.pieSlice} style={{ '--angle': '120deg', '--color': '#8b5cf6' }} />
                                    <div className={styles.pieSlice} style={{ '--angle': '90deg', '--color': '#06b6d4', '--start': '120deg' }} />
                                    <div className={styles.pieSlice} style={{ '--angle': '150deg', '--color': '#10b981', '--start': '210deg' }} />
                                </div>
                            </div>
                            <div className={styles.dashCard + ' ' + styles.wideCard}>
                                <div className={styles.dashLine}>
                                    <svg viewBox="0 0 200 60">
                                        <path d="M0 50 Q50 40 80 30 T120 25 T160 20 T200 10" fill="none" stroke="#8b5cf6" strokeWidth="2" />
                                        <path d="M0 55 Q50 50 80 45 T120 40 T160 35 T200 30" fill="none" stroke="#06b6d4" strokeWidth="2" />
                                    </svg>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );

        case 'ml':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>ML Pipeline</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.mlPipeline}>
                            <div className={styles.pipelineStep + ' ' + styles.stepComplete}>
                                <div className={styles.stepIcon}>✓</div>
                                <div className={styles.stepInfo}>
                                    <span className={styles.stepName}>Data Prep</span>
                                    <span className={styles.stepMeta}>15,420 rows</span>
                                </div>
                            </div>
                            <div className={styles.pipelineConnector} />
                            <div className={styles.pipelineStep + ' ' + styles.stepComplete}>
                                <div className={styles.stepIcon}>✓</div>
                                <div className={styles.stepInfo}>
                                    <span className={styles.stepName}>Features</span>
                                    <span className={styles.stepMeta}>24 selected</span>
                                </div>
                            </div>
                            <div className={styles.pipelineConnector} />
                            <div className={styles.pipelineStep + ' ' + styles.stepActive}>
                                <div className={styles.stepIcon}>
                                    <div className={styles.spinner} />
                                </div>
                                <div className={styles.stepInfo}>
                                    <span className={styles.stepName}>Training</span>
                                    <span className={styles.stepMeta}>87% complete</span>
                                </div>
                            </div>
                            <div className={styles.pipelineConnector + ' ' + styles.connectorPending} />
                            <div className={styles.pipelineStep + ' ' + styles.stepPending}>
                                <div className={styles.stepIcon}>○</div>
                                <div className={styles.stepInfo}>
                                    <span className={styles.stepName}>Deploy</span>
                                    <span className={styles.stepMeta}>Waiting</span>
                                </div>
                            </div>
                        </div>
                        <div className={styles.modelComparison}>
                            <div className={styles.modelCard}>
                                <span>XGBoost</span>
                                <span className={styles.modelScore}>94.2%</span>
                            </div>
                            <div className={styles.modelCard + ' ' + styles.modelBest}>
                                <span>Random Forest</span>
                                <span className={styles.modelScore}>96.8%</span>
                                <span className={styles.bestBadge}>Best</span>
                            </div>
                            <div className={styles.modelCard}>
                                <span>Neural Net</span>
                                <span className={styles.modelScore}>93.1%</span>
                            </div>
                        </div>
                    </div>
                </div>
            );

        case 'security':
            return (
                <div className={styles.visualWindow}>
                    <div className={styles.windowHeader}>
                        <div className={styles.windowDots}>
                            <span style={{ background: '#FF5F57' }} />
                            <span style={{ background: '#FEBC2E' }} />
                            <span style={{ background: '#28C840' }} />
                        </div>
                        <span>Security Center</span>
                    </div>
                    <div className={styles.windowContent}>
                        <div className={styles.securityDashboard}>
                            <div className={styles.securityShield}>
                                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="1.5">
                                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                                    <path d="M9 12l2 2 4-4" />
                                </svg>
                                <span>Protected</span>
                            </div>
                            <div className={styles.securityStats}>
                                <div className={styles.securityStat}>
                                    <span className={styles.statIcon}>🔐</span>
                                    <span className={styles.statLabel}>Encryption</span>
                                    <span className={styles.statValue}>AES-256</span>
                                </div>
                                <div className={styles.securityStat}>
                                    <span className={styles.statIcon}>🔑</span>
                                    <span className={styles.statLabel}>API Keys</span>
                                    <span className={styles.statValue}>Your Own</span>
                                </div>
                                <div className={styles.securityStat}>
                                    <span className={styles.statIcon}>✓</span>
                                    <span className={styles.statLabel}>SOC 2</span>
                                    <span className={styles.statValue}>Certified</span>
                                </div>
                                <div className={styles.securityStat}>
                                    <span className={styles.statIcon}>📋</span>
                                    <span className={styles.statLabel}>GDPR</span>
                                    <span className={styles.statValue}>Compliant</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            );

        default:
            return null;
    }
}
