'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
    IconRobot,
    IconBrain,
    IconChart,
    IconLightning,
    IconSearch,
    IconTarget,
    IconFlask,
    IconStats,
    IconNetwork,
    IconTime,
    IconText,
    IconChat,
    IconTrend,
    IconAlert,
    IconSurvival,
    IconPalette,
    IconMoney,
    IconPortfolio,
    IconDice,
    IconShield,
    IconDatabase,
    IconSparkles,
    IconDashboard,
    IconArrowRight,
} from '@/components/icons';
import styles from './Features.module.css';

const featureCategories = [
    { id: 'all', label: 'All Features' },
    { id: 'ml', label: 'Machine Learning' },
    { id: 'stats', label: 'Statistics' },
    { id: 'viz', label: 'Visualization' },
    { id: 'nlp', label: 'NLP & Text' },
    { id: 'finance', label: 'Financial' },
];

const features = [
    // ML Features
    { id: 1, title: 'AutoML', desc: 'Automated model selection & hyperparameter tuning', category: 'ml', Icon: IconRobot },
    { id: 2, title: 'Random Forest', desc: 'Ensemble classification & regression models', category: 'ml', Icon: IconNetwork },
    { id: 3, title: 'XGBoost', desc: 'State-of-the-art gradient boosting', category: 'ml', Icon: IconLightning },
    { id: 4, title: 'Deep Learning', desc: 'MLP, LSTM, and Transformer networks', category: 'ml', Icon: IconBrain },
    { id: 5, title: 'Model Explainability', desc: 'SHAP values & feature importance', category: 'ml', Icon: IconSearch },
    { id: 6, title: 'Ensemble Methods', desc: 'Voting, stacking, and blending', category: 'ml', Icon: IconTarget },

    // Stats Features
    { id: 7, title: 'A/B Testing', desc: 'Frequentist & Bayesian experiments', category: 'stats', Icon: IconFlask },
    { id: 8, title: 'Hypothesis Testing', desc: 't-test, ANOVA, chi-square analysis', category: 'stats', Icon: IconStats },
    { id: 9, title: 'Causal Inference', desc: 'Propensity matching & treatment effects', category: 'stats', Icon: IconNetwork },
    { id: 10, title: 'Anomaly Detection', desc: 'Multi-method outlier identification', category: 'stats', Icon: IconAlert },
    { id: 11, title: 'Time Series', desc: 'Prophet, ARIMA, and forecasting', category: 'stats', Icon: IconTrend },
    { id: 12, title: 'Survival Analysis', desc: 'Kaplan-Meier & hazard modeling', category: 'stats', Icon: IconSurvival },

    // Viz Features
    { id: 13, title: 'Smart Charts', desc: 'AI-selected optimal visualizations', category: 'viz', Icon: IconChart },
    { id: 14, title: 'Interactive Dashboards', desc: 'Real-time analytics & KPI tracking', category: 'viz', Icon: IconDashboard },
    { id: 15, title: '15+ Chart Types', desc: 'Bar, line, scatter, heatmap & more', category: 'viz', Icon: IconPalette },
    { id: 16, title: 'Plotly Integration', desc: 'Publication-ready interactive graphics', category: 'viz', Icon: IconSparkles },

    // NLP Features
    { id: 17, title: 'Sentiment Analysis', desc: 'Emotion & opinion extraction', category: 'nlp', Icon: IconChat },
    { id: 18, title: 'Named Entity Recognition', desc: 'Extract entities from text', category: 'nlp', Icon: IconDatabase },
    { id: 19, title: 'Topic Modeling', desc: 'Discover hidden patterns in text', category: 'nlp', Icon: IconText },
    { id: 20, title: 'Natural Language Queries', desc: 'Ask questions in plain English', category: 'nlp', Icon: IconShield },

    // Finance Features
    { id: 21, title: 'Sharpe & Sortino', desc: 'Risk-adjusted return metrics', category: 'finance', Icon: IconTrend },
    { id: 22, title: 'Value at Risk', desc: 'VaR & CVaR calculations', category: 'finance', Icon: IconAlert },
    { id: 23, title: 'Monte Carlo', desc: 'Scenario simulation & modeling', category: 'finance', Icon: IconDice },
    { id: 24, title: 'Portfolio Analysis', desc: 'Beta, alpha, and drawdown metrics', category: 'finance', Icon: IconPortfolio },
];

export default function Features() {
    const [activeCategory, setActiveCategory] = useState('all');

    const filteredFeatures = activeCategory === 'all'
        ? features
        : features.filter(f => f.category === activeCategory);

    return (
        <section className={styles.section}>
            <div className={styles.container}>
                {/* Header */}
                <motion.div
                    className={styles.header}
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                >
                    <span className={styles.label}>Capabilities</span>
                    <h2 className={styles.title}>
                        <span className={styles.gradient}>244+</span> Production-Ready Features
                    </h2>
                    <p className={styles.subtitle}>
                        From basic EDA to advanced ML, everything a data scientist needs — and more
                    </p>
                </motion.div>

                {/* Category Tabs */}
                <div className={styles.tabs}>
                    {featureCategories.map((cat) => (
                        <button
                            key={cat.id}
                            className={`${styles.tab} ${activeCategory === cat.id ? styles.active : ''}`}
                            onClick={() => setActiveCategory(cat.id)}
                        >
                            {cat.label}
                        </button>
                    ))}
                </div>

                {/* Feature Grid */}
                <motion.div
                    className={styles.grid}
                    layout
                >
                    {filteredFeatures.map((feature, index) => (
                        <motion.div
                            key={feature.id}
                            className={styles.card}
                            initial={{ opacity: 0, scale: 0.9 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.3, delay: index * 0.05 }}
                            whileHover={{ y: -5, transition: { duration: 0.2 } }}
                        >
                            <div className={styles.cardIcon}>
                                <feature.Icon size={28} />
                            </div>
                            <h3 className={styles.cardTitle}>{feature.title}</h3>
                            <p className={styles.cardDesc}>{feature.desc}</p>
                        </motion.div>
                    ))}
                </motion.div>

                {/* See All Link */}
                <motion.div
                    className={styles.seeAll}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                >
                    <a href="/features" className={styles.seeAllLink}>
                        View all 244+ features
                        <IconArrowRight size={20} />
                    </a>
                </motion.div>
            </div>
        </section>
    );
}
