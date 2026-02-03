import Link from 'next/link';
import {
    IconRobot, IconBrain, IconChart, IconLightning, IconSearch, IconTarget,
    IconFlask, IconStats, IconNetwork, IconTime, IconText, IconChat,
    IconTrend, IconAlert, IconSurvival, IconPalette, IconMoney, IconPortfolio,
    IconDice, IconShield, IconDatabase, IconSparkles, IconDashboard, IconArrowRight,
    IconKey, IconUpload, IconGlobe, IconCode, IconUsers, IconFolder
} from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Features - 244+ AI Analytics Capabilities',
    description: 'Explore all 244+ features of NeuralAnalyst: AutoML, deep learning, statistical testing, forecasting, NLP, financial analytics, and more.',
    openGraph: {
        title: 'Features - 244+ AI Analytics Capabilities | NeuralAnalyst',
        description: 'Complete feature list of the most comprehensive AI data analyst platform.',
    },
};

const categories = [
    {
        id: 'data-ingestion',
        title: 'Data Ingestion & Parsing',
        description: 'Connect to any data source with intelligent parsing',
        features: [
            { title: 'CSV Auto-Parsing', desc: 'Automatic delimiter and encoding detection', Icon: IconUpload },
            { title: 'Excel Support', desc: 'Multi-sheet XLSX and legacy XLS files', Icon: IconFolder },
            { title: 'JSON/JSONL', desc: 'Nested JSON with automatic flattening', Icon: IconCode },
            { title: 'Parquet & ORC', desc: 'Columnar formats for big data', Icon: IconDatabase },
            { title: 'Fixed-Width Files', desc: 'Mainframe and legacy data formats', Icon: IconText },
            { title: 'XML Streaming', desc: 'Memory-efficient large XML parsing', Icon: IconGlobe },
            { title: 'Multi-Encoding', desc: 'Auto-detect UTF-8, Latin-1, CP1252', Icon: IconSearch },
            { title: 'Database Connectors', desc: 'PostgreSQL, MySQL, SQLite, and more', Icon: IconNetwork },
        ],
    },
    {
        id: 'data-cleaning',
        title: 'Data Cleaning & Preprocessing',
        description: 'Production-grade data quality and transformation',
        features: [
            { title: 'Missing Value Detection', desc: 'MCAR, MAR, MNAR pattern analysis', Icon: IconSearch },
            { title: 'Smart Imputation', desc: 'KNN, MICE, time-series, hot-deck', Icon: IconTarget },
            { title: 'Outlier Detection', desc: 'Z-score, IQR, Isolation Forest, LOF', Icon: IconAlert },
            { title: 'Duplicate Detection', desc: 'Exact and fuzzy matching', Icon: IconUsers },
            { title: 'Type Inference', desc: 'Automatic data type detection', Icon: IconBrain },
            { title: 'Feature Scaling', desc: 'Standard, MinMax, Robust scaling', Icon: IconStats },
            { title: 'Encoding', desc: 'Label, one-hot, target encoding', Icon: IconCode },
            { title: 'Datetime Features', desc: 'Extract day, month, year, weekday', Icon: IconTime },
        ],
    },
    {
        id: 'machine-learning',
        title: 'Machine Learning',
        description: 'State-of-the-art algorithms with AutoML',
        features: [
            { title: 'AutoML Pipeline', desc: 'Automated model selection and tuning', Icon: IconRobot },
            { title: 'Random Forest', desc: 'Ensemble trees for classification/regression', Icon: IconNetwork },
            { title: 'XGBoost', desc: 'Gradient boosting with regularization', Icon: IconLightning },
            { title: 'Deep Learning', desc: 'MLP, LSTM, Transformer architectures', Icon: IconBrain },
            { title: 'SVM', desc: 'Support Vector Machines with kernels', Icon: IconTarget },
            { title: 'K-Means Clustering', desc: 'Unsupervised pattern discovery', Icon: IconPalette },
            { title: 'Model Explainability', desc: 'SHAP, permutation importance, PDP', Icon: IconSearch },
            { title: 'Ensemble Methods', desc: 'Voting, stacking, blending', Icon: IconStats },
        ],
    },
    {
        id: 'statistical-analysis',
        title: 'Statistical Analysis',
        description: 'Rigorous hypothesis testing and inference',
        features: [
            { title: 'A/B Testing', desc: 'Frequentist, Bayesian, sequential', Icon: IconFlask },
            { title: 'Hypothesis Testing', desc: 't-test, ANOVA, chi-square, Fisher', Icon: IconStats },
            { title: 'Correlation Analysis', desc: 'Pearson, Spearman, Kendall', Icon: IconNetwork },
            { title: 'Effect Size', desc: "Cohen's d, Hedges' g, Cramér's V", Icon: IconChart },
            { title: 'Causal Inference', desc: 'Propensity matching, ATE estimation', Icon: IconTarget },
            { title: 'Power Analysis', desc: 'Sample size calculation', Icon: IconTrend },
            { title: 'Normality Tests', desc: 'Shapiro-Wilk, Anderson-Darling', Icon: IconSearch },
            { title: 'Variance Tests', desc: "Levene's, Bartlett's test", Icon: IconAlert },
        ],
    },
    {
        id: 'forecasting',
        title: 'Time Series & Forecasting',
        description: 'Predict the future with confidence intervals',
        features: [
            { title: 'Prophet', desc: 'Facebook/Meta forecasting library', Icon: IconTime },
            { title: 'ARIMA/SARIMA', desc: 'Classical time series models', Icon: IconTrend },
            { title: 'Exponential Smoothing', desc: 'Holt-Winters methods', Icon: IconChart },
            { title: 'Seasonality Detection', desc: 'Multiple seasonality handling', Icon: IconSurvival },
            { title: 'Trend Analysis', desc: 'Linear and polynomial trends', Icon: IconStats },
            { title: 'Ensemble Forecasting', desc: 'Combine multiple models', Icon: IconNetwork },
            { title: 'Confidence Intervals', desc: 'Uncertainty quantification', Icon: IconShield },
            { title: 'Anomaly Detection', desc: 'Real-time time series anomalies', Icon: IconAlert },
        ],
    },
    {
        id: 'nlp',
        title: 'NLP & Text Analytics',
        description: 'Extract insights from unstructured text',
        features: [
            { title: 'Sentiment Analysis', desc: 'Positive, negative, neutral detection', Icon: IconChat },
            { title: 'Named Entity Recognition', desc: 'Extract people, orgs, locations', Icon: IconDatabase },
            { title: 'Topic Modeling', desc: 'Discover hidden themes', Icon: IconText },
            { title: 'Keyword Extraction', desc: 'Important term identification', Icon: IconSearch },
            { title: 'Text Summarization', desc: 'Extractive and abstractive', Icon: IconSparkles },
            { title: 'Natural Language Queries', desc: 'Ask in plain English', Icon: IconBrain },
            { title: 'SQL Generation', desc: 'Convert questions to SQL', Icon: IconCode },
            { title: 'Text Preprocessing', desc: 'Tokenization, stemming, lemmatization', Icon: IconPalette },
        ],
    },
    {
        id: 'financial',
        title: 'Financial Analytics',
        description: 'Portfolio analysis and risk management',
        features: [
            { title: 'Sharpe Ratio', desc: 'Risk-adjusted return measurement', Icon: IconTrend },
            { title: 'Sortino Ratio', desc: 'Downside risk-adjusted returns', Icon: IconChart },
            { title: 'Value at Risk', desc: 'VaR and Conditional VaR', Icon: IconAlert },
            { title: 'Monte Carlo Simulation', desc: 'Scenario analysis and forecasting', Icon: IconDice },
            { title: 'Portfolio Metrics', desc: 'Beta, alpha, maximum drawdown', Icon: IconPortfolio },
            { title: 'Correlation Analysis', desc: 'Asset correlation matrices', Icon: IconNetwork },
            { title: 'Rolling Metrics', desc: 'Time-windowed calculations', Icon: IconTime },
            { title: 'Risk Assessment', desc: 'Comprehensive risk profiling', Icon: IconShield },
        ],
    },
    {
        id: 'customer',
        title: 'Customer Analytics',
        description: 'Understand and retain your customers',
        features: [
            { title: 'RFM Analysis', desc: 'Recency, Frequency, Monetary segmentation', Icon: IconUsers },
            { title: 'Cohort Analysis', desc: 'Track user behavior over time', Icon: IconStats },
            { title: 'CLV Prediction', desc: 'Customer Lifetime Value modeling', Icon: IconMoney },
            { title: 'Churn Prediction', desc: 'Identify at-risk customers', Icon: IconAlert },
            { title: 'Survival Analysis', desc: 'Time-to-event modeling', Icon: IconSurvival },
            { title: 'Market Basket', desc: 'Association rules and recommendations', Icon: IconPortfolio },
            { title: 'Customer Segmentation', desc: 'K-means and hierarchical clustering', Icon: IconPalette },
            { title: 'Retention Analysis', desc: 'Track and improve retention rates', Icon: IconTarget },
        ],
    },
];

export default function FeaturesPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        <span className={styles.gradient}>244+</span> Features
                    </h1>
                    <p className={styles.heroSubtitle}>
                        The most comprehensive AI data analyst platform. Every feature you need,
                        from basic EDA to advanced deep learning.
                    </p>
                    <div className={styles.heroCta}>
                        <Link href="/register" className={styles.primaryBtn}>
                            Start Free Trial
                            <IconArrowRight size={20} />
                        </Link>
                        <Link href="/pricing" className={styles.secondaryBtn}>
                            View Pricing
                        </Link>
                    </div>
                </div>
            </section>

            {/* Categories */}
            {categories.map((category, categoryIndex) => (
                <section
                    key={category.id}
                    className={`${styles.category} ${categoryIndex % 2 === 1 ? styles.categoryAlt : ''}`}
                    id={category.id}
                >
                    <div className={styles.container}>
                        <div className={styles.categoryHeader}>
                            <h2 className={styles.categoryTitle}>{category.title}</h2>
                            <p className={styles.categoryDesc}>{category.description}</p>
                        </div>
                        <div className={styles.featureGrid}>
                            {category.features.map((feature, index) => (
                                <div key={index} className={styles.featureCard}>
                                    <div className={styles.featureIcon}>
                                        <feature.Icon size={24} />
                                    </div>
                                    <div className={styles.featureContent}>
                                        <h3 className={styles.featureTitle}>{feature.title}</h3>
                                        <p className={styles.featureDesc}>{feature.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            ))}

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <h2 className={styles.ctaTitle}>Ready to analyze your data?</h2>
                    <p className={styles.ctaDesc}>
                        Start with our free tier. No credit card required.
                    </p>
                    <div className={styles.ctaButtons}>
                        <Link href="/register" className={styles.primaryBtn}>
                            Get Started Free
                            <IconArrowRight size={20} />
                        </Link>
                        <Link href="/setup-keys" className={styles.secondaryBtn}>
                            <IconKey size={20} />
                            Use Your Own API Keys
                        </Link>
                    </div>
                </div>
            </section>
        </main>
    );
}
