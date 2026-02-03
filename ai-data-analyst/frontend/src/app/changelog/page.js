import { IconSparkles, IconChart, IconBrain, IconShield } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Changelog - What\'s New in NeuralAnalyst',
    description: 'See the latest features, improvements, and updates to NeuralAnalyst.',
};

const releases = [
    {
        version: '2.5.0',
        date: 'December 2024',
        type: 'Feature Release',
        changes: [
            { type: 'feature', title: 'Natural Language SQL', desc: 'Generate SQL queries from plain English descriptions' },
            { type: 'feature', title: 'Advanced A/B Testing', desc: 'Sequential testing and Bayesian analysis options' },
            { type: 'improvement', title: 'Faster Data Upload', desc: '3x faster CSV parsing for large files' },
            { type: 'fix', title: 'Chart Export', desc: 'Fixed high-DPI image export issues' },
        ],
    },
    {
        version: '2.4.0',
        date: 'November 2024',
        type: 'Feature Release',
        changes: [
            { type: 'feature', title: 'AutoML Pipeline', desc: 'Automated machine learning with hyperparameter tuning' },
            { type: 'feature', title: 'Model Explainability', desc: 'SHAP values and feature importance visualization' },
            { type: 'improvement', title: 'Dashboard Performance', desc: 'Improved loading times for large datasets' },
            { type: 'fix', title: 'Time Zone Handling', desc: 'Fixed time zone conversion in time series' },
        ],
    },
    {
        version: '2.3.0',
        date: 'October 2024',
        type: 'Feature Release',
        changes: [
            { type: 'feature', title: 'BYOK Mode', desc: 'Bring Your Own Key for complete privacy control' },
            { type: 'feature', title: 'Anomaly Detection', desc: 'Multi-method anomaly detection for time series' },
            { type: 'improvement', title: 'API Rate Limits', desc: 'Increased API rate limits for Pro users' },
        ],
    },
    {
        version: '2.2.0',
        date: 'September 2024',
        type: 'Feature Release',
        changes: [
            { type: 'feature', title: 'Customer Segmentation', desc: 'RFM analysis and behavioral clustering' },
            { type: 'feature', title: 'Financial Analytics', desc: 'Portfolio metrics and risk analysis' },
            { type: 'improvement', title: 'Data Quality', desc: 'Enhanced data quality scoring and recommendations' },
        ],
    },
];

export default function ChangelogPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        <span className={styles.gradient}>Changelog</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Track the latest updates, features, and improvements to NeuralAnalyst.
                    </p>
                </div>
            </section>

            {/* Releases */}
            <section className={styles.releases}>
                <div className={styles.container}>
                    {releases.map((release) => (
                        <div key={release.version} className={styles.release}>
                            <div className={styles.releaseHeader}>
                                <div className={styles.versionBadge}>v{release.version}</div>
                                <span className={styles.releaseDate}>{release.date}</span>
                                <span className={styles.releaseType}>{release.type}</span>
                            </div>
                            <div className={styles.changesList}>
                                {release.changes.map((change, i) => (
                                    <div key={i} className={styles.changeItem}>
                                        <span className={`${styles.changeType} ${styles[change.type]}`}>
                                            {change.type}
                                        </span>
                                        <div className={styles.changeContent}>
                                            <h3>{change.title}</h3>
                                            <p>{change.desc}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </section>
        </main>
    );
}
