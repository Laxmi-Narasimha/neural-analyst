import Link from 'next/link';
import { IconBook, IconCode, IconArrowRight, IconSearch, IconPlay, IconDownload, IconKey, IconDatabase, IconChart, IconBrain } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Documentation - Learn NeuralAnalyst',
    description: 'Comprehensive documentation for NeuralAnalyst. Quick start guides, API reference, tutorials, and best practices for AI-powered data analysis.',
    openGraph: {
        title: 'Documentation | NeuralAnalyst',
        description: 'Everything you need to master NeuralAnalyst.',
    },
};

const sections = [
    {
        title: 'Getting Started',
        Icon: IconPlay,
        links: [
            { title: 'Quick Start Guide', href: '/docs/quickstart', time: '5 min' },
            { title: 'Your First Analysis', href: '/docs/first-analysis', time: '10 min' },
            { title: 'Understanding the Interface', href: '/docs/interface', time: '8 min' },
            { title: 'BYOK Setup', href: '/docs/byok-setup', time: '5 min' },
        ],
    },
    {
        title: 'Data Management',
        Icon: IconDatabase,
        links: [
            { title: 'Uploading Datasets', href: '/docs/uploading', time: '5 min' },
            { title: 'Supported Formats', href: '/docs/formats', time: '3 min' },
            { title: 'Data Cleaning', href: '/docs/cleaning', time: '12 min' },
            { title: 'Missing Value Handling', href: '/docs/missing-values', time: '10 min' },
        ],
    },
    {
        title: 'Analysis Features',
        Icon: IconChart,
        links: [
            { title: 'Exploratory Data Analysis', href: '/docs/eda', time: '15 min' },
            { title: 'Statistical Testing', href: '/docs/statistics', time: '20 min' },
            { title: 'A/B Testing', href: '/docs/ab-testing', time: '12 min' },
            { title: 'Time Series Forecasting', href: '/docs/forecasting', time: '18 min' },
        ],
    },
    {
        title: 'Machine Learning',
        Icon: IconBrain,
        links: [
            { title: 'AutoML Overview', href: '/docs/automl', time: '10 min' },
            { title: 'Classification Models', href: '/docs/classification', time: '15 min' },
            { title: 'Regression Models', href: '/docs/regression', time: '15 min' },
            { title: 'Model Explainability', href: '/docs/explainability', time: '12 min' },
        ],
    },
    {
        title: 'API Reference',
        Icon: IconCode,
        links: [
            { title: 'Authentication', href: '/docs/api/auth', time: '5 min' },
            { title: 'Datasets API', href: '/docs/api/datasets', time: '10 min' },
            { title: 'Analysis API', href: '/docs/api/analysis', time: '15 min' },
            { title: 'Webhooks', href: '/docs/api/webhooks', time: '8 min' },
        ],
    },
    {
        title: 'Advanced Topics',
        Icon: IconKey,
        links: [
            { title: 'Custom Integrations', href: '/docs/integrations', time: '20 min' },
            { title: 'Enterprise Setup', href: '/docs/enterprise', time: '15 min' },
            { title: 'Security & Compliance', href: '/docs/security', time: '10 min' },
            { title: 'Best Practices', href: '/docs/best-practices', time: '12 min' },
        ],
    },
];

export default function DocsPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        <span className={styles.gradient}>Documentation</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Everything you need to master NeuralAnalyst
                    </p>

                    {/* Search */}
                    <div className={styles.searchWrapper}>
                        <IconSearch size={20} className={styles.searchIcon} />
                        <input
                            type="text"
                            placeholder="Search documentation..."
                            className={styles.searchInput}
                        />
                        <span className={styles.searchHint}>⌘K</span>
                    </div>
                </div>
            </section>

            {/* Quick Links */}
            <section className={styles.quickLinks}>
                <div className={styles.container}>
                    <div className={styles.quickGrid}>
                        <Link href="/docs/quickstart" className={styles.quickCard}>
                            <IconPlay size={24} />
                            <span>Quick Start</span>
                        </Link>
                        <Link href="/docs/api" className={styles.quickCard}>
                            <IconCode size={24} />
                            <span>API Reference</span>
                        </Link>
                        <Link href="/docs/tutorials" className={styles.quickCard}>
                            <IconBook size={24} />
                            <span>Tutorials</span>
                        </Link>
                        <Link href="/docs/examples" className={styles.quickCard}>
                            <IconDownload size={24} />
                            <span>Examples</span>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Documentation Sections */}
            <section className={styles.docs}>
                <div className={styles.container}>
                    <div className={styles.docsGrid}>
                        {sections.map((section, i) => (
                            <div key={i} className={styles.docsSection}>
                                <div className={styles.sectionHeader}>
                                    <section.Icon size={20} className={styles.sectionIcon} />
                                    <h2 className={styles.sectionTitle}>{section.title}</h2>
                                </div>
                                <ul className={styles.linkList}>
                                    {section.links.map((link, j) => (
                                        <li key={j}>
                                            <Link href={link.href} className={styles.docLink}>
                                                <span>{link.title}</span>
                                                <span className={styles.linkTime}>{link.time}</span>
                                            </Link>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Help */}
            <section className={styles.help}>
                <div className={styles.container}>
                    <div className={styles.helpCard}>
                        <h2 className={styles.helpTitle}>Need Help?</h2>
                        <p className={styles.helpDesc}>
                            Can't find what you're looking for? Our team is here to help.
                        </p>
                        <div className={styles.helpButtons}>
                            <Link href="/contact" className={styles.primaryBtn}>
                                Contact Support
                                <IconArrowRight size={18} />
                            </Link>
                            <a href="https://github.com/neuralanalyst/docs" className={styles.secondaryBtn} target="_blank" rel="noopener noreferrer">
                                GitHub Discussions
                            </a>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    );
}
