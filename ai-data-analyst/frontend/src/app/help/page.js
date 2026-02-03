import Link from 'next/link';
import { IconSearch, IconBook, IconChat, IconCode, IconPlay, IconArrowRight } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Help Center - NeuralAnalyst Support',
    description: 'Find answers, tutorials, and support resources for NeuralAnalyst.',
};

const categories = [
    {
        Icon: IconPlay,
        title: 'Getting Started',
        desc: 'Quick start guides and tutorials',
        links: [
            { title: 'Creating your first analysis', href: '/docs/quickstart' },
            { title: 'Uploading datasets', href: '/docs/uploading' },
            { title: 'Understanding the dashboard', href: '/docs/dashboard' },
        ],
    },
    {
        Icon: IconBook,
        title: 'Documentation',
        desc: 'Detailed feature documentation',
        links: [
            { title: 'Exploratory Data Analysis', href: '/docs/eda' },
            { title: 'Machine Learning', href: '/docs/automl' },
            { title: 'Statistical Testing', href: '/docs/statistics' },
        ],
    },
    {
        Icon: IconCode,
        title: 'API Reference',
        desc: 'REST API documentation',
        links: [
            { title: 'Authentication', href: '/docs/api/auth' },
            { title: 'Datasets API', href: '/docs/api/datasets' },
            { title: 'Analysis API', href: '/docs/api/analysis' },
        ],
    },
    {
        Icon: IconChat,
        title: 'Community',
        desc: 'Get help from the community',
        links: [
            { title: 'GitHub Discussions', href: 'https://github.com/neuralanalyst' },
            { title: 'Discord Server', href: 'https://discord.gg/neuralanalyst' },
            { title: 'Feature Requests', href: '/feedback' },
        ],
    },
];

const faqs = [
    { q: 'How do I reset my password?', a: 'Go to the login page and click "Forgot password" to receive a reset link.' },
    { q: 'What file formats are supported?', a: 'We support CSV, Excel (.xlsx, .xls), JSON, and Parquet files.' },
    { q: 'Is my data secure?', a: 'Yes. We use 256-bit encryption at rest and TLS 1.3 in transit. BYOK mode ensures your data never touches our servers.' },
    { q: 'How do I upgrade my plan?', a: 'Go to Settings > Billing and click "Upgrade Plan" to see available options.' },
];

export default function HelpPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        How can we <span className={styles.gradient}>help</span>?
                    </h1>
                    <div className={styles.searchWrapper}>
                        <IconSearch size={20} className={styles.searchIcon} />
                        <input
                            type="text"
                            placeholder="Search for help articles..."
                            className={styles.searchInput}
                        />
                    </div>
                </div>
            </section>

            {/* Categories */}
            <section className={styles.categories}>
                <div className={styles.container}>
                    <div className={styles.categoriesGrid}>
                        {categories.map((category, i) => (
                            <div key={i} className={styles.categoryCard}>
                                <category.Icon size={28} className={styles.categoryIcon} />
                                <h2>{category.title}</h2>
                                <p>{category.desc}</p>
                                <ul className={styles.categoryLinks}>
                                    {category.links.map((link, j) => (
                                        <li key={j}>
                                            <Link href={link.href}>{link.title}</Link>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* FAQ */}
            <section className={styles.faq}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Frequently Asked Questions</h2>
                    <div className={styles.faqList}>
                        {faqs.map((faq, i) => (
                            <div key={i} className={styles.faqItem}>
                                <h3>{faq.q}</h3>
                                <p>{faq.a}</p>
                            </div>
                        ))}
                    </div>
                    <Link href="/faq" className={styles.viewAllLink}>
                        View all FAQs
                        <IconArrowRight size={16} />
                    </Link>
                </div>
            </section>

            {/* Contact */}
            <section className={styles.contact}>
                <div className={styles.container}>
                    <div className={styles.contactCard}>
                        <h2>Still need help?</h2>
                        <p>Our support team is available to assist you.</p>
                        <div className={styles.contactButtons}>
                            <Link href="/contact" className={styles.primaryBtn}>
                                Contact Support
                            </Link>
                            <a href="mailto:support@neuralanalyst.ai" className={styles.secondaryBtn}>
                                Email Us
                            </a>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    );
}
