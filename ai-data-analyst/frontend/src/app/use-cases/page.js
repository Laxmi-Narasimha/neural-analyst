import Link from 'next/link';
import { IconArrowRight, IconBuilding, IconUsers, IconChart, IconTrend, IconTarget, IconDatabase, IconBrain, IconShield } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Use Cases - AI Data Analysis for Every Industry',
    description: 'Discover how NeuralAnalyst transforms data analysis across industries: marketing, finance, healthcare, e-commerce, and more.',
    openGraph: {
        title: 'Use Cases | NeuralAnalyst',
        description: 'AI-powered data analysis for marketing, finance, healthcare, e-commerce, and research.',
    },
};

const useCases = [
    {
        id: 'marketing',
        title: 'Marketing & Growth',
        description: 'Optimize campaigns, understand customer behavior, and maximize ROI with AI-powered insights.',
        Icon: IconTrend,
        features: [
            'A/B test analysis with statistical significance',
            'Customer segmentation with RFM analysis',
            'Attribution modeling and funnel analysis',
            'Sentiment analysis on customer feedback',
            'Campaign performance forecasting',
        ],
        stat: { value: '34%', label: 'Average ROI Increase' },
    },
    {
        id: 'finance',
        title: 'Finance & Banking',
        description: 'Risk assessment, portfolio optimization, and fraud detection with enterprise-grade security.',
        Icon: IconChart,
        features: [
            'Portfolio analysis with Sharpe and Sortino ratios',
            'Value at Risk (VaR) calculations',
            'Monte Carlo simulation for stress testing',
            'Fraud detection with anomaly detection',
            'Financial forecasting and trend analysis',
        ],
        stat: { value: '99.9%', label: 'Fraud Detection Rate' },
    },
    {
        id: 'healthcare',
        title: 'Healthcare & Life Sciences',
        description: 'Clinical trial analysis, patient outcomes, and research insights with HIPAA compliance.',
        Icon: IconShield,
        features: [
            'Clinical trial data analysis',
            'Survival analysis with Kaplan-Meier curves',
            'Patient cohort identification',
            'Treatment efficacy comparison',
            'Adverse event detection',
        ],
        stat: { value: '45%', label: 'Faster Trial Analysis' },
    },
    {
        id: 'ecommerce',
        title: 'E-Commerce & Retail',
        description: 'Inventory optimization, demand forecasting, and customer lifetime value prediction.',
        Icon: IconDatabase,
        features: [
            'Demand forecasting with Prophet and ARIMA',
            'Customer lifetime value (CLV) prediction',
            'Churn prediction and prevention',
            'Market basket analysis',
            'Price optimization modeling',
        ],
        stat: { value: '28%', label: 'Revenue Growth' },
    },
    {
        id: 'saas',
        title: 'SaaS & Technology',
        description: 'Product analytics, user behavior analysis, and growth metrics tracking.',
        Icon: IconBrain,
        features: [
            'Cohort retention analysis',
            'Feature adoption tracking',
            'User journey mapping',
            'Churn prediction modeling',
            'MRR and ARR forecasting',
        ],
        stat: { value: '52%', label: 'Reduced Churn' },
    },
    {
        id: 'research',
        title: 'Research & Academia',
        description: 'Statistical analysis, hypothesis testing, and publication-ready visualizations.',
        Icon: IconTarget,
        features: [
            'Comprehensive hypothesis testing',
            'Effect size calculations',
            'Correlation and regression analysis',
            'Publication-ready visualizations',
            'Reproducible analysis workflows',
        ],
        stat: { value: '10x', label: 'Faster Analysis' },
    },
];

export default function UseCasesPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        AI Analytics for <span className={styles.gradient}>Every Industry</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        From startups to Fortune 500 companies, NeuralAnalyst adapts to your
                        industry-specific data analysis needs.
                    </p>
                </div>
            </section>

            {/* Use Cases Grid */}
            <section className={styles.casesSection}>
                <div className={styles.container}>
                    {useCases.map((useCase, index) => (
                        <div
                            key={useCase.id}
                            className={`${styles.caseCard} ${index % 2 === 1 ? styles.reversed : ''}`}
                        >
                            <div className={styles.caseContent}>
                                <div className={styles.caseIcon}>
                                    <useCase.Icon size={28} />
                                </div>
                                <h2 className={styles.caseTitle}>{useCase.title}</h2>
                                <p className={styles.caseDesc}>{useCase.description}</p>
                                <ul className={styles.caseFeatures}>
                                    {useCase.features.map((feature, i) => (
                                        <li key={i}>{feature}</li>
                                    ))}
                                </ul>
                                <Link href="/register" className={styles.caseBtn}>
                                    Start Analyzing
                                    <IconArrowRight size={18} />
                                </Link>
                            </div>
                            <div className={styles.caseStat}>
                                <div className={styles.statValue}>{useCase.stat.value}</div>
                                <div className={styles.statLabel}>{useCase.stat.label}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Testimonials */}
            <section className={styles.testimonials}>
                <div className={styles.container}>
                    <h2 className={styles.testimonialTitle}>Trusted by Data Teams</h2>
                    <div className={styles.testimonialGrid}>
                        {[
                            { quote: 'NeuralAnalyst reduced our analysis time from days to minutes.', author: 'Sarah Chen', role: 'Data Lead, FinTech Startup' },
                            { quote: 'The AutoML feature found patterns our team had missed for months.', author: 'Michael Torres', role: 'VP Analytics, E-Commerce' },
                            { quote: 'Finally, an AI tool that actually understands statistical rigor.', author: 'Dr. Emily Watson', role: 'Research Director' },
                        ].map((testimonial, i) => (
                            <div key={i} className={styles.testimonialCard}>
                                <p className={styles.quote}>"{testimonial.quote}"</p>
                                <div className={styles.author}>
                                    <strong>{testimonial.author}</strong>
                                    <span>{testimonial.role}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <h2 className={styles.ctaTitle}>Ready to transform your data analysis?</h2>
                    <p className={styles.ctaDesc}>Join thousands of teams making better decisions with AI.</p>
                    <div className={styles.ctaButtons}>
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
        </main>
    );
}
