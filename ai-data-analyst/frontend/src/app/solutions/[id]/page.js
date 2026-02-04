import Link from 'next/link';
import { notFound } from 'next/navigation';
import { IconArrowRight, IconCheck, IconChart, IconBrain, IconDatabase, IconTrend, IconShield, IconUsers } from '@/components/icons';
import styles from './page.module.css';

const solutionsData = {
    'data-teams': {
        title: 'For Data Teams',
        subtitle: 'Accelerate your data pipeline',
        description: 'Free your data scientists from repetitive work. NeuralAnalyst handles the 80% of analysis that follows common patterns, letting your team focus on complex, high-impact problems.',
        Icon: IconDatabase,
        benefits: [
            { title: 'AutoML Pipeline', desc: 'Build production-ready models in minutes, not weeks. Automated feature engineering, algorithm selection, and hyperparameter tuning.' },
            { title: 'SQL Generation', desc: 'Convert natural language questions to optimized SQL queries. Connect to any database and explore data without writing code.' },
            { title: 'Collaborative Notebooks', desc: 'Share analysis with your team. Version control, comments, and reproducible workflows built in.' },
            { title: 'Data Quality Monitoring', desc: 'Automatic detection of data drift, schema changes, and quality issues. Get alerts before problems reach production.' },
        ],
        features: ['Automated EDA', 'Feature Engineering', 'Model Explainability', 'A/B Testing', 'Time Series Forecasting', 'Anomaly Detection', 'API Access', 'Jupyter Integration'],
        stats: [{ value: '10x', label: 'Faster Analysis' }, { value: '80%', label: 'Less Manual Work' }, { value: '99.9%', label: 'Uptime SLA' }],
    },
    'business-analysts': {
        title: 'For Business Analysts',
        subtitle: 'Insights without coding',
        description: 'No SQL. No Python. Just ask questions in plain English and get instant answers. NeuralAnalyst brings the power of data science to everyone.',
        Icon: IconChart,
        benefits: [
            { title: 'Natural Language Queries', desc: 'Ask "What were our top products last quarter?" and get instant charts and insights. No technical skills required.' },
            { title: 'Self-Service Dashboards', desc: 'Create interactive dashboards by describing what you want. Share with stakeholders in one click.' },
            { title: 'Automated Reports', desc: 'Schedule weekly and monthly reports. AI-generated summaries highlight what matters most.' },
            { title: 'Excel Integration', desc: 'Import from and export to Excel. Work with familiar tools while leveraging AI power.' },
        ],
        features: ['Point-and-Click Analysis', 'Smart Visualizations', 'Trend Detection', 'Comparative Analysis', 'PDF Export', 'Email Scheduling', 'Team Sharing', 'Template Library'],
        stats: [{ value: '5 min', label: 'Time to First Insight' }, { value: 'Zero', label: 'Code Required' }, { value: '85%', label: 'Self-Service Rate' }],
    },
    'marketing': {
        title: 'For Marketing Teams',
        subtitle: 'Data-driven campaigns',
        description: 'Make every campaign count. From A/B testing to customer segmentation, get the insights you need to optimize marketing performance.',
        Icon: IconTrend,
        benefits: [
            { title: 'A/B Test Analysis', desc: 'Statistically rigorous experiment analysis. Know when results are significant and make confident decisions.' },
            { title: 'Customer Segmentation', desc: 'RFM analysis, behavioral clustering, and cohort tracking. Understand who your customers really are.' },
            { title: 'Attribution Modeling', desc: 'Multi-touch attribution to understand which channels drive conversions. Optimize spend across channels.' },
            { title: 'Campaign Analytics', desc: 'Track performance across all campaigns. Identify what works and replicate success.' },
        ],
        features: ['Funnel Analysis', 'Cohort Retention', 'CLV Prediction', 'Churn Prevention', 'Sentiment Analysis', 'Social Listening', 'ROI Tracking', 'Creative Testing'],
        stats: [{ value: '34%', label: 'Average ROI Lift' }, { value: '2x', label: 'Faster Insights' }, { value: '50%', label: 'Better Targeting' }],
    },
    'finance': {
        title: 'For Finance Teams',
        subtitle: 'Risk analysis & forecasting',
        description: 'Enterprise-grade security meets powerful analytics. Portfolio analysis, financial modeling, and risk assessment you can trust.',
        Icon: IconShield,
        benefits: [
            { title: 'Portfolio Metrics', desc: 'Sharpe ratio, Sortino ratio, beta, alpha, and more. Comprehensive risk-adjusted return analysis.' },
            { title: 'Risk Modeling', desc: 'Value at Risk (VaR), Monte Carlo simulations, and stress testing. Understand your exposure.' },
            { title: 'Financial Forecasting', desc: 'Revenue projections, budget planning, and scenario analysis with confidence intervals.' },
            { title: 'Compliance Reports', desc: 'Audit-ready reports with full data lineage. Meet regulatory requirements with ease.' },
        ],
        features: ['Rolling Metrics', 'Correlation Analysis', 'Volatility Tracking', 'Benchmark Comparison', 'Scenario Planning', 'Sensitivity Analysis', 'Audit Trails', 'SOC 2 Compliant'],
        stats: [{ value: '99.9%', label: 'Accuracy' }, { value: 'SOC 2', label: 'Compliance' }, { value: '256-bit', label: 'Encryption' }],
    },
    'enterprise': {
        title: 'For Enterprise',
        subtitle: 'Scale with confidence',
        description: 'Built for the demands of large organizations. SSO, role-based access, dedicated support, and custom integrations.',
        Icon: IconUsers,
        benefits: [
            { title: 'SSO & RBAC', desc: 'Single sign-on with your identity provider. Fine-grained role-based access control for data security.' },
            { title: 'Custom Integrations', desc: 'Connect to your data warehouse, BI tools, and workflows. REST API and webhooks for automation.' },
            { title: 'Dedicated Support', desc: 'Named account manager, priority support, and custom onboarding for your team.' },
            { title: 'SLA Guarantees', desc: '99.9% uptime SLA with enterprise-grade infrastructure and disaster recovery.' },
        ],
        features: ['SAML/OIDC SSO', 'Custom Domains', 'Data Residency', 'Private Cloud', 'Custom Training', 'Security Reviews', 'Procurement Support', 'Volume Discounts'],
        stats: [{ value: '99.9%', label: 'Uptime SLA' }, { value: '24/7', label: 'Support' }, { value: 'SOC 2', label: 'Certified' }],
    },
};

export async function generateStaticParams() {
    return Object.keys(solutionsData).map((id) => ({ id }));
}

export async function generateMetadata({ params }) {
    const { id } = await Promise.resolve(params);
    const solution = solutionsData[id];
    if (!solution) {
        return {
            title: 'NeuralAnalyst Solutions',
            description: 'NeuralAnalyst solutions for different teams.',
        };
    }
    return {
        title: `${solution.title} - NeuralAnalyst Solutions`,
        description: solution.description,
    };
}

export default async function SolutionDetailPage({ params }) {
    const { id } = await Promise.resolve(params);
    const solution = solutionsData[id];
    if (!solution) notFound();

    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <div className={styles.icon}>
                        <solution.Icon size={32} />
                    </div>
                    <span className={styles.subtitle}>{solution.subtitle}</span>
                    <h1 className={styles.title}>{solution.title}</h1>
                    <p className={styles.desc}>{solution.description}</p>
                    <div className={styles.ctas}>
                        <Link href="/register" className={styles.primaryBtn}>
                            Start Free Trial
                            <IconArrowRight size={18} />
                        </Link>
                        <Link href="/contact" className={styles.secondaryBtn}>
                            Talk to Sales
                        </Link>
                    </div>
                </div>
            </section>

            {/* Stats */}
            <section className={styles.statsSection}>
                <div className={styles.container}>
                    <div className={styles.statsGrid}>
                        {solution.stats.map((stat, i) => (
                            <div key={i} className={styles.statCard}>
                                <span className={styles.statValue}>{stat.value}</span>
                                <span className={styles.statLabel}>{stat.label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Benefits */}
            <section className={styles.benefits}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Key Benefits</h2>
                    <div className={styles.benefitsGrid}>
                        {solution.benefits.map((benefit, i) => (
                            <div key={i} className={styles.benefitCard}>
                                <h3>{benefit.title}</h3>
                                <p>{benefit.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className={styles.features}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Included Features</h2>
                    <div className={styles.featuresGrid}>
                        {solution.features.map((feature, i) => (
                            <div key={i} className={styles.featureItem}>
                                <IconCheck size={18} className={styles.featureCheck} />
                                {feature}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.ctaSection}>
                <div className={styles.container}>
                    <h2>Ready to get started?</h2>
                    <p>Start your free trial today. No credit card required.</p>
                    <Link href="/register" className={styles.primaryBtn}>
                        Start Free Trial
                        <IconArrowRight size={18} />
                    </Link>
                </div>
            </section>
        </main>
    );
}
