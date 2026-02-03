import Link from 'next/link';
import { IconArrowRight, IconChart, IconBrain, IconDatabase, IconTrend, IconShield, IconUsers } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Solutions - AI Analytics for Every Team',
    description: 'Discover NeuralAnalyst solutions for data teams, business analysts, marketing, finance, and enterprise organizations.',
};

const solutions = [
    {
        id: 'data-teams',
        title: 'For Data Teams',
        subtitle: 'Accelerate your data pipeline',
        description: 'Automate repetitive analysis, build ML models faster, and focus on high-impact work.',
        Icon: IconDatabase,
        features: ['AutoML & Model Training', 'SQL Generation', 'Data Quality Monitoring', 'Collaborative Notebooks'],
        cta: 'Explore for Data Teams',
    },
    {
        id: 'business-analysts',
        title: 'For Business Analysts',
        subtitle: 'Insights without coding',
        description: 'Ask questions in plain English, get answers in seconds. No SQL or Python required.',
        Icon: IconChart,
        features: ['Natural Language Queries', 'Self-Service Dashboards', 'Automated Reports', 'Excel Integration'],
        cta: 'Explore for Analysts',
    },
    {
        id: 'marketing',
        title: 'For Marketing Teams',
        subtitle: 'Data-driven campaigns',
        description: 'A/B testing, customer segmentation, and attribution modeling made easy.',
        Icon: IconTrend,
        features: ['A/B Test Analysis', 'Customer Segmentation', 'Attribution Modeling', 'Campaign Analytics'],
        cta: 'Explore for Marketing',
    },
    {
        id: 'finance',
        title: 'For Finance Teams',
        subtitle: 'Risk analysis & forecasting',
        description: 'Portfolio analysis, financial modeling, and risk assessment with enterprise security.',
        Icon: IconShield,
        features: ['Portfolio Metrics', 'Risk Modeling', 'Financial Forecasting', 'Compliance Reports'],
        cta: 'Explore for Finance',
    },
    {
        id: 'enterprise',
        title: 'For Enterprise',
        subtitle: 'Scale with confidence',
        description: 'SSO, RBAC, dedicated support, and custom integrations for large organizations.',
        Icon: IconUsers,
        features: ['SSO & RBAC', 'Custom Integrations', 'Dedicated Support', 'SLA Guarantees'],
        cta: 'Explore for Enterprise',
    },
];

export default function SolutionsPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Solutions for <span className={styles.gradient}>Every Team</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Whether you're a solo analyst or a global enterprise, NeuralAnalyst adapts to your workflow.
                    </p>
                </div>
            </section>

            {/* Solutions */}
            <section className={styles.solutions}>
                <div className={styles.container}>
                    {solutions.map((solution, index) => (
                        <Link
                            key={solution.id}
                            href={`/solutions/${solution.id}`}
                            className={`${styles.solutionCard} ${index % 2 === 1 ? styles.reversed : ''}`}
                        >
                            <div className={styles.solutionContent}>
                                <div className={styles.solutionIcon}>
                                    <solution.Icon size={28} />
                                </div>
                                <span className={styles.solutionSubtitle}>{solution.subtitle}</span>
                                <h2 className={styles.solutionTitle}>{solution.title}</h2>
                                <p className={styles.solutionDesc}>{solution.description}</p>
                                <ul className={styles.featureList}>
                                    {solution.features.map((feature, i) => (
                                        <li key={i}>{feature}</li>
                                    ))}
                                </ul>
                                <span className={styles.solutionCta}>
                                    {solution.cta}
                                    <IconArrowRight size={16} />
                                </span>
                            </div>
                            <div className={styles.solutionVisual}>
                                <div className={styles.visualGlow}></div>
                            </div>
                        </Link>
                    ))}
                </div>
            </section>

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <h2>Not sure which solution is right for you?</h2>
                    <p>Talk to our team and get a personalized recommendation.</p>
                    <Link href="/contact" className={styles.ctaBtn}>
                        Contact Sales
                        <IconArrowRight size={18} />
                    </Link>
                </div>
            </section>
        </main>
    );
}
