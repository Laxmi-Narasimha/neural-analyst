import Link from 'next/link';
import { IconCheck, IconX, IconArrowRight, IconKey, IconShield, IconUsers, IconBuilding } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Pricing - Simple, Transparent Plans',
    description: 'Choose the perfect plan for your data analysis needs. Free tier available. BYOK option for full control with your own API keys.',
    openGraph: {
        title: 'Pricing - NeuralAnalyst',
        description: 'Flexible pricing for startups, professionals, and enterprises. Start free today.',
    },
};

const plans = [
    {
        id: 'free',
        name: 'Free',
        price: '0',
        period: 'forever',
        description: 'Perfect for exploring NeuralAnalyst',
        features: [
            { text: '100 queries per month', included: true },
            { text: 'Basic EDA & statistics', included: true },
            { text: 'Up to 10MB datasets', included: true },
            { text: '5 visualizations per day', included: true },
            { text: 'Community support', included: true },
            { text: 'ML & AutoML', included: false },
            { text: 'Forecasting', included: false },
            { text: 'API access', included: false },
        ],
        cta: 'Start Free',
        ctaLink: '/register',
        Icon: null,
    },
    {
        id: 'pro',
        name: 'Pro',
        price: '29',
        period: '/month',
        description: 'For professionals and growing teams',
        popular: true,
        features: [
            { text: 'Unlimited queries', included: true },
            { text: 'All 244+ features', included: true },
            { text: 'Up to 1GB datasets', included: true },
            { text: 'ML & AutoML included', included: true },
            { text: 'Time series forecasting', included: true },
            { text: 'Priority email support', included: true },
            { text: 'Export to PDF/Excel', included: true },
            { text: 'API access', included: false },
        ],
        cta: 'Start Pro Trial',
        ctaLink: '/register?plan=pro',
        Icon: IconUsers,
    },
    {
        id: 'enterprise',
        name: 'Enterprise',
        price: '99',
        period: '/month',
        description: 'For large teams and organizations',
        features: [
            { text: 'Everything in Pro', included: true },
            { text: 'Unlimited datasets', included: true },
            { text: 'Full API access', included: true },
            { text: 'Custom integrations', included: true },
            { text: 'Team collaboration', included: true },
            { text: 'SSO & RBAC', included: true },
            { text: 'Dedicated support', included: true },
            { text: 'SLA guarantee', included: true },
        ],
        cta: 'Contact Sales',
        ctaLink: '/contact',
        Icon: IconBuilding,
    },
];

const faqs = [
    {
        question: 'What is BYOK?',
        answer: 'BYOK (Bring Your Own Key) lets you use your own OpenAI, Gemini, or Anthropic API keys. You get full access to all features and only pay for what you use through your provider.',
    },
    {
        question: 'Can I upgrade or downgrade anytime?',
        answer: 'Yes, you can change your plan at any time. When upgrading, you only pay the prorated difference. When downgrading, your credit carries over.',
    },
    {
        question: 'What happens when I reach my query limit?',
        answer: 'On the Free plan, you can wait for the next month or upgrade to Pro for unlimited queries. We will notify you before you reach your limit.',
    },
    {
        question: 'Is there a discount for annual billing?',
        answer: 'Yes! Choose annual billing and save 20%. Pro becomes $23/month and Enterprise becomes $79/month when billed annually.',
    },
    {
        question: 'Do you offer a free trial?',
        answer: 'All paid plans include a 14-day free trial. No credit card required to start. Cancel anytime during the trial period.',
    },
];

export default function PricingPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Simple, <span className={styles.gradient}>Transparent</span> Pricing
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Start free, scale as you grow. No hidden fees, no surprises.
                    </p>
                </div>
            </section>

            {/* Plans */}
            <section className={styles.plansSection}>
                <div className={styles.container}>
                    <div className={styles.plansGrid}>
                        {plans.map((plan) => (
                            <div
                                key={plan.id}
                                className={`${styles.planCard} ${plan.popular ? styles.popular : ''}`}
                            >
                                {plan.popular && (
                                    <div className={styles.popularBadge}>Most Popular</div>
                                )}
                                <div className={styles.planHeader}>
                                    <h3 className={styles.planName}>{plan.name}</h3>
                                    <div className={styles.priceRow}>
                                        <span className={styles.currency}>$</span>
                                        <span className={styles.price}>{plan.price}</span>
                                        <span className={styles.period}>{plan.period}</span>
                                    </div>
                                    <p className={styles.planDesc}>{plan.description}</p>
                                </div>
                                <ul className={styles.featureList}>
                                    {plan.features.map((feature, i) => (
                                        <li key={i} className={styles.featureItem}>
                                            <span className={feature.included ? styles.checkIcon : styles.xIcon}>
                                                {feature.included ? <IconCheck size={18} /> : <IconX size={18} />}
                                            </span>
                                            <span className={feature.included ? '' : styles.notIncluded}>
                                                {feature.text}
                                            </span>
                                        </li>
                                    ))}
                                </ul>
                                <Link
                                    href={plan.ctaLink}
                                    className={plan.popular ? styles.primaryBtn : styles.secondaryBtn}
                                >
                                    {plan.cta}
                                    <IconArrowRight size={18} />
                                </Link>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* BYOK Section */}
            <section className={styles.byokSection}>
                <div className={styles.container}>
                    <div className={styles.byokCard}>
                        <div className={styles.byokContent}>
                            <div className={styles.byokBadge}>
                                <IconKey size={16} />
                                Open Source
                            </div>
                            <h2 className={styles.byokTitle}>
                                <span className={styles.gradient}>BYOK</span> — Bring Your Own Keys
                            </h2>
                            <p className={styles.byokDesc}>
                                Use your own OpenAI, Gemini, or Anthropic API keys. Full access to all 244+ features.
                                You only pay for what you use through your provider. Perfect for developers who want
                                complete control and privacy.
                            </p>
                            <div className={styles.byokFeatures}>
                                <div className={styles.byokFeature}>
                                    <IconShield size={20} />
                                    <span>Full Privacy Control</span>
                                </div>
                                <div className={styles.byokFeature}>
                                    <IconKey size={20} />
                                    <span>Your Own API Costs</span>
                                </div>
                                <div className={styles.byokFeature}>
                                    <IconBuilding size={20} />
                                    <span>Self-Hosted Option</span>
                                </div>
                            </div>
                        </div>
                        <Link href="/setup-keys" className={styles.byokBtn}>
                            Setup BYOK
                            <IconArrowRight size={20} />
                        </Link>
                    </div>
                </div>
            </section>

            {/* FAQ */}
            <section className={styles.faqSection}>
                <div className={styles.container}>
                    <h2 className={styles.faqTitle}>Frequently Asked Questions</h2>
                    <div className={styles.faqGrid}>
                        {faqs.map((faq, i) => (
                            <div key={i} className={styles.faqItem}>
                                <h3 className={styles.faqQuestion}>{faq.question}</h3>
                                <p className={styles.faqAnswer}>{faq.answer}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.ctaSection}>
                <div className={styles.container}>
                    <h2 className={styles.ctaTitle}>Ready to get started?</h2>
                    <p className={styles.ctaDesc}>Start your free account today. No credit card required.</p>
                    <Link href="/register" className={styles.primaryBtn}>
                        Create Free Account
                        <IconArrowRight size={20} />
                    </Link>
                </div>
            </section>
        </main>
    );
}
