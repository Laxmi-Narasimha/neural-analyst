'use client';

import Link from 'next/link';
import styles from './Pricing.module.css';

const plans = [
    {
        name: 'Free',
        price: '$0',
        period: 'forever',
        description: 'For individuals exploring data analysis',
        features: [
            '10 analyses per month',
            '100MB data storage',
            'Basic visualizations',
            'Community support',
        ],
        cta: 'Start free',
        href: '/register',
        popular: false,
    },
    {
        name: 'Pro',
        price: '$29',
        period: 'per month',
        description: 'For professionals and small teams',
        features: [
            'Unlimited analyses',
            '10GB data storage',
            'Advanced visualizations',
            'ML model building',
            'Priority support',
            'API access',
        ],
        cta: 'Start free trial',
        href: '/register?plan=pro',
        popular: true,
    },
    {
        name: 'Enterprise',
        price: 'Custom',
        period: 'per org',
        description: 'For organizations with advanced needs',
        features: [
            'Everything in Pro',
            'Unlimited storage',
            'SSO & SAML',
            'Dedicated support',
            'Custom integrations',
            'SLA guarantee',
        ],
        cta: 'Contact sales',
        href: '/contact',
        popular: false,
    },
];

export default function Pricing() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                <div className={styles.header}>
                    <h2 className={styles.title}>Simple, transparent pricing</h2>
                    <p className={styles.subtitle}>
                        Start free, scale as you grow. No hidden fees.
                    </p>
                </div>

                <div className={styles.cards}>
                    {plans.map((plan) => (
                        <div
                            key={plan.name}
                            className={`${styles.card} ${plan.popular ? styles.popular : ''}`}
                        >
                            {plan.popular && (
                                <span className={styles.popularBadge}>Most popular</span>
                            )}

                            <div className={styles.cardHeader}>
                                <h3 className={styles.planName}>{plan.name}</h3>
                                <div className={styles.priceRow}>
                                    <span className={styles.price}>{plan.price}</span>
                                    <span className={styles.period}>/{plan.period}</span>
                                </div>
                                <p className={styles.planDesc}>{plan.description}</p>
                            </div>

                            <ul className={styles.features}>
                                {plan.features.map((feature, index) => (
                                    <li key={index} className={styles.feature}>
                                        <svg className={styles.checkIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <polyline points="20 6 9 17 4 12" />
                                        </svg>
                                        {feature}
                                    </li>
                                ))}
                            </ul>

                            <Link
                                href={plan.href}
                                className={`${styles.cta} ${plan.popular ? styles.ctaPrimary : styles.ctaSecondary}`}
                            >
                                {plan.cta}
                            </Link>
                        </div>
                    ))}
                </div>

                <div className={styles.byok}>
                    <div className={styles.byokContent}>
                        <span className={styles.byokBadge}>BYOK</span>
                        <h3 className={styles.byokTitle}>Bring Your Own Keys</h3>
                        <p className={styles.byokDesc}>
                            Use your own API keys from OpenAI, Anthropic, or Google.
                            Your data, your keys, your control.
                        </p>
                    </div>
                    <Link href="/setup-keys" className={styles.byokCta}>
                        Learn more
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                </div>
            </div>
        </section>
    );
}
