import Link from 'next/link';
import { IconArrowRight, IconGlobe, IconCode, IconDatabase } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Partners - NeuralAnalyst Partner Program',
    description: 'Join the NeuralAnalyst partner program. Technology partners, consultants, and resellers welcome.',
};

const partnerTypes = [
    {
        Icon: IconCode,
        title: 'Technology Partners',
        desc: 'Build integrations with NeuralAnalyst. Access our API, get co-marketing opportunities, and be featured in our marketplace.',
        benefits: ['API access', 'Technical support', 'Co-marketing', 'Marketplace listing'],
    },
    {
        Icon: IconDatabase,
        title: 'Consulting Partners',
        desc: 'Help clients implement and get the most from NeuralAnalyst. Get certified, access deal registration, and earn referral fees.',
        benefits: ['Partner certification', 'Deal registration', 'Referral commissions', 'Training resources'],
    },
    {
        Icon: IconGlobe,
        title: 'Resellers',
        desc: 'Sell NeuralAnalyst to your customers. Get volume discounts, white-label options, and dedicated support.',
        benefits: ['Volume discounts', 'White-label options', 'Sales support', 'Partner portal'],
    },
];

const partners = [
    'Snowflake', 'Databricks', 'AWS', 'Google Cloud', 'Microsoft Azure',
    'Tableau', 'Power BI', 'Slack', 'Stripe', 'Auth0',
];

export default function PartnersPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Partner with <span className={styles.gradient}>NeuralAnalyst</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Join our partner ecosystem and grow your business with AI-powered analytics.
                    </p>
                    <a href="/contact?type=partnership" className={styles.heroBtn}>
                        Become a Partner
                        <IconArrowRight size={18} />
                    </a>
                </div>
            </section>

            {/* Partner Types */}
            <section className={styles.types}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Partnership Programs</h2>
                    <div className={styles.typesGrid}>
                        {partnerTypes.map((type, i) => (
                            <div key={i} className={styles.typeCard}>
                                <type.Icon size={32} className={styles.typeIcon} />
                                <h3>{type.title}</h3>
                                <p>{type.desc}</p>
                                <ul className={styles.benefitsList}>
                                    {type.benefits.map((benefit, j) => (
                                        <li key={j}>{benefit}</li>
                                    ))}
                                </ul>
                                <Link href="/contact?type=partnership" className={styles.typeBtn}>
                                    Learn More
                                </Link>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Current Partners */}
            <section className={styles.current}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Our Partners</h2>
                    <div className={styles.partnerLogos}>
                        {partners.map((partner, i) => (
                            <div key={i} className={styles.partnerLogo}>
                                {partner}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <div className={styles.ctaCard}>
                        <h2>Ready to Partner?</h2>
                        <p>Contact us to learn more about partnership opportunities.</p>
                        <a href="mailto:partners@neuralanalyst.ai" className={styles.ctaBtn}>
                            partners@neuralanalyst.ai
                        </a>
                    </div>
                </div>
            </section>
        </main>
    );
}
