import Link from 'next/link';
import { IconShield, IconLock, IconDatabase, IconCheck, IconArrowRight } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Security - Enterprise-Grade Protection',
    description: 'Learn about NeuralAnalyst security practices, compliance certifications, and data protection measures.',
};

const certifications = [
    { name: 'SOC 2 Type II', status: 'In Progress', desc: 'Enterprise security controls audit' },
    { name: 'GDPR', status: 'Compliant', desc: 'EU data protection compliance' },
    { name: 'CCPA', status: 'Compliant', desc: 'California privacy law' },
    { name: 'HIPAA', status: 'Enterprise Plan', desc: 'Healthcare data protection' },
];

const practices = [
    { icon: IconLock, title: 'Encryption', desc: 'AES-256 encryption at rest, TLS 1.3 in transit' },
    { icon: IconShield, title: 'Access Control', desc: 'Role-based permissions, SSO, and MFA' },
    { icon: IconDatabase, title: 'Data Isolation', desc: 'Tenant isolation and dedicated infrastructure options' },
];

const features = [
    '256-bit AES encryption at rest',
    'TLS 1.3 encryption in transit',
    'SOC 2 Type II audit (in progress)',
    'GDPR and CCPA compliant',
    'SSO via SAML/OIDC',
    'Role-based access control',
    'Audit logging',
    'Automatic backups',
    'DDoS protection',
    '99.9% uptime SLA',
    'Vulnerability scanning',
    'Penetration testing',
];

export default function SecurityPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <div className={styles.heroIcon}>
                        <IconShield size={40} />
                    </div>
                    <h1 className={styles.heroTitle}>
                        Enterprise-Grade <span className={styles.gradient}>Security</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Your data is protected by industry-leading security practices.
                        We take security seriously so you can focus on insights.
                    </p>
                </div>
            </section>

            {/* Practices */}
            <section className={styles.practices}>
                <div className={styles.container}>
                    <div className={styles.practicesGrid}>
                        {practices.map((practice, i) => (
                            <div key={i} className={styles.practiceCard}>
                                <practice.icon size={32} className={styles.practiceIcon} />
                                <h3>{practice.title}</h3>
                                <p>{practice.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* BYOK */}
            <section className={styles.byok}>
                <div className={styles.container}>
                    <div className={styles.byokCard}>
                        <h2>Maximum Privacy with BYOK</h2>
                        <p>
                            With Bring Your Own Key mode, your data never touches our servers.
                            API keys are stored locally, and all processing happens through your
                            own provider accounts. Complete privacy, complete control.
                        </p>
                        <Link href="/setup-keys" className={styles.byokBtn}>
                            Learn More About BYOK
                            <IconArrowRight size={18} />
                        </Link>
                    </div>
                </div>
            </section>

            {/* Certifications */}
            <section className={styles.certifications}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Compliance & Certifications</h2>
                    <div className={styles.certGrid}>
                        {certifications.map((cert, i) => (
                            <div key={i} className={styles.certCard}>
                                <h3>{cert.name}</h3>
                                <span className={`${styles.certStatus} ${cert.status === 'Compliant' ? styles.compliant : ''}`}>
                                    {cert.status}
                                </span>
                                <p>{cert.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className={styles.features}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Security Features</h2>
                    <div className={styles.featuresGrid}>
                        {features.map((feature, i) => (
                            <div key={i} className={styles.featureItem}>
                                <IconCheck size={16} className={styles.featureCheck} />
                                {feature}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Contact */}
            <section className={styles.contact}>
                <div className={styles.container}>
                    <h2>Have Security Questions?</h2>
                    <p>Contact our security team for detailed information or to request our security documentation.</p>
                    <a href="mailto:security@neuralanalyst.ai" className={styles.contactBtn}>
                        Contact Security Team
                    </a>
                </div>
            </section>
        </main>
    );
}
