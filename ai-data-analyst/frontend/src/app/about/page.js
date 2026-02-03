import Link from 'next/link';
import { IconArrowRight, IconHeart, IconStar, IconAward, IconUsers, IconGlobe, IconShield, IconBrain } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'About - Our Mission to Democratize Data Analysis',
    description: 'Learn about NeuralAnalyst, our mission to make advanced data analysis accessible to everyone, and the team behind the technology.',
    openGraph: {
        title: 'About NeuralAnalyst',
        description: 'Our mission: Make advanced data analysis accessible to every team.',
    },
};

const values = [
    { Icon: IconShield, title: 'Privacy First', desc: 'Your data stays yours. BYOK ensures complete control over your information.' },
    { Icon: IconBrain, title: 'AI Excellence', desc: '244+ features refined through millions of analyses and continuous improvement.' },
    { Icon: IconUsers, title: 'Accessibility', desc: 'No coding required. Natural language queries make data analysis for everyone.' },
    { Icon: IconGlobe, title: 'Open Source', desc: 'Transparent development. Contribute, audit, and customize to your needs.' },
];

const milestones = [
    { year: '2023', title: 'Founded', desc: 'Started with a vision to democratize data analysis' },
    { year: '2024', title: 'Launch', desc: 'Public launch with 100+ features and BYOK support' },
    { year: '2024', title: '10K Users', desc: 'Reached 10,000 active users across 50 countries' },
    { year: '2025', title: '244+ Features', desc: 'Expanded to enterprise-grade capability set' },
];

export default function AboutPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Data Analysis for <span className={styles.gradient}>Everyone</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        We believe every team deserves access to enterprise-grade data analysis,
                        regardless of their technical expertise or budget.
                    </p>
                </div>
            </section>

            {/* Mission */}
            <section className={styles.mission}>
                <div className={styles.container}>
                    <div className={styles.missionContent}>
                        <h2 className={styles.sectionTitle}>Our Mission</h2>
                        <p className={styles.missionText}>
                            The gap between data and insights shouldn't require a team of data scientists.
                            We're building AI that understands your questions, analyzes your data, and delivers
                            actionable insights — all through natural conversation.
                        </p>
                        <p className={styles.missionText}>
                            NeuralAnalyst combines cutting-edge machine learning with intuitive design,
                            making professional-grade analysis accessible to marketing teams, researchers,
                            small businesses, and enterprises alike.
                        </p>
                    </div>
                </div>
            </section>

            {/* Values */}
            <section className={styles.values}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>What We Stand For</h2>
                    <div className={styles.valuesGrid}>
                        {values.map((value, i) => (
                            <div key={i} className={styles.valueCard}>
                                <div className={styles.valueIcon}>
                                    <value.Icon size={24} />
                                </div>
                                <h3 className={styles.valueTitle}>{value.title}</h3>
                                <p className={styles.valueDesc}>{value.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Timeline */}
            <section className={styles.timeline}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Our Journey</h2>
                    <div className={styles.timelineTrack}>
                        {milestones.map((milestone, i) => (
                            <div key={i} className={styles.milestone}>
                                <div className={styles.milestoneYear}>{milestone.year}</div>
                                <div className={styles.milestoneContent}>
                                    <h3 className={styles.milestoneTitle}>{milestone.title}</h3>
                                    <p className={styles.milestoneDesc}>{milestone.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Stats */}
            <section className={styles.stats}>
                <div className={styles.container}>
                    <div className={styles.statsGrid}>
                        <div className={styles.statCard}>
                            <div className={styles.statValue}>244+</div>
                            <div className={styles.statLabel}>Features</div>
                        </div>
                        <div className={styles.statCard}>
                            <div className={styles.statValue}>10K+</div>
                            <div className={styles.statLabel}>Users</div>
                        </div>
                        <div className={styles.statCard}>
                            <div className={styles.statValue}>50+</div>
                            <div className={styles.statLabel}>Countries</div>
                        </div>
                        <div className={styles.statCard}>
                            <div className={styles.statValue}>99.9%</div>
                            <div className={styles.statLabel}>Uptime</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <h2 className={styles.ctaTitle}>Join the Data Revolution</h2>
                    <p className={styles.ctaDesc}>Start analyzing your data today — no credit card required.</p>
                    <div className={styles.ctaButtons}>
                        <Link href="/register" className={styles.primaryBtn}>
                            Get Started Free
                            <IconArrowRight size={20} />
                        </Link>
                        <Link href="/features" className={styles.secondaryBtn}>
                            Explore Features
                        </Link>
                    </div>
                </div>
            </section>
        </main>
    );
}
