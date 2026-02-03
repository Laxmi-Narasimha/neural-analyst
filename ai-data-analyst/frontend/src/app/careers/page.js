import Link from 'next/link';
import { IconArrowRight, IconUsers, IconHeart, IconGlobe, IconBrain } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Careers - Join the NeuralAnalyst Team',
    description: 'Join us in democratizing data analysis. We are building the future of AI-powered analytics.',
};

const openings = [
    { id: 1, title: 'Senior Backend Engineer', dept: 'Engineering', location: 'Remote', type: 'Full-time' },
    { id: 2, title: 'ML Engineer', dept: 'AI/ML', location: 'Remote', type: 'Full-time' },
    { id: 3, title: 'Frontend Engineer', dept: 'Engineering', location: 'Remote', type: 'Full-time' },
    { id: 4, title: 'Product Designer', dept: 'Design', location: 'Remote', type: 'Full-time' },
    { id: 5, title: 'DevOps Engineer', dept: 'Engineering', location: 'Remote', type: 'Full-time' },
    { id: 6, title: 'Technical Writer', dept: 'Product', location: 'Remote', type: 'Contract' },
];

const values = [
    { Icon: IconBrain, title: 'Innovation First', desc: 'We push the boundaries of what AI can do for data analysis.' },
    { Icon: IconUsers, title: 'Remote-First', desc: 'Work from anywhere. We trust our team to deliver great work.' },
    { Icon: IconHeart, title: 'User Obsessed', desc: 'Every feature starts with understanding user needs.' },
    { Icon: IconGlobe, title: 'Diverse & Inclusive', desc: 'We celebrate different perspectives and backgrounds.' },
];

const benefits = [
    'Competitive salary + equity',
    'Remote-first culture',
    'Unlimited PTO',
    'Health, dental, vision',
    '$1,000 home office stipend',
    'Learning & development budget',
    'Mental health support',
    'Flexible hours',
];

export default function CareersPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.heroGlow}></div>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Build the Future of <span className={styles.gradient}>Data Analysis</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Join our mission to make advanced analytics accessible to everyone.
                        We're a remote-first team building something extraordinary.
                    </p>
                    <a href="#openings" className={styles.heroBtn}>
                        View Open Positions
                        <IconArrowRight size={18} />
                    </a>
                </div>
            </section>

            {/* Values */}
            <section className={styles.values}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Our Values</h2>
                    <div className={styles.valuesGrid}>
                        {values.map((value, i) => (
                            <div key={i} className={styles.valueCard}>
                                <value.Icon size={28} className={styles.valueIcon} />
                                <h3>{value.title}</h3>
                                <p>{value.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Benefits */}
            <section className={styles.benefits}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Benefits & Perks</h2>
                    <div className={styles.benefitsGrid}>
                        {benefits.map((benefit, i) => (
                            <div key={i} className={styles.benefitItem}>
                                {benefit}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Openings */}
            <section id="openings" className={styles.openings}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Open Positions</h2>
                    <div className={styles.openingsList}>
                        {openings.map((job) => (
                            <Link key={job.id} href={`/careers/${job.id}`} className={styles.openingCard}>
                                <div className={styles.openingInfo}>
                                    <h3>{job.title}</h3>
                                    <div className={styles.openingMeta}>
                                        <span>{job.dept}</span>
                                        <span>{job.location}</span>
                                        <span>{job.type}</span>
                                    </div>
                                </div>
                                <IconArrowRight size={18} className={styles.openingArrow} />
                            </Link>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA */}
            <section className={styles.cta}>
                <div className={styles.container}>
                    <h2>Don't see a fit?</h2>
                    <p>We're always looking for talented people. Send us your resume.</p>
                    <a href="mailto:careers@neuralanalyst.ai" className={styles.ctaBtn}>
                        Get in Touch
                    </a>
                </div>
            </section>
        </main>
    );
}
