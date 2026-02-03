import Link from 'next/link';
import { IconArrowRight } from '@/components/icons';
import styles from '../page.module.css';

const jobsData = {
    '1': {
        title: 'Senior Backend Engineer',
        dept: 'Engineering',
        location: 'Remote (Worldwide)',
        type: 'Full-time',
        about: 'We are looking for a Senior Backend Engineer to help build our AI-powered data analysis platform. You will work on designing and implementing scalable APIs, data pipelines, and ML infrastructure.',
        responsibilities: [
            'Design and implement RESTful APIs using Python/FastAPI',
            'Build and optimize data processing pipelines',
            'Collaborate with ML engineers on model deployment',
            'Ensure code quality through testing and code reviews',
            'Mentor junior engineers and contribute to team growth',
        ],
        requirements: [
            '5+ years of backend development experience',
            'Strong Python experience (FastAPI, Django preferred)',
            'Experience with PostgreSQL and Redis',
            'Understanding of containerization (Docker, Kubernetes)',
            'Excellent communication skills',
        ],
        niceToHave: [
            'Experience with ML/AI systems',
            'Background in data engineering',
            'Open source contributions',
        ],
    },
    '2': {
        title: 'ML Engineer',
        dept: 'AI/ML',
        location: 'Remote (Worldwide)',
        type: 'Full-time',
        about: 'Join our ML team to build intelligent data analysis capabilities. You will work on AutoML, time series forecasting, and natural language understanding.',
        responsibilities: [
            'Develop and deploy machine learning models',
            'Build automated ML pipelines (AutoML)',
            'Implement model explainability features (SHAP, etc.)',
            'Optimize model performance and latency',
            'Stay current with ML research and best practices',
        ],
        requirements: [
            '3+ years of ML engineering experience',
            'Strong Python and scikit-learn/PyTorch experience',
            'Experience with tabular data and time series',
            'Understanding of statistical methods',
            'Experience deploying models to production',
        ],
        niceToHave: [
            'Experience with LLMs and NLP',
            'Publications or research background',
            'Experience with Ray or distributed computing',
        ],
    },
};

export async function generateStaticParams() {
    return Object.keys(jobsData).map((id) => ({ id }));
}

export async function generateMetadata({ params }) {
    const job = jobsData[params.id] || { title: 'Job Opening' };
    return {
        title: `${job.title} - Careers at NeuralAnalyst`,
        description: job.about?.substring(0, 160),
    };
}

export default function JobDetailPage({ params }) {
    const job = jobsData[params.id];

    if (!job) {
        return (
            <main className={styles.main}>
                <div className={styles.container}>
                    <h1>Position not found</h1>
                    <Link href="/careers">View all positions</Link>
                </div>
            </main>
        );
    }

    return (
        <main className={styles.main}>
            <section style={{ padding: '80px 24px', maxWidth: '800px', margin: '0 auto' }}>
                <Link href="/careers" style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-400)', marginBottom: '32px' }}>
                    <IconArrowRight size={14} style={{ transform: 'rotate(180deg)' }} />
                    Back to all positions
                </Link>

                <h1 style={{ fontSize: '2.5rem', fontWeight: 800, color: 'var(--text-50)', marginBottom: '16px' }}>{job.title}</h1>
                <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', fontSize: '14px', color: 'var(--text-500)' }}>
                    <span>{job.dept}</span>
                    <span>{job.location}</span>
                    <span>{job.type}</span>
                </div>

                <div style={{ color: 'var(--text-300)', lineHeight: 1.8 }}>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-50)', marginBottom: '16px' }}>About the Role</h2>
                    <p style={{ marginBottom: '32px' }}>{job.about}</p>

                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-50)', marginBottom: '16px' }}>Responsibilities</h2>
                    <ul style={{ marginBottom: '32px', paddingLeft: '24px' }}>
                        {job.responsibilities.map((item, i) => (
                            <li key={i} style={{ marginBottom: '8px' }}>{item}</li>
                        ))}
                    </ul>

                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-50)', marginBottom: '16px' }}>Requirements</h2>
                    <ul style={{ marginBottom: '32px', paddingLeft: '24px' }}>
                        {job.requirements.map((item, i) => (
                            <li key={i} style={{ marginBottom: '8px' }}>{item}</li>
                        ))}
                    </ul>

                    <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-50)', marginBottom: '16px' }}>Nice to Have</h2>
                    <ul style={{ marginBottom: '48px', paddingLeft: '24px' }}>
                        {job.niceToHave.map((item, i) => (
                            <li key={i} style={{ marginBottom: '8px' }}>{item}</li>
                        ))}
                    </ul>

                    <a
                        href={`mailto:careers@neuralanalyst.ai?subject=Application: ${job.title}`}
                        style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '16px 32px',
                            background: 'var(--gradient-primary)',
                            color: 'white',
                            fontWeight: 600,
                            borderRadius: '9999px',
                        }}
                    >
                        Apply Now
                        <IconArrowRight size={18} />
                    </a>
                </div>
            </section>
        </main>
    );
}
