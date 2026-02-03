import Link from 'next/link';
import { IconArrowRight, IconDatabase, IconCode, IconChart, IconGlobe } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Integrations - Connect Your Data Stack',
    description: 'Connect NeuralAnalyst with your existing tools: databases, BI platforms, data warehouses, and more.',
};

const categories = [
    {
        name: 'Databases',
        integrations: [
            { id: 'postgresql', name: 'PostgreSQL', desc: 'Connect to PostgreSQL databases', status: 'available' },
            { id: 'mysql', name: 'MySQL', desc: 'MySQL and MariaDB support', status: 'available' },
            { id: 'mongodb', name: 'MongoDB', desc: 'Document database integration', status: 'available' },
            { id: 'sqlserver', name: 'SQL Server', desc: 'Microsoft SQL Server', status: 'available' },
            { id: 'sqlite', name: 'SQLite', desc: 'Local SQLite databases', status: 'available' },
            { id: 'oracle', name: 'Oracle', desc: 'Oracle Database', status: 'coming' },
        ],
    },
    {
        name: 'Data Warehouses',
        integrations: [
            { id: 'snowflake', name: 'Snowflake', desc: 'Cloud data warehouse', status: 'available' },
            { id: 'bigquery', name: 'BigQuery', desc: 'Google BigQuery', status: 'available' },
            { id: 'redshift', name: 'Redshift', desc: 'Amazon Redshift', status: 'available' },
            { id: 'databricks', name: 'Databricks', desc: 'Lakehouse platform', status: 'available' },
            { id: 'synapse', name: 'Azure Synapse', desc: 'Microsoft analytics', status: 'coming' },
        ],
    },
    {
        name: 'BI & Visualization',
        integrations: [
            { id: 'tableau', name: 'Tableau', desc: 'Export to Tableau', status: 'available' },
            { id: 'powerbi', name: 'Power BI', desc: 'Microsoft Power BI', status: 'available' },
            { id: 'looker', name: 'Looker', desc: 'Google Looker', status: 'coming' },
            { id: 'metabase', name: 'Metabase', desc: 'Open source BI', status: 'available' },
        ],
    },
    {
        name: 'Cloud Storage',
        integrations: [
            { id: 's3', name: 'Amazon S3', desc: 'AWS object storage', status: 'available' },
            { id: 'gcs', name: 'Google Cloud Storage', desc: 'GCP storage', status: 'available' },
            { id: 'azure-blob', name: 'Azure Blob', desc: 'Azure storage', status: 'available' },
            { id: 'dropbox', name: 'Dropbox', desc: 'Dropbox integration', status: 'coming' },
        ],
    },
    {
        name: 'APIs & Tools',
        integrations: [
            { id: 'rest-api', name: 'REST API', desc: 'Full REST API access', status: 'available' },
            { id: 'webhooks', name: 'Webhooks', desc: 'Event notifications', status: 'available' },
            { id: 'jupyter', name: 'Jupyter', desc: 'Notebook integration', status: 'available' },
            { id: 'slack', name: 'Slack', desc: 'Slack notifications', status: 'available' },
            { id: 'zapier', name: 'Zapier', desc: 'Workflow automation', status: 'coming' },
        ],
    },
];

export default function IntegrationsPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        <span className={styles.gradient}>Integrations</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Connect NeuralAnalyst to your existing data stack. Import from anywhere,
                        export to your favorite tools.
                    </p>
                </div>
            </section>

            {/* Categories */}
            <section className={styles.categories}>
                <div className={styles.container}>
                    {categories.map((category) => (
                        <div key={category.name} className={styles.category}>
                            <h2 className={styles.categoryTitle}>{category.name}</h2>
                            <div className={styles.integrationsGrid}>
                                {category.integrations.map((integration) => (
                                    <div key={integration.id} className={styles.integrationCard}>
                                        <div className={styles.integrationIcon}>
                                            <IconDatabase size={24} />
                                        </div>
                                        <div className={styles.integrationContent}>
                                            <h3>{integration.name}</h3>
                                            <p>{integration.desc}</p>
                                        </div>
                                        {integration.status === 'coming' ? (
                                            <span className={styles.comingSoon}>Coming Soon</span>
                                        ) : (
                                            <span className={styles.available}>Available</span>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* API */}
            <section className={styles.apiSection}>
                <div className={styles.container}>
                    <div className={styles.apiCard}>
                        <div className={styles.apiContent}>
                            <IconCode size={32} className={styles.apiIcon} />
                            <h2>Build Custom Integrations</h2>
                            <p>
                                Use our comprehensive REST API to build custom integrations
                                with any tool in your stack.
                            </p>
                            <Link href="/docs/api" className={styles.apiBtn}>
                                View API Docs
                                <IconArrowRight size={18} />
                            </Link>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    );
}
