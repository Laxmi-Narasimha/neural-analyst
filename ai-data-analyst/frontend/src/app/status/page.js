import { IconCheck, IconTime, IconAlertTriangle } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'System Status - NeuralAnalyst',
    description: 'Check the current status of NeuralAnalyst services and view incident history.',
};

const services = [
    { name: 'Web Application', status: 'operational', uptime: '99.99%' },
    { name: 'API', status: 'operational', uptime: '99.98%' },
    { name: 'Analysis Engine', status: 'operational', uptime: '99.97%' },
    { name: 'Database', status: 'operational', uptime: '99.99%' },
    { name: 'File Storage', status: 'operational', uptime: '99.99%' },
    { name: 'Authentication', status: 'operational', uptime: '99.99%' },
];

const incidents = [
    {
        date: 'Nov 28, 2024',
        title: 'Scheduled Maintenance',
        status: 'resolved',
        desc: 'Planned infrastructure upgrade. All services maintained during maintenance window.',
    },
    {
        date: 'Nov 15, 2024',
        title: 'API Latency',
        status: 'resolved',
        desc: 'Elevated API response times for ~15 minutes. Root cause identified and fixed.',
    },
    {
        date: 'Oct 22, 2024',
        title: 'File Upload Issues',
        status: 'resolved',
        desc: 'Some users experienced file upload failures. Issue resolved within 30 minutes.',
    },
];

const uptimeData = [
    { day: 'Mon', status: 100 },
    { day: 'Tue', status: 100 },
    { day: 'Wed', status: 100 },
    { day: 'Thu', status: 99.9 },
    { day: 'Fri', status: 100 },
    { day: 'Sat', status: 100 },
    { day: 'Sun', status: 100 },
];

export default function StatusPage() {
    const allOperational = services.every(s => s.status === 'operational');

    return (
        <main className={styles.main}>
            {/* Status Banner */}
            <section className={styles.banner}>
                <div className={styles.container}>
                    <div className={`${styles.statusCard} ${allOperational ? styles.operational : styles.degraded}`}>
                        <div className={styles.statusIcon}>
                            {allOperational ? <IconCheck size={32} /> : <IconAlertTriangle size={32} />}
                        </div>
                        <h1>{allOperational ? 'All Systems Operational' : 'Some Systems Degraded'}</h1>
                        <p>Last updated: {new Date().toLocaleString()}</p>
                    </div>
                </div>
            </section>

            {/* Uptime Chart */}
            <section className={styles.uptime}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>7-Day Uptime</h2>
                    <div className={styles.uptimeChart}>
                        {uptimeData.map((day, i) => (
                            <div key={i} className={styles.uptimeDay}>
                                <div
                                    className={`${styles.uptimeBar} ${day.status === 100 ? styles.full : styles.partial}`}
                                    style={{ height: `${day.status}%` }}
                                />
                                <span className={styles.dayLabel}>{day.day}</span>
                            </div>
                        ))}
                    </div>
                    <div className={styles.uptimeStats}>
                        <span>Overall: <strong>99.99%</strong></span>
                    </div>
                </div>
            </section>

            {/* Services */}
            <section className={styles.services}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Service Status</h2>
                    <div className={styles.servicesList}>
                        {services.map((service, i) => (
                            <div key={i} className={styles.serviceItem}>
                                <div className={styles.serviceInfo}>
                                    <span className={styles.serviceName}>{service.name}</span>
                                    <span className={styles.serviceUptime}>{service.uptime} uptime</span>
                                </div>
                                <span className={`${styles.serviceStatus} ${styles[service.status]}`}>
                                    <span className={styles.statusDot}></span>
                                    {service.status.charAt(0).toUpperCase() + service.status.slice(1)}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Incidents */}
            <section className={styles.incidents}>
                <div className={styles.container}>
                    <h2 className={styles.sectionTitle}>Incident History</h2>
                    <div className={styles.incidentsList}>
                        {incidents.map((incident, i) => (
                            <div key={i} className={styles.incidentItem}>
                                <div className={styles.incidentHeader}>
                                    <span className={styles.incidentDate}>{incident.date}</span>
                                    <span className={`${styles.incidentStatus} ${styles[incident.status]}`}>
                                        {incident.status}
                                    </span>
                                </div>
                                <h3>{incident.title}</h3>
                                <p>{incident.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Subscribe */}
            <section className={styles.subscribe}>
                <div className={styles.container}>
                    <h2>Get Status Updates</h2>
                    <p>Subscribe to receive notifications about service disruptions.</p>
                    <form className={styles.subscribeForm}>
                        <input type="email" placeholder="your@email.com" />
                        <button type="submit">Subscribe</button>
                    </form>
                </div>
            </section>
        </main>
    );
}
