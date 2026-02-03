import Link from 'next/link';
import styles from '../terms/page.module.css';

export const metadata = {
    title: 'Cookie Policy - NeuralAnalyst',
    description: 'Cookie Policy for NeuralAnalyst. Learn about the cookies we use and how to manage your preferences.',
};

export default function CookiesPage() {
    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <h1 className={styles.title}>Cookie Policy</h1>
                <p className={styles.lastUpdated}>Last updated: December 1, 2024</p>

                <div className={styles.content}>
                    <section className={styles.section}>
                        <h2>1. What Are Cookies</h2>
                        <p>
                            Cookies are small text files stored on your device when you visit a website.
                            They help websites remember information about your visit, making your next visit
                            easier and the site more useful to you.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>2. How We Use Cookies</h2>
                        <p>We use cookies for the following purposes:</p>
                        <ul>
                            <li><strong>Essential Cookies:</strong> Required for the website to function (authentication, security)</li>
                            <li><strong>Functional Cookies:</strong> Remember your preferences and settings</li>
                            <li><strong>Analytics Cookies:</strong> Help us understand how visitors use our site</li>
                            <li><strong>Performance Cookies:</strong> Help us improve site speed and performance</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>3. Types of Cookies We Use</h2>
                        <p><strong>Essential Cookies:</strong></p>
                        <ul>
                            <li>Session cookies for authentication</li>
                            <li>Security cookies for CSRF protection</li>
                            <li>Load balancer cookies</li>
                        </ul>
                        <p><strong>Analytics Cookies:</strong></p>
                        <ul>
                            <li>Google Analytics (anonymized IP)</li>
                            <li>Internal usage analytics</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>4. Third-Party Cookies</h2>
                        <p>We may use cookies from these third-party services:</p>
                        <ul>
                            <li>Stripe (payment processing)</li>
                            <li>Google Analytics (anonymized)</li>
                            <li>Intercom (customer support)</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>5. Managing Cookies</h2>
                        <p>
                            You can control and manage cookies through your browser settings. Please note that
                            removing or blocking cookies may impact your user experience and parts of our website
                            may no longer be fully accessible.
                        </p>
                        <p>
                            To manage cookies in your browser:
                        </p>
                        <ul>
                            <li><strong>Chrome:</strong> Settings → Privacy and security → Cookies</li>
                            <li><strong>Firefox:</strong> Settings → Privacy & Security → Cookies</li>
                            <li><strong>Safari:</strong> Preferences → Privacy → Cookies</li>
                            <li><strong>Edge:</strong> Settings → Privacy → Cookies</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>6. Do Not Track</h2>
                        <p>
                            We respect Do Not Track (DNT) signals. When DNT is enabled, we disable non-essential analytics cookies.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>7. Updates to This Policy</h2>
                        <p>
                            We may update this Cookie Policy from time to time. Changes will be posted on this page
                            with an updated revision date.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>8. Contact Us</h2>
                        <p>
                            If you have questions about our use of cookies, please contact us at{' '}
                            <a href="mailto:privacy@neuralanalyst.ai">privacy@neuralanalyst.ai</a> or visit our{' '}
                            <Link href="/contact">Contact Page</Link>.
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
