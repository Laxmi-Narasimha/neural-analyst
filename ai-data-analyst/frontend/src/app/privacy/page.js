import Link from 'next/link';
import styles from '../terms/page.module.css';

export const metadata = {
    title: 'Privacy Policy - NeuralAnalyst',
    description: 'Privacy Policy for NeuralAnalyst. Learn how we collect, use, and protect your personal information.',
};

export default function PrivacyPage() {
    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <h1 className={styles.title}>Privacy Policy</h1>
                <p className={styles.lastUpdated}>Last updated: December 1, 2024</p>

                <div className={styles.content}>
                    <section className={styles.section}>
                        <h2>1. Introduction</h2>
                        <p>
                            NeuralAnalyst ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy
                            explains how we collect, use, disclose, and safeguard your information when you use our AI-powered
                            data analysis platform.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>2. Information We Collect</h2>
                        <p><strong>Account Information:</strong></p>
                        <ul>
                            <li>Name and email address</li>
                            <li>Company name (optional)</li>
                            <li>Password (encrypted)</li>
                            <li>Payment information (processed by Stripe)</li>
                        </ul>
                        <p><strong>Usage Data:</strong></p>
                        <ul>
                            <li>Analysis queries and results</li>
                            <li>Datasets you upload (for managed plans)</li>
                            <li>Feature usage and preferences</li>
                            <li>Log data and device information</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>3. BYOK (Bring Your Own Key) Users</h2>
                        <p>
                            If you use our BYOK option, your data processing occurs directly through your own API keys.
                            We do NOT store, process, or have access to:
                        </p>
                        <ul>
                            <li>Your datasets or analysis results</li>
                            <li>Your API keys (stored locally in your browser)</li>
                            <li>Any data sent to AI providers</li>
                        </ul>
                        <p>
                            BYOK provides maximum privacy as your data never touches our servers.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>4. How We Use Your Information</h2>
                        <ul>
                            <li>To provide and maintain the Service</li>
                            <li>To process your transactions</li>
                            <li>To send service-related communications</li>
                            <li>To improve and personalize the Service</li>
                            <li>To respond to your inquiries and support requests</li>
                            <li>To comply with legal obligations</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>5. Data Sharing</h2>
                        <p>We do NOT sell your personal information. We may share data with:</p>
                        <ul>
                            <li><strong>Service Providers:</strong> Stripe for payments, cloud providers for hosting</li>
                            <li><strong>AI Providers:</strong> OpenAI, Google, Anthropic (only for managed plan users)</li>
                            <li><strong>Legal Requirements:</strong> When required by law or to protect our rights</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>6. Data Security</h2>
                        <p>
                            We implement industry-standard security measures including:
                        </p>
                        <ul>
                            <li>256-bit SSL/TLS encryption in transit</li>
                            <li>AES-256 encryption at rest</li>
                            <li>SOC 2 Type II compliance (in progress)</li>
                            <li>Regular security audits and penetration testing</li>
                            <li>Access controls and employee training</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>7. Data Retention</h2>
                        <p>
                            We retain your data for as long as your account is active. Upon account deletion:
                        </p>
                        <ul>
                            <li>Personal data is deleted within 30 days</li>
                            <li>Datasets are deleted immediately</li>
                            <li>Anonymized analytics may be retained</li>
                            <li>Legal and compliance records may be retained as required</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>8. Your Rights</h2>
                        <p>Depending on your location, you may have the right to:</p>
                        <ul>
                            <li>Access your personal data</li>
                            <li>Correct inaccurate data</li>
                            <li>Delete your data</li>
                            <li>Export your data</li>
                            <li>Object to or restrict processing</li>
                            <li>Withdraw consent</li>
                        </ul>
                        <p>
                            To exercise these rights, contact us at <a href="mailto:privacy@neuralanalyst.ai">privacy@neuralanalyst.ai</a>.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>9. Cookies</h2>
                        <p>
                            We use cookies and similar technologies. See our <Link href="/cookies">Cookie Policy</Link> for details.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>10. International Transfers</h2>
                        <p>
                            Your data may be transferred to and processed in countries other than your own.
                            We ensure appropriate safeguards are in place, including Standard Contractual Clauses
                            for transfers from the EU.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>11. Children's Privacy</h2>
                        <p>
                            The Service is not intended for users under 16 years of age. We do not knowingly
                            collect data from children.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>12. Contact Us</h2>
                        <p>
                            For privacy inquiries, contact our Data Protection Officer:
                        </p>
                        <p>
                            Email: <a href="mailto:privacy@neuralanalyst.ai">privacy@neuralanalyst.ai</a><br />
                            Or visit our <Link href="/contact">Contact Page</Link>
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
