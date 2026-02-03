import Link from 'next/link';
import styles from './page.module.css';

export const metadata = {
    title: 'Terms of Service - NeuralAnalyst',
    description: 'Terms of Service for NeuralAnalyst. Read our terms and conditions for using our AI data analysis platform.',
};

export default function TermsPage() {
    return (
        <main className={styles.main}>
            <div className={styles.container}>
                <h1 className={styles.title}>Terms of Service</h1>
                <p className={styles.lastUpdated}>Last updated: December 1, 2024</p>

                <div className={styles.content}>
                    <section className={styles.section}>
                        <h2>1. Acceptance of Terms</h2>
                        <p>
                            By accessing or using NeuralAnalyst ("Service"), you agree to be bound by these Terms of Service ("Terms").
                            If you do not agree to these Terms, please do not use the Service. These Terms apply to all visitors, users,
                            and others who access or use the Service.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>2. Description of Service</h2>
                        <p>
                            NeuralAnalyst provides an AI-powered data analysis platform that enables users to analyze datasets,
                            generate insights, build machine learning models, and create visualizations. The Service includes
                            both managed hosting options and BYOK (Bring Your Own Key) self-hosted options.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>3. User Accounts</h2>
                        <p>
                            To access certain features of the Service, you must register for an account. You agree to:
                        </p>
                        <ul>
                            <li>Provide accurate, current, and complete information during registration</li>
                            <li>Maintain and promptly update your account information</li>
                            <li>Maintain the security of your password and accept all risks of unauthorized access</li>
                            <li>Immediately notify us if you discover any unauthorized use of your account</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>4. Acceptable Use</h2>
                        <p>You agree not to use the Service to:</p>
                        <ul>
                            <li>Violate any applicable laws or regulations</li>
                            <li>Infringe upon the rights of others</li>
                            <li>Upload malicious code, viruses, or harmful data</li>
                            <li>Attempt to gain unauthorized access to our systems</li>
                            <li>Interfere with the proper working of the Service</li>
                            <li>Use the Service for competitive analysis or benchmarking</li>
                        </ul>
                    </section>

                    <section className={styles.section}>
                        <h2>5. Data and Privacy</h2>
                        <p>
                            Your use of the Service is also governed by our <Link href="/privacy">Privacy Policy</Link>.
                            You retain all rights to your data. We process your data only to provide the Service.
                            For BYOK users, data processing occurs through your own API keys and we do not store your data.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>6. Intellectual Property</h2>
                        <p>
                            The Service and its original content, features, and functionality are owned by NeuralAnalyst
                            and are protected by international copyright, trademark, patent, trade secret, and other
                            intellectual property laws. You retain ownership of any data you upload to the Service.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>7. Payment Terms</h2>
                        <p>
                            For paid plans, you agree to pay all fees according to the pricing plan you select.
                            Fees are non-refundable except as required by law. We may change our fees with 30 days notice.
                            Free tier users are subject to usage limits as specified on our pricing page.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>8. Termination</h2>
                        <p>
                            We may terminate or suspend your account at any time for violation of these Terms.
                            Upon termination, your right to use the Service will immediately cease.
                            You may export your data before termination. We will retain your data for 30 days
                            after account termination.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>9. Disclaimer of Warranties</h2>
                        <p>
                            THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND. WE DO NOT WARRANT
                            THAT THE SERVICE WILL BE UNINTERRUPTED, SECURE, OR ERROR-FREE. AI-GENERATED INSIGHTS
                            SHOULD BE VERIFIED BY QUALIFIED PROFESSIONALS BEFORE MAKING BUSINESS DECISIONS.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>10. Limitation of Liability</h2>
                        <p>
                            TO THE MAXIMUM EXTENT PERMITTED BY LAW, NEURALANALYST SHALL NOT BE LIABLE FOR ANY
                            INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES RESULTING FROM
                            YOUR USE OF THE SERVICE.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>11. Changes to Terms</h2>
                        <p>
                            We reserve the right to modify these Terms at any time. We will notify users of any
                            material changes via email or through the Service. Continued use of the Service after
                            changes constitutes acceptance of the new Terms.
                        </p>
                    </section>

                    <section className={styles.section}>
                        <h2>12. Contact Us</h2>
                        <p>
                            If you have any questions about these Terms, please contact us at:
                        </p>
                        <p>
                            Email: <a href="mailto:legal@neuralanalyst.ai">legal@neuralanalyst.ai</a><br />
                            Or visit our <Link href="/contact">Contact Page</Link>
                        </p>
                    </section>
                </div>
            </div>
        </main>
    );
}
